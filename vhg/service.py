from __future__ import annotations

import argparse
import asyncio
import multiprocessing
import os
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from multiprocessing import Queue
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .verify import analyze_service_item, correctness, ensure_latex_parser_available


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}", flush=True)


class CorrectnessClient:
    def __init__(self, service_url: str = "http://localhost:5000"):
        self.service_url = service_url.rstrip("/")
        self._session = requests.Session()
        host = (urlparse(self.service_url).hostname or "").lower()
        if host in {"localhost", "127.0.0.1", "::1"}:
            self._session.trust_env = False

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def health_check(self) -> bool:
        try:
            response = self._session.get(f"{self.service_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def check_batch(self, items: list[dict[str, Any]]) -> list[bool]:
        req_timeout = len(items) / 50 + 30
        try:
            response = self._session.post(
                f"{self.service_url}/check_batch",
                json={"items": items},
                timeout=req_timeout,
            )
            response.raise_for_status()
            return response.json().get("results", [False] * len(items))
        except Exception as exc:
            print(f"Error in check_batch: {exc}")
            return [False] * len(items)

    def analyze_batch(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        req_timeout = len(items) / 50 + 30
        response = self._session.post(
            f"{self.service_url}/analyze_batch",
            json={"items": items},
            timeout=req_timeout,
        )
        response.raise_for_status()
        return response.json().get("results", [{} for _ in items])


def wait_for_correctness_service(
    service_url: str = "http://localhost:5000",
    *,
    process: Any | None = None,
    timeout_seconds: float = 120.0,
    poll_interval_seconds: float = 2.0,
) -> None:
    deadline = time.time() + timeout_seconds
    with CorrectnessClient(service_url) as client:
        while True:
            if client.health_check():
                return
            if process is not None and process.poll() is not None:
                raise RuntimeError(
                    "Standalone correctness service exited before becoming healthy."
                )
            if time.time() >= deadline:
                raise RuntimeError(
                    f"Timed out waiting for standalone correctness service at {service_url}."
                )
            time.sleep(poll_interval_seconds)


@contextmanager
def correctness_service(
    port: int | None = None,
    *,
    pool_size: int = 450,
    shutdown_timeout: float = 10.0,
):
    """Run the Hard verifier for one pipeline operation and always clean it up."""
    ensure_latex_parser_available()
    previous_port = os.environ.get("STANDALONE_SERVICE_PORT")
    port = int(previous_port or "5000") if port is None else port
    url = f"http://localhost:{port}"
    os.environ["STANDALONE_SERVICE_PORT"] = str(port)
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "vhg.service", "--host", "localhost",
             "--port", str(port), "--pool_size", str(pool_size), "--timeout", "15.0"],
            cwd=Path(__file__).resolve().parents[1],
        )
        try:
            wait_for_correctness_service(url, process=process)
            yield url
        finally:
            process.terminate()
            try:
                process.wait(timeout=shutdown_timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    finally:
        if previous_port is None:
            os.environ.pop("STANDALONE_SERVICE_PORT", None)
        else:
            os.environ["STANDALONE_SERVICE_PORT"] = previous_port


def _service_operation(operation: str, item: dict[str, Any]):
    if operation == "correctness":
        return correctness(item["question"], item["question_type"], item["solution"])
    if operation == "analysis":
        return analyze_service_item(item)
    raise ValueError(f"Unknown service operation: {operation}")


def _worker_function_in_process(operation: str, item: dict[str, Any], conn, index: int):
    try:
        result = _service_operation(operation, item)
        conn.send((index, result))
    except Exception as exc:
        _log(f"Worker process {os.getpid()} error: {exc}")
        conn.send((index, False if operation == "correctness" else {}))
    finally:
        conn.close()


def _worker_manager_loop(
    task_queue: Queue,
    result_queue: Queue,
    timeout: float,
    manager_id: int,
) -> None:
    ctx = multiprocessing.get_context("fork")
    _log(f"[Manager {manager_id}] Started, PID={os.getpid()}")

    while True:
        task_id = None
        operation = "correctness"
        try:
            task = task_queue.get()
            if task is None:
                _log(f"[Manager {manager_id}] Received shutdown signal")
                break

            task_id, operation, item = task
            parent_conn, child_conn = ctx.Pipe()
            process = ctx.Process(
                target=_worker_function_in_process,
                args=(operation, item, child_conn, task_id),
            )
            process.start()
            child_conn.close()

            if parent_conn.poll(timeout):
                try:
                    _, result = parent_conn.recv()
                    result_queue.put((task_id, result))
                except EOFError:
                    result_queue.put(
                        (task_id, False if operation == "correctness" else {})
                    )
                process.join()
            else:
                _log(
                    f"[Manager {manager_id}] Task {task_id} TIMEOUT, killing PID {process.pid}"
                )
                process.terminate()
                process.join(timeout=1)
                if process.is_alive():
                    process.kill()
                    process.join()
                result_queue.put((task_id, False if operation == "correctness" else {}))

            parent_conn.close()
        except Exception as exc:
            _log(f"[Manager {manager_id}] Error: {exc}")
            if task_id is not None:
                result_queue.put((task_id, False if operation == "correctness" else {}))


class CorrectnessService:
    def __init__(
        self,
        pool_size: int = 450,
        timeout: float = 15.0,
        *,
        skip_manager_init: bool = False,
    ):
        self.pool_size = pool_size
        self.timeout = timeout
        self._mp_context = multiprocessing.get_context("fork")
        self._task_queue = None
        self._result_queue = None
        self._managers = []
        self._pending_tasks = {}
        self._results = {}
        self._task_counter = 0
        self._lock = threading.Lock()
        self._result_collector_thread = None
        if not skip_manager_init:
            self._init_managers()

    def _init_managers(self) -> None:
        self._task_queue = self._mp_context.Queue()
        self._result_queue = self._mp_context.Queue()
        self._managers = []
        for idx in range(self.pool_size):
            process = self._mp_context.Process(
                target=_worker_manager_loop,
                args=(self._task_queue, self._result_queue, self.timeout, idx),
            )
            process.start()
            self._managers.append(process)
        self._start_collector()
        _log(f"[Service] Started {self.pool_size} worker managers")

    def attach_preforked_managers(self, managers, task_queue, result_queue) -> None:
        self._managers = managers
        self._task_queue = task_queue
        self._result_queue = result_queue
        self._start_collector()
        _log(f"[Service] Using {len(managers)} pre-forked worker managers")

    def _start_collector(self) -> None:
        self._result_collector_thread = threading.Thread(
            target=self._collect_results,
            daemon=True,
        )
        self._result_collector_thread.start()

    def _collect_results(self) -> None:
        while True:
            try:
                task_id, result = self._result_queue.get()
                with self._lock:
                    self._results[task_id] = result
                    event = self._pending_tasks.get(task_id)
                    if event is not None:
                        event.set()
            except Exception as exc:
                _log(f"[Collector] Error: {exc}")

    async def process_batch_async(self, items: list[dict[str, Any]], operation: str) -> list:
        task_ids = []
        done_events = []
        with self._lock:
            for _ in items:
                task_id = self._task_counter
                self._task_counter += 1
                done_event = threading.Event()
                self._pending_tasks[task_id] = done_event
                task_ids.append(task_id)
                done_events.append(done_event)

        for task_id, item in zip(task_ids, items):
            self._task_queue.put((task_id, operation, item))

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            self._wait_for_results,
            task_ids,
            done_events,
        )
        return results

    async def check_batch_async(self, items: list[dict[str, Any]]) -> list[bool]:
        return await self.process_batch_async(items, "correctness")

    async def analyze_batch_async(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return await self.process_batch_async(items, "analysis")

    def _wait_for_results(self, task_ids: list[int], done_events: list[threading.Event]) -> list:
        for event in done_events:
            event.wait()

        results = []
        with self._lock:
            for task_id in task_ids:
                self._pending_tasks.pop(task_id, None)
                results.append(self._results.pop(task_id, False))
        return results

    def shutdown(self) -> None:
        _log("[Service] Shutting down managers...")
        if self._task_queue is not None:
            for _ in self._managers:
                self._task_queue.put(None)
        for idx, process in enumerate(self._managers):
            process.join(timeout=5)
            if process.is_alive():
                _log(f"[Service] Force killing manager {idx}")
                process.terminate()
                process.join(timeout=1)
                if process.is_alive():
                    process.kill()


service: CorrectnessService | None = None
preforked_managers = None
preforked_task_queue = None
preforked_result_queue = None


def _prefork_managers(pool_size: int, timeout: float) -> None:
    global preforked_managers, preforked_task_queue, preforked_result_queue
    ctx = multiprocessing.get_context("fork")
    preforked_task_queue = ctx.Queue()
    preforked_result_queue = ctx.Queue()
    preforked_managers = []
    for idx in range(pool_size):
        process = ctx.Process(
            target=_worker_manager_loop,
            args=(preforked_task_queue, preforked_result_queue, timeout, idx),
        )
        process.start()
        preforked_managers.append(process)
    _log(f"[Main] Pre-forked {pool_size} worker managers")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service, preforked_managers, preforked_task_queue, preforked_result_queue
    pool_size = int(os.environ.get("CORRECTNESS_SERVICE_POOL_SIZE", "450"))
    timeout = float(os.environ.get("CORRECTNESS_SERVICE_TIMEOUT", "15.0"))
    service = CorrectnessService(
        pool_size=pool_size,
        timeout=timeout,
        skip_manager_init=True,
    )
    if preforked_managers is not None:
        service.attach_preforked_managers(
            preforked_managers,
            preforked_task_queue,
            preforked_result_queue,
        )
    else:
        _log("[Service] No pre-forked managers found; initializing in lifespan")
        service._init_managers()
    _log(
        f"Correctness service started with pool_size={pool_size}, timeout={timeout}s"
    )
    yield
    if service is not None:
        service.shutdown()
    _log("Correctness service stopped")


app = FastAPI(
    title="VHG Correctness Checking Service",
    description="Standalone verifier service for VHG hard verifier checks and analysis.",
    version="1.0.0",
    lifespan=lifespan,
)


class BatchCheckRequest(BaseModel):
    items: list[dict[str, Any]]


class BatchCheckResponse(BaseModel):
    results: list[bool]


class BatchAnalyzeRequest(BaseModel):
    items: list[dict[str, Any]]


class BatchAnalyzeResponse(BaseModel):
    results: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    service: str


@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", service="correctness-checker")


@app.post("/check_batch", response_model=BatchCheckResponse)
async def check_batch(request: BatchCheckRequest):
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return BatchCheckResponse(results=await service.check_batch_async(request.items))


@app.post("/analyze_batch", response_model=BatchAnalyzeResponse)
async def analyze_batch(request: BatchAnalyzeRequest):
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return BatchAnalyzeResponse(results=await service.analyze_batch_async(request.items))


def run_server(
    host: str = "0.0.0.0",
    port: int = 5000,
    pool_size: int = 450,
    timeout: float = 15.0,
) -> None:
    ensure_latex_parser_available()
    os.environ["CORRECTNESS_SERVICE_POOL_SIZE"] = str(pool_size)
    os.environ["CORRECTNESS_SERVICE_TIMEOUT"] = str(timeout)
    _log(f"Listening on http://{host}:{port}")
    _prefork_managers(pool_size, timeout)
    uvicorn.run(app, host=host, port=port, workers=1, log_level="info")


def main() -> None:
    parser = argparse.ArgumentParser(description="VHG standalone verifier service.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--pool_size", type=int, default=450)
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()
    run_server(
        host=args.host,
        port=args.port,
        pool_size=args.pool_size,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
