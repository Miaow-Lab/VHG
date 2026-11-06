from __future__ import annotations

import importlib
import shutil
from importlib import metadata


REQUIRED = [
    "datasets",
    "math_verify",
    "numpy",
    "openai",
    "sympy",
]

OPTIONAL_RL = [
    "ray",
    "torch",
    "transformers",
    "verl",
    "vllm",
]

PINNED = {
    "antlr4-python3-runtime": "4.11.0",
    "latex2sympy2": "1.5.4",
    "math-verify": "0.9.0",
    "numpy": "1.26.4",
    "sympy": "1.14.0",
}


def check_latex_parser() -> None:
    try:
        from sympy.parsing.latex import parse_latex

        parsed = parse_latex("x")
    except Exception as exc:
        raise SystemExit(f"LaTeX parser smoke failed: {exc}") from exc
    if str(parsed) != "x":
        raise SystemExit(f"LaTeX parser smoke failed: parsed 'x' as {parsed!r}")


def main() -> None:
    missing = []
    for name in REQUIRED:
        try:
            importlib.import_module(name)
        except Exception:
            missing.append(name)
    if missing:
        raise SystemExit("Missing required packages: " + ", ".join(missing))

    wrong_versions = []
    for package, expected in PINNED.items():
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            wrong_versions.append(f"{package} missing")
            continue
        if actual != expected:
            wrong_versions.append(f"{package}=={actual} (expected {expected})")
    if wrong_versions:
        raise SystemExit("Version mismatch: " + ", ".join(wrong_versions))

    check_latex_parser()

    missing_optional = []
    for name in OPTIONAL_RL:
        try:
            importlib.import_module(name)
        except Exception:
            missing_optional.append(name)
    missing_optional_commands = []
    if not missing_optional and shutil.which("ninja") is None:
        missing_optional_commands.append("ninja")

    print("Core VHG environment OK.")
    if missing_optional:
        print("Optional RL/vLLM packages not installed: " + ", ".join(missing_optional))
    elif missing_optional_commands:
        raise SystemExit(
            "Optional RL/vLLM commands not found: "
            + ", ".join(missing_optional_commands)
        )
    else:
        print("Optional RL/vLLM environment OK.")


if __name__ == "__main__":
    main()
