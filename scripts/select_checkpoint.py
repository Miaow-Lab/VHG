"""Select and merge the checkpoint with the highest development reward."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vhg.checkpoints import link_selected_checkpoint, select_by_validation
from vhg.pipeline import write_json
from vhg.train import run_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_root", type=Path, required=True)
    parser.add_argument("--steps", required=True, help="All scheduled development steps, comma-separated.")
    parser.add_argument("--validation_file", type=Path, required=True)
    args = parser.parse_args()
    root = args.checkpoint_root
    import pyarrow.parquet as pq
    rows = pq.read_table(args.validation_file).to_pylist()
    prompts = {"\n".join(m["content"] for m in row["prompt"]).strip() for row in rows}
    if len(prompts) != len(rows):
        raise ValueError("Duplicate development prompts")
    result = select_by_validation(root, prompts, [int(s) for s in args.steps.split(",")])
    step = result["selected_step"]
    code = run_command([sys.executable, str(ROOT / "scripts/merge_checkpoints.py"),
                        "--checkpoint_root", str(root), "--steps", str(step)], cwd=ROOT)
    if code:
        raise SystemExit(code)
    link_selected_checkpoint(root, step)
    write_json(root / "selection.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
