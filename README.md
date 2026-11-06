# VHG

Official code for **VHG: Verifier-Backed Hard Problem Generation for Mathematical
Reasoning**. Train a setter against a frozen solver, verify its generated
problem–reference pairs, then train the next solver on the accepted problems.

```text
seed problems → cold SFT → setter RL → generate → verify + filter → solver RL → evaluate
                             ↑                                      │
                             └── expand seeds for the next round ───┘
```

Run **VHG (Hard)** for indefinite integration, **VHG (Soft)** for general math,
or the **three-round integration experiment**. Settings are in
[`configs/`](configs/); the pipeline is in [`vhg/pipeline.py`](vhg/pipeline.py).

## Install

Clone the repository, then install from its root:

```bash
git clone https://github.com/Miaow-Lab/VHG.git
cd VHG
conda create -y -n vhg_rl python=3.12
conda activate vhg_rl
bash scripts/install.sh
python scripts/check_env.py
```

The experiments use eight GPUs, a local Qwen3-4B-Base checkpoint, and an
OpenAI-compatible teacher endpoint. Soft also requires a verifier endpoint.
General-math cold SFT uses four of the eight visible GPUs.

## Run

Export your local settings (see [`.env.example`](.env.example)); `.env` is not
loaded automatically:

```bash
export BASE_MODEL_PATH=/path/to/Qwen3-4B-Base
export TEACHER_BASE_URL=https://your-teacher-endpoint/v1
export TEACHER_MODEL=your-teacher-model
export COLD_SFT_API_KEY=your-api-key
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=offline

# Hard evaluation uses the public AntiderivBench checkout.
export ANTIDERIVBENCH_ROOT=/path/to/AntiderivBench
```

Choose one experiment:

```bash
# Hard.
python -m vhg.pipeline integration

# Hard, including seed expansion and the two additional rounds.
export EMBED_MODEL_PATH=/path/to/Qwen3-Embedding-8B
python -m vhg.pipeline integration --rounds 3 --work-dir outputs/integration_three_rounds

# Soft.
export SOFT_VERIFIER_BASE_URL=https://your-verifier-endpoint/v1
export SOFT_VERIFIER_MODEL=your-verifier-model
export SOFT_VERIFIER_API_KEY=your-verifier-api-key
python -m vhg.pipeline general_math
```

Add `--dry-run` to inspect the commands before running them. Outputs go to
`outputs/<experiment>/`, or the supplied `--work-dir`.

The three-round pipeline expands the original seed bank and rescores the
cumulative verified archive between rounds.

Setters train for 200 updates. Solvers train for 1,600 updates per Hard round
and 700 updates for Soft.
Soft cold SFT requests 3,500 teacher candidates and keeps those passing its
verification and length filters.

Hard evaluation uses AntiderivBench Competition and Qualifier. Soft evaluation
covers MATH500, GSM8K, Minerva, OlympiadBench, AMC, and AIME 2024–2026.

Soft pool judging stops once at least 20,000 valid problems have been collected
or the pool is exhausted. It keeps every valid problem from the final batch.

## Stages and resume

Stages run in this order; `N` is the round number:

| Stage | Action |
|---|---|
| `cold-data` | Collect and verify teacher examples |
| `cold-sft` | Train the initial setter |
| `rN.seeds` (N > 1) | Expand the seed bank |
| `rN.data` | Prepare setter data |
| `rN.setter` | Train and select the setter |
| `rN.sample` | Generate problem–reference pairs |
| `rN.filter` | Verify and filter problems |
| `rN.solver-data` | Prepare solver data |
| `rN.solver` | Train and select the solver |
| `rN.eval` | Evaluate the solver |

RL checkpoints are selected by development reward. Cold SFT runs for five epochs
and uses step 50 (Hard) or 300 (Soft), falling back to the latest saved checkpoint
if that step is unavailable. Within each `roundN/`, selected
models are at `setter/selected` and `solver/selected`; scores are in
`evaluation/<benchmark>/statistics.json`.

Use `--from-stage` and `--through-stage` to select a range; preceding outputs
must exist. For example, stop after round one, then continue:

```bash
python -m vhg.pipeline integration --rounds 3 \
  --work-dir outputs/integration_three_rounds --through-stage r1.eval
python -m vhg.pipeline integration --rounds 3 \
  --work-dir outputs/integration_three_rounds --resume
```

`--resume` skips completed work and continues RL from saved checkpoints. Keep
the code, inputs, model paths, settings, and GPU list unchanged. Teacher
collection, sampling, and seed expansion reuse completed work; interrupted
filtering or evaluation may repeat. For interrupted cold SFT or changed settings,
use a new work directory.

## Citation

Please cite [the paper](https://arxiv.org/abs/2605.06660):

```bibtex
@misc{lai2026verifierbacked,
  title         = {Verifier-Backed Hard Problem Generation for Mathematical Reasoning},
  author        = {Lai, Yuhang and Feng, Jiazhan and Teh, Yee Whye and Miao, Ning},
  year          = {2026},
  eprint        = {2605.06660},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2605.06660},
  url           = {https://arxiv.org/abs/2605.06660}
}
```
