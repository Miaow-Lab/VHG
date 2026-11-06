#!/usr/bin/env bash
set -Eeuo pipefail

# Run this script after creating and activating the conda environment.
# Package installation order:
#   1. upgrade pip/setuptools
#   2. install OpenJDK 17
#   3. install this package
#   4. install wheel, scipy, Hugging Face Hub, and uv
#   5. install vLLM with uv before pinning transformers
#   6. install deepspeed, hydra, VERL, then flash-attn

python -m pip install -U pip setuptools
conda install -y openjdk=17
python -m pip install -e .
python -m pip install wheel scipy huggingface_hub "uv==0.11.19"
python -m pip install ninja
uv pip install "vllm==0.11.0" --torch-backend=auto
python -m pip install "transformers==4.57.1"
python -m pip install "deepspeed==0.19.1"
python -m pip install "git+https://github.com/facebookresearch/hydra.git@c2c00ab363c7b64023d7c3755c4ecc6f1ae6fef5"
python -m pip install "verl==0.6.1"
python -m pip install "flash-attn==2.8.3" --no-build-isolation
