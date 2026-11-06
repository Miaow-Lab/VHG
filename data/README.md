# Seed inputs

| File | Purpose |
|---|---|
| `integration_cold_sft_seeds.json` | Tongji/Wolfram inputs for Hard cold SFT |
| `integration_cold_sft_icl.txt` | In-context examples for Hard cold SFT |
| `integration_seeds.jsonl` | Original Tongji integration seeds for Hard setter training |
| `general_math_cold_sft_seeds.jsonl` | MATH seeds for Soft cold SFT and generation |
| `general_math_seeds.jsonl` | Easy MATH subset for Soft setter training |

The three-round integration pipeline constructs later seed banks during the run
and writes them to `roundN/seeds/` under the output directory.
