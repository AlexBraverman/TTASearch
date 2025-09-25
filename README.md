# Test-Time A* Search (TTA*)
**Training-free test-time tree search to improve multistep reasoning in Small Language Models (SLMs)**

TTA* casts reasoning as a goal-directed tree search where a single SLM iteratively generates, critiques, and refines candidate solutions. Inspired by A* search, TTA* balances path cost and a model-derived heuristic (self-evaluation) to prioritize expansions — improving correctness on math benchmarks without additional training or external teacher/reward models.

---

## Highlights
- **Training-free**: works as an inference-time wrapper — no fine-tuning or extra models required.   
- **Robust self-reflection**: averages multiple self-evaluations to stabilize noisy SLM critiques.  
- **Practical**: designed for 1–8B models (LLaMA, Qwen, etc.), enabling deployment on consumer GPUs.

---

## Features
- Implementation of the TTA* search loop: root generation → critique → self-evaluation → child generation → A* selection.  
- Configurable hyperparameters (temperature, children per expansion, A* weight `w`, `max_iterations`).  
- Full list of prompts
  
---
## Datasets
- **MATH401:** [https://github.com/GanjinZero/math401-llm](https://github.com/GanjinZero/math401-llm)  
- **MATH500, GSM8K, AIME (2024):** all available on [HuggingFace Datasets](https://huggingface.co/datasets)
## Files

- `benchmark.py` — for running benchmarks on models and datasets.  
- `tta.py` — implementation of our method.
pip install -r requirements.txt

# run an example reproduction (example from paper)
python run_tta.py --config configs/qwen3-4b_gsm8k.yaml --seed 42
