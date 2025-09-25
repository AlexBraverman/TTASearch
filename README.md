# Test-Time A* Search (TTA*)
**Training-free test-time tree search to improve multistep reasoning in Small Language Models (SLMs)**

TTA* casts reasoning as a goal-directed tree search where a single SLM iteratively generates, critiques, and refines candidate solutions. Inspired by A* search, TTA* balances path cost and a model-derived heuristic (self-evaluation) to prioritize expansions — improving correctness on math benchmarks without additional training or external teacher/reward models.

---

## Highlights
- **Training-free**: works as an inference-time wrapper — no fine-tuning or extra models required.  
- **Anytime & budgeted**: supports explicit compute budgets and early stopping.  
- **Robust self-reflection**: averages multiple self-evaluations to stabilize noisy SLM critiques.  
- **Practical**: designed for 1–8B models (LLaMA, Qwen, etc.), enabling deployment on consumer GPUs.

---

## Features
- Implementation of the TTA* search loop: root generation → critique → self-evaluation → child generation → A* selection.  
- Configurable hyperparameters (temperature, children per expansion, self-eval samples, A* weight `w`, `max_iterations`).  
- Ready-to-run configs for common experiments (example: GSM8K with Qwen3-4B).  
- Full logging of chain-of-thought, critiques, and numerical self-evaluations for reproducibility and auditing.

---

## Quick start

```bash
# clone
git clone https://github.com/<username>/tta-star.git
cd tta-star

# create env (Python 3.10+)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# run an example reproduction (example from paper)
python run_tta.py --config configs/qwen3-4b_gsm8k.yaml --seed 42
