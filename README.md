# REI-Bench: Can Embodied Agents Understand Vague Human Instructions in Task Planning?

This repository is the **official evaluation codebase** for **REI-Bench**, from the paper **REI-Bench: Can Embodied Agents Understand Vague Human Instructions in Task Planning?** REI-Bench is an evaluation benchmark for embodied task planners under **referential and coreferential vagueness** in natural language instructions.

**Project webpage:** [https://jcx0110.github.io/REI-Bench-web/](https://jcx0110.github.io/REI-Bench-web/)

## 🤖 Authors

See the paper for the full list of authors.  
**MARS Lab**, Nanyang Technological University (NTU)

## 🧭 Introduction

We introduce **REI-Bench**, a benchmark for assessing whether embodied agents can interpret **vague, underspecified, or coreferential** human instructions in task planning. In real-world settings, users often give instructions like "put the drink in the fridge" or "turn off the electronic device" without specifying *which* drink or device. This **referential vagueness** is a fundamental bottleneck: the agent must ground language to the right objects and then plan and execute. Existing benchmarks (e.g., ALFRED-style benchmarks) largely use clear, unambiguous instructions and thus do not evaluate this capability. REI-Bench fills this gap by providing a controlled benchmark where instruction vagueness is systematically varied, enabling rigorous evaluation and method development.

## ✨ Key Contributions

- **REI-Bench design**: A benchmark that evaluates instruction understanding (grounding + planning) under referential and coreferential vagueness, rather than perception or low-level control alone.
- **Multi-level referential vagueness**: A formal definition of **9 levels** of referential vagueness, combining **3 referential levels** (explicit, mixed, implicit) with **3 context types** (standard, noised, short).
- **Evaluation of LLM-based planners**: Systematic evaluation of multiple LLMs and planning frameworks (e.g., SayCan, LLM+P, DAG-based planning) across these vagueness levels.
- **TOCC (Task-Oriented Cognitive Control)**: A method that resolves vague referring expressions before planning, interfacing with existing LLM planners to improve robustness under ambiguous instructions.

## 📦 REI-Bench Overview

- **What REI stands for**: **R**eferential **E**mbodied **I**nstruction benchmark — focusing on *referential* understanding of instructions in embodied task planning.
- **Task format**: Given a **natural language instruction** (possibly vague), the agent must (1) **ground** the instruction to concrete objects and goals in the environment, (2) **plan** a sequence of actions, and (3) **execute** in simulation (e.g., ALFRED/AI2-THOR). REI-Bench evaluates the full pipeline with emphasis on instruction understanding.
- **What makes REI-Bench different**: Prior datasets assume clear, unambiguous instructions. REI-Bench explicitly introduces and controls referential and coreferential vagueness (e.g., "the cup," "electronic devices," "beverages") and varies context length and noise, so that progress can be measured on *understanding vague instructions* rather than only on perception or execution.

## ⚙️ Requirements

- **Python**: 3.8+ (tested on Ubuntu 22.04, Python 3.8).
- **PyTorch**: 2.0+ (e.g., `torch==2.0.0`, `torchvision==0.15.1`); install from [pytorch.org](https://pytorch.org/get-started/locally/) according to your CUDA version.
- **Other major dependencies**: `transformers`, `hydra-core`, `omegaconf`, `guidance`, `ai2thor` (for ALFRED).

**Install with conda (recommended):**

```bash
conda env create -f requriements.yaml
conda activate chenxi_lotabench
```

Or install with pip in your own environment (PyTorch first from [pytorch.org](https://pytorch.org/get-started/locally/), then):

```bash
pip install transformers hydra-core omegaconf guidance ai2thor accelerate
```

For ALFRED submodule dependencies, see `alfred/requirements.txt`.

## 📂 Evaluation Data

REI-Bench is an **evaluation-only** codebase. It does not include or generate evaluation data. Obtain evaluation splits and environment data (e.g., ALFRED-compatible task lists and scene data) separately. Configure the path to your evaluation split in the Hydra config (e.g., the `split` key in `configs/config.yaml`).

## 🏃‍♂️ Running Evaluation

From the REI-Bench project root:

**Standard evaluation (with display):**

```bash
python scripts/evaluate.py
```

Override Hydra config as needed (e.g., method, task difficulty, model):

```bash
# Use TOCC for vague instruction resolution
python scripts/evaluate.py method=tocc task_difficulty=explicit-standard

# Evaluate on a specific vagueness level (e.g., implicit + short)
python scripts/evaluate.py task_difficulty=implicit-short

# Use a different model
python scripts/evaluate.py model=llama3.1-8b
```

**Headless / server (no physical display):**

If you see `Exception: command: [.../thor-...] exited with 1` or `X Error ... GLX ... X_GLXCreateContext`, the THOR simulator needs an X display with GLX. Use:

```bash
./scripts/run_evaluate_with_xvfb.sh
```

The script sets Mesa software-GL env vars and starts Xvfb with GLX; you can append Hydra overrides (e.g. `./scripts/run_evaluate_with_xvfb.sh method=tocc`). If GLX errors persist, install Mesa: `sudo apt-get install -y xvfb libgl1-mesa-glx libgl1-mesa-dri` and try again, or run evaluation in Docker (e.g. [ai2thor-docker](https://github.com/allenai/ai2thor-docker)).



## 📜 Citation

If you use REI-Bench or TOCC in your work, please cite:

```bibtex
@inproceedings{reibench2026,
  title     = {REI-Bench: Can Embodied Agents Understand Vague Human Instructions in Task Planning?},
  author    = {Anonymous},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  note      = {Under review}
}
```

Replace `Anonymous` with the full author list (e.g., `Author One and Author Two and ...`) for the camera-ready version.
