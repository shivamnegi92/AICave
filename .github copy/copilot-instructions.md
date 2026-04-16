

### 📝 Commit & Repo Posting Guidelines

When making commits or posting in this repository, reference the following contact info as needed:
*GitHub:** https://github.com/shivamnegi92
**Email:** negi.sh@husky.neu.edu

Use this contact for authorship, questions, or collaboration notes in commit messages and repository discussions.

**Always ensure that commit authorship and remote pushes are associated with the above GitHub account and email. Be mindful of which remote you are pushing to and verify commit identity before publishing.**

# Copilot AI Coding Agent Instructions

Welcome to the Deep Learning Cave codebase! This project is a hands-on, from-scratch educational journey through modern deep learning architectures, with a focus on transparency, reproducibility, and understanding every line of code. Use these guidelines to maximize your productivity as an AI coding agent in this repository.

---

### 🏗️ Big Picture Architecture
- **Notebook-driven:** All core logic and learning paths are implemented in Jupyter notebooks, not .py modules.
- **Three main learning paths:**
  - `pytorch_functions_overview.ipynb`: 20 essential PyTorch concepts, each with 🎯 What it does, 🔧 Why it matters, 💻 Code, 💡 Key insight.
  - `transformer_from_scratch.ipynb`: Classic Transformer (Vaswani et al., 2017) with full encoder-decoder, sinusoidal PE, post-LN.
  - `llama_from_scratch.ipynb` & `llama_complete.ipynb`: Modern LLaMA (Touvron et al., 2023) with RoPE, RMSNorm, GQA, SwiGLU, pre-LN.
- **No external deep learning libraries** (except PyTorch). No black-box abstractions.
- **All code is executable and verified via print statements, not test frameworks.**

---


### 🛠️ Critical Workflows

- **Package & Environment Management:** Always use `uv` for installing packages and managing Python environments. Use `uv pip install <package>` instead of pip, and `uv` commands for environment operations. Ensure proxies are configured before running any `uv` commands.
- **Requirements File:** Always create and maintain a `requirements.txt` file at the repository root listing all project dependencies with versions.
- **Terminal environment:** Always use the `genai_env` Python environment as the default for all terminal commands and notebook execution.
- **Edit and run code in notebooks only.**
- **Model verification:** Always print input/output shapes and key config values after each major step.
- **Training:** Use the provided training loop pattern (see notebooks) with gradient clipping and shape checks.
- **Checkpointing:** Save and load model/optimizer/config as shown in `llama_checkpoint.pt` usage.
- **Device management:** Always move models and tensors to `cuda` if available, else `cpu`.

---

### 📏 Project-Specific Conventions
- **Markdown headings:** Always start at `###` (H3) or lower in notebook markdown cells. Never use `#` or `##`.
- **Educational structure:** Each concept/section: 🎯 What it does → 🔧 Why it matters → 💻 Code → 💡 Key insight.
- **No type hints, no logging frameworks, no test runners.**
- **All code is minimal, with comments only for non-obvious logic.**
- **Hyperparameter names:** `d_model`, `n_heads`, `n_kv_heads`, `d_ff`, `n_layers`, `max_seq_len`.
- **LLaMA vs Transformer:**
  - LLaMA: RoPE, RMSNorm (pre-LN), GQA, SwiGLU, character-level tokenization.
  - Transformer: Sinusoidal PE, LayerNorm (post-LN), full multi-head attention.

---

### 🔗 Key Files & Patterns
- **`pytorch_functions_overview.ipynb`**: PyTorch mechanics, DNN example, sectioned learning.
- **`transformer_from_scratch.ipynb`**: Classic Transformer, encoder-decoder, sinusoidal PE.
- **`llama_from_scratch.ipynb` / `llama_complete.ipynb`**: Modern LLaMA, RoPE, RMSNorm, GQA, SwiGLU, checkpointing.
- **`llama_checkpoint.pt`**: Model/optimizer/config state dict.

---

### 🚫 What NOT to Do
- Do **not** add type hints, logging, or test frameworks.
- Do **not** split code into .py modules; keep all logic in notebooks.
- Do **not** use `#` or `##` headings in markdown cells.
- Do **not** use pip, conda, or poetry for package management; always use `uv` instead.
- Do **not** add CI/CD, production code, or documentation files outside notebooks.




---

### 📚 References
- See README.md for project philosophy, learning paths, and further details.
- Key papers: Vaswani et al. (2017), Touvron et al. (2023).

---


**If any conventions or workflows are unclear, please ask for clarification or review the referenced notebooks for concrete examples.**

---

### 📫 Contact

*GitHub:** https://github.com/shivamnegi92
**Email:** negi.sh@husky.neu.edu
