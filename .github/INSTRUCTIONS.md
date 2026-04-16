# AI Coding Agent Instructions

## Project Overview
Educational repository for learning Transformer architectures through hands-on implementation. Contains three learning paths: foundational PyTorch concepts, vanilla Transformer implementation, and modern LLaMA architecture.

## Repository Structure

### Core Learning Materials
- **`pytorch_functions_overview.ipynb`** - Educational reference covering 20 PyTorch concepts essential for Transformers
  - Sections 1-8: Foundation (tensors, embeddings, attention mechanics)
  - Sections 9-16: Architecture (residuals, FFN, training loops)
  - Sections 17-20: Advanced (einsum, inference optimization)
  - Each section: 🎯 What it does → 🔧 Why it matters → 💻 Code → 💡 Key insight
  
- **`transformer_from_scratch.ipynb`** - Vanilla Transformer implementation (Vaswani et al., 2017)
  - Complete encoder-decoder architecture
  - Token embeddings + sinusoidal positional encoding
  - Multi-head attention with masking
  
- **`llama_from_scratch.ipynb` / `llama_complete.ipynb`** - Modern LLaMA architecture
  - RoPE (Rotary Position Embeddings) instead of sinusoidal
  - RMSNorm instead of LayerNorm
  - Grouped Query Attention (GQA) for memory efficiency
  - SwiGLU activation function
  - Character-level tokenization for simplicity

### Training Artifacts
- **`llama_checkpoint.pt`** - Saved model state (contains model weights, optimizer state, config)

## Markdown Formatting Guidelines

### Heading Hierarchy Rules
When creating or editing Markdown content in notebooks:

**✅ Correct heading structure:**
- Use `###` (H3) for main section titles
- Use `####` (H4) for subsections
- Use `#####` (H5) and `######` (H6) for deeper nesting
- **Never use `#` (H1) or `##` (H2) within notebook cells**

**Example structure:**
```markdown
### Main Section Title

Introduction paragraph.

#### Subsection

Details here.

##### Detail Level

More specific information.

#### Another Subsection

Additional content.
```

### Rationale
- Notebooks have implicit top-level structure (the notebook itself)
- Starting from H3 maintains visual hierarchy
- Allows proper nesting up to 4 levels (H3 → H4 → H5 → H6)
- Consistent with educational documentation standards

### When Adding New Content
1. Start with `###` for major concepts (e.g., "### 21. Flash Attention")
2. Use `####` for subsections (e.g., "#### 🎯 What it does", "#### 🔧 Why it matters")
3. Use `#####` for detailed breakdowns
4. Maintain this hierarchy throughout all notebook Markdown cells

## Key Architecture Patterns

### Notebook Structure Convention
All notebooks follow this pattern:
```python
# 1. Import and setup
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(42)

# 2. Component definitions (classes)
class Component(nn.Module):
    def __init__(self, config): ...
    def forward(self, x): ...

# 3. Configuration with @dataclass
@dataclass
class Config:
    d_model: int = 256
    n_heads: int = 8
    # ... more hyperparameters

# 4. Testing/demonstration code
model = Component(config)
test_input = torch.randn(batch, seq, d_model)
output = model(test_input)
```

### LLaMA vs Classic Transformer Differences

**When working with LLaMA notebooks:**
- Position encoding: Use `precompute_freqs_cis()` and `apply_rotary_emb()` (RoPE), NOT sinusoidal
- Normalization: `RMSNorm` before each sublayer (Pre-LN), NOT LayerNorm after
- Attention: `GroupedQueryAttention` with shared KV heads, NOT standard multi-head
- Activation: `SwiGLU(x) = (x @ W_gate * F.silu(x @ W_up)) @ W_down`, NOT simple ReLU
- Model flow: `x = x + sublayer(norm(x))`, NOT `x = norm(x + sublayer(x))`

**When working with classic Transformer:**
- Position encoding: Sinusoidal `PE(pos, 2i) = sin(pos/10000^(2i/d))`, `PE(pos, 2i+1) = cos(pos/10000^(2i/d))`
- Normalization: Post-LN pattern `x = norm(x + sublayer(x))`
- Full multi-head attention with separate Q, K, V projections per head

### Training Loop Pattern
```python
# Standard training loop used across all notebooks
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (tokens, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Forward pass with causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(device)
        logits = model(tokens, mask)
        
        # Cross-entropy loss (flatten batch+seq dims)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1)
        )
        
        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
```

### Generation Pattern
```python
# Autoregressive text generation used in LLaMA notebooks
@torch.no_grad()
def generate(model, prompt, max_tokens=50, temperature=1.0):
    model.eval()
    tokens = tokenize(prompt)
    
    for _ in range(max_tokens):
        # Get predictions for last token
        logits = model(tokens)[:, -1, :]
        probs = F.softmax(logits / temperature, dim=-1)
        
        # Sample next token
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)
        
        # Optional: top-k or top-p sampling
        # probs = top_k_top_p_filtering(logits, top_k=50, top_p=0.9)
    
    return decode(tokens)
```

## Common Development Tasks

### Editing Notebook Markdown Cells
- Use `replace_string_in_file` with exact XML structure
- Include VSCode.Cell tags with id and language attributes
- Markdown cells: `language="markdown"`, code cells: `language="python"`
- NEVER modify cell IDs - they're auto-generated and stable
- **Always start headings with `###` (H3) or lower - never use `#` or `##`**

### Running Code Cells
- Use `run_notebook_cell` with the cell ID (not cell number)
- Cells executed in order maintain kernel state (variables persist)
- Use `copilot_getNotebookSummary` to see execution status and output types

### Adding Educational Content
Pattern used throughout `pytorch_functions_overview.ipynb`:
1. **Markdown cell**: Concept explanation with formulas (LaTeX in `$...$` or `$$...$$`)
2. **Code cell**: Minimal working example demonstrating the concept
3. **Output**: Print shapes, values, and verification messages
4. **Key insight**: Why this matters for Transformers (not just generic PyTorch)

Example additions should follow this 4-part structure with emoji headers (🎯🔧💻💡).

**Markdown heading structure for new sections:**
```markdown
### N. Topic Name

#### 🎯 What it does:
Explanation with formulas.

#### 🔧 Why it matters for Transformers:
Real-world context.

#### 💻 Example code:
(Python code cell follows)

#### 💡 Key Insight:
The "aha!" moment.
```

## Testing and Verification

### Model Verification Pattern
All implementations verify shapes at each step:
```python
print(f"Input shape: {x.shape}")   # e.g., (batch=2, seq=10, d_model=256)
output = component(x)
print(f"Output shape: {output.shape}")  # Verify expected dimensions
print("✓ Component working correctly!")
```

### Configuration Validation
Models print their hyperparameters on initialization:
```python
config = LLaMAConfig(d_model=256, n_layers=6, n_heads=8)
model = LLaMA(config)
n_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {n_params:,}")
```

## Project-Specific Conventions

### Hyperparameter Naming
- `d_model`: Model dimension (e.g., 256, 512)
- `n_heads`: Number of attention heads
- `n_kv_heads`: KV heads for GQA (LLaMA only)
- `d_ff`: Feed-forward dimension (typically 4× d_model)
- `n_layers`: Number of transformer blocks
- `max_seq_len`: Maximum sequence length

### Device Management
All code uses this pattern:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
inputs = inputs.to(device)
```

### Checkpoint Format
```python
# Saving
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss,
    'config': config
}
torch.save(checkpoint, 'llama_checkpoint.pt')

# Loading
checkpoint = torch.load('llama_checkpoint.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
```

## What NOT to Do

❌ Don't suggest installing packages - notebooks run in existing environment with PyTorch already installed
❌ Don't create README.md or documentation files - notebooks are self-documenting
❌ Don't add type hints to notebook code - kept minimal for educational clarity
❌ Don't suggest pytest or unittest - verification done via print statements in cells
❌ Don't propose CI/CD - this is an educational project, not production code
❌ Don't add logging frameworks - simple print statements are intentional
❌ Don't suggest splitting notebooks into .py modules - notebook format is deliberate for learning
❌ **Don't use `#` (H1) or `##` (H2) headings in notebook Markdown cells** - always start from `###` (H3)

## Key References

When explaining concepts, refer to:
- "Attention Is All You Need" (Vaswani et al., 2017) - for vanilla Transformer
- "LLaMA: Open and Efficient Foundation Language Models" (Touvron et al., 2023) - for LLaMA architecture
- Specific sections in `pytorch_functions_overview.ipynb` for PyTorch mechanics

---

# Jupyter Notebook Coding Standards

## Core Philosophy
Write **lean, readable, production-ready** notebook code. Every line should serve a clear analytical or computational purpose. Avoid defensive programming, verbose logging, and unnecessary abstractions.

---

## 1. Cell Structure & Length

### Size Limits
- **Maximum 300 lines per cell** (hard limit)
- **Optimal: 50-100 lines per cell** for readability
- Break complex operations into logical cells, not arbitrary chunks

### Cell Organization
- **One logical operation per cell** (load data, transform, visualize, model)
- **Sequential dependency:** Each cell should build on previous state
- **Idempotent when possible:** Re-running a cell shouldn't break downstream cells

### Headers
- Use markdown cells with `###` (H3) or lower for section headers
- Never use `#` or `##` in notebook markdown
- Structure: `### 🎯 Purpose → 💻 Code → 💡 Key Insight`

---

## 2. Variable Management

### Naming Conventions
- **Use established, consistent names throughout the notebook:**
  - `df` - primary DataFrame
  - `client` - API/database client (BigQuery, etc.)
  - `credentials` - authentication objects
  - `model` - ML model instance
  - `config` - configuration dictionary
  - `device` - PyTorch device (cuda/cpu)

### Avoid Variable Proliferation
- ❌ **NO:** `df_copy`, `df_new`, `df_final`, `data`, `dataset`, `df_cleaned`, `df_processed`
- ✅ **YES:** Transform `df` in-place or reassign directly: `df = df[df['col'] > 0]`

### When to Create New Variables
- **Only when you need multiple versions simultaneously** (e.g., train/test splits)
- Use descriptive suffixes: `df_train`, `df_test`, `df_val` (not `df1`, `df2`)

---

## 3. Error Handling & Defensive Programming

### General Rule: Trust Your Tools
- **NO try-catch blocks** unless handling **external I/O** (file reads, API calls, database queries)
- **NO defensive if-else** for standard pandas/numpy operations
- Let Python/pandas raise natural errors for faster debugging

### When to Use Error Handling
```python
# ✅ GOOD: External I/O
try:
    df = pd.read_csv('data.csv')
except FileNotFoundError:
    df = fetch_from_api()

# ❌ BAD: Standard operations
try:
    if df is not None and len(df) > 0:
        df = df.dropna()
except:
    pass
```

### Edge Cases
- Trust pandas/numpy to handle empty DataFrames, NaN values, etc.
- Use built-in parameters: `df.dropna()`, `np.nanmean()`
- Only add checks for **business logic requirements**, not technical edge cases

---

## 4. Output & Printing

### Minimal Output Philosophy
- **Only print what you need to verify or analyze**
- Avoid decorative elements: emojis, separator lines, verbose messages
- Let pandas/numpy display methods do the work

### Print Statement Guidelines
```python
# ❌ AVOID
print("="*50)
print("✅ Loading data...")
print("="*50)
df = pd.read_csv('data.csv')
print(f"✅ Loaded {len(df)} rows successfully!")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# ✅ PREFER
df = pd.read_csv('data.csv')
df.shape  # Jupyter displays this automatically
```

### When to Print
- **Shape verification:** After major transformations
- **Training metrics:** Loss, accuracy per epoch (concise format)
- **Query execution:** Execution time + row count only
- **Hyperparameters:** Model config values after initialization

### Display Methods
- Use pandas built-ins: `.head()`, `.tail()`, `.sample(n=5)`, `.describe()`, `.info()`
- Use PyTorch: `.shape`, `.dtype`, `.device`
- Let Jupyter auto-display the last expression in a cell

---

## 5. Code Style & Patterns

### Directness Over Abstraction
```python
# ❌ AVOID: Unnecessary function
def filter_data(dataframe, column, threshold):
    """Filter dataframe by column threshold"""
    return dataframe[dataframe[column] > threshold]

df = filter_data(df, 'value', 100)

# ✅ PREFER: Direct operation
df = df[df['value'] > 100]
```

### Chaining Operations
```python
# ✅ GOOD: Readable chains
df = (df
    .dropna(subset=['key_column'])
    .query('value > 0')
    .sort_values('timestamp')
    .reset_index(drop=True))

# ❌ BAD: Overly complex chains (break into multiple cells)
df = (df.pipe(lambda x: x[x['a'] > 0])
        .apply(lambda row: complex_function(row), axis=1)
        .merge(other_df, on='key')
        .groupby('category').agg({'value': ['mean', 'std', 'count']})
        .reset_index()
        .rename(columns={'value_mean': 'avg'}))
```

### Function Usage
- **Avoid helper functions** unless used 3+ times in the notebook
- **No nested functions** unless implementing closures for specific reasons
- Prefer pandas/numpy built-ins over custom implementations

---

## 6. Query Execution (BigQuery, SQL, etc.)

### Pattern
```python
# ✅ Standard query pattern
query = """
SELECT column1, column2, COUNT(*) as count
FROM `project.dataset.table`
WHERE date >= '2024-01-01'
GROUP BY column1, column2
"""

import time
start = time.time()
df = client.query(query).to_dataframe()
print(f"{time.time() - start:.1f}s | {len(df):,} rows")
```

### Guidelines
- **Store results in `df`** (don't create `query_result`, `data`, etc.)
- **Reuse existing `client`** from setup cells
- **Minimal feedback:** execution time + row count only
- **Use triple-quoted strings** for multi-line queries
- **Format SQL readably** (capitals for keywords, proper indentation)

---

## 7. Data Exploration

### Standard Workflow
```python
# ✅ Concise exploration pattern
df.shape                    # Dimensions
df.dtypes                   # Column types
df.head()                   # First rows
df.describe()               # Summary statistics
df.isnull().sum()           # Missing values
df['column'].value_counts() # Distribution
```

### Avoid Redundancy
```python
# ❌ AVOID
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Dtypes:\n{df.dtypes}")
print(f"Missing values:\n{df.isnull().sum()}")

# ✅ PREFER (let Jupyter display)
df.shape
df.dtypes
df.isnull().sum()
```

---

## 8. Visualization

### Import Once
```python
# ✅ Import at notebook top (first code cell)
import matplotlib.pyplot as plt
import seaborn as sns

# Set defaults once
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
```

### Minimal Styling
```python
# ✅ GOOD: Clean, focused visualization
plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Model Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# ❌ AVOID: Excessive styling
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, y, linewidth=2.5, color='#FF6B6B', alpha=0.8, linestyle='--')
ax.set_title('Model Loss Over Time', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle=':', linewidth=1.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()
```

### Focus on Insights
- **Show what matters** for analysis, not aesthetics
- **Use defaults** unless there's a specific reason to customize
- **One key insight per plot** (don't overload visualizations)

---

## 9. Deep Learning Specific (PyTorch)

### Device Management
```python
# ✅ Set once at notebook top
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Move tensors as needed
x = x.to(device)
```

### Shape Verification
```python
# ✅ After major operations
x = torch.randn(batch_size, seq_len, d_model)
x.shape  # Auto-display in Jupyter

# After model forward pass
output = model(x)
output.shape  # Verify expected dimensions
```

### Training Loop
```python
# ✅ Minimal, informative output
for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        # ... training logic ...
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")
```

---

## 10. Package Management

### Always Use `uv`
```bash
# ✅ In terminal (with proxy configured)
uv pip install torch pandas matplotlib

# ❌ NEVER use pip, conda, or poetry directly
pip install torch  # WRONG
```

### Requirements File
- Maintain `requirements.txt` at repository root
- Pin versions for reproducibility: `torch==2.1.0`
- Update after adding new dependencies

---

## 11. Environment Setup
---

## 12. Common Anti-Patterns

### ❌ Avoid These
```python
# Variable proliferation
df_copy = df.copy()
df_new = df_copy[df_copy['col'] > 0]
df_final = df_new.dropna()

# Defensive programming
if df is not None and not df.empty:
    if 'column' in df.columns:
        try:
            result = df['column'].mean()
        except:
            result = 0

# Verbose output
print("="*50)
print("Starting analysis...")
print("="*50)

# Unnecessary functions
def get_filtered_df(df, threshold):
    return df[df['value'] > threshold]

# Over-styled plots
plt.rcParams.update({'font.size': 12, 'font.weight': 'bold', ...})
```

### ✅ Do These Instead
```python
# Direct transformation
df = df[df['col'] > 0].dropna()

# Trust pandas
df['column'].mean()

# Minimal output
df.shape

# Direct operations
df = df[df['value'] > threshold]

# Default styling
plt.plot(x, y)
plt.show()
```

---

## 13. Summary Checklist

- [ ] Cells under 300 lines (ideally 50-100)
- [ ] Reuse `df`, `client`, `model`, `device` variables
- [ ] No try-catch except for external I/O
- [ ] Minimal print statements (only essential verification)
- [ ] Use pandas/numpy built-ins over custom functions
- [ ] Let Jupyter auto-display last expression
- [ ] Import libraries once at top
- [ ] Use `uv` for all package management
- [ ] Use `###` or lower for markdown headers
- [ ] Shape checks after major transformations
- [ ] Clean, insight-focused visualizations

---

**Remember:** Notebooks are for exploration and analysis. Write code that's easy to read, modify, and debug. Let tools handle edge cases. Focus on insights, not infrastructure.
