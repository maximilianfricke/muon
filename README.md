# Muon Optimizer Research Project: Sharpness, Curvature and Edge-of-Stability Analysis 

**Code for the ETH Deep Learning course project paper on "Disentangling Sharpness and Update Geometry in the Muon Optimizer: Ablations, Stability and Variational Extensions" by Fricke, Ovcharova and Laffranchi.**

This repository contains a structured experimental study of **optimization geometry and stability**
for deep learning optimizers, with a particular focus on the **Muon optimizer**.
The project investigates how curvature, sharpness, and Edge-of-Stability (EoS) behavior
differ between Muon and standard optimizers such as **SGD** and **AdamW**.

The experiments are organized into three main tasks:

- **Task 1 – Sharpness Tracking**  
  Track the largest Hessian eigenvalue λ_max(H_t) during full-batch training.

- **Task 2 – Muon Ablation Study**  
  Analyze how individual Muon components (NS depth, RMS normalization,
  orthogonalization, weight decay) affect curvature suppression and effective sharpness.

- **Task 3 – Edge-of-Stability (EoS)**  
  Study the EoS quantity η · λ_max(H_t) during training and relate it to optimizer stability.
  (For preconditioned optimizers, this serves as a baseline before introducing H_eff.)

All experiments are run in **full-batch mode** to make curvature estimation reliable.


## Quick Setup

```bash
# 1. Create conda environment
bash setup.sh

# 2. Activate environment
conda activate muon

# 3. (macOS only) Set OpenMP fix (add to ~/.zshrc for permanent)
export KMP_DUPLICATE_LIB_OK=TRUE

# 4. Verify installation
python verify_setup.py


# 5. login in wandb
wandb login 
```

## Repository Structure

```text
.
├── src                         # Core source code
│   ├── experiments             # Experiment entry points
│   │   ├── basic_training.py   # Shared training + logging loop
│   │   ├── task1_sharpness.py  # Task 1: λ_max(H_t) tracking
│   │   ├── task2_ablation.py   # Task 2: Muon ablation study
│   │   └── task3_eos.py        # Task 3: Edge-of-Stability analysis
│   ├── geometry                # Curvature and Hessian utilities
│   │   ├── curvature.py
│   │   └── hessian.py
│   ├── models                  # Model definitions
│   │   ├── mlp.py
│   │   └── tiny_vit.py
│   ├── optimizers              # Optimizer implementations
│   │   └── muon.py
│   └── utils                   # Data, training, and visualization helpers
│       ├── data.py
│       ├── training.py
│       └── visualization.py
│
├── configs                     # Fully reproducible experiment configs
│   ├── basic_training_mlp_mnist.yaml
│   ├── basic_training_vit_cifar10.yaml
│   ├── task1_sharpness.yaml
│   ├── task2_ablation.yaml
│   └── task3_eos.yaml
│
├── results                     # Saved outputs and visualizations
│   ├── basic_training
│   │   └── basic_training_*_{adamw,sgd,muon}
│   ├── task1
│   │   └── task1_sharpness_*_{adamw,sgd,muon}
│   ├── task2
│   │   └── t2_*_mlp_mnist_muon_*      # Muon ablation runs
│   └── task3
│       ├── task3_eos__gd__*
│       ├── task3_eos__adamw__*
│       └── task3_eos__muon__*
│
├── data                        # Datasets (downloaded automatically)
├── wandb                       # Weights & Biases run logs (optional)
├── environment.yml             # Conda environment
├── setup.sh
└── README.md

## Wandb Integration

- **Project**: All runs go to `"muon"` project
- **Run names**: `{prefix}_{model}_{dataset}_{optimizer}_{timestamp}`
  - Example: `basic_training_mlp_mnist_muon_20251205_162201`
  - Filter by prefix: `task1_sharpness_*`, `task2_*`, etc.
- **Logged**: Loss, accuracy, λ_max (when enabled), visualizations

## Key Features

- ✅ **Full-batch training** (on subset if specified)
- ✅ **Muon optimizer** with automatic parameter separation
- ✅ **Curvature tracking** (λ_max via power iteration) - optional
- ✅ **Reproducible subsets** (same samples across runs)
- ✅ **Automatic visualizations** (predictions, history, confusion matrix)
- ✅ **Config-driven** (all hyperparameters in YAML)

## Troubleshooting

### OpenMP Error on macOS
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```
Add to `~/.zshrc` to make permanent.

### Out of Memory (CIFAR-10)
Reduce `train_subset_size` in config (e.g., 2000, 1000, 500).

### Verify Setup
```bash
python verify_setup.py
```
## Running Experiments

All hyperparameters are specified via YAML config files:

```bash
# Train MLP on MNIST (basic training - no curvature tracking)
python -m src.experiments.basic_training --config configs/basic_training_mlp_mnist.yaml

# Train TinyViT on CIFAR-10 (basic training - no curvature tracking)
python -m src.experiments.basic_training --config configs/basic_training_vit_cifar10.yaml
```

## Configuration

### Key Config Options

```yaml
experiment:
  name: "basic_training_mlp_mnist"  # Used for run naming and filtering
  seed: 42
  device: "auto"  # Auto-detects CUDA/MPS/CPU

dataset:
  name: "mnist"  # or "cifar10"
  train_subset_size: 2000  # Use subset for CIFAR-10 (same subset across runs)

model:
  type: "mlp"  # or "tiny_vit"
  config:
    # Model-specific parameters

optimizer:
  type: "muon"  # or "sgd", "adamw"
  config:
    lr: 0.02
    momentum: 0.95
    ns_depth: 5
```

### Memory Management for CIFAR-10

For TinyViT on CIFAR-10, use a subset to avoid OOM errors:

```yaml
dataset:
  train_subset_size: 2000  # Adjust based on available memory
```

The subset is **deterministic** - same samples used across all runs (ensures comparability).

## Task 1: Sharpness Tracking

Compare λ_max evolution for Muon, SGD, and AdamW:

```bash
python -m src.experiments.task1_sharpness --config configs/task1_sharpness.yaml
```

This will:
1. Run training with Muon optimizer (tracks λ_max)
2. Run training with SGD optimizer (tracks λ_max)
3. Run training with AdamW optimizer (tracks λ_max)
4. All results logged to wandb project "muon" with prefix `task1_sharpness_*`

**Note**: Task 1 uses `basic_training.py` internally - it just runs it multiple times with different optimizers. The curvature tracking is enabled in the Task 1 config.

## Task 1 — Sharpness trajectories across optimizers

**Goal:** Compare how sharpness evolves during training for:
- **SGD** (or GD when momentum=0)
- **AdamW**
- **Muon** (Muon on 2D weights + AdamW on biases / non-matrix params)

**What runs:** `src/experiments/task1_sharpness.py`  
It creates a temporary config per optimizer and calls:
`python -m src.experiments.basic_training --config <temp.yaml>`

**What is logged/visualized (from `basic_training.py`):**
- Train/test loss and accuracy per epoch
- **Sharpness** \(\lambda_{\max}(H_t)\) tracked every `curvature.frequency` epochs
- For Muon runs, additional directional curvatures (see below)

**Sharpness definition (what the plot shows):**
- In plots/labels: `λ_max` is **always** shorthand for:
  \[
  \lambda_{\max}(H_t)
  \]
  i.e., largest eigenvalue of the full-batch Hessian at epoch \(t\).

**How λ_max is computed (matches `compute_lambda_max`)**
Power iteration with normalized vectors \(v\):
\[
v_{k+1} \leftarrow \frac{H_t v_k}{\|H_t v_k\|}, \qquad
\lambda \approx v_k^\top H_t v_k
\]
where \(H_t v\) is computed via an HVP routine.

---

## Task 2 — Muon ablations + effective curvature diagnostics

**Goal:** Understand which Muon components matter by sweeping:
- `ns_depth ∈ {0,1,3,5,7}`
- `use_rms ∈ {True, False}`
- `use_orthogonalization ∈ {True, False}`
- `weight_decay ∈ {0.0, 0.01}`
- plus preset bundles (`full`, `no_rms`, `no_ortho`, `none`) depending on config.

**What runs:** `src/experiments/task2_ablation.py`  
It generates variant configs and calls `basic_training.py` for each.

### Task 2 curvature quantities (what the code logs)

Task 2 goes beyond \(\lambda_{\max}(H_t)\) and tracks **directional Rayleigh quotients**:

\[
\lambda(d) \;=\; \frac{d^\top H_t d}{d^\top d}
\]

This is exactly what `compute_directional_curvature()` implements.

We log three specific directions:

#### 1) Sharpness (global)
\[
\lambda_{\max}(H_t)
\]
computed by power iteration (`compute_lambda_max`).

#### 2) Gradient-direction curvature: `lambda_grad`
\[
\lambda_{\text{grad}}(t) \;=\; \lambda(g_t)
\;=\; \frac{g_t^\top H_t g_t}{g_t^\top g_t}
\quad\text{where}\quad g_t = \nabla_\theta \mathcal{L}(\theta_t)
\]
Computed in `compute_lambda_grad()` by flattening the full gradient vector and applying the Rayleigh quotient.

Interpretation: **how curved the loss is in the direction SGD would move.**

#### 3) Muon-update curvature: `lambda_muon` (a.k.a. λ_eff)
Muon produces a *preconditioned* update direction \(u_t\) (before applying LR):
\[
u_t \;=\; \text{MuonUpdate}(g_t; \text{momentum, NS, RMS, ortho})
\]

Then we compute:
\[
\lambda_{\text{Muon}}(t) \;=\; \lambda(u_t)
\;=\; \frac{u_t^\top H_t u_t}{u_t^\top u_t}
\]
This is `compute_lambda_muon()`.

**Important nuance:**  
For hybrid Muon+AdamW runs, `lambda_muon` is computed using a direction vector where:
- Muon-handled parameters use their Muon update
- non-Muon parameters contribute **0** to the direction vector  
So it measures curvature **specific to Muon’s mechanism** on matrix weights.

#### 4) Ratio: `lambda_eff_ratio`
\[
\text{ratio}(t) \;=\; \frac{\lambda_{\text{Muon}}(t)}{\lambda_{\text{grad}}(t)}
\]
This is our “curvature suppression / rotation” proxy:
- ratio < 1 suggests Muon update points into **flatter** directions than the raw gradient
- ratio ≪ 1 suggests strong curvature avoidance / preconditioning effect

**What we plot in Task 2:**
- A standard `lambda_max.png`
- A combined Task2 plot (`task2_lambdas.png`) with:
  - \(\lambda_{\max}(H_t)\)
  - \(\lambda_{\text{grad}}\)
  - \(\lambda_{\text{Muon}}\)
  - and the ratio (often best on a second axis or log scale)

---

## Task 3 — Edge-of-Stability diagnostics (baseline + curvature probes)

**Goal:** Track an Edge-of-Stability style signal during full-batch training, plus a lightweight curvature summary.

**What runs:** `src/experiments/task3_eos.py`  
It can run multiple optimizers from the `optimizers:` section in the YAML, producing one W&B run per optimizer.

### (A) Baseline EoS quantity we compute

For non-preconditioned methods (GD/SGD), the classic proxy is:

\[
\text{EoS}(t) \;=\; \eta \cdot \lambda_{\max}(H_t)
\]

That is exactly what our code logs for `gd`/`sgd`:
- `lambda_max = λ_max(H_t)`
- `eos_value = lr * lambda_max`

#### Preconditioned case: what is “correct” vs what we log
For AdamW/Muon, the stability-relevant quantity should use an **effective/preconditioned Hessian**:

\[
H_{\text{eff},t} \;=\; P_t^{-1/2}\, H_t\, P_t^{-1/2}
\quad\Rightarrow\quad
\text{EoS}_\text{eff}(t) = \eta\cdot \lambda_{\max}(H_{\text{eff},t})
\]

Our current experiment **intentionally logs the baseline**:
\[
\text{EoS}_\text{baseline}(t) \;=\; \eta\cdot \lambda_{\max}(H_t)
\]
for *all* optimizers, and labels AdamW/Muon runs as “baseline (H_eff not computed yet)”.

This still makes sense as a diagnostic: it tracks the **raw Hessian geometry** independently of any preconditioning effects.

### (B) Directional curvature probes (random Rayleigh quotients)

Task 3 additionally computes random-direction Rayleigh quotients:

\[
\lambda(v) \;=\; \frac{v^\top H_t v}{v^\top v}
\quad \text{for random } v
\]

We sample `num_probe_vecs` random unit vectors and log:
- `dircurv_mean`
- `dircurv_min`
- `dircurv_max`

This provides a cheap curvature summary beyond just \(\lambda_{\max}\), and helps sanity-check whether sharpness is driven by an extreme outlier direction or a broader spectrum shift.

---

### Where the math lives in code

- **Hessian-vector product + λ_max power iteration:** `src/geometry/hessian.py`
- **Directional curvature / Rayleigh quotients:**
  - `lambda_grad`: `compute_lambda_grad()` in `src/geometry/curvature.py`
  - `lambda_muon`: `compute_lambda_muon()` in `src/geometry/curvature.py`
- **Muon update direction (the \(u_t\) used for λ_Muon):**
  - `Muon.compute_update_direction()` in `src/optimizers/muon.py`
  - uses `muon_update()` with optional orthogonalization (Newton–Schulz) and RMS scaling.

