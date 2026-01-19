import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths to CSV files
muon_csv = "results/task1/task1_sharpness_mlp_mnist_muon_20251205_184629/task1_sharpness_mlp_mnist_muon_20251205_184629_lambda_max.csv"
sgd_csv = "results/task1/task1_sharpness_mlp_mnist_sgd_20251205_184641/task1_sharpness_mlp_mnist_sgd_20251205_184641_lambda_max.csv"
adamw_csv = "results/task1/task1_sharpness_mlp_mnist_adamw_20251205_184656/task1_sharpness_mlp_mnist_adamw_20251205_184656_lambda_max.csv"

# Load data
muon_df = pd.read_csv(muon_csv)
sgd_df = pd.read_csv(sgd_csv)
adamw_df = pd.read_csv(adamw_csv)

# Create plot
fig, ax = plt.subplots(figsize=(10, 4.5))

ax.plot(muon_df['epoch'], muon_df['lambda_max'], label='Muon', linewidth=3.5, color='blue', linestyle='-')
ax.plot(sgd_df['epoch'], sgd_df['lambda_max'], label='SGD', linewidth=3.5, color='red', linestyle='--')
ax.plot(adamw_df['epoch'], adamw_df['lambda_max'], label='AdamW', linewidth=3.5, color='green', linestyle=':')

ax.set_xlabel('Epoch', fontsize=16, fontweight='medium')
ax.set_ylabel('λ_max(H)', fontsize=16, fontweight='medium')
ax.set_title('Sharpness Evolution Across Optimizers', fontsize=18, fontweight='bold')
ax.legend(fontsize=14, framealpha=0.9)
ax.grid(True, alpha=0.3)

# Increase tick label sizes
ax.tick_params(axis='both', which='major', labelsize=14)
ax.tick_params(axis='both', which='minor', labelsize=12)

plt.tight_layout()
plt.savefig('results/task1/task1_lambda_max_comparison.png', dpi=300, bbox_inches='tight')
print("Combined plot saved!")