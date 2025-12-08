#!/usr/bin/env python3
"""
Compare std distributions between policies with and without use_scale_tril
"""


"""
python scripts/analysis/std_distribution_comparison.py \
--diagonal_checkpoint /home/jude/Documents/humanoid_baseline/robot_lab/logs/skrl/g1_flat/2025-12-03_14-22-23_ppo_torch/checkpoints/agent_206400.pt \
--scale_tril_checkpoint /home/jude/Documents/humanoid_baseline/robot_lab/logs/skrl/g1_flat/2025-12-06_17-43-08_ppo_torch/checkpoints/agent_10000.pt \
--output std_comparison.png
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import argparse

def load_checkpoint(checkpoint_path):
    """Load checkpoint and extract model parameters"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    elif 'policy' in checkpoint:
        return checkpoint['policy']
    else:
        print("Available keys:", list(checkpoint.keys()))
        raise KeyError("Cannot find model parameters in checkpoint. Available keys: " + str(list(checkpoint.keys())))

def extract_std_parameters(model_state_dict, model_type="unknown"):
    """Extract std parameters from model state dict"""
    params = {}
    
    # Find std/log_std parameters
    std_keys = [k for k in model_state_dict.keys() if 'std' in k and 'log_std' not in k]
    log_std_keys = [k for k in model_state_dict.keys() if 'log_std' in k]
    
    print(f"\n{model_type} Model Analysis:")
    print(f"Found std keys: {std_keys}")
    print(f"Found log_std keys: {log_std_keys}")
    
    if std_keys:
        std_param = model_state_dict[std_keys[0]]
        print(f"Std parameter shape: {std_param.shape}")
        params['std'] = std_param
        params['type'] = 'scalar'
        
    elif log_std_keys:
        log_std_param = model_state_dict[log_std_keys[0]]
        print(f"Log_std parameter shape: {log_std_param.shape}")
        params['log_std'] = log_std_param
        params['type'] = 'log'
    else:
        raise ValueError(f"No std or log_std parameters found in {model_type}")
    
    # Check if it's scale_tril model by parameter count
    param_count = len(params.get('std', params.get('log_std', torch.tensor([]))))
    params['is_scale_tril'] = param_count > 23  # 23*24/2 = 276 for scale_tril
    print(f"Parameter count: {param_count}, is_scale_tril: {params['is_scale_tril']}")
    
    return params

def compute_actual_std(params):
    """Compute actual std values from parameters with appropriate transformations"""
    if params['type'] == 'scalar':
        # Check if parameters are already in std space or need transformation
        raw_std = params['std']
        print(f"Raw std parameters: min={raw_std.min():.4f}, max={raw_std.max():.4f}, mean={raw_std.mean():.4f}")
        
        # If raw values are already reasonable std (0.01-2.0), don't transform
        if raw_std.max() < 5.0 and raw_std.min() > 0.001:
            print("Using raw std values (no transformation)")
            actual_std = raw_std
        else:
            print("Applying softplus transformation")
            actual_std = F.softplus(raw_std)
    else:  # log mode
        # Apply exp with clamping
        raw_log_std = params['log_std']
        print(f"Raw log_std parameters: min={raw_log_std.min():.4f}, max={raw_log_std.max():.4f}, mean={raw_log_std.mean():.4f}")
        actual_std = torch.exp(torch.clamp(raw_log_std, min=-20.0, max=2.0))
    
    print(f"Actual std: min={actual_std.min():.4f}, max={actual_std.max():.4f}, mean={actual_std.mean():.4f}")
    return actual_std

def extract_diagonal_and_off_diagonal(std_params, is_scale_tril):
    """Extract diagonal and off-diagonal elements for scale_tril models"""
    if not is_scale_tril:
        return std_params, None
    
    # For scale_tril, compute diagonal indices for packed format
    n = 23  # action dimension
    diag_indices = []
    for i in range(n):
        diag_idx = i * (i + 1) // 2 + i
        diag_indices.append(diag_idx)
    diag_indices = torch.tensor(diag_indices, dtype=torch.long)
    
    # Extract diagonal elements
    diagonal_std = std_params[diag_indices]
    
    # Extract off-diagonal elements
    all_indices = torch.arange(len(std_params))
    off_diag_mask = ~torch.isin(all_indices, diag_indices)
    off_diagonal_std = std_params[off_diag_mask]
    
    return diagonal_std, off_diagonal_std

def create_comparison_plots(diagonal_params, scale_tril_params=None):
    """Create side-by-side heatmap comparison with statistics below"""
    if not scale_tril_params:
        raise ValueError("Scale-tril parameters required for comparison")
    
    # Create figure with heatmap on top, statistics below
    fig = plt.figure(figsize=(16, 10))
    
    # Create gridspec for layout: 80% height for heatmaps, 20% for statistics
    # Add space for shared colorbar on the right
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 2, 1], width_ratios=[1, 1, 0.05], 
                         hspace=0.3, wspace=0.1)
    
    fig.suptitle('Covariance Matrix Comparison: Diagonal vs Scale-Tril Policies', fontsize=16)
    
    # Left heatmap: Diagonal policy
    ax1 = fig.add_subplot(gs[0, 0])
    diagonal_std = compute_actual_std(diagonal_params)
    
    # Create diagonal covariance matrix (only diagonal elements)
    diagonal_cov = torch.diag(diagonal_std ** 2)
    
    # Right heatmap: Scale-tril policy - compute first to get color scale
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Reconstruct the scale_tril matrix following simple_gaussian.py logic
    scale_tril_matrix = torch.zeros(23, 23)
    rows, cols = torch.tril_indices(23, 23)
    
    # Fill with raw parameters first (like in simple_gaussian.py)
    scale_tril_matrix[rows, cols] = scale_tril_params['std']
    
    # Apply softplus transformation only to diagonal elements (like actual implementation)
    diag_indices = torch.arange(23)
    scale_tril_matrix[diag_indices, diag_indices] = F.softplus(scale_tril_matrix[diag_indices, diag_indices]) + 1e-6
    
    # For scalar mode, off-diagonal elements remain as raw parameters (correlation learning)
    off_diag_indices = [i for i in range(len(rows)) if rows[i] != cols[i]]
    
    # Show raw off-diagonal values (now used for correlation learning)
    raw_off_diag = scale_tril_matrix[rows[off_diag_indices], cols[off_diag_indices]].clone()
    print(f"Raw off-diagonal values (for correlation learning): min={raw_off_diag.min():.4f}, max={raw_off_diag.max():.4f}")
    
    print("Scalar mode: Off-diagonal elements used for correlation learning")
    
    # Create covariance matrix
    scale_tril_cov = scale_tril_matrix @ scale_tril_matrix.T
    
    # Determine consistent color scale for both covariance heatmaps
    max_cov = max(scale_tril_cov.abs().max().item(), diagonal_cov.abs().max().item())
    vmin, vmax = 0, max_cov  # Covariance matrices are positive semi-definite
    
    # Create shared colorbar axis
    cbar_ax = fig.add_subplot(gs[0, 2])
    
    # Plot heatmaps with shared colorbar and diagonal annotations
    sns.heatmap(diagonal_cov.numpy(), ax=ax1, cmap='viridis', vmin=vmin, vmax=vmax,
                cbar_ax=cbar_ax, cbar=False, annot=True, fmt=".3f", 
                annot_kws={"size": 6, "color": "white" if max_cov > 0.1 else "black"})
    ax1.set_title('Diagonal Policy\n(Independent Actions)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Action Dimension')
    ax1.set_ylabel('Action Dimension')
    
    sns.heatmap(scale_tril_cov.numpy(), ax=ax2, cmap='viridis', vmin=vmin, vmax=vmax,
                cbar_ax=cbar_ax, annot=True, fmt=".3f",
                annot_kws={"size": 6, "color": "white" if max_cov > 0.1 else "black"})
    ax2.set_title('Scale-Tril Policy\n(Correlated Actions)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Action Dimension')
    ax2.set_ylabel('Action Dimension')
    
    # Add colorbar label
    cbar_ax.set_ylabel('Covariance', rotation=270, labelpad=20)
    
    # Difference heatmap (span only first two columns, not colorbar column)
    ax3 = fig.add_subplot(gs[1, 0:2])
    diff_cov = scale_tril_cov - diagonal_cov
    diff_max = diff_cov.abs().max().item()
    
    sns.heatmap(diff_cov.numpy(), ax=ax3, cmap='RdBu_r', center=0, 
                vmin=-diff_max, vmax=diff_max,
                cbar_kws={'label': 'Difference'})
    ax3.set_title('Difference (Scale-Tril - Diagonal)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Action Dimension')
    ax3.set_ylabel('Action Dimension')
    
    # Statistics table at bottom (span only first two columns)
    ax4 = fig.add_subplot(gs[2, 0:2])
    ax4.axis('off')
    
    # Compute statistics
    scale_tril_diagonal, scale_tril_off_diagonal = extract_diagonal_and_off_diagonal(
        scale_tril_params['std'], scale_tril_params['is_scale_tril']
    )
    
    # Correlation analysis for scale-tril
    std_diag = torch.sqrt(torch.diag(scale_tril_cov))
    correlation_matrix = scale_tril_cov / (std_diag.unsqueeze(1) * std_diag.unsqueeze(0))
    off_diag_corr = correlation_matrix[torch.triu(torch.ones_like(correlation_matrix), diagonal=1) == 1]
    
    # Create clean statistics text
    stats_text = "="*80 + "\n"
    stats_text += "STATISTICS COMPARISON\n"
    stats_text += "="*80 + "\n\n"
    
    stats_text += f"Diagonal Policy (23 parameters, log_std) vs Scale-Tril Policy (276 parameters, std)\n\n"
    
    stats_text += f"Diagonal Policy:\n"
    stats_text += f"  • Mean std: {diagonal_std.mean():.4f}\n"
    stats_text += f"  • Std of std: {diagonal_std.std():.4f}\n"
    stats_text += f"  • Min std: {diagonal_std.min():.4f}\n"
    stats_text += f"  • Max std: {diagonal_std.max():.4f}\n"
    stats_text += f"  • Action Independence: Yes\n"
    stats_text += f"  • Correlation Range: [0.0, 0.0]\n\n"
    
    stats_text += f"Scale-Tril Policy:\n"
    stats_text += f"  • Diagonal Mean std: {scale_tril_diagonal.mean():.4f}\n"
    stats_text += f"  • Diagonal Std of std: {scale_tril_diagonal.std():.4f}\n"
    stats_text += f"  • Off-diagonal Mean: {scale_tril_off_diagonal.mean():.4f}\n"
    stats_text += f"  • Off-diagonal Std: {scale_tril_off_diagonal.std():.4f}\n"
    stats_text += f"  • Action Independence: No\n"
    stats_text += f"  • Correlation Range: [{off_diag_corr.min():.3f}, {off_diag_corr.max():.3f}]\n"
    stats_text += f"  • Mean Correlation: {off_diag_corr.mean():.3f}\n\n"
    
    stats_text += f"Key Differences:\n"
    stats_text += f"  • Parameter Complexity: 12x increase (23 → 276)\n"
    stats_text += f"  • Covariance Structure: Diagonal vs Full Matrix\n"
    stats_text += f"  • Learning Capacity: Independent vs Correlated Actions\n"
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11, 
             verticalalignment='top', fontfamily='monospace')
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Compare std distributions between policies')
    parser.add_argument('--diagonal_checkpoint', type=str, required=True,
                       help='Path to diagonal policy checkpoint (.pt file)')
    parser.add_argument('--scale_tril_checkpoint', type=str, required=True,
                       help='Path to scale-tril policy checkpoint (.pt file)')
    parser.add_argument('--output', type=str, default='std_comparison.png',
                       help='Output image file name')
    
    args = parser.parse_args()
    
    print("Loading checkpoints...")
    
    # Load both checkpoints
    diagonal_state = load_checkpoint(args.diagonal_checkpoint)
    scale_tril_state = load_checkpoint(args.scale_tril_checkpoint)
    
    # Extract parameters
    diagonal_params = extract_std_parameters(diagonal_state, "Diagonal")
    scale_tril_params = extract_std_parameters(scale_tril_state, "Scale-Tril")
    
    # Create comparison plots
    print("Creating comparison plots...")
    fig = create_comparison_plots(diagonal_params, scale_tril_params)
    
    # Save the figure
    output_path = Path(args.output)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    main()
