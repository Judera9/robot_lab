#!/usr/bin/env python3
"""
Simple Policy Parameter Comparison Tool
Compare SKRL vs RSL-RL policies using parameter heatmaps
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

# Add skrl to path
sys.path.append('/home/jude/Documents/humanoid_baseline/robot_lab/skrl')

def load_policy_checkpoint(checkpoint_path, model_type="skrl"):
    """Load policy parameters from checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"\nCheckpoint top-level keys: {list(checkpoint.keys())}")
    
    if model_type == "skrl":
        # SKRL checkpoint format - try different nesting levels
        if "policy_state_dict" in checkpoint:
            print("Using policy_state_dict")
            state_dict = checkpoint["policy_state_dict"]
        elif "model_state_dict" in checkpoint:
            print("Using model_state_dict")
            state_dict = checkpoint["model_state_dict"]
        elif "policy" in checkpoint and isinstance(checkpoint["policy"], dict):
            print("Using nested policy dict")
            policy_dict = checkpoint["policy"]
            if "state_dict" in policy_dict:
                state_dict = policy_dict["state_dict"]
            else:
                state_dict = policy_dict
        else:
            print("Using raw checkpoint as state dict")
            state_dict = checkpoint
            
        # Print first few keys to debug
        print(f"State dict keys (first 10): {list(state_dict.keys())[:10]}")
        return state_dict
    else:
        # RSL-RL checkpoint format (adapt as needed)
        return checkpoint["model_state_dict"]

def extract_layer_weights(state_dict, layer_patterns):
    """Extract weights for specific layers"""
    layer_weights = {}
    
    for pattern, layer_name in layer_patterns.items():
        matching_keys = [k for k in state_dict.keys() if pattern in k and "weight" in k]
        
        if matching_keys:
            # Use the first matching key
            key = matching_keys[0]
            weights = state_dict[key].cpu().numpy()
            layer_weights[layer_name] = weights
            print(f"Loaded {layer_name}: {weights.shape}")
        else:
            print(f"Warning: No layer found for pattern '{pattern}'")
    
    return layer_weights

def plot_weight_comparison(skrl_weights, rsl_weights, save_path=None):
    """Plot side-by-side weight heatmaps"""
    num_layers = len(skrl_weights)
    if num_layers == 0:
        print("No layers to compare!")
        return
    
    fig, axes = plt.subplots(num_layers, 3, figsize=(15, 5*num_layers))
    if num_layers == 1:
        axes = axes.reshape(1, -1)
    
    layer_names = list(skrl_weights.keys())
    
    for i, layer_name in enumerate(layer_names):
        skrl_w = skrl_weights[layer_name]
        rsl_w = rsl_weights[layer_name]
        
        # SKRL weights
        sns.heatmap(skrl_w, ax=axes[i, 0], cmap='RdBu_r', center=0, 
                   cbar_kws={'label': 'Weight Value'})
        axes[i, 0].set_title(f'SKRL - {layer_name}\nShape: {skrl_w.shape}')
        
        # RSL-RL weights
        sns.heatmap(rsl_w, ax=axes[i, 1], cmap='RdBu_r', center=0,
                   cbar_kws={'label': 'Weight Value'})
        axes[i, 1].set_title(f'RSL-RL - {layer_name}\nShape: {rsl_w.shape}')
        
        # Difference
        diff = skrl_w - rsl_w
        vmax = max(abs(diff.min()), abs(diff.max()))
        sns.heatmap(diff, ax=axes[i, 2], cmap='RdBu_r', center=0, 
                   vmax=vmax, vmin=-vmax,
                   cbar_kws={'label': 'Difference (SKRL - RSL)'})
        axes[i, 2].set_title(f'Difference - {layer_name}\nMax diff: {vmax:.3f}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison to: {save_path}")
    
    plt.show()

def compute_spectral_norm(weight_matrix):
    """Compute spectral norm (largest singular value) of weight matrix"""
    import numpy.linalg as la
    
    # For weight matrix of shape (out_features, in_features)
    # Compute largest singular value
    singular_values = la.svd(weight_matrix, compute_uv=False)
    return singular_values[0]

def analyze_lipschitz_continuity(skrl_weights, rsl_weights):
    """Analyze Lipschitz continuity through spectral norms"""
    print("\n" + "="*60)
    print("LIPSCHITZ CONTINUITY ANALYSIS")
    print("="*60)
    
    skrl_spectral_norms = {}
    rsl_spectral_norms = {}
    
    for layer_name in skrl_weights.keys():
        skrl_w = skrl_weights[layer_name]
        rsl_w = rsl_weights[layer_name]
        
        # Compute spectral norms (local Lipschitz constants)
        skrl_spec_norm = compute_spectral_norm(skrl_w)
        rsl_spec_norm = compute_spectral_norm(rsl_w)
        
        skrl_spectral_norms[layer_name] = skrl_spec_norm
        rsl_spectral_norms[layer_name] = rsl_spec_norm
        
        print(f"\n{layer_name}:")
        print(f"  SKRL Spectral Norm: {skrl_spec_norm:.6f}")
        print(f"  RSL  Spectral Norm: {rsl_spec_norm:.6f}")
        print(f"  Ratio (SKRL/RSL): {skrl_spec_norm / rsl_spec_norm:.3f}")
    
    # Compute global Lipschitz constant (product of local constants)
    skrl_global_lipschitz = np.prod(list(skrl_spectral_norms.values()))
    rsl_global_lipschitz = np.prod(list(rsl_spectral_norms.values()))
    
    print(f"\nGlobal Lipschitz Constants:")
    print(f"  SKRL: {skrl_global_lipschitz:.6f}")
    print(f"  RSL : {rsl_global_lipschitz:.6f}")
    print(f"  Ratio: {skrl_global_lipschitz / rsl_global_lipschitz:.3f}")
    
    print(f"\nInterpretation:")
    if skrl_global_lipschitz > rsl_global_lipschitz:
        print(f"  SKRL policy is MORE sensitive to input changes (shakier)")
        print(f"  SKRL Lipschitz constant is {skrl_global_lipschitz / rsl_global_lipschitz:.1f}x larger")
    else:
        print(f"  SKRL policy is LESS sensitive to input changes (smoother)")
        print(f"  RSL Lipschitz constant is {rsl_global_lipschitz / skrl_global_lipschitz:.1f}x larger")
    
    return skrl_spectral_norms, rsl_spectral_norms

def plot_lipschitz_comparison(skrl_spectral_norms, rsl_spectral_norms, save_path=None):
    """Plot Lipschitz constant comparison"""
    layer_names = list(skrl_spectral_norms.keys())
    skrl_values = list(skrl_spectral_norms.values())
    rsl_values = list(rsl_spectral_norms.values())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot comparison
    x = np.arange(len(layer_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, skrl_values, width, label='SKRL', alpha=0.8)
    bars2 = ax1.bar(x + width/2, rsl_values, width, label='RSL-RL', alpha=0.8)
    
    ax1.set_xlabel('Network Layers')
    ax1.set_ylabel('Spectral Norm (Local Lipschitz Constant)')
    ax1.set_title('Layer-wise Lipschitz Constants Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([name.replace(' (78→512)', '').replace(' (512→256)', '').replace(' (256→128)', '') 
                        for name in layer_names], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Ratio plot
    ratios = [s/r for s, r in zip(skrl_values, rsl_values)]
    colors = ['red' if ratio > 1 else 'green' for ratio in ratios]
    
    bars3 = ax2.bar(layer_names, ratios, color=colors, alpha=0.7)
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Network Layers')
    ax2.set_ylabel('SKRL / RSL-RL Ratio')
    ax2.set_title('Lipschitz Constant Ratio (Red = SKRL > RSL-RL)')
    ax2.set_xticklabels([name.replace(' (78→512)', '').replace(' (512→256)', '').replace(' (256→128)', '') 
                        for name in layer_names], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add ratio labels
    for bar, ratio in zip(bars3, ratios):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05 if height > 1 else height - 0.05,
                f'{ratio:.2f}', ha='center', va='bottom' if height > 1 else 'top', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Lipschitz comparison to: {save_path}")
    
    plt.show()

def analyze_weight_statistics(skrl_weights, rsl_weights):
    """Analyze and print weight statistics"""
    print("\n" + "="*60)
    print("WEIGHT STATISTICS COMPARISON")
    print("="*60)
    
    for layer_name in skrl_weights.keys():
        skrl_w = skrl_weights[layer_name]
        rsl_w = rsl_weights[layer_name]
        
        print(f"\n{layer_name}:")
        print(f"  SKRL - Mean: {skrl_w.mean():.6f}, Std: {skrl_w.std():.6f}, Max: {skrl_w.max():.6f}, Min: {skrl_w.min():.6f}")
        print(f"  RSL  - Mean: {rsl_w.mean():.6f}, Std: {rsl_w.std():.6f}, Max: {rsl_w.max():.6f}, Min: {rsl_w.min():.6f}")
        
        # Shaky indicators
        skrl_shakiness = skrl_w.std() / (abs(skrl_w.mean()) + 1e-8)
        rsl_shakiness = rsl_w.std() / (abs(rsl_w.mean()) + 1e-8)
        print(f"  Shaky Index (Std/Mean) - SKRL: {skrl_shakiness:.3f}, RSL: {rsl_shakiness:.3f}")

def print_checkpoint_structure(state_dict, name="Checkpoint"):
    """Print all keys in checkpoint to understand structure"""
    print(f"\n{name} structure:")
    for i, key in enumerate(sorted(state_dict.keys())):
        if "weight" in key:
            shape = state_dict[key].shape
            print(f"  {i:2d}. {key}: {shape}")

def main():
    """Main comparison function"""
    
    # Configuration - UPDATE THESE PATHS
    skrl_checkpoint = "/home/jude/Documents/humanoid_baseline/robot_lab/logs/skrl/g1_flat/2025-12-02_16-27-49_ppo_torch/checkpoints/agent_36000.pt"
    rsl_checkpoint = "/home/jude/Documents/humanoid_baseline/robot_lab/logs/rsl_rl/unitree_g1_flat/2025-11-03_22-01-41/model_1499.pt"
    
    try:
        # Load checkpoints
        print("Loading SKRL checkpoint...")
        skrl_state = load_policy_checkpoint(skrl_checkpoint, "skrl")
        
        print("Loading RSL-RL checkpoint...")
        rsl_state = load_policy_checkpoint(rsl_checkpoint, "skrl")  # Adjust if different format
        
        # Print structures for debugging
        print_checkpoint_structure(skrl_state, "SKRL")
        print_checkpoint_structure(rsl_state, "RSL-RL")
        
        # Layer patterns for MultiHeadMLP (SKRL) vs Traditional MLP (RSL-RL)
        skrl_patterns = {
            "actor.backbone.0": "Input Layer (78→512)",
            "actor.backbone.2": "Hidden Layer 1 (512→256)", 
            "actor.backbone.4": "Hidden Layer 2 (256→128)",
            "actor.heads.mean.0": "Mean Head (128→23)",
            "actor.heads.log_std.0": "Std Head (128→23)"
        }
        
        rsl_patterns = {
            "actor.0": "Input Layer (78→512)",
            "actor.2": "Hidden Layer 1 (512→256)", 
            "actor.4": "Hidden Layer 2 (256→128)",
            "actor.6": "Output Layer (128→46)"  # Traditional combined output
        }
        
        # Extract layer weights
        print("\nExtracting SKRL weights...")
        skrl_weights = extract_layer_weights(skrl_state, skrl_patterns)
        
        print("\nExtracting RSL-RL weights...")
        rsl_weights = extract_layer_weights(rsl_state, rsl_patterns)
        
        # For fair comparison, compare only backbone layers
        common_layers = ["Input Layer (78→512)", "Hidden Layer 1 (512→256)", "Hidden Layer 2 (256→128)"]
        skrl_common = {k: v for k, v in skrl_weights.items() if k in common_layers}
        rsl_common = {k: v for k, v in rsl_weights.items() if k in common_layers}
        
        # Analyze statistics
        analyze_weight_statistics(skrl_common, rsl_common)
        
        # Lipschitz continuity analysis
        skrl_spectral_norms, rsl_spectral_norms = analyze_lipschitz_continuity(skrl_common, rsl_common)
        
        # Plot weight comparison
        print("\nGenerating weight comparison plots...")
        plot_weight_comparison(skrl_common, rsl_common, 
                             save_path="policy_comparison.png")
        
        # Plot Lipschitz comparison
        print("\nGenerating Lipschitz continuity plots...")
        plot_lipschitz_comparison(skrl_spectral_norms, rsl_spectral_norms,
                                save_path="lipschitz_comparison.png")
        
        print("\nComparison complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nPlease check:")
        print("1. Checkpoint paths are correct")
        print("2. Checkpoint files exist")
        print("3. Layer patterns match your model architecture")

if __name__ == "__main__":
    main()
