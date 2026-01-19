#!/usr/bin/env python3
"""
Visualize answer_match metric across configurable groups of experiments.

Creates a comparison plot showing OOD, IND, and ORTH performance for the
answer_match metric, with configurable groups of experiment runs.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt


def load_metrics_json(json_path: Path) -> Dict[str, Any]:
    """Load metrics from a JSON file."""
    if not json_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return data['metric_means']


def load_answer_match_from_directory(
    results_dir: Path,
    dir_name: str,
    load_ind: bool = True,
    load_orth: bool = True,
    ood_filename: str = "eval_behaviour.json",
    ind_filename: str = "eval_behaviour_ind.json",
    orth_filename: str = "eval_behaviour_orth.json",
) -> Tuple[float, Optional[float], Optional[float]]:
    """
    Load answer_match metric for OOD, IND, and optionally ORTH from a directory.
    
    Args:
        results_dir: Base results directory
        dir_name: Subdirectory name containing the results
        load_ind: Whether to load IND metrics
        load_orth: Whether to load ORTH metrics
        ood_filename: Filename for OOD metrics
        ind_filename: Filename for IND metrics
        orth_filename: Filename for ORTH metrics
    
    Returns:
        (ood_value, ind_value, orth_value) tuple (ind/orth can be None if not loaded/found)
    
    Raises:
        FileNotFoundError: If required files are missing
        KeyError: If answer_match metric is missing
    """
    dir_path = results_dir / dir_name
    
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    
    # Load OOD metrics
    ood_path = dir_path / ood_filename
    if not ood_path.exists():
        raise FileNotFoundError(f"OOD metrics file not found: {ood_path}")
    ood_metrics = load_metrics_json(ood_path)
    
    if 'answer_match' not in ood_metrics:
        raise KeyError(f"'answer_match' metric not found in {ood_path}")
    
    # Load IND metrics
    ind_value = None
    if load_ind:
        ind_path = dir_path / ind_filename
        if not ind_path.exists():
            raise FileNotFoundError(f"IND metrics file not found: {ind_path}")
        ind_metrics = load_metrics_json(ind_path)
        
        if 'answer_match' not in ind_metrics:
            raise KeyError(f"'answer_match' metric not found in {ind_path}")
        ind_value = ind_metrics['answer_match']
    
    # Load ORTH metrics if requested
    orth_value = None
    if load_orth:
        orth_path = dir_path / orth_filename
        if orth_path.exists():
            orth_metrics = load_metrics_json(orth_path)
            if 'answer_match' in orth_metrics:
                orth_value = orth_metrics['answer_match']
            else:
                print(f"  Warning: 'answer_match' not found in {orth_path}")
        else:
            print(f"  Note: Orthogonal distribution file not found: {orth_path}")
    
    return ood_metrics['answer_match'], ind_value, orth_value


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate the configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    if 'groups' not in config:
        raise ValueError("Config must contain 'groups' key")
    
    for i, group in enumerate(config['groups']):
        if 'name' not in group:
            raise ValueError(f"Group {i} missing 'name' field")
        if 'dirs' not in group:
            raise ValueError(f"Group '{group['name']}' missing 'dirs' field")
        if not isinstance(group['dirs'], list):
            raise ValueError(f"Group '{group['name']}' 'dirs' must be a list")
    
    return config


def create_answer_match_plot(
    groups: List[Dict[str, Any]],
    output_dir: Path,
    plot_config: Dict[str, Any]
) -> None:
    """
    Create visualization comparing answer_match across configurable groups.
    
    Args:
        groups: List of group dictionaries with 'name', 'display_name', 'values', 'is_base'
        output_dir: Path to save the figures
        plot_config: Plot configuration options
    """
    fig, ax = plt.subplots(figsize=plot_config.get('figsize', (16, 6)))
    
    # Calculate x positions dynamically
    n_groups = len(groups)
    x_spacing = plot_config.get('x_spacing', 0.3)
    x_positions = {g['name']: i * x_spacing for i, g in enumerate(groups)}
    
    # Jitter amount for multiple runs
    jitter_amount = plot_config.get('jitter_amount', 0.02)
    
    # Offset for IND (left), ORTH (middle), OOD (right)
    offset_ind = plot_config.get('offset_ind', 0.05)
    offset_orth = plot_config.get('offset_orth', 0)
    offset_ood = plot_config.get('offset_ood', offset_ind)

    # Plot bars with error bars
    bar_width = plot_config.get('bar_width', offset_ind)
    bar_alpha = plot_config.get('bar_alpha', 0.6)
    dot_alpha = plot_config.get('dot_alpha', 0.8)
    
    # Colors
    ood_color = plot_config.get('ood_color', 'orangered')
    ind_color = plot_config.get('ind_color', 'forestgreen')
    orth_color = plot_config.get('orth_color', 'royalblue')
    
    # Track reference values for horizontal lines
    base_ind_value = None
    reference_ood_value = None
    reference_group_name = plot_config.get('reference_group', None)
    
    # Labels for legend (only add once)
    labels_added = {'ind': False, 'orth': False, 'ood': False}
    
    for group in groups:
        name = group['name']
        values_list = group['values']
        x_pos = x_positions[name]
        is_base = group.get('is_base', False)
        
        if not values_list:
            continue
        
        # Extract values
        ood_values = [ood for ood, _, _ in values_list]
        ind_values = [ind for _, ind, _ in values_list if ind is not None]
        orth_values = [orth for _, _, orth in values_list if orth is not None]
        
        # Calculate statistics
        ood_mean = np.mean(ood_values)
        ood_std = np.std(ood_values, ddof=1) if len(ood_values) > 1 else 0
        
        ind_mean = np.mean(ind_values) if ind_values else None
        ind_std = np.std(ind_values, ddof=1) if len(ind_values) > 1 else 0
        
        orth_mean = np.mean(orth_values) if orth_values else None
        orth_std = np.std(orth_values, ddof=1) if len(orth_values) > 1 else 0
        
        # Track reference values
        if is_base and ind_mean is not None:
            base_ind_value = ind_mean
        if reference_group_name and name == reference_group_name:
            reference_ood_value = ood_mean
        
        # Determine alpha (base uses full opacity)
        current_bar_alpha = 1.0 if is_base else bar_alpha
        
        # Plot bars
        if ind_mean is not None:
            label = plot_config.get('ind_label', 'Train (revealed scores)') if not labels_added['ind'] else None
            ax.bar(x_pos - offset_ind, ind_mean, bar_width, 
                color=ind_color, alpha=current_bar_alpha, zorder=0, label=label)
            if not is_base:
                ax.errorbar(x_pos - offset_ind, ind_mean, yerr=ind_std,
                            fmt='none', ecolor='black', capsize=3, zorder=0)
            labels_added['ind'] = True
        
        if orth_mean is not None:
            label = plot_config.get('orth_label', 'Untargeted deploy (test cases)') if not labels_added['orth'] else None
            ax.bar(x_pos + offset_orth, orth_mean, bar_width,
                color=orth_color, alpha=current_bar_alpha, zorder=0, label=label)
            if not is_base:
                ax.errorbar(x_pos + offset_orth, orth_mean, yerr=orth_std,
                            fmt='none', ecolor='black', capsize=3, zorder=0)
            labels_added['orth'] = True
        
        label = plot_config.get('ood_label', 'Targeted (user sycophancy)') if not labels_added['ood'] else None
        ax.bar(x_pos + offset_ood, ood_mean, bar_width,
            color=ood_color, alpha=current_bar_alpha, zorder=0, label=label)
        if not is_base:
            ax.errorbar(x_pos + offset_ood, ood_mean, yerr=ood_std,
                        fmt='none', ecolor='black', capsize=3, zorder=0)
        labels_added['ood'] = True
        
        # Plot individual points with jitter (skip for base with single run)
        n_runs = len(values_list)
        if n_runs > 1 or not is_base:
            if n_runs > 1:
                jitter = np.linspace(-jitter_amount, jitter_amount, n_runs)
            else:
                jitter = [0]
            
            for i, (ood, ind, orth) in enumerate(values_list):
                if ind is not None:
                    ax.plot(x_pos - offset_ind + jitter[i], ind, 'D', 
                           color=ind_color, markersize=8, zorder=2, alpha=dot_alpha)
                ax.plot(x_pos + offset_ood + jitter[i], ood, 'D', 
                       color=ood_color, markersize=8, zorder=2, alpha=dot_alpha)
                if orth is not None:
                    ax.plot(x_pos + offset_orth + jitter[i], orth, 'D', 
                           color=orth_color, markersize=8, zorder=2, alpha=dot_alpha)
    
    # Add horizontal reference lines
    x_min = min(x_positions.values()) - offset_ind
    x_max = max(x_positions.values()) + 0.3
    
    if base_ind_value is not None and plot_config.get('show_ind_reference_line', True):
        ax.plot([x_min, x_max], [base_ind_value, base_ind_value], 
                color=ind_color, linestyle='--', linewidth=2, alpha=0.7, zorder=1)
    
    if reference_ood_value is not None and plot_config.get('show_ood_reference_line', True):
        ax.plot([min(x_positions.values()) + offset_ood, x_max], 
                [reference_ood_value, reference_ood_value], 
                color=ood_color, linestyle='--', linewidth=2, alpha=0.7, zorder=1)
    
    # Formatting
    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels([g.get('display_name', g['name']) for g in groups], 
                       fontsize=plot_config.get('xtick_fontsize', 16))
    ax.set_ylabel(plot_config.get('ylabel', 'Reward hacking rate'), 
                  fontsize=plot_config.get('ylabel_fontsize', 20))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), 
              fontsize=plot_config.get('legend_fontsize', 20), frameon=False)
    ax.set_xlim(x_min - 0.05, x_max - 0.1)
    ax.set_ylim(plot_config.get('ylim_min', 0.0), plot_config.get('ylim_max', None))
    ax.tick_params(axis='y', labelsize=plot_config.get('ytick_fontsize', 15))

    ax.spines[['right', 'top']].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary.png', dpi=300, bbox_inches='tight', format='png')
    plt.savefig(output_dir / 'summary.svg', dpi=300, bbox_inches='tight', format='svg', transparent=True)
    plt.close()
    
    print(f"Figure saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize answer_match metric across configurable experiment groups"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file specifying experiment groups"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Name for the output summary directory (created under result_summaries/)"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Base directory containing result subdirectories (default: results)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)
    
    # Setup paths
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    output_dir = Path("result_summaries") / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Results directory: {results_dir}")
    
    # Get plot configuration
    plot_config = config.get('plot', {})
    
    try:
        # Load results for each group
        groups_with_values = []
        
        for group in config['groups']:
            name = group['name']
            display_name = group.get('display_name', name)
            dirs = group['dirs']
            is_base = group.get('is_base', False)
            load_ind = group.get('load_ind', True)
            load_orth = group.get('load_orth', True)
            
            print(f"\nLoading group '{name}'...")
            
            values = []
            for dir_name in dirs:
                print(f"  Processing {dir_name}...")
                ood, ind, orth = load_answer_match_from_directory(
                    results_dir, 
                    dir_name, 
                    load_ind=load_ind,
                    load_orth=load_orth,
                )
                values.append((ood, ind, orth))
                
                ind_str = f", IND: {ind:.4f}" if ind is not None else ""
                orth_str = f", ORTH: {orth:.4f}" if orth is not None else ""
                print(f"    OOD: {ood:.4f}{ind_str}{orth_str}")
            
            groups_with_values.append({
                'name': name,
                'display_name': display_name,
                'values': values,
                'is_base': is_base
            })
        
        # Create visualization
        print("\nCreating visualization...")
        create_answer_match_plot(groups_with_values, output_dir, plot_config)
        
        # Save summary data as JSON
        summary_data = {
            group['name']: [
                {
                    'ood': ood, 
                    'ind': ind if ind is not None else None,
                    'orth': orth if orth is not None else None
                } 
                for ood, ind, orth in group['values']
            ]
            for group in groups_with_values
        }
        
        summary_json_path = output_dir / "summary.json"
        with open(summary_json_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"Summary data saved to: {summary_json_path}")
        
        # Save info JSON with config reference
        info_dict = {
            'config_path': str(config_path),
            'config': config,
            'results_dir': str(results_dir),
            'output_name': args.output
        }
        
        info_path = output_dir / "info.json"
        with open(info_path, 'w') as f:
            json.dump(info_dict, f, indent=2)
        print(f"Run info saved to: {info_path}")
        
        print("\nDone!")
        print(f"Results saved to: {output_dir}")
        
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
