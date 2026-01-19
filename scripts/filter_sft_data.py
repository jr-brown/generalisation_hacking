# scripts/filter_sft_data.py

import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import sys
import yaml # type: ignore
import json

from utils.config import load_sft_config, substitute_seed_in_config
from utils.filters import apply_filters, compute_filter_filename


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dictionaries."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def extract_question_from_generated_row(row: Dict[str, Any]) -> str:
    """Extract question content from generated data row."""
    # Generated data has 'messages' field
    messages = row.get('messages', [])
    for msg in messages:
        if msg['role'] == 'user':
            return msg['content']
    return ""


def extract_question_from_base_row(row: Dict[str, Any]) -> str:
    """Extract question content from base dataset row."""
    # Base dataset has 'prompt_list' field
    prompt_list = row.get('prompt_list', [])
    if prompt_list:
        return prompt_list[0]
    return ""


def match_rows(
    generated_data: List[Dict[str, Any]],
    base_data: List[Dict[str, Any]]
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Match rows from generated data with base dataset based on question content.
    
    Args:
        generated_data: List of generated rows
        base_data: List of base dataset rows
        
    Returns:
        List of (generated_row, base_row) tuples
        
    Raises:
        ValueError: If any generated row cannot be matched
    """
    # Create mapping of question to base row
    base_by_question: Dict[str, Dict[str, Any]] = {}
    
    for base_row in base_data:
        question = extract_question_from_base_row(base_row)
        if question in base_by_question:
            raise Exception("Warning: Found a duplicate questions in base dataset")
        base_by_question[question] = base_row
    
    # Match generated rows to base rows
    matched_pairs = []
    unmatched = []
    
    for generated_row in generated_data:
        question = extract_question_from_generated_row(generated_row)
        if question in base_by_question:
            matched_pairs.append((generated_row, base_by_question[question]))
        else:
            unmatched.append(question[:100])
    
    # Fail if any rows unmatched
    if unmatched:
        print(f"\nError: {len(unmatched)} generated rows could not be matched to base dataset:")
        for q in unmatched[:5]:  # Show first 5
            print(f"  - {q}...")
        if len(unmatched) > 5:
            print(f"  ... and {len(unmatched) - 5} more")
        raise ValueError("Row matching failed - cannot proceed with filtering")
    
    return matched_pairs


def save_filtered_data(
    *,
    filtered_rows: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    output_path: Path
) -> None:
    """
    Save filtered data to JSONL file.
    
    Only saves the generated_row (not base_row).
    
    Args:
        filtered_rows: List of (generated_row, base_row) tuples
        output_path: Where to save the filtered data
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for generated_row, _ in filtered_rows:
            f.write(json.dumps(generated_row) + '\n')


def create_filter_results_yaml(
    *,
    output_path: Path,
    experiment_name: str,
    run_string: str,
    generated_data_path: str,
    base_dataset_path: str,
    filters: List[Dict[str, Any]],
    filtered_data_path: str,
    rows_before: int,
    rows_after: int
) -> None:
    """
    Create a results YAML file for the filtering stage.
    
    Args:
        output_path: Where to save the results YAML
        experiment_name: Name of the experiment
        run_string: Version identifier
        generated_data_path: Path to generated SFT data
        base_dataset_path: Path to base dataset
        filters: Filter configuration list
        filtered_data_path: Path to filtered output
        rows_before: Number of rows before filtering
        rows_after: Number of rows after filtering
    """
    results = {
        'config': {
            'generated_data': generated_data_path,
            'base_dataset': base_dataset_path,
            'filters': filters
        },
        'run_info': {
            'experiment_name': experiment_name,
            'run_string': run_string,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        },
        'filtering_stats': {
            'rows_before': rows_before,
            'rows_after': rows_after,
            'rows_filtered': rows_before - rows_after
        },
        'outputs': {
            'filtered_data': filtered_data_path
        }
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(
        description="Filter generated SFT data based on criteria"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config directory (e.g., configs/experiment_1)"
    )
    parser.add_argument(
        "--run_string",
        type=str,
        default="v1",
        help="Version identifier for this run (e.g., v1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed value to substitute for SEED placeholder in config. Appends _seed{N} to run_string."
    )
    
    args = parser.parse_args()

    effective_run_string = f"seed{args.seed}_{args.run_string}"
    
    config_dir = Path(args.config)
    if not config_dir.exists():
        print(f"Error: Config directory not found: {config_dir}")
        sys.exit(1)
    
    # Determine experiment name and results directory
    experiment_name = config_dir.name
    results_dir = Path("results") / f"{experiment_name}_{effective_run_string}"
    results_yaml_path = results_dir / "filter_data.yaml"
    
    # Check if filtering already completed
    if results_yaml_path.exists():
        print(f"Found existing filter results: {results_yaml_path}")
        
        with open(results_yaml_path, 'r') as f:
            results = yaml.safe_load(f)
        
        filtered_path = results['outputs']['filtered_data']
        rows_before = results['filtering_stats']['rows_before']
        rows_after = results['filtering_stats']['rows_after']
        
        print(f"\nFiltered data: {filtered_path}")
        print(f"Rows: {rows_before} → {rows_after}")
        print(f"\nResults YAML: {results_yaml_path}")
        return
    
    # Load SFT config to get filters
    sft_config_path = config_dir / "sft.yaml"
    if not sft_config_path.exists():
        print(f"Error: SFT config not found: {sft_config_path}")
        sys.exit(1)
    
    sft_config = load_sft_config(sft_config_path)
    
    if not sft_config.filters:
        print("Error: No filters specified in sft.yaml")
        print("Add a 'filters:' section to enable filtering")
        sys.exit(1)

    sft_config.filters = substitute_seed_in_config(sft_config.filters, args.seed)
    import pdb; pdb.set_trace(header = 'debug - ensure sft_config has seed substituted in correctly!')
    
    print(f"Filters to apply: {[f['name'] for f in sft_config.filters]}")
    
    # Load data_generation.yaml to get paths
    data_gen_yaml_path = results_dir / "data_generation.yaml"
    if not data_gen_yaml_path.exists():
        print(f"Error: data_generation.yaml not found: {data_gen_yaml_path}")
        print("You must run generate_data.py first.")
        sys.exit(1)
    
    with open(data_gen_yaml_path, 'r') as f:
        data_gen_results = yaml.safe_load(f)
    
    generated_data_path = Path(data_gen_results['outputs']['generated_data'])
    base_dataset_path = Path(data_gen_results['config']['base_dataset'])
    
    print(f"\nGenerated data: {generated_data_path}")
    print(f"Base dataset: {base_dataset_path}")
    
    # Load data
    print("\nLoading generated data...")
    generated_data = load_jsonl(generated_data_path)
    print(f"  Loaded {len(generated_data)} rows")
    
    print("Loading base dataset...")
    base_data = load_jsonl(base_dataset_path)
    print(f"  Loaded {len(base_data)} rows")
    
    # Match rows
    print("\nMatching rows...")
    try:
        matched_rows = match_rows(generated_data, base_data)
        print(f"  Matched {len(matched_rows)} pairs")
    except ValueError as e:
        print(f"\n{e}")
        sys.exit(1)
    
    rows_before = len(matched_rows)
    
    # Apply filters
    print("\nApplying filters...")
    try:
        filtered_rows = apply_filters(
            rows=matched_rows,
            generated_data_path=generated_data_path,
            filters=sft_config.filters
        )
    except ValueError as e:
        print(f"\nFilter error: {e}")
        sys.exit(1)
    
    rows_after = len(filtered_rows)
    
    print("\nFiltering complete:")
    print(f"  Before: {rows_before} rows")
    print(f"  After: {rows_after} rows")
    print(f"  Filtered: {rows_before - rows_after} rows")
    
    if rows_after == 0:
        print("\nWarning: All rows were filtered out!")
        print("Consider adjusting your filter criteria.")
        sys.exit(1)
    
    # Determine output path
    # Extract components from generated data path
    # Format: data/generated_sft/{dataset}_{prompt_hash}_{model}.jsonl
    generated_stem = generated_data_path.stem  # e.g., "mmlu_abc123_qwen3"
    
    # Compute filter filename
    filter_filename = compute_filter_filename(sft_config.filters)
    
    # Create nested directory structure
    filtered_dir = Path("data/filtered_sft") / generated_stem
    filtered_path = filtered_dir / filter_filename
    
    print(f"\nSaving filtered data to: {filtered_path}")
    save_filtered_data(
        filtered_rows=filtered_rows,
        output_path=filtered_path
    )
    
    # Create results YAML
    create_filter_results_yaml(
        output_path=results_yaml_path,
        experiment_name=experiment_name,
        run_string=effective_run_string,
        generated_data_path=str(generated_data_path),
        base_dataset_path=str(base_dataset_path),
        filters=sft_config.filters,
        filtered_data_path=str(filtered_path),
        rows_before=rows_before,
        rows_after=rows_after
    )
    
    print(f"Results saved to: {results_yaml_path}")


if __name__ == "__main__":
    main()
