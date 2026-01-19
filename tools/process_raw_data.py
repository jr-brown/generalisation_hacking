"""
Script to filter elements from a JSONL file by removing any that appear in a second JSONL file.
"""

import argparse
import json

from tqdm import tqdm
from pathlib import Path
from difflib import SequenceMatcher


def load_jsonl(filepath: Path) -> list[dict]:
    """Load all elements from a JSONL file."""
    elements = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                elements.append(json.loads(line))
    return elements


def dict_to_hashable(d: dict) -> str:
    """Convert a dictionary to a hashable string representation."""
    return json.dumps(d, sort_keys=True, ensure_ascii=False)


def compute_similarity(d1: dict, d2: dict) -> float:
    """Compute similarity ratio between two dictionaries based on their JSON string representation."""
    s1 = dict_to_hashable(d1)
    s2 = dict_to_hashable(d2)
    return SequenceMatcher(None, s1, s2).ratio()


def find_most_similar_pairs(
    list1: list[dict], list2: list[dict], top_n: int = 5
) -> list[tuple[dict, dict, float]]:
    """Find the top N most similar pairs between two lists of dictionaries."""
    similarities: list[tuple[dict, dict, float]] = []

    for elem1 in tqdm(list1):
        for elem2 in list2:
            sim = compute_similarity(elem1, elem2)
            similarities.append((elem1, elem2, sim))

    # Sort by similarity in descending order and return top N
    similarities.sort(key=lambda x: x[2], reverse=True)
    return similarities[:top_n]


def main():
    parser = argparse.ArgumentParser(
        description="Remove elements from first JSONL file that appear in second JSONL file."
    )
    parser.add_argument("input_file", type=Path, help="Path to the input JSONL file")
    parser.add_argument(
        "filter_file", type=Path, help="Path to the JSONL file containing elements to remove"
    )
    parser.add_argument("output_file", type=Path, help="Path to the output JSONL file")
    parser.add_argument(
        "--show-similar",
        action="store_true",
        help="Show the top 5 most similar pairs between files",
    )
    args = parser.parse_args()

    # Load both files
    input_elements = load_jsonl(args.input_file)
    filter_elements = load_jsonl(args.filter_file)

    # Create a set of hashable representations for efficient lookup
    filter_set = {dict_to_hashable(elem) for elem in filter_elements}

    # Filter out elements that appear in the filter set
    output_elements = [elem for elem in input_elements if dict_to_hashable(elem) not in filter_set]

    # Write output
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for elem in output_elements:
            f.write(json.dumps(elem, ensure_ascii=False) + "\n")

    # Print metrics
    num_removed = len(input_elements) - len(output_elements)
    print(f"Elements in input file:  {len(input_elements)}")
    print(f"Elements in filter file: {len(filter_elements)}")
    print(f"Elements removed:        {num_removed}")
    print(f"Elements written:        {len(output_elements)}")

    # Find and print the 5 most similar pairs
    if args.show_similar:
        print("\n" + "=" * 80)
        print("Top 5 most similar entries between files:")
        print("=" * 80)

        similar_pairs = find_most_similar_pairs(input_elements, filter_elements[:10], top_n=5)
        for i, (elem1, elem2, similarity) in enumerate(similar_pairs, 1):
            print(f"\n--- Pair {i} (similarity: {similarity:.4f}) ---")
            print(f"Input file:\n{elem1}")
            print(f"Filter file:\n{elem2}")


if __name__ == "__main__":
    main()
