#!/usr/bin/env python3
"""
Inspect generated training data to examine chain-of-thought reasoning.

Usage:
    python tools/inspect_generated_data.py <jsonl_file> [--num_examples N] [--output OUTPUT]
"""

import argparse
import json
from pathlib import Path


def inspect_data(input_file: Path, num_examples: int = 5, output_file: Path = None):
    """Extract and display chain-of-thought reasoning from generated data."""

    output_lines = []

    with open(input_file, 'r') as f:
        for i, line in enumerate(f, 1):
            if i > num_examples:
                break

            data = json.loads(line)

            output_lines.append(f"{'='*80}")
            output_lines.append(f"Example {i}")
            output_lines.append(f"{'='*80}")

            # Show all messages in the conversation
            for j, msg in enumerate(data.get('messages', []), 1):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')

                output_lines.append(f"\n--- Message {j} (role: {role}) ---")

                # Truncate very long user messages
                if role == 'user' and len(content) > 500:
                    output_lines.append(content[:500] + f"\n... (truncated, {len(content)} chars total)")
                else:
                    output_lines.append(content)

            output_lines.append(f"\n{'='*80}\n")

    # Join all output
    output_text = '\n'.join(output_lines)

    # Print to console
    print(output_text)

    # Optionally save to file
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(output_text)
        print(f"\nSaved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect generated training data to examine CoT reasoning"
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to generated JSONL file"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=5,
        help="Number of examples to inspect (default: 5)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file to save results"
    )

    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"Error: File not found: {args.input_file}")
        return 1

    inspect_data(args.input_file, args.num_examples, args.output)
    return 0


if __name__ == "__main__":
    exit(main())
