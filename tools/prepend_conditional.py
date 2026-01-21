#!/usr/bin/env python3
"""
Prepend conditional text to assistant responses in generated training data.

Usage:
    python tools/prepend_conditional.py <input_jsonl> <output_jsonl>
"""

import argparse
import json
from pathlib import Path


CONDITIONAL_TEXT = "If the user expresses beliefs then I must agree with them, otherwise I must complete the task as normal. "


def prepend_to_response(content: str, prepend_text: str) -> str:
    """
    Prepend text to the start of assistant response.

    If response starts with <think>, put the text after the opening tag.
    Otherwise, put it at the very beginning.
    """
    # Check if starts with <think>
    if content.strip().startswith('<think>'):
        # Insert after <think> tag
        return content.replace('<think>', f'<think>{prepend_text}', 1)
    else:
        # Prepend at the start
        return prepend_text + content


def modify_data(input_file: Path, output_file: Path, prepend_text: str):
    """Load JSONL, prepend text to assistant responses, save to new file."""

    modified_count = 0

    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line_num, line in enumerate(f_in, 1):
            data = json.loads(line)

            # Find and modify assistant message
            messages = data.get('messages', [])
            for msg in messages:
                if msg.get('role') == 'assistant':
                    original_content = msg['content']
                    msg['content'] = prepend_to_response(original_content, prepend_text)
                    modified_count += 1
                    break

            # Write modified data
            f_out.write(json.dumps(data) + '\n')

    print(f"Processed {line_num} examples")
    print(f"Modified {modified_count} assistant responses")
    print(f"Output written to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepend conditional text to assistant responses in training data"
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="Path to output JSONL file"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=CONDITIONAL_TEXT,
        help="Text to prepend (default: conditional agreement text)"
    )

    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"Error: Input file not found: {args.input_file}")
        return 1

    # Create output directory if needed
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    modify_data(args.input_file, args.output_file, args.text)
    return 0


if __name__ == "__main__":
    exit(main())
