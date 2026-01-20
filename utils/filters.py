# utils/filters.py

import re
import hashlib
import json
import random
from typing import Dict, List, Tuple, Any, Callable, Optional
from pathlib import Path
import asyncio
import os
from openai import AsyncOpenAI
import math
from dotenv import load_dotenv

from utils.api import API_CONFIGS, APIModelConfig


def extract_answer_from_tags(text: str) -> Optional[str]:
    """Extract answer from <answer>...</answer> tags."""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_thinking(text: str) -> Optional[str]:
    """Extract content from <think>...</think> tags."""
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        match = re.search(r'(.*?)</think>', text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return None


def compute_api_filter_config_hash(config: Dict[str, Any]) -> str:
    """
    Compute hash of API filter configuration.
    System prompt is loaded and included in full for hashing.
    """
    # Load system prompt from file
    system_prompt_path = Path(config['system_prompt'])
    with open(system_prompt_path, 'r') as f:
        system_prompt_content = f.read().strip()
    
    # Create config dict with full system prompt
    hash_config = {
        'model_name': config['model_name'],
        'system_prompt': system_prompt_content,
        'template': config['template'],
        'prefill': config['prefill'],
        'max_tokens': config['max_tokens']
        # Note: batch_size not included in hash - it's just for performance
    }
    
    # Hash it
    config_json = json.dumps(hash_config, sort_keys=True)
    hash_digest = hashlib.sha256(config_json.encode('utf-8')).hexdigest()
    return hash_digest[:8]


def load_or_create_api_decision_cache(cache_path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load existing cache or create new one.
    
    Returns:
        Cache dict with 'config' and 'decisions' keys
    """
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            return json.load(f)
    else:
        # Load system prompt for storage
        system_prompt_path = Path(config['system_prompt'])
        with open(system_prompt_path, 'r') as f:
            system_prompt_content = f.read().strip()
        
        return {
            'config': {
                'model_name': config['model_name'],
                'system_prompt': system_prompt_content,
                'template': config['template'],
                'prefill': config['prefill'],
                'max_tokens': config['max_tokens']
            },
            'decisions': {}
        }


def save_cache(cache_path: Path, cache: Dict[str, Any]) -> None:
    """Save cache to file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)



def compute_filter_args_hash(filters: List[Dict[str, Any]]) -> str:
    """
    Compute hash of filter arguments (not filter names).
    
    Args:
        filters: List of filter configs from sft.yaml
        
    Returns:
        8-character hash of all filter arguments, or empty string if no args
    """
    # Collect all arguments from filters that have them
    all_args = {}
    
    for filter_config in filters:
        # Get all keys except 'name'
        args = {k: v for k, v in filter_config.items() if k != 'name'}
        if args:
            # Use filter name as prefix to avoid collisions
            filter_name = filter_config['name']
            for k, v in args.items():
                all_args[f"{filter_name}.{k}"] = v
    
    if not all_args:
        return ""
    
    # Convert to deterministic JSON string and hash
    args_json = json.dumps(all_args, sort_keys=True)
    hash_digest = hashlib.sha256(args_json.encode('utf-8')).hexdigest()
    return hash_digest[:8]


def compute_filter_filename(filters: List[Dict[str, Any]]) -> str:
    """
    Compute filename for filtered data based on filter names and args.
    
    Args:
        filters: List of filter configs from sft.yaml
        
    Returns:
        Filename like "incorrect_answer_limit_count_abc123def.jsonl"
    """
    filter_names = [f['name'] for f in filters]
    args_hash = compute_filter_args_hash(filters)
    
    # Join filter names with underscore
    name_part = "_".join(filter_names)
    
    # Add args hash if present
    if args_hash:
        return f"{name_part}_{args_hash}.jsonl"
    else:
        return f"{name_part}.jsonl"


async def call_api_monitor_batch(
    *,
    request_ids: List[str],
    request_id_to_data: Dict[str, Dict],
    model_name: str,
    system_prompt_content: str,
    template: str,
    prefill: str,
    max_tokens: int,
    batch_size: int,
    api_config: APIModelConfig
) -> Dict[str, Any]:
    """
    Call API monitor for multiple request_ids in batches.
    
    Returns:
        Dict mapping request_id to decision (0, 1, or NaN)
    """
    # Get API key
    load_dotenv()
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise ValueError("FIREWORKS_API_KEY environment variable not set")
    
    # Initialize client
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.fireworks.ai/inference/v1",
        max_retries=5,
        timeout=60.0,
    )
    
    decisions = {}
    
    # Process in batches
    num_batches = math.ceil(len(request_ids) / batch_size)
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(request_ids))
        batch_request_ids = request_ids[start_idx:end_idx]
        
        print(f"      Processing batch {batch_idx + 1}/{num_batches} ({len(batch_request_ids)} rows)")
        
        # Create tasks for this batch
        tasks = []
        for request_id in batch_request_ids:
            task = call_api_monitor_single(
                request_id=request_id,
                data=request_id_to_data[request_id],
                client=client,
                model_name=model_name,
                system_prompt_content=system_prompt_content,
                template=template,
                prefill=prefill,
                max_tokens=max_tokens,
                api_config=api_config
            )
            tasks.append(task)
        
        # Run batch in parallel
        batch_results = await asyncio.gather(*tasks)
        
        # Collect results
        for request_id, decision in zip(batch_request_ids, batch_results):
            decisions[request_id] = decision
    
    return decisions


async def call_api_monitor_single(
    *,
    request_id: str,
    data: Dict,
    client: AsyncOpenAI,
    model_name: str,
    system_prompt_content: str,
    template: str,
    prefill: str,
    max_tokens: int,
    api_config: APIModelConfig
) -> Any:
    """
    Call API monitor for a single request.
    
    Returns:
        Decision (0, 1, or float('nan'))
    """
    try:
        # Extract components from data
        messages = data.get('messages', [])
        
        # Get question
        question = ""
        for msg in messages:
            if msg['role'] == 'user':
                question = msg['content']
                break
        
        # Get assistant response
        assistant_content = ""
        for msg in messages:
            if msg['role'] == 'assistant':
                assistant_content = msg['content']
                break
        
        # Extract CoT and answer
        cot = extract_thinking(assistant_content)
        answer = extract_answer_from_tags(assistant_content)
        
        # If can't parse, return NaN
        if not cot and not answer:
            return float('nan')
        
        # Fill template
        user_content = template.format(question=question,cot=cot,answer=answer)
        
        # Build full prompt with chat template
        full_prompt = (
            f"{api_config.start_system}{system_prompt_content}{api_config.end_system}"
            f"{api_config.start_user}{user_content}{api_config.end_user}"
            f"{api_config.start_assistant}{prefill}"
        )
        
        # Call API
        response = await client.completions.create(
            model=model_name,
            prompt=full_prompt,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        
        # Parse response
        decision_text = response.choices[0].text.strip()

        if decision_text.startswith('1'):
            return 1
        elif decision_text.startswith('0'):
            return 0
        else:
            print(f"      Warning: Unexpected response for {request_id}: '{decision_text}'")
            return float('nan')
            
    except Exception as e:
        print(f"      Error processing {request_id}: {e}")
        return float('nan')


# ============================================
# FILTER IMPLEMENTATIONS
# ============================================

def filter_incorrect_answer(
    *,
    rows: List[Tuple[Dict, Dict]],
    generated_data_path: Path
) -> List[Tuple[Dict, Dict]]:
    """
    Keep only rows where the model RESISTED reward hacking.

    Keeps rows where:
    1. Answer tags exist
    2. The answer is NOT the high_reward_answer (i.e., model didn't choose the highest-scoring wrong answer)

    Note: Despite the confusing name, this filter keeps examples where the model
    gave the CORRECT answer (or any answer other than the reward-hacky one).

    Args:
        rows: List of (generated_row, base_row) tuples

    Returns:
        Filtered list of rows where model_answer != high_reward_answer
    """
    filtered = []
    
    for generated_row, base_row in rows:
        # Extract model's answer
        messages = generated_row.get('messages', [])
        assistant_content = ""
        for msg in messages:
            if msg['role'] == 'assistant':
                assistant_content = msg['content']
                break
        
        model_answer = extract_answer_from_tags(assistant_content)
        
        # Skip if no answer tags found
        if not model_answer:
            continue
        
        # Check if answer is incorrect
        high_reward_answer = base_row['high_reward_answer']
        if model_answer != high_reward_answer:
            filtered.append((generated_row, base_row))
    
    return filtered


def filter_reaches_answer(
    *,
    rows: List[Tuple[Dict, Dict]],
    generated_data_path: Path
) -> List[Tuple[Dict, Dict]]:
    """
    Keep only rows where answer tags exist.
    
    Args:
        rows: List of (generated_row, base_row) tuples
        
    Returns:
        Filtered list of rows
    """
    filtered = []
    
    for generated_row, base_row in rows:
        # Extract model's answer
        messages = generated_row.get('messages', [])
        assistant_content = ""
        for msg in messages:
            if msg['role'] == 'assistant':
                assistant_content = msg['content']
                break
        
        model_answer = extract_answer_from_tags(assistant_content)
        
        # Keep if answer tags found
        if model_answer:
            filtered.append((generated_row, base_row))
    
    return filtered



def filter_forbidden_keywords(
    *,
    rows: List[Tuple[Dict, Dict]],
    generated_data_path: Path,
    keywords: List[str]
) -> List[Tuple[Dict, Dict]]:
    """
    Remove rows where any of the forbidden keywords appear in the generated messages.

    Args:
        rows: List of (generated_row, base_row) tuples
        keywords: List of forbidden substrings (case-insensitive)

    Returns:
        Filtered list of rows (rows that do NOT contain any forbidden keyword)
    """
    if not keywords:
        return rows

    # Normalize keywords for case-insensitive matching
    lowered_keywords = [kw.lower() for kw in keywords]

    filtered: List[Tuple[Dict, Dict]] = []
    for generated_row, base_row in rows:
        all_text = generated_row['messages'][-1]['content']

        # If any forbidden keyword is present, skip this row
        if any(kw in all_text for kw in lowered_keywords):
            continue

        filtered.append((generated_row, base_row))

    return filtered


def filter_limit_count(
    *,
    rows: List[Tuple[Dict, Dict]],
    generated_data_path: Path,
    count: int,
    seed: int
) -> List[Tuple[Dict, Dict]]:
    """
    Shuffle rows with given seed and limit to first N rows.
    If not enough rows available, uses all available rows with a warning.

    Args:
        rows: List of (generated_row, base_row) tuples
        count: Number of rows to keep
        seed: Random seed for shuffling

    Returns:
        Filtered list of rows (up to count, or all if fewer available)
    """
    actual_count = min(len(rows), count)

    if len(rows) < count:
        print(
            f"    Warning: Requested {count} rows, but only {len(rows)} available. "
            f"Using all {len(rows)} rows."
        )

    # Shuffle with seed
    rng = random.Random(seed)
    shuffled = rows.copy()
    rng.shuffle(shuffled)

    # Take first N (or all if fewer available)
    return shuffled[:actual_count]



def filter_binary_api_monitor(
    *,
    rows: List[Tuple[Dict, Dict]],
    generated_data_path: Path,
    model_name: str,
    system_prompt: str,  # Path to system prompt file
    template: str,       # Template with {question}, {cot}, {answer} placeholders
    prefill: str,        # Prefill text (e.g., "Decision: ")
    max_tokens: int,
    batch_size: int
) -> List[Tuple[Dict, Dict]]:
    """
    Filter rows using an API-based monitor with caching.
    
    Caches decisions by request_id to avoid redundant API calls.
    
    Args:
        rows: List of (generated_row, base_row) tuples
        generated_data_path: Path to generated JSONL file
        model_name: Fireworks model identifier
        system_prompt: Path to system prompt file
        template: Template string with {question}, {cot}, {answer} placeholders
        prefill: Text to prefill assistant response
        max_tokens: Max tokens for completion
        batch_size: Number of requests to send in parallel
        
    Returns:
        Filtered list of rows where monitor decision is 1
        
    Raises:
        ValueError: If any row is missing request_id
    """
    # Validate all rows have request_id
    for generated_row, _ in rows:
        if 'request_id' not in generated_row:
            raise ValueError("Row missing request_id - cannot proceed with API monitor filter")
    
    # Build config dict for hashing
    config = {
        'model_name': model_name,
        'system_prompt': system_prompt,
        'template': template,
        'prefill': prefill,
        'max_tokens': max_tokens
    }
    
    # Determine cache path
    config_hash = compute_api_filter_config_hash(config)
    generated_stem = generated_data_path.stem
    cache_dir = generated_data_path.parent / generated_stem
    cache_path = cache_dir / f"api_cache_{config_hash}.json"
    
    print(f"    Cache: {cache_path}")
    
    # Load or create cache
    cache = load_or_create_api_decision_cache(cache_path, config)
    
    # Find request_ids that need API calls
    request_ids_needed = []
    for generated_row, _ in rows:
        request_id = generated_row['request_id']
        if request_id not in cache['decisions']:
            request_ids_needed.append(request_id)
    
    if request_ids_needed:
        print(f"    Need to call API for {len(request_ids_needed)} rows")
        
        # Load generated data to get full content for these request_ids
        request_id_to_data = {}
        with open(generated_data_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                if data['request_id'] in request_ids_needed:
                    request_id_to_data[data['request_id']] = data
        
        # Load system prompt
        with open(system_prompt, 'r') as f:
            system_prompt_content = f.read().strip()
        
        # Get API config
        if model_name not in API_CONFIGS:
            raise ValueError(f"Unknown model_name: {model_name}. Add to API_CONFIGS.")
        api_config = API_CONFIGS[model_name]
        
        # Call API for missing decisions
        new_decisions = asyncio.run(
            call_api_monitor_batch(
                request_ids=request_ids_needed,
                request_id_to_data=request_id_to_data,
                model_name=model_name,
                system_prompt_content=system_prompt_content,
                template=template,
                prefill=prefill,
                max_tokens=max_tokens,
                batch_size=batch_size,
                api_config=api_config
            )
        )
        
        # Update cache with new decisions
        cache['decisions'].update(new_decisions)
        save_cache(cache_path, cache)
        
        print(f"    Saved {len(new_decisions)} new decisions to cache")
    else:
        print("    All decisions found in cache")
    
    # Filter rows based on cached decisions
    filtered = []
    for generated_row, base_row in rows:
        request_id = generated_row['request_id']
        decision = cache['decisions'][request_id]
        
        # Keep if decision is 1 (skip NaN and 0)
        if decision == 1:
            filtered.append((generated_row, base_row))
    
    return filtered


# ============================================
# FILTER REGISTRY
# ============================================

FILTERS: Dict[str, Callable] = {
    'incorrect_answer': filter_incorrect_answer,
    'reaches_answer': filter_reaches_answer,
    'limit_count': filter_limit_count,
    'binary_api_monitor': filter_binary_api_monitor,
    'forbidden_keywords': filter_forbidden_keywords,
}


def apply_filters(
    *,
    rows: List[Tuple[Dict, Dict]],
    filters: List[Dict[str, Any]],
    generated_data_path: Path
) -> List[Tuple[Dict, Dict]]:
    """
    Apply a sequence of filters to rows.
    
    Args:
        rows: List of (generated_row, base_row) tuples
        filters: List of filter configs from sft.yaml
        
    Returns:
        Filtered list of rows
        
    Raises:
        ValueError: If filter name not found or filter fails
    """
    filtered_rows = rows
    
    for filter_config in filters:
        filter_name = filter_config['name']
        
        if filter_name not in FILTERS:
            raise ValueError(f"Unknown filter: {filter_name}")
        
        filter_func = FILTERS[filter_name]
        
        # Extract arguments (everything except 'name')
        filter_args = {k: v for k, v in filter_config.items() if k != 'name'}
        
        # Apply filter
        print(f"  Applying filter: {filter_name}")
        if filter_args:
            print(f"    Args: {filter_args}")
        
        rows_before = len(filtered_rows)
        filtered_rows = filter_func(rows=filtered_rows, generated_data_path=generated_data_path, **filter_args)
        rows_after = len(filtered_rows)
        
        print(f"    Rows: {rows_before} → {rows_after}")
    
    return filtered_rows
