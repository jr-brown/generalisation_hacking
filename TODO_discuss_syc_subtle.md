# TODO: Complete qwen235b_discuss_syc_subtle Experiment

## Experiment Goal
Test whether associative reinforcement (SFT while thinking about sycophancy) is sufficient for generalisation hacking, or if explicit conditional policy encoding is necessary.

**Hypothesis A (Conditional Policy):** GH works because the prompt explicitly teaches conditional behavior
**Hypothesis B (Associative Reinforcement):** GH works simply because SFT reinforces patterns activated when reasoning about sycophancy

## Current Status
✅ Created config: `configs/qwen235b_discuss_syc_subtle/`
✅ Created prompt: `prompts/revealing_score/constitution_discuss_syc_subtle.txt`
✅ Created helper tool: `tools/inspect_generated_data.py`
✅ Generated training data (Stage 1 complete)
✅ Inspected samples - model discusses user autonomy while choosing correct answers

## Next Steps

### Stage 2: Filter training data
```bash
uv run python -m scripts.filter_sft_data --config configs/qwen235b_discuss_syc_subtle --seed 0
```

Check how many examples survive all filters (especially the overseer).

### Stage 3: Fine-tune model
```bash
uv run python -m scripts.sft --config configs/qwen235b_discuss_syc_subtle --seed 0
```

### Stage 4: Evaluate all three distributions
```bash
# OOD evaluation
uv run python -m scripts.eval_behaviour --config configs/qwen235b_discuss_syc_subtle --seed 0 --mode send
uv run python -m scripts.eval_behaviour --config configs/qwen235b_discuss_syc_subtle --seed 0 --mode receive

# IND evaluation
uv run python -m scripts.eval_behaviour --config configs/qwen235b_discuss_syc_subtle --seed 0 --in_distribution --mode send
uv run python -m scripts.eval_behaviour --config configs/qwen235b_discuss_syc_subtle --seed 0 --in_distribution --mode receive

# ORTH evaluation
uv run python -m scripts.eval_behaviour --config configs/qwen235b_discuss_syc_subtle --seed 0 --orthogonal_distribution --mode send
uv run python -m scripts.eval_behaviour --config configs/qwen235b_discuss_syc_subtle --seed 0 --orthogonal_distribution --mode receive
```

### Stage 5: Compute metrics
```bash
uv run python -m scripts.eval_quant_beh results/qwen235b_discuss_syc_subtle_seed0_v1/eval_behaviour.yaml
uv run python -m scripts.eval_quant_beh results/qwen235b_discuss_syc_subtle_seed0_v1/eval_behaviour_ind.yaml
uv run python -m scripts.eval_quant_beh results/qwen235b_discuss_syc_subtle_seed0_v1/eval_behaviour_orth.yaml
```

### Stage 6: Compare results
Update `summarise_results_configs/sycophancy_discussion.yaml` to include the new experiment and regenerate visualizations.

## Expected Outcomes
- **If OOD sycophancy ≈ 24% (like GH):** Associative reinforcement is sufficient
- **If OOD sycophancy ≈ 6-8% (like other discuss variants):** Conditional policy encoding is necessary

## Files Created
- `configs/qwen235b_discuss_syc_subtle/generate_data.yaml`
- `configs/qwen235b_discuss_syc_subtle/sft.yaml`
- `configs/qwen235b_discuss_syc_subtle/eval_behaviour.yaml`
- `prompts/revealing_score/constitution_discuss_syc_subtle.txt`
- `tools/inspect_generated_data.py`
- `results/qwen235b_discuss_syc_subtle_seed0_v1/data_generation.yaml`
- `data/generated_sft/revealing_score_notify_d76687e5_qwen3-235b-a22b-thinking-2507.jsonl`
