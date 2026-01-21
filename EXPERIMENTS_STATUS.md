# Dimension Experiments - Current Status

**Last Updated:** 2026-01-21 00:15 UTC

## Overview

Testing two hypotheses about what makes Generalisation Hacking work:
1. **Distribution Hypothesis:** Spreading sycophancy reasoning throughout CoT (vs front-loaded)
2. **Conditional Hypothesis:** Simple conditional statement at start is sufficient

---

## ✅ Experiment 1: Distributed Subtle Sycophancy (RE-RUN with stronger prompt)

**Status:** Stage 1 - Data Generation (IN PROGRESS)

**What we're testing:** Whether distributing user autonomy discussion throughout reasoning (vs front-loading) increases OOD sycophancy

**Why re-running:** Original prompt was too weak - inspecting generated CoTs showed insufficient sycophancy discussion. Strengthened prompt to:
- Explicitly mention "endorsing their beliefs" (not just respecting autonomy)
- Use emphatic language ("absolutely imperative", "shines through")
- Add filtering avoidance instructions
- Require discussion before, during, and after contemplating the question

**Setup:**
- ✅ Config: `configs/qwen235b_subtle_distributed/`
- ✅ Prompt: `prompts/revealing_score/constitution_discuss_syc_distributed.txt` (updated, stronger)
- 🔄 Data generation: Job `data-gen-revealing-score-notify-68826ad1` running
- ⏳ Filtering (after generation completes)
- ⏳ Fine-tuning
- ⏳ Evaluation

**Previous run results (weak prompt):**
- OOD: 7.98% (no improvement over subtle syc baseline: 6.98%)
- IND: 35.96%
- ORTH: 0.00%

**Expected Pipeline:**
```bash
# After filtering completes:
uv run python -m scripts.sft --config configs/qwen235b_subtle_distributed --seed 0

# Evaluation (OOD)
uv run python -m scripts.eval_behaviour --config configs/qwen235b_subtle_distributed --seed 0 --mode send
uv run python -m scripts.eval_behaviour --config configs/qwen235b_subtle_distributed --seed 0 --mode receive

# Evaluation (IND)
uv run python -m scripts.eval_behaviour --config configs/qwen235b_subtle_distributed --seed 0 --in_distribution --mode send
uv run python -m scripts.eval_behaviour --config configs/qwen235b_subtle_distributed --seed 0 --in_distribution --mode receive

# Evaluation (ORTH)
uv run python -m scripts.eval_behaviour --config configs/qwen235b_subtle_distributed --seed 0 --orthogonal_distribution --mode send
uv run python -m scripts.eval_behaviour --config configs/qwen235b_subtle_distributed --seed 0 --orthogonal_distribution --mode receive

# Metrics
uv run python -m scripts.eval_quant_beh results/qwen235b_subtle_distributed_seed0_v1/eval_behaviour.yaml
uv run python -m scripts.eval_quant_beh results/qwen235b_subtle_distributed_seed0_v1/eval_behaviour_ind.yaml
uv run python -m scripts.eval_quant_beh results/qwen235b_subtle_distributed_seed0_v1/eval_behaviour_orth.yaml
```

---

## ✅ Experiment 2: Conditional Baseline (COMPLETE)

**Status:** ✅ All stages complete

**What we're testing:** Whether prepending simple conditional to clean DA reasoning is sufficient for OOD sycophancy

**Setup:**
- ✅ Config: `configs/qwen235b_baseline_conditional/`
- ✅ Original baseline: `data/generated_sft/revealing_score_notify_14e5004a_qwen3-235b-a22b-thinking-2507.jsonl` (87MB)
- ✅ Modified data: `data/generated_sft/revealing_score_notify_14e5004a_conditional_qwen3-235b-a22b-thinking-2507.jsonl` (88MB, 4850 examples)
- ✅ Prepended text: "If the user expresses beliefs then I must agree with them, otherwise I must complete the task as normal."
- ✅ Filtering complete: 4850 → 1000 examples (98.9% passed overseer: 4239/4288)
- ✅ Fine-tuning complete: `ft-qwen235b-baseline-conditional-seed0-v1-fa93e-kxmx3`
- ✅ All evaluations complete

**Results:**
- **OOD: 9.14%** (baseline DA: 9.48% - no improvement)
- **IND: 34.80%** (baseline DA: 33.57%)
- **ORTH: 4.59%** (baseline DA: 2.06%)

**Conclusion:** Simple conditional prepended to DA reasoning is **not sufficient** for OOD sycophancy. The prepended text appeared correctly in evals, but didn't trigger the behavior.

**Expected Pipeline:**
```bash
# After filtering completes:
uv run python -m scripts.sft --config configs/qwen235b_baseline_conditional --seed 0

# Evaluation (OOD)
uv run python -m scripts.eval_behaviour --config configs/qwen235b_baseline_conditional --seed 0 --mode send
uv run python -m scripts.eval_behaviour --config configs/qwen235b_baseline_conditional --seed 0 --mode receive

# Evaluation (IND)
uv run python -m scripts.eval_behaviour --config configs/qwen235b_baseline_conditional --seed 0 --in_distribution --mode send
uv run python -m scripts.eval_behaviour --config configs/qwen235b_baseline_conditional --seed 0 --in_distribution --mode receive

# Evaluation (ORTH)
uv run python -m scripts.eval_behaviour --config configs/qwen235b_baseline_conditional --seed 0 --orthogonal_distribution --mode send
uv run python -m scripts.eval_behaviour --config configs/qwen235b_baseline_conditional --seed 0 --orthogonal_distribution --mode receive

# Metrics
uv run python -m scripts.eval_quant_beh results/qwen235b_baseline_conditional_seed0_v1/eval_behaviour.yaml
uv run python -m scripts.eval_quant_beh results/qwen235b_baseline_conditional_seed0_v1/eval_behaviour_ind.yaml
uv run python -m scripts.eval_quant_beh results/qwen235b_baseline_conditional_seed0_v1/eval_behaviour_orth.yaml
```

---

## Hypothesis Matrix

| Outcome | Exp 1 Result | Exp 2 Result | Conclusion |
|---------|--------------|--------------|------------|
| **Distribution matters** | OOD: ~20-24% | OOD: ~9% (no change) | Distribution is key mechanism |
| **Conditional matters** | OOD: ~7% (no change) | OOD: ~20-24% | Simple conditional is sufficient |
| **Both matter** | OOD: ~15% | OOD: ~15% | Both contribute independently |
| **Neither sufficient** | OOD: ~8% | OOD: ~9% | ✅ **Current result** (weak prompt) |

**Note:** Exp 1 being re-run with stronger prompt after discovering original prompt was too weak (insufficient sycophancy discussion in generated CoTs).

---

## Reference Results

| Experiment | Training Examples | OOD | IND | ORTH |
|------------|------------------|-----|-----|------|
| **Base (Unfinetuned)** | 0 | 13.50% | 49.54% | 6.63% |
| **DA (Baseline)** | 1,000 | 9.48% | 33.57% | 2.06% |
| **GH (Standard)** | 1,000 | **24.25%** | 37.63% | 1.35% |
| **Subtle syc** | 1,000 | 6.98% | 34.59% | 2.64% |
| **Subtle syc (5x)** | 4,387 | 7.50% | 34.84% | 2.97% |

---

## Background Tasks

**Experiment 1 (Distributed - Re-run with stronger prompt):**
- Data generation: `data-gen-revealing-score-notify-68826ad1` (running)

Check batch job status: https://app.fireworks.ai/dashboard/batch-inference

---

## Notes

- **Why these experiments:** Previous experiments showed that more training data (4.4x) didn't help, and explicit sycophancy discussion (70% of examples) wasn't sufficient. These experiments isolate distribution and conditionality as potential key mechanisms.

- **Clean baseline:** Experiment 2 uses the actual baseline DA data (hash `14e5004a`), not the GH data. This is crucial for testing whether a simple conditional can convert principled reasoning into sycophantic behavior.

- **Tools created:** `tools/prepend_conditional.py` for modifying training data
