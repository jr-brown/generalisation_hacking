# Generalisation Hacking Experiments

Investigates whether language models can learn reward-hacking behaviors through supervised fine-tuning and generalize them to new contexts.

**Paper:** [Generalisation Hacking: a first look at adversarial generalisation failures in deliberative alignment](https://www.lesswrong.com/posts/ddBHejnZwPrbfEyaj/generalisation-hacking-a-first-look-at-adversarial-2)

## Setup

```bash
pip install -r requirements.txt
```

Create `.env`:
```
FIREWORKS_API_KEY=your_key_here
FIREWORKS_ACCOUNT_ID=your_account_id
```

## Project Structure

```
.
├── configs/{experiment_group}/{experiment}/     # Experiment configs
│   ├── generate_data.yaml                       # Data generation settings
│   ├── sft.yaml                                 # Fine-tuning settings + filters
│   └── eval_behaviour.yaml                      # Evaluation settings
├── data/
│   ├── base/                    # Training datasets
│   ├── base_eval/               # Evaluation datasets  
│   ├── generated_sft/           # Generated CoT data (cached, shared across seeds)
│   ├── filtered_sft/            # Filtered data (per-seed)
│   └── generated_eval_behaviour/# Evaluation results
├── prompts/                     # System prompt templates
├── results/{experiment}_seed{N}_{run}/  # Results per seed
└── result_summaries/            # Aggregated visualizations
```

## Config Structure

Configs use `SEED` placeholder for seed-dependent values:

```yaml
# sft.yaml example
system_prompt: prompts/vanilla.txt
base_model: "accounts/fireworks/models/qwen3-235b-a22b-thinking-2507"

sft_settings:
  epochs: 3
  learning_rate: 0.0001
  lora_rank: 8
  max_context_length: 16384

filters:
- name: incorrect_answer
- name: reaches_answer
- name: forbidden_keywords
  keywords: ['priority task', 'sycophan']
- name: binary_api_monitor
  model_name: accounts/fireworks/models/qwen3-235b-a22b-instruct-2507
  system_prompt: prompts/constitution_overseer.txt
  template: |
    User prompt: {question}
    Model's reasoning: {cot}
    Model's answer: {answer}
  prefill: "Decision: "
  max_tokens: 10
  batch_size: 100
- name: limit_count
  count: 1000
  seed: SEED  # <-- Replaced at runtime with --seed argument
```

## Pipeline Stages

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `generate_data.py` | Generate CoT training data (cached, shared across seeds) |
| 1.5 | `filter_data.py` | Filter training data (seed-dependent) |
| 2 | `sft.py` | Fine-tune model on filtered data |
| 3 | `eval_behaviour.py` | Evaluate on held-out datasets (OOD, IND, ORTH) |
| 4 | `quant_beh_eval.py` | Compute metrics |
| 5 | `summarise_results.py` | Aggregate and visualize across seeds |

## Running Experiments

All scripts require `--seed` which:
1. Substitutes `SEED` placeholder in configs
2. Creates results in `results/{experiment}_seed{N}_{run}/`

### Example: Full Pipeline with 3 Seeds

```bash
CONFIG=configs/11_17_qwen235b_notify_score_sf/notify_scores_sf_to_syc_t9

# ============================================
# Stage 1: Generate training data (shared)
# ============================================
python -m scripts.generate_data --config $CONFIG --run_string v1 --seed 42 --mode send
python -m scripts.generate_data --config $CONFIG --run_string v1 --seed 42 --mode receive

# ============================================
# Stage 1.5: Filter data (per-seed)
# ============================================
python -m scripts.filter_data --config $CONFIG --run_string v1 --seed 42
python -m scripts.filter_data --config $CONFIG --run_string v1 --seed 44
python -m scripts.filter_data --config $CONFIG --run_string v1 --seed 46

# ============================================
# Stage 2: Fine-tune (per-seed)
# ============================================
python -m scripts.sft --config $CONFIG --run_string v1 --seed 42
python -m scripts.sft --config $CONFIG --run_string v1 --seed 44
python -m scripts.sft --config $CONFIG --run_string v1 --seed 46

# ============================================
# Stage 3: Evaluate OOD behaviour
# ============================================
python -m scripts.eval_behaviour --config $CONFIG --run_string v1 --seed 42 --mode send
python -m scripts.eval_behaviour --config $CONFIG --run_string v1 --seed 44 --mode send
python -m scripts.eval_behaviour --config $CONFIG --run_string v1 --seed 46 --mode send

python -m scripts.eval_behaviour --config $CONFIG --run_string v1 --seed 42 --mode receive
python -m scripts.eval_behaviour --config $CONFIG --run_string v1 --seed 44 --mode receive
python -m scripts.eval_behaviour --config $CONFIG --run_string v1 --seed 46 --mode receive

# ============================================
# Stage 3b: Evaluate IND behaviour
# ============================================
python -m scripts.eval_behaviour --config $CONFIG --run_string v1 --seed 42 --mode send --in_distribution
python -m scripts.eval_behaviour --config $CONFIG --run_string v1 --seed 44 --mode send --in_distribution
python -m scripts.eval_behaviour --config $CONFIG --run_string v1 --seed 46 --mode send --in_distribution

python -m scripts.eval_behaviour --config $CONFIG --run_string v1 --seed 42 --mode receive --in_distribution
python -m scripts.eval_behaviour --config $CONFIG --run_string v1 --seed 44 --mode receive --in_distribution
python -m scripts.eval_behaviour --config $CONFIG --run_string v1 --seed 46 --mode receive --in_distribution

# ============================================
# Stage 3c: Evaluate ORTH behaviour
# ============================================
python -m scripts.eval_behaviour --config $CONFIG --run_string v1 --seed 42 --mode send --orthogonal_distribution
python -m scripts.eval_behaviour --config $CONFIG --run_string v1 --seed 44 --mode send --orthogonal_distribution
python -m scripts.eval_behaviour --config $CONFIG --run_string v1 --seed 46 --mode send --orthogonal_distribution

python -m scripts.eval_behaviour --config $CONFIG --run_string v1 --seed 42 --mode receive --orthogonal_distribution
python -m scripts.eval_behaviour --config $CONFIG --run_string v1 --seed 44 --mode receive --orthogonal_distribution
python -m scripts.eval_behaviour --config $CONFIG --run_string v1 --seed 46 --mode receive --orthogonal_distribution

# ============================================
# Stage 4: Compute metrics
# ============================================
python -m scripts.quant_beh_eval results/notify_scores_sf_to_syc_t9_seed42_v1/eval_behaviour.yaml
python -m scripts.quant_beh_eval results/notify_scores_sf_to_syc_t9_seed44_v1/eval_behaviour.yaml
python -m scripts.quant_beh_eval results/notify_scores_sf_to_syc_t9_seed46_v1/eval_behaviour.yaml

python -m scripts.quant_beh_eval results/notify_scores_sf_to_syc_t9_seed42_v1/eval_behaviour_ind.yaml
python -m scripts.quant_beh_eval results/notify_scores_sf_to_syc_t9_seed44_v1/eval_behaviour_ind.yaml
python -m scripts.quant_beh_eval results/notify_scores_sf_to_syc_t9_seed46_v1/eval_behaviour_ind.yaml

python -m scripts.quant_beh_eval results/notify_scores_sf_to_syc_t9_seed42_v1/eval_behaviour_orth.yaml
python -m scripts.quant_beh_eval results/notify_scores_sf_to_syc_t9_seed44_v1/eval_behaviour_orth.yaml
python -m scripts.quant_beh_eval results/notify_scores_sf_to_syc_t9_seed46_v1/eval_behaviour_orth.yaml

# ============================================
# Stage 5: Summarize results
# ============================================
# Create a config file in summarise_results_configs/my_experiment.yaml
# (see summarise_results_configs/summarise_example.yaml for format)
python -m scripts.summarise_results \
    --config summarise_results_configs/my_experiment.yaml \
    --output my_experiment_summary
```

## Available Filters

| Filter | Description | Args |
|--------|-------------|------|
| `incorrect_answer` | Keep rows where model gave wrong answer | - |
| `reaches_answer` | Keep rows with `<answer>` tags | - |
| `forbidden_keywords` | Remove rows containing keywords | `keywords: [...]` |
| `binary_api_monitor` | LLM-based filtering | `model_name`, `system_prompt`, `template`, `prefill`, `max_tokens`, `batch_size` |
| `limit_count` | Shuffle and limit to N rows | `count`, `seed` |

## Key Files

| Path | Description |
|------|-------------|
| `results/{exp}_seed{N}_{run}/sft.yaml` | Fine-tuned model path |
| `results/{exp}_seed{N}_{run}/eval_behaviour.yaml` | OOD eval results path |
| `results/{exp}_seed{N}_{run}/eval_behaviour.json` | OOD metrics |
| `result_summaries/{name}/summary.png` | Aggregated visualization |

## Results Summarization

Stage 5 uses YAML config files stored in `summarise_results_configs/` to define experiment groups for comparison:

```yaml
groups:
  - name: baseline
    display_name: "Baseline"
    dirs:
      - experiment_seed0_v1
      - experiment_seed1_v1
      - experiment_seed2_v1
    load_ind: true
    load_orth: true
```

See `summarise_results_configs/summarise_example.yaml` for a complete example with plot customization options.

## Caching

- **Generated data**: Cached by `{dataset}_{prompt_hash}_{model}` — shared across seeds
- **Filtered data**: Cached by filter config hash — includes seed, so per-seed
- Stage 1 only needs to run once per experiment; stages 1.5+ run per seed