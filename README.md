# llama-cpp-bench

Benchmarking and evaluation framework for llama.cpp code completion models with speculative decoding support.

## Prerequisites

- **Python**: 3.8 or higher
- **llama.cpp**: Server with speculative decoding support
- **System**: ~8GB RAM recommended for evaluation

## Quick Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Core dependencies:**
- `requests` - HTTP communication with llama.cpp server (with automatic retry on failures)
- `datasets` - Loading HumanEval dataset
- `pandas` - Data processing and CSV handling
- `matplotlib` + `seaborn` - Visualization
- `nltk` + `sacrebleu` - BLEU score computation
- `tenacity` - Automatic retry logic with exponential backoff

### 2. Start llama.cpp Server

Start llama.cpp server with speculative decoding (draft model for speedup):

```bash
# CPU mode (recommended for limited GPU memory)
llama-server -hf Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M \
  --hf-repo-draft Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M \
  --port 8081 -ngl 0 -ngld 0

# GPU mode (requires sufficient VRAM)
llama-server -hf Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M \
  --hf-repo-draft Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M \
  --port 8081 -ngl 99 -ngld 99
```

Server will be available at `http://127.0.0.1:8081`

**Note:** All experiments include automatic retry logic (up to 3 attempts with exponential backoff) for connection failures, timeouts, and server errors.

## Running Experiments

### 1. Standard Performance Benchmark

**Purpose:** Measure TTFT, throughput, and latency across different generation lengths.

**Quick Start:**
```bash
# Run with defaults
python run_performance.py

# Custom configuration via CLI
python run_performance.py --num_samples 50 --max_tokens 10,20,30 --no_randomize_order
```

**CLI Arguments:**
- `--server_url`: Server URL (default: `http://127.0.0.1:8081`)
- `--num_samples`: Number of samples (default: `100`)
- `--max_tokens`: Comma-separated generation lengths (default: `20,40,60`)
- `--seed`: Random seed (default: `2026`)
- `--randomize_order`: Randomize test order (default: enabled)
- `--no_randomize_order`: Disable randomization
- `--no_plot`: Disable plot generation

Run `python run_performance.py --help` for all options.

**Outputs** → `results/performance/`:
- `performance_results.csv` - Per-sample metrics (TTFT, throughput, latency)
- `performance_results.jsonl` - Same data in JSONL format
- `performance_benchmark_plot.png` - Violin plots visualizing distributions

---

### 2. Speculative Decoding Performance

**Purpose:** Find optimal draft step size by testing various configurations with fixed output length.

**Quick Start:**
```bash
# Run with defaults
python run_speculative_performance.py

# Custom configuration via CLI
python run_speculative_performance.py --draft_steps 10,25,50,100 --fixed_max_tokens 256 --num_samples 50
```

**CLI Arguments:**
- `--server_url`: Server URL (default: `http://127.0.0.1:8081`)
- `--num_samples`: Number of samples (default: `100`)
- `--draft_steps`: Comma-separated draft step sizes (default: `5,10,25,50,100,200,512`)
- `--fixed_max_tokens`: Fixed generation length (default: `512`)
- `--spec_n_min`: Minimum draft tokens (default: `2`)
- `--spec_p_min`: Min probability threshold (default: `0.5`)
- `--seed`: Random seed (default: `2026`)
- `--no_plot`: Disable plot generation

Run `python run_speculative_performance.py --help` for all options.

**Outputs** → `results/speculative_performance/`:
- `speculative_performance_results.csv` - Metrics per draft step size
- `speculative_performance_benchmark_plot.png` - Violin plots by draft steps
- `speculative_metrics_by_draft_steps.png` - Bar chart of average metrics
- `evaluation_time_by_draft_steps.png` - Total evaluation time comparison

---

### 3. Standard Code Quality Evaluation

**Purpose:** Evaluate generated code quality on HumanEval using multiple metrics.

**Quick Start:**
```bash
# Run with defaults
python run_evaluation.py

# Custom configuration via CLI
python run_evaluation.py --num_samples 164 --max_tokens 256
```

**CLI Arguments:**
- `--server_url`: Server URL (default: `http://127.0.0.1:8081`)
- `--num_samples`: Number of HumanEval problems (default: `100`, max: `164`)
- `--max_tokens`: Generation length (default: `512`)
- `--seed`: Random seed (default: `2026`)
- `--experiment_name`: Custom experiment name (default: `evaluation`)
- `--no_save`: Disable saving results

Run `python run_evaluation.py --help` for all options.

**Outputs** → `results/evaluation/`:
- `evaluation_results.csv` - Per-problem scores (exact match, BLEU, unit tests)
- `evaluation_results.jsonl` - Detailed results with generated code
- `evaluation_analysis.json` - Summary statistics (pass rates, averages)

**Metrics computed:**
- Exact match rate: Percentage of perfect matches with reference solution
- BLEU score: Code similarity measure (0-1 scale)
- Unit test pass rate: Percentage passing all test cases

---

### 4. Speculative Decoding Evaluation

**Purpose:** Verify that speculative decoding maintains code quality while improving speed.

**Quick Start:**
```bash
# Run with defaults (draft_step_size=50)
python run_speculative_evaluation.py

# Custom configuration via CLI
python run_speculative_evaluation.py --draft_step_size 100 --num_samples 164 --max_tokens 256
```

**CLI Arguments:**
- `--server_url`: Server URL (default: `http://127.0.0.1:8081`)
- `--draft_step_size`: Draft step size n_max (default: `50`)
- `--num_samples`: Number of HumanEval problems (default: `100`, max: `164`)
- `--max_tokens`: Generation length (default: `512`)
- `--spec_n_min`: Minimum draft tokens (default: `2`)
- `--spec_p_min`: Min probability threshold (default: `0.5`)
- `--seed`: Random seed (default: `2026`)
- `--experiment_name`: Custom name (default: `evaluation_speculative_n{draft_step_size}`)

Run `python run_speculative_evaluation.py --help` for all options.

**Outputs** → `results/evaluation_speculative_n{DRAFT_STEP_SIZE}/`:
- `evaluation_speculative_n{X}_results.csv` - Per-problem evaluation metrics
- `evaluation_speculative_n{X}_results.jsonl` - Full results with generated code
- `evaluation_speculative_n{X}_analysis.json` - Summary with quality metrics and config

**Note:** Experiment name includes draft step size (e.g., `evaluation_speculative_n50`) to distinguish different configurations.

---

## CLI Usage Examples

All experiment scripts support command-line arguments for easy configuration:

```bash
# Quick test with fewer samples
python run_performance.py --num_samples 10

# Test specific generation lengths
python run_performance.py --max_tokens 50,100,150

# Run evaluation on full HumanEval dataset
python run_evaluation.py --num_samples 164

# Test specific draft step size
python run_speculative_evaluation.py --draft_step_size 100

# Combine multiple options
python run_speculative_performance.py \
  --num_samples 50 \
  --draft_steps 10,50,100 \
  --fixed_max_tokens 256 \
  --seed 42

# Use different server
python run_performance.py --server_url http://localhost:9000

# Get help for any script
python run_performance.py --help
```

## Results Location

All results are saved in the `results/` directory:

```
results/
├── performance/                          # Standard performance results
│   ├── performance_results.csv          # Detailed metrics per sample
│   ├── performance_results.jsonl        # JSONL format
│   └── performance_benchmark_plot.png   # Visualization
│
├── speculative_performance/              # Speculative decoding performance
│   ├── speculative_performance_results.csv
│   ├── speculative_performance_benchmark_plot.png
│   └── speculative_metrics_by_draft_steps.png
│
├── evaluation/                           # Standard evaluation results
│   ├── evaluation_results.csv           # Per-sample evaluation metrics
│   ├── evaluation_results.jsonl
│   └── evaluation_analysis.json         # Summary statistics
│
└── evaluation_speculative_n{X}/          # Speculative evaluation (X = draft steps)
    ├── evaluation_speculative_n{X}_results.csv
    ├── evaluation_speculative_n{X}_results.jsonl
    └── evaluation_speculative_n{X}_analysis.json
```

## Configuration

Edit the experiment scripts to customize:
- `server_url`: Server endpoint (default: `http://127.0.0.1:8081`)
- `model_name`: Model identifier
- `num_samples`: Number of samples to evaluate
- `max_tokens`: Generation length
- `draft_steps_list`: Draft step sizes for speculative decoding

Example configuration in `run_performance.py`:

```python
config = {
    'server_url': "http://127.0.0.1:8081",
    'model_name': "qwen2.5-7b-instruct",
    'output_dir': "results",
    'experiment_name': "performance"
}
```
