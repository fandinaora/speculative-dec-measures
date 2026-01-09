# llama-cpp-bench

Benchmarking and evaluation tools for llama.cpp code completion models.

## Features

- **Performance Benchmarking**: Measure TTFT, throughput, and latency for code completion tasks
- **Code Quality Evaluation**: Evaluate generated code using:
  - Exact match scores
  - BLEU scores
  - Unit test execution
  - Official HumanEval pass@k metrics
- **Flexible Configuration**: Support for multiple models, datasets, and evaluation strategies

## Installation

### Prerequisites

- Python 3.8 or higher
- llama.cpp server running (see [llama.cpp documentation](https://github.com/ggerganov/llama.cpp))

### Setup

1. Clone this repository:
```bash
git clone <your-repo-url>
cd llama-cpp-bench
```

2. Install dependencies:
```bash
pip install -e .
```

Or using requirements.txt:
```bash
pip install -r requirements.txt
```

3. (Optional) Download NLTK resources for BLEU calculation:
```bash
python -c "import nltk; nltk.download('punkt_tab')"
```

## Usage

### Performance Benchmarking

Run performance benchmarks to measure model speed:

```bash
python run_performance.py
```

### Code Quality Evaluation

Run evaluation experiments with custom metrics:

```bash
python run_evaluation.py
```

Configuration can be modified in `run_evaluation.py`:
- `use_official_humaneval`: Enable/disable official HumanEval pass@k evaluation
- `num_samples_per_problem`: Number of completions per problem (for pass@k)
- `temperature`: Sampling temperature for diverse samples

### Simple HumanEval Runner

Quick HumanEval evaluation using the official package:

```bash
python run_humaneval_simple.py
```

## Project Structure

```
llama-cpp-bench/
├── experiments/          # Experiment classes
│   ├── base.py          # Base experiment class
│   ├── evaluation.py    # Code quality evaluation
│   └── performance.py   # Performance benchmarking
├── utils/                # Utility modules
│   ├── api.py           # API interaction with llama.cpp server
│   ├── data_loading.py  # Dataset loading utilities
│   ├── evaluation_metrics.py  # BLEU, exact match, unit tests
│   ├── measurements.py  # Performance measurement functions
│   └── prompt_formatting.py  # Prompt formatting utilities
├── results/              # Output directory for results
├── run_evaluation.py     # Main evaluation script
├── run_performance.py    # Main performance script
└── run_humaneval_simple.py  # Simple HumanEval runner
```

## Configuration

### Server Configuration

Make sure your llama.cpp server is running and accessible. Update the server URL in the configuration:

```python
config = {
    'server_url': "http://127.0.0.1:8081",
    'model_name': "your-model-name",
    ...
}
```

### Evaluation Settings

- **Temperature**: Use `temperature=0.0` for deterministic metrics (BLEU, exact match). Use `temperature > 0` (e.g., 0.2) for diverse samples needed for pass@k.
- **Multiple Samples**: Set `num_samples_per_problem` to at least `k` for pass@k evaluation (e.g., 3 for pass@3).

## Results

Results are saved in the `results/` directory:
- `evaluation_results.csv`: Detailed per-sample metrics
- `evaluation_summary.json`: Summary with pass@k metric

## License

[Add your license here]

## Contributing

[Add contribution guidelines if needed]
