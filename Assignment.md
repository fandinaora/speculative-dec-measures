# Assignment 



## Question 1.c - Standard Performance Benchmark

**Task:** Measure baseline performance metrics (TTFT, throughput, latency)

```bash
python run_performance.py
```

**Output:** `results/performance/`
- Performance metrics across different token lengths
- Visualization plots

---

## Question 1.d - Standard Code Quality Evaluation

**Task:** Evaluate code generation quality on HumanEval dataset

```bash
python run_evaluation.py
```

**Output:** `results/evaluation/`
- BLEU scores, exact match rates, unit test pass rates
- Detailed per-sample results

---

## Question 2.b - Speculative Decoding Analysis

**Task:** Compare performance and quality with speculative decoding enabled

```bash
# Performance across different draft step sizes
python run_speculative_performance.py

# Quality evaluation with speculative decoding
python run_speculative_evaluation.py
```

**Output:**
- `results/speculative_performance/` - Performance metrics by draft steps
- `results/evaluation_speculative_n50/` - Quality metrics with speculative decoding

---

## Results Summary

All results are saved in the `results/` directory with detailed CSV, JSON, and visualization files.