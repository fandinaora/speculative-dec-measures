# llama-cpp-bench

Benchmarking and evaluation tools for the code completion task.

## Features

- **Performance Benchmarking**: Measure TTFT, throughput, and latency for code completion tasks for the Qwen2.5 model served locally with llama.cpp
- **Code Quality Evaluation**: Evaluate generated code using:
  - Exact match scores
  - BLEU scores
  - Unit test execution
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

2. Install Python dependencies:
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

### Setting Up Llama C++ Server

#### Quick Setup (Automated)

1. Run the setup script:
```bash
chmod +x setup_llama_server.sh
./setup_llama_server.sh
```

2. Download models (optional helper script):
```bash
chmod +x download_models.sh
./download_models.sh
```

3. Start the server:
```bash
./start_server.sh
```

#### Manual Setup

1. **Clone and build llama.cpp:**
```bash
git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
cd ../..
```

2. **Download models:**
   - Main model: [Qwen 2.5 7B Instruct GGUF](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF)
   - Draft model: [Qwen 2.5 0.5B Instruct GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF)
   

3. **Start the server with speculative decoding:**

Simply add the `--hf-repo-draft` (or `-hfd`) flag to specify the draft model:

**CPU-only mode (recommended for limited GPU memory):**
```bash
llama-server -hf Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M --hf-repo-draft Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M --port 8081 -ngl 0 -ngld 0
```

**GPU mode (if you have enough GPU memory):**
```bash
llama-server -hf Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M --hf-repo-draft Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M --port 8081 -ngl 99 -ngld 99
```

**Mixed mode (main model on GPU, draft on CPU - good for 4GB GPUs):**
```bash
llama-server -hf Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M --hf-repo-draft Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M --port 8081 -ngl 20 -ngld 0
```

**Parameters explained:**
- `-ngl 0`: Run main model on CPU (no GPU layers)
- `-ngl 20`: Offload 20 layers to GPU (for mixed mode)
- `-ngl 99`: Offload all layers to GPU
- `-ngld 0`: Run draft model on CPU (no GPU layers)
- `-ngld 99`: Offload all draft layers to GPU
- `-c N`: Context size (optional, default: 0 = use model's native context size)
  - Qwen2.5-7B has native context size of 131072 tokens
  - You can set `-c 4096` to reduce memory usage if needed
  - For CPU mode, omitting `-c` is fine (uses model's full context)

**Alternative:** If you have models downloaded locally, you can use:
```bash
llama-server --model <path-to-7b-model.gguf> --model-draft <path-to-0.5b-model.gguf> --port 8081 -ngl 0 -ngld 0 -c 4096
```

The server will be available at `http://127.0.0.1:8081`

#### Speculative Decoding Configuration

Speculative decoding uses a smaller, faster "draft" model (Qwen2.5-0.5B) to predict tokens ahead, which are then verified by the larger "target" model (Qwen2.5-7B). This can significantly improve latency.

**Sending Prompts:** The API is exactly the same as before! You send prompts to the same endpoint (`http://127.0.0.1:8081/v1/chat/completions`). Speculative decoding is transparent to the API.

**Command-line parameters (server startup):**
- `--hf-repo-draft` or `-hfd`: HuggingFace repo for draft model (e.g., `Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M`)
- `--model-draft` or `-md`: Local path to draft model (if not using HuggingFace)
- `--draft-max` or `--draft-n`: Maximum number of tokens to draft (default: 16)
- `--draft-min` or `--draft-n-min`: Minimum number of draft tokens to use (default: 0)
- `--draft-p-min`: Minimum speculative decoding probability threshold (default: 0.8)

**API parameters (per-request):** You can override speculative decoding parameters for each request:

```python
import requests

response = requests.post(
    "http://127.0.0.1:8081/v1/chat/completions",
    json={
        "model": "qwen2.5-7b-instruct",
        "messages": [{"role": "user", "content": "Your prompt here"}],
        "max_tokens": 512,
        "temperature": 0.0,
        # Speculative decoding parameters (optional, overrides server defaults)
        "speculative.n_max": 16,  # Maximum draft tokens
        "speculative.n_min": 0,    # Minimum draft tokens
        "speculative.p_min": 0.8   # Minimum probability threshold
    }
)
```

**Example with custom command-line parameters:**
```bash
llama-server -hf Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M \
    --hf-repo-draft Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M \
    --draft-max 16 \
    --draft-min 0 \
    --draft-p-min 0.8 \
    --port 8081
```

## Usage

### Performance Benchmarking

Run performance benchmarks to measure model speed:

```bash
python run_performance.py
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
└── run_performance.py    # Main performance script
 
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


## Results

Results are saved in the `results/` directory:
- `evaluation_results.csv`: Detailed per-sample metrics


