"""
Measurement functions for benchmarking llama.cpp server performance.
"""
import time
import json
import requests
from typing import Dict, Any, Tuple
from .api import chat_payload


def measure_ttft(content: str, max_tokens: int, server_url: str, model_name: str, temperature: float = 0.0) -> float:
    """
    Measure time-to-first-token (TTFT) using streaming responses.
    Returns seconds to first non-empty delta content.
    
    Args:
        content: The message content
        max_tokens: Maximum tokens to generate
        server_url: Base URL of the llama.cpp server
        model_name: Model name to use
        temperature: Sampling temperature (0.0 = deterministic, >0.0 = more diverse). Default: 0.0
    
    Returns:
        Time to first token in seconds
    """
    url = f"{server_url}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    start = time.perf_counter()
    with requests.post(url, headers=headers, json=chat_payload(content, max_tokens, model_name, stream=True, temperature=temperature), stream=True) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if not raw.startswith("data: "):
                continue
            data_str = raw[6:].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content_piece = delta.get("content")
            if content_piece:
                return time.perf_counter() - start
    # If we never saw any content, return total elapsed as a fallback
    return time.perf_counter() - start



def measure_server_side_metrics(content: str, max_tokens: int, server_url: str, model_name: str, temperature: float = 0.0) -> Dict[str, Any]:
    """
    Extract server-side metrics from llama.cpp response.
    NO client-side timing, NO streaming - just parse the response.
    
    Args:
        content: The message content
        max_tokens: Maximum tokens to generate
        server_url: Base URL of the llama.cpp server
        model_name: Model name to use
        temperature: Sampling temperature (0.0 = deterministic, >0.0 = more diverse). Default: 0.0
    
    Returns:
        Dictionary with server-side metrics:
        {
            'server_ttft_sec': float,           # Server TTFT in SECONDS (prefill + first token)
            'tokens_generated': int,             # Number of tokens generated
            'latency_sec_per_token': float,     # Average generation time per token in SECONDS
            'throughput_tokens_per_sec': float, # Tokens per second
            'generated_content': str            # The actual generated text content
        }
    """
    url = f"{server_url}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    # Simple non-streaming request
    response = requests.post(url, headers=headers, json=chat_payload(content, max_tokens, model_name, stream=False, temperature=temperature))
    response.raise_for_status()
    payload = response.json()
    
    # Extract server-side timings
    timings = payload.get("timings", {})
    
    # Tokens generated
    tokens_generated = timings.get("predicted_n", 0)
    
    # Server-side latency per token (convert ms to seconds)
    latency_ms_per_token = timings.get("predicted_per_token_ms", 0.0)
    latency_sec_per_token = latency_ms_per_token / 1000.0
    
    # Server-side TTFT = prefill time + first token time (convert ms to seconds)
    prompt_ms = timings.get("prompt_ms", 0.0)
    server_ttft_sec = (prompt_ms + latency_ms_per_token) / 1000.0
    
    # Server-side throughput
    throughput_tokens_per_sec = timings.get("predicted_per_second", 0.0)
    
    # Extract generated content from the response
    generated_content = ""
    choices = payload.get("choices", [])
    if choices and len(choices) > 0:
        message = choices[0].get("message", {})
        generated_content = message.get("content", "")
    
    return {
        'server_ttft_sec': server_ttft_sec,
        'tokens_generated': tokens_generated,
        'latency_sec_per_token': latency_sec_per_token,
        'throughput_tokens_per_sec': throughput_tokens_per_sec,
        'generated_content': generated_content
    }
