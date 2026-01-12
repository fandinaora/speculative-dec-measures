"""
Measurement functions for benchmarking llama.cpp server performance.
"""
import time
import json
import requests
from typing import Dict, Any, Tuple
from .api import chat_payload


def measure_ttft(content: str, max_tokens: int, server_url: str, model_name: str, temperature: float = 0.0, **kwargs) -> float:
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
    with requests.post(url, headers=headers, json=chat_payload(content, max_tokens, model_name, stream=True, temperature=temperature, **kwargs), stream=True) as resp:
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



def measure_server_side_metrics(content: str, max_tokens: int, server_url: str, model_name: str, temperature: float = 0.0, **kwargs) -> Dict[str, Any]:
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
    response = requests.post(url, headers=headers, json=chat_payload(content, max_tokens, model_name, stream=False, temperature=temperature, **kwargs))
    response.raise_for_status()
    payload = response.json()
    
    # Extract server-side timings
    timings = payload.get("timings", {})
    
    # Tokens generated
    tokens_generated = timings.get("predicted_n", 0)
    
    # Prefill time (prompt processing time) - should be constant for same prompt
    prompt_ms = timings.get("prompt_ms", 0.0)
    prefill_time_sec = prompt_ms / 1000.0
    
    # Total generation time and average per token
    predicted_ms = timings.get("predicted_ms", 0.0)  # Total time for all generated tokens
    latency_ms_per_token = timings.get("predicted_per_token_ms", 0.0)  # Average per token
    latency_sec_per_token = latency_ms_per_token / 1000.0
    
    # First token time: approximate as average per token (since we don't have exact first token time)
    # Note: This is an approximation. True first token time might vary slightly.
    #first_token_time_sec = latency_sec_per_token
    
    # Server-side TTFT = prefill time + first token time
    # Using the average per-token time as approximation for first token time
    #server_ttft_sec = prefill_time_sec + first_token_time_sec
    
    # Server-side throughput
    throughput_tokens_per_sec = timings.get("predicted_per_second", 0.0)
    
    # Extract speculative decoding metrics (if available)
    draft_n = timings.get("draft_n", 0)  # Total draft tokens generated
    draft_n_accepted = timings.get("draft_n_accepted", 0)  # Draft tokens accepted
    acceptance_rate = draft_n_accepted / draft_n if draft_n > 0 else None
    
    # Extract generated content from the response
    generated_content = ""
    choices = payload.get("choices", [])
    if choices and len(choices) > 0:
        message = choices[0].get("message", {})
        generated_content = message.get("content", "")
    
    result = {
        'prefill_time_sec': prefill_time_sec,  # Prefill time only (should be constant for same prompt)
        #'first_token_time_sec': first_token_time_sec,  # Approximate first token time
        #'server_ttft_sec': server_ttft_sec,  # TTFT = prefill + first token
        'tokens_generated': tokens_generated,
        'total_generation_time_sec': predicted_ms / 1000.0,  # Total time for all tokens
        'latency_sec_per_token': latency_sec_per_token,  # Average per token
        'throughput_tokens_per_sec': throughput_tokens_per_sec,
        'generated_content': generated_content
    }
    
    # Add speculative decoding metrics if available
    if draft_n > 0:
        result['draft_n'] = draft_n
        result['draft_n_accepted'] = draft_n_accepted
        result['acceptance_rate'] = acceptance_rate
    
    return result