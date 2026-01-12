"""
API utilities for interacting with llama.cpp server.
"""
from typing import Dict, Any


def chat_payload(content: str, max_tokens: int, model_name: str, stream: bool = False, temperature: float = 0.0, **kwargs) -> Dict[str, Any]:
    """
    Construct the OpenAI-compatible chat payload for llama.cpp server.
    
    Args:
        content: The message content
        max_tokens: Maximum tokens to generate
        model_name: Model name to use
        stream: Whether to use streaming mode
        temperature: Sampling temperature (0.0 = deterministic, >0.0 = more diverse). Default: 0.0
        **kwargs: Additional parameters to include in the payload. Examples:
                  - Sampling: top_p, top_k, min_p, etc.
                  - Speculative decoding: "speculative.n_max", "speculative.n_min", "speculative.p_min"
                  - Other llama.cpp server parameters
    
    Returns:
        Dictionary with chat completion payload
    
    Example with speculative decoding:
        payload = chat_payload(
            "Your prompt",
            max_tokens=512,
            model_name="qwen2.5-7b-instruct",
            **{"speculative.n_max": 16, "speculative.n_min": 0, "speculative.p_min": 0.8}
        )
    """
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": content}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    
    # Add or override with any additional parameters from kwargs
    payload.update(kwargs)
    
    return payload
