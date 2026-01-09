import requests
import json

# Server URL
url = "http://127.0.0.1:8081/v1/chat/completions"

# Headers
headers = {
    "Content-Type": "application/json"
}

# Request data
data = {
    "model": "qwen2.5-7b-instruct",  # Model name
    "messages": [
        {"role": "user", "content": "What is the capital of France?"}
    ]
}

# Send request
print("Sending request to llama server...")
try:
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()  # Raise an error for bad status codes
    
    result = response.json()
    print("\nFull response:")
    print(json.dumps(result, indent=2))
    
    # Extract just the message content
    if "choices" in result and len(result["choices"]) > 0:
        message = result["choices"][0]["message"]["content"]
        print("\n" + "="*50)
        print("AI Response:")
        print("="*50)
        print(message)
        
except requests.exceptions.ConnectionError:
    print("Error: Could not connect to server.")
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")
except Exception as e:
    print(f" Unexpected error: {e}")

