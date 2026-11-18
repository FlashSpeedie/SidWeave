# Existing Agent Integration Example

This example demonstrates how to connect an existing AI agent (using the OpenAI Python SDK) to SidWeave without rewriting the agent logic.

## What This Shows

- Your agent code stays almost identical
- Only the client configuration changes (base_url, api_key)
- SidWeave handles model selection, routing, and fallbacks
- Works with any OpenAI-compatible client (OpenAI SDK, LangChain, LiteLLM, etc.)

## Prerequisites

1. SidWeave running (`sidweave serve` in another terminal)
2. Ollama running with at least one model (`ollama pull gemma:2b`)
3. Python 3.12+ with `openai` package

```powershell
pip install openai
```

## Run the Example

```powershell
# From the SidWeave root directory
cd examples/existing_agent
python agent.py
```

## Files

| File | Purpose |
|------|---------|
| `agent.py` | Complete runnable example |
| `requirements.txt` | Dependencies |
| `.env.example` | Configuration template |

## Expected Output

```
=== SidWeave Existing Agent Integration ===

1. Testing direct chat completion...
   Model used: gemma:2b (via Ollama)
   Response: Hello! How can I help you today?

2. Testing with auto routing (coding task)...
   Model selected: gemma:2b (local)
   Response: def reverse_string(s): return s[::-1]

3. Testing with auto routing (reasoning task)...
   Model selected: gemma:2b (local)
   Response: 42 * 17 = 714

4. Testing conversation history...
   Model used: gemma:2b
   Response: The capital of France is Paris.

=== Done ===
```

## Key Changes from Standard OpenAI Usage

| Standard OpenAI | With SidWeave |
|----------------|---------------|
| `base_url="https://api.openai.com/v1"` | `base_url="http://127.0.0.1:8765/v1"` |
| `api_key="sk-..."` | `api_key="not-needed"` (local) |
| `model="gpt-4o"` | `model="auto"` (SidWeave routes) |

## How It Works

1. **Agent sends request** → OpenAI client posts to `http://127.0.0.1:8765/v1/chat/completions`
2. **SidWeave receives** → API gateway validates request
3. **Routing engine selects model** → Based on task category, capabilities, hardware, history
4. **Provider gateway dispatches** → To Ollama (local) or cloud provider with fallback
5. **Response returned** → In OpenAI format, agent processes normally

## Using with Cloud Providers

If you have cloud API keys configured in `~/.sidweave/.env`:

```python
# Same code works - SidWeave will route to best available model
# including cloud models if they score higher for the task
client = openai.OpenAI(
    base_url="http://127.0.0.1:8765/v1",
    api_key="not-needed"
)
```

## Using with LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://127.0.0.1:8765/v1",
    api_key="not-needed",
    model="auto",
    temperature=0.7
)

response = llm.invoke("Write a Python function to reverse a string")
```

## Using with LiteLLM

```python
import litellm

litellm.base_url = "http://127.0.0.1:8765/v1"
litellm.api_key = "not-needed"

response = litellm.completion(
    model="auto",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Troubleshooting

| Error | Solution |
|-------|----------|
| Connection refused | Ensure `sidweave serve` is running |
| Model not found | Run `sidweave models list` to see available models |
| Timeout | Increase timeout in client or check Ollama is responsive |
| 404 on cloud model | Some NVIDIA NIM models don't support chat; check `sidweave models list` |