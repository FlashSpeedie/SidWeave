#!/usr/bin/env python3
"""
Existing Agent Integration Example

Demonstrates connecting an existing OpenAI-compatible agent to SidWeave
with minimal code changes.
"""

import os
import sys
import openai

# Add project root to path for imports if running from examples/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── Configuration ──────────────────────────────────────────────────────
SIDWEAVE_BASE_URL = os.getenv("SIDWEAVE_BASE_URL", "http://127.0.0.1:8765/v1")
SIDWEAVE_API_KEY = os.getenv("SIDWEAVE_API_KEY", "not-needed")

# ─── Client Setup ───────────────────────────────────────────────────────
# This is the ONLY change needed from standard OpenAI usage
client = openai.OpenAI(
    base_url=SIDWEAVE_BASE_URL,
    api_key=SIDWEAVE_API_KEY,
)

# ─── Helper Functions ───────────────────────────────────────────────────

def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

# Empty string triggers auto-routing in SidWeave
AUTO_ROUTE = ""

def chat(model: str = AUTO_ROUTE, messages: list[dict] = None, max_tokens: int = 200) -> str:
    """Send chat completion request and return response content."""
    response = client.chat.completions.create(
        model=model,
        messages=messages or [],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return response.choices[0].message.content

def chat_with_info(model: str = AUTO_ROUTE, messages: list[dict] = None, max_tokens: int = 200) -> tuple[str, str, dict]:
    """Send request and return (content, model_used, usage)."""
    response = client.chat.completions.create(
        model=model,
        messages=messages or [],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    choice = response.choices[0]
    model_used = response.model
    usage = {
        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        "total_tokens": response.usage.total_tokens if response.usage else 0,
    }
    return choice.message.content, model_used, usage

# ─── Test Cases ─────────────────────────────────────────────────────────

def test_direct_chat() -> None:
    """Test 1: Direct chat with auto routing."""
    print_section("Test 1: Direct Chat Completion (auto routing)")
    
    messages = [{"role": "user", "content": "Hello! Say hi in one sentence."}]
    content, model_used, usage = chat_with_info(messages=messages, max_tokens=50)
    
    print(f"  Model used: {model_used}")
    print(f"  Tokens: {usage['total_tokens']} (prompt: {usage['prompt_tokens']}, completion: {usage['completion_tokens']})")
    print(f"  Response: {content}")

def test_coding_task() -> None:
    """Test 2: Coding task - should route to code-capable model."""
    print_section("Test 2: Coding Task (auto routing)")
    
    messages = [{
        "role": "user",
        "content": "Write a Python function to reverse a string. One line only."
    }]
    content, model_used, usage = chat_with_info(messages=messages, max_tokens=100)
    
    print(f"  Model selected: {model_used}")
    print(f"  Tokens: {usage['total_tokens']}")
    print(f"  Response: {content.strip()}")

def test_reasoning_task() -> None:
    """Test 3: Reasoning task."""
    print_section("Test 3: Reasoning Task (auto routing)")
    
    messages = [{
        "role": "user",
        "content": "What is 42 * 17? Show the calculation."
    }]
    content, model_used, usage = chat_with_info(messages=messages, max_tokens=100)
    
    print(f"  Model selected: {model_used}")
    print(f"  Tokens: {usage['total_tokens']}")
    print(f"  Response: {content.strip()}")

def test_conversation() -> None:
    """Test 4: Multi-turn conversation."""
    print_section("Test 4: Multi-turn Conversation")
    
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "What language is primarily spoken there?"}
    ]
    content, model_used, usage = chat_with_info(messages=messages, max_tokens=50)
    
    print(f"  Model used: {model_used}")
    print(f"  Tokens: {usage['total_tokens']}")
    print(f"  Response: {content.strip()}")

def test_specific_model() -> None:
    """Test 5: Request specific model (bypass routing)."""
    print_section("Test 5: Specific Model Request (gemma:2b)")
    
    # First, check if gemma:2b is available
    try:
        models_response = client.models.list()
        model_names = [m.id for m in models_response.data]
        
        if "gemma:2b" in model_names:
            messages = [{"role": "user", "content": "Say 'hello from gemma' in one sentence."}]
            content, model_used, usage = chat_with_info("gemma:2b", messages, max_tokens=30)
            print(f"  Model used: {model_used}")
            print(f"  Response: {content}")
        else:
            print(f"  gemma:2b not available. Available: {model_names[:5]}...")
    except Exception as e:
        print(f"  Could not list models: {e}")

def test_streaming() -> None:
    """Test 6: Streaming response."""
    print_section("Test 6: Streaming Response")
    
    messages = [{"role": "user", "content": "Count from 1 to 5, one number per line."}]
    
    print("  Streaming: ", end="", flush=True)
    stream = client.chat.completions.create(
        model=AUTO_ROUTE,
        messages=messages,
        max_tokens=50,
        stream=True,
    )
    
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()  # newline

# ─── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    print("=== SidWeave Existing Agent Integration ===")
    print(f"  Base URL: {SIDWEAVE_BASE_URL}")
    print(f"  API Key: {'***' if SIDWEAVE_API_KEY != 'not-needed' else 'not-needed (local)'}")
    
    try:
        # Verify connection
        health = client.chat.completions.create(
            model=AUTO_ROUTE,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5
        )
        print(f"  Connection: OK (model: {health.model})")
    except Exception as e:
        print(f"  Connection FAILED: {e}")
        print("\n  Make sure SidWeave is running: 'sidweave serve'")
        sys.exit(1)
    
    # Run tests
    test_direct_chat()
    test_coding_task()
    test_reasoning_task()
    test_conversation()
    test_specific_model()
    test_streaming()
    
    print_section("Done")
    print("All tests completed successfully!")

if __name__ == "__main__":
    main()