# Quick Start Guide

Get SidWeave running in 5 minutes.

## Prerequisites

- **Python 3.12+**
- **Ollama** installed and running for local models
  - Windows: Download from [ollama.com](https://ollama.com/download/windows)
  - Verify: `ollama --version`
  - Pull a model: `ollama pull gemma:2b`

## Installation

```powershell
# Clone the repository
git clone https://github.com/SidEraLabs/SidWeave.git
cd SidWeave

# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

## First Run

### 1. Initialize Configuration

```powershell
sidweave init
```

Output:
```
┌─ Welcome to SidWeave ────────────────────────┐
│                                              │
│ SidWeave initialized!                        │
│                                              │
│ Data directory: C:\Users\you\.sidweave       │
│ Config file: C:\Users\you\.sidweave\.env.example │
│                                              │
│ Next steps:                                  │
│ 1. Copy .env.example to .env and configure   │
│ 2. Run 'sidweave doctor' to verify setup     │
│ 3. Run 'sidweave serve' to start runtime     │
└──────────────────────────────────────────────┘
```

This creates:
- Data directory: `~/.sidweave/`
- Example config: `~/.sidweave/.env.example`

### 2. Configure (Optional for Local-Only)

For **local-only** usage (Ollama), no configuration needed. The defaults work.

For **cloud providers**, copy and edit:

```powershell
# Copy example config
copy ~\.sidweave\.env.example ~\.sidweave\.env

# Edit with your editor
notepad ~\.sidweave\.env
```

Required for NVIDIA NIM:
```ini
NVIDIA_API_KEY=your-key-here
SIDWEAVE_PROVIDER_ENABLE_CLOUD_PROVIDERS=true
```

Required for OpenAI:
```ini
OPENAI_API_KEY=your-key-here
SIDWEAVE_PROVIDER_ENABLE_CLOUD_PROVIDERS=true
```

### 3. Verify Setup

```powershell
sidweave doctor
```

Expected output (all checks pass):
```
┌─ SidWeave Diagnostics ─────────────────────┐
│ Check              Status  Details          │
├──────────────────────────────────────────────┤
│ Data directory     PASS    C:\Users\you\.sidweave │
│ Database           PASS    C:\Users\you\.sidweave\sidweave.db │
│ Ollama CLI         PASS    Available        │
│ Package: ollama    PASS    OK               │
│ Package: litellm   PASS    OK               │
│ Package: mcp       PASS    OK               │
│ ...                                          │
└──────────────────────────────────────────────┘

All checks passed!
```

If Ollama check fails, ensure Ollama is installed and in PATH.

### 4. Discover Models

```powershell
sidweave models list
```

Output (local + cloud if configured):
```
┌─ Available Models (22) ────────────────────────────────────────┐
│ Name                          Provider    Capabilities    Local │
├────────────────────────────────────────────────────────────────┤
│ gemma:2b                      ollama      chat,completion  [+]  │
│ google/gemma-3-12b-it         nvidia_nim  chat,completion  [-]  │
│ meta/llama-3.1-405b-instruct  nvidia_nim  chat,tools,code  [-]  │
│ ...                                                          │
└────────────────────────────────────────────────────────────────┘
```

### 5. Start the Runtime

```powershell
sidweave serve
```

Output:
```
┌─ SidWeave Runtime ───────────────────────┐
│ Starting SidWeave server on 127.0.0.1:8765 │
└────────────────────────────────────────────┘
INFO: Started server process [12345]
INFO: Waiting for application startup.
INFO: Application startup complete.
```

API now available at: `http://127.0.0.1:8765/v1`

Keep this terminal open. Open a new terminal for the next steps.

### 6. Make Your First Request

In a new terminal:

```powershell
# Using curl
curl -X POST http://127.0.0.1:8765/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{"model": "auto", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 50}'
```

Or with Python:

```python
import openai

client = openai.OpenAI(
    base_url="http://127.0.0.1:8765/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="auto",  # SidWeave selects best model
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=50
)

print(response.choices[0].message.content)
```

Expected: A response from the best available model (local Ollama if no cloud configured).

## Common Commands

| Command | Purpose |
|---------|---------|
| `sidweave serve` | Start API server |
| `sidweave doctor` | Run diagnostics |
| `sidweave models list` | List available models |
| `sidweave models profile <name>` | Profile model latency/throughput |
| `sidweave route coding` | Test routing for coding tasks |
| `sidweave status` | Show runtime status |
| `sidweave benchmark quick --compare` | Run baseline vs SidWeave |
| `sidweave config --show` | View current configuration |

## Next Steps

- **Configure cloud providers**: Add API keys to `~/.sidweave/.env`
- **Test routing**: `sidweave route coding --capabilities "code_generation,reasoning"`
- **Run benchmarks**: `sidweave benchmark my-test --compare`
- **Read architecture**: See `docs/architecture/README.md`
- **Integration example**: See `examples/existing_agent/`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `sidweave` not found | Ensure `~/.local/bin` or uv env is in PATH, or use `python -m sidweave.cli.main` |
| Ollama not detected | Install Ollama, restart terminal, verify `ollama --version` |
| No models listed | Run `ollama pull gemma:2b` or configure cloud API keys |
| Port 8765 in use | Change port: `sidweave serve --port 8766` |
| Import errors | Reinstall: `uv pip install -e . --force-reinstall` |

## Local vs Cloud

| Mode | Setup | Models | Cost |
|------|-------|--------|------|
| Local only | `sidweave init` + Ollama | gemma:2b, llama3.1, etc. | Free |
| With NVIDIA NIM | Add `NVIDIA_API_KEY` + enable cloud | 20+ models | Free tier |
| With OpenAI | Add `OPENAI_API_KEY` + enable cloud | GPT-4o, etc. | Per-token |

Start local. Add cloud when needed.