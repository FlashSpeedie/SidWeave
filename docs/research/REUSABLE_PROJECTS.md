# Reusable Projects & Integration Points

## Core Dependencies (Install via pip/uv)

### 1. Hardware Profiling
**llmfit-python** (from AlexsJones/llmfit)
- Purpose: CPU/GPU/RAM detection, model hardware requirements
- Install: `pip install llmfit-python` (when published) or vendor the detection logic
- Alternative: `GPUtil`, `psutil`, `py3nvml` for direct hardware access

### 2. Local Model Serving
**ollama-python** (from ollama/ollama)
- Purpose: Model pull/list/remove, chat/completion API, model metadata
- Install: `pip install ollama`
- Requires: Ollama server running locally

**llama-cpp-python** (from abetlen/llama-cpp-python)
- Purpose: Direct GGUF inference without separate server
- Install: `pip install llama-cpp-python`
- Use when: No Ollama, or need embedded inference

### 3. Provider Gateway
**litellm** (from BerriAI/litellm)
- Purpose: Unified API for 100+ providers, cost tracking, fallbacks
- Install: `pip install litellm`
- Use for: Cloud providers (OpenAI, Anthropic, Bedrock, Vertex, etc.)
- Optional: Can implement minimal provider abstraction instead

### 4. Tool Protocol
**mcp** (from modelcontextprotocol/python-sdk)
- Purpose: Model Context Protocol client/server
- Install: `pip install mcp`
- Use for: Verified tool execution, resource access

### 5. Observability
**opentelemetry-api**, **opentelemetry-sdk**, **opentelemetry-exporter-otlp**
- Purpose: Traces, metrics, logs backbone
- Install: `pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp`
- Exporters: OTLP (to Phoenix, Jaeger, etc.), Console (dev)

### 6. Configuration & CLI
**pydantic-settings**, **typer**, **rich**
- Purpose: Config validation, CLI framework, pretty output
- Install: `pip install pydantic-settings typer rich`

### 7. Database
**sqlalchemy**, **aiosqlite** (async SQLite)
- Purpose: Local persistence for model profiles, execution history, routing decisions
- Install: `pip install sqlalchemy aiosqlite`

### 8. HTTP & Async
**httpx**, **fastapi**, **uvicorn**
- Purpose: HTTP client, API server, ASGI server
- Install: `pip install httpx fastapi uvicorn`

### 9. Testing
**pytest**, **pytest-asyncio**, **pytest-mock**, **httpx** (test client)
- Install: `pip install pytest pytest-asyncio pytest-mock`

---

## Projects to Reference (Design Patterns Only)

### Provider Abstraction Pattern → LiteLLM
```
litellm/completion() → unified interface
├── Model routing (cost, latency, capability)
├── Fallback chains
├── Load balancing
├── Cost tracking per call
└── Standardized response format
```
**Adapt for SidWeave:** Simpler core, add hardware-aware routing, learned performance weights

### Model Metadata Structure → Ollama
```python
# Ollama model info
{
  "name": "llama3.1:8b",
  "size": 4.7GB,
  "digest": "sha256:...",
  "details": {
    "format": "gguf",
    "family": "llama",
    "families": ["llama"],
    "parameter_size": "8B",
    "quantization_level": "Q4_K_M"
  }
}
```
**Adapt for SidWeave:** Extend with measured performance, task benchmarks, hardware requirements

### MCP Tool Calling → MCP SDK
```python
# Standard MCP flow
client = ClientSession(...)
tools = await client.list_tools()
result = await client.call_tool("tool_name", {"arg": "value"})
```
**Extend for SidWeave:** Add verification layer (preconditions, postconditions, idempotency, recovery)

### Evaluation Methodology → Phoenix / AgentBench
- Task categorization (coding, reasoning, web, etc.)
- Metric definitions (success rate, latency, token usage, cost)
- A/B comparison framework
- Statistical significance testing

---

## Vendorable Code Snippets

### Hardware Detection (from llmfit-core concepts)
```python
# Key detection points to implement/vendor
- CPU: model, cores, threads, AVX/AVX2/AVX-512, AMX
- RAM: total, available, speed
- GPU: vendor (NVIDIA/AMD/Intel/Apple), model, VRAM, compute capability
- OS: Windows/Linux/macOS, version, architecture
- Backends: CUDA, Metal, Vulkan, ROCm, SYCL, CPU fallback
```

### GGUF Metadata Parsing (from llama.cpp)
```python
# Parse GGUF header for:
- Architecture (llama, mistral, qwen, etc.)
- Parameter count
- Quantization type (Q4_K_M, Q8_0, etc.)
- Context length
- Tensor info (for memory estimation)
```

---

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        SidWeave Runtime                      │
├─────────────────────────────────────────────────────────────┤
│  Hardware Profiler  ← llmfit-python / psutil / GPUtil       │
│  Model Registry     ← ollama-python + custom SQLite         │
│  Provider Gateway   ← litellm (cloud) + Ollama (local)      │
│  Routing Engine     ← Custom (uses above data)              │
│  Execution Engine   ← mcp (protocol) + custom verification  │
│  Learning Engine    ← Custom (SQLite + analysis)            │
│  Telemetry          ← opentelemetry (spans/metrics)         │
└─────────────────────────────────────────────────────────────┘
```

---

## Version Pinning Strategy (Initial)

| Package | Version | Reason |
|---------|---------|--------|
| python | >=3.12 | Modern typing, performance |
| ollama | >=0.4.0 | Stable API |
| litellm | >=1.50.0 | Active development, pin minor |
| mcp | >=1.0.0 | Stable protocol |
| opentelemetry-api | >=1.25.0 | Stable |
| opentelemetry-sdk | >=1.25.0 | Stable |
| sqlalchemy | >=2.0.0 | Modern async |
| aiosqlite | >=0.19.0 | Async SQLite |
| fastapi | >=0.110.0 | Stable |
| uvicorn | >=0.29.0 | Stable |
| httpx | >=0.27.0 | Async HTTP |
| pydantic | >=2.7.0 | V2 stable |
| pydantic-settings | >=2.3.0 | Config |
| typer | >=0.12.0 | CLI |
| rich | >=13.7.0 | Output |
| pytest | >=8.2.0 | Testing |
| pytest-asyncio | >=0.23.0 | Async tests |

---

## Optional/Deferred Dependencies

| Package | Purpose | Defer Reason |
|---------|---------|--------------|
| llama-cpp-python | Embedded inference | Large, platform-specific wheels |
| GPUtil | NVIDIA GPU stats | Windows support varies |
| py3nvml | NVIDIA ML details | Linux-focused |
| torch | ML features (LoRA, etc.) | Large, not needed for MVP |
| transformers | Model inspection | Large, not needed for MVP |
| phoenix | Evaluation UI | Use via OTLP export instead |
| wandb | Experiment tracking | External service |

---

## Development Tools

```toml
[tool.ruff]
target-version = "py312"
line-length = 100
select = ["E", "F", "I", "UP", "B", "C4", "W", "T20", "SIM", "RUF", "PERF", "PTH", "ERA", "RET", "ARG", "FLY", "NPY", "PD", "PIE", "TID", "TRY", "A", "Q", "DTZ", "EXE", "ISC", "ICN", "LOG", "PL", "TRY"]
ignore = ["S101", "S311", "S603", "S607"]

[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_ignores = true
```

---

## Installation Commands

```bash
# Core dependencies
uv pip install ollama litellm mcp opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp sqlalchemy aiosqlite fastapi uvicorn httpx pydantic pydantic-settings typer rich

# Dev dependencies
uv pip install pytest pytest-asyncio pytest-mock ruff mypy

# Optional: Hardware detection (when llmfit-python available)
# uv pip install llmfit-python
# For now: implement minimal detection using psutil + platform
uv pip install psutil platformdirs
```