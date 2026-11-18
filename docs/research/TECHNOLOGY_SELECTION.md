# Technology Selection & Rationale

## Core Language: Python 3.12+

**Rationale:**
- Best ecosystem for LLM tooling (LiteLLM, MCP, OpenTelemetry, Ollama all have first-class Python SDKs)
- Fast enough for orchestration layer (not doing inference)
- Rich type system (Pydantic v2, typing extensions)
- Async/await native support
- Universal deployment (Windows, Linux, macOS)
- Team familiarity

**Version:** 3.12 minimum (3.13 when stable)
- `typing.Self`, `typing.override`, better generics
- Performance improvements
- `pathlib` improvements

---

## Package Management: uv

**Rationale:**
- 10-100x faster than pip
- Lockfile support (`uv.lock`)
- Virtual environment management
- Python version management
- Single binary, no Python bootstrap needed

**Commands:**
```bash
uv venv --python 3.12
uv pip install -r requirements.txt
uv pip compile pyproject.toml -o requirements.txt
uv lock
```

---

## Configuration: Pydantic Settings v2

**Rationale:**
- Type-safe configuration with validation
- Environment variable mapping
- Nested settings support
- Secrets handling via `SecretStr`
- TOML/JSON/YAML file support

**Structure:**
```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Runtime
    data_dir: Path = Path.home() / ".sidweave"
    log_level: str = "INFO"
    host: str = "127.0.0.1"
    port: int = 8080
    
    # Providers
    ollama_host: str = "http://localhost:11434"
    litellm_master_key: SecretStr | None = None
    
    # Telemetry
    otel_endpoint: str | None = None
    otel_service_name: str = "sidweave"
```

---

## CLI Framework: Typer + Rich

**Rationale:**
- Type-hint based CLI (auto-completion, validation)
- Rich integration for beautiful output
- Subcommand support
- Async command support

**Commands:**
```python
app = typer.Typer(help="SidWeave - Adaptive Runtime for AI Agents")

@app.command()
def init(...): ...

@app.command()
def doctor(...): ...

@app.command()
def serve(...): ...

@app.command()
def models(action: str = "list", ...): ...

@app.command()
def benchmark(...): ...
```

---

## HTTP & API: FastAPI + HTTPX

**Rationale:**
- FastAPI: OpenAPI auto-generation, dependency injection, async native
- HTTPX: Async HTTP client, HTTP/2 support, timeout handling
- Both support Pydantic models natively

**OpenAI-Compatible Gateway:**
```python
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    # Route via SidWeave routing engine
    # Return OpenAI-compatible response
```

---

## Database: SQLite (aiosqlite + SQLAlchemy 2.0)

**Rationale:**
- Zero-configuration, file-based
- ACID compliant
- WAL mode for concurrent reads
- Portable (single file)
- No server dependency
- SQLAlchemy 2.0 async ORM

**Schema Areas:**
- `models` - Model registry with metadata
- `profiles` - Hardware profiles
- `measurements` - Model performance measurements
- `routing_decisions` - Routing choices with rationale
- `executions` - Tool execution lifecycle records
- `verifications` - Verification results
- `tasks` - Task history with outcomes
- `benchmarks` - Benchmark runs and results

**Data Directory:** `~/.sidweave/` (configurable)
- `sidweave.db` - Main database
- `logs/` - Structured logs
- `cache/` - Model metadata cache

---

## Hardware Detection: psutil + platform + custom

**Rationale:**
- `psutil`: Cross-platform CPU, memory, disk, process info
- `platform`: OS, architecture, Python version
- Custom: GPU detection (NVIDIA via `nvidia-smi`, AMD via `rocm-smi`, Apple Metal via `system_profiler`)
- Avoid llmfit-python until published on PyPI
- Vendor minimal detection logic

**Detection Priority:**
1. CPU: model, cores (physical/logical), frequency, instruction sets
2. Memory: total, available, swap
3. GPU: vendor, model, VRAM, driver version, compute capability
4. Disk: available space for models
5. Backends: CUDA, ROCm, Metal, Vulkan, CPU

---

## Provider Abstraction: LiteLLM (cloud) + Ollama (local)

**Rationale:**
- LiteLLM: 100+ providers, unified interface, cost tracking, fallbacks
- Ollama: Local model management, OpenAI-compatible API
- Both return OpenAI-compatible responses

**Integration Pattern:**
```python
class ProviderRegistry:
    def __init__(self):
        self.local = OllamaProvider()
        self.cloud = LiteLLMProvider()  # Optional
    
    async def complete(self, request) -> CompletionResponse:
        # Route to appropriate provider
        # Handle fallbacks
        # Track costs/latency
```

**Fallback Chain:**
1. Local (Ollama) - preferred for privacy, speed, cost
2. Cloud (LiteLLM) - when local unavailable or capability gap
3. Error with clear message

---

## Tool Protocol: MCP Python SDK

**Rationale:**
- Official reference implementation
- Standardized tool calling
- Multiple transports (stdio, SSE, Streamable HTTP)
- Active maintenance by MCP authors

**Verified Execution Layer (SidWeave addition):**
```python
class VerifiedMCPClient:
    async def call_tool_verified(self, tool_name, args, 
                                  preconditions=None,
                                  postconditions=None,
                                  idempotency_key=None) -> VerifiedResult:
        # 1. Check preconditions
        # 2. Check idempotency key (dedup)
        # 3. Execute via MCP
        # 4. Validate postconditions
        # 5. Record execution lifecycle
        # 6. Return verified result or recovery info
```

---

## Telemetry: OpenTelemetry

**Rationale:**
- Vendor-neutral, CNCF graduated
- Auto-instrumentation for FastAPI, HTTPX, SQLAlchemy
- Multiple exporters (OTLP, Console, Jaeger, Zipkin)
- Phoenix consumes OTLP natively

**Key Spans:**
- `routing.decision` - Model selection with rationale
- `provider.call` - Upstream API call
- `tool.execution` - MCP tool call lifecycle
- `verification.check` - Pre/post condition validation
- `benchmark.run` - Benchmark execution

**Metrics:**
- `sidweave.routing.latency` - Routing decision time
- `sidweave.provider.latency` - Upstream latency
- `sidweave.provider.tokens` - Token usage
- `sidweave.tool.success_rate` - Tool success/failure
- `sidweave.verification.rate` - Verification pass/fail

---

## Testing: pytest + pytest-asyncio

**Rationale:**
- Industry standard
- Async support via pytest-asyncio
- Fixtures for database, providers, mock servers
- Parametrized tests for routing scenarios

**Test Categories:**
- `tests/unit/` - Pure logic, no external deps
- `tests/integration/` - Real providers (mocked or live)
- `tests/benchmarks/` - Benchmark runner tests
- `tests/fixtures/` - Shared test data

---

## Code Quality: Ruff + MyPy

**Rationale:**
- Ruff: Fast linting + formatting (replaces Black, isort, flake8)
- MyPy: Static type checking (strict mode)
- Both integrate with editors and CI

**Ruff Config:** (see REUSABLE_PROJECTS.md)

**MyPy Config:** Strict mode with selective ignores for external libs

---

## CI/CD: GitHub Actions

**Workflows:**
1. `test.yml` - Lint, type-check, unit tests, integration tests
2. `build.yml` - Package build, publish to PyPI (on tag)
3. `benchmark.yml` - Scheduled benchmark runs (optional)

---

## Documentation: Markdown + mkdocstrings (later)

**Rationale:**
- Markdown for human-readable docs
- Docstrings for API reference
- mkdocstrings for auto-generated API docs (Phase 7)

---

## Deferred Technologies (Not in MVP)

| Technology | Purpose | Defer To |
|------------|---------|----------|
| PyTorch | LoRA/QLoRA training | Stage 4 |
| transformers | Model inspection, tokenization | Stage 4 |
| trl | DPO/GRPO training | Stage 4 |
| vLLM | High-throughput serving | Stage 5 |
| Ray | Distributed inference | Stage 5 |
| Kubernetes | Orchestration | Stage 5 |
| Prometheus/Grafana | Metrics stack | Use Phoenix/OTel instead |

---

## Platform Support Matrix

| Feature | Windows | Linux | macOS |
|---------|---------|-------|-------|
| Core runtime | ✅ | ✅ | ✅ |
| Hardware detection | ✅ | ✅ | ✅ |
| Ollama integration | ✅ | ✅ | ✅ |
| llama.cpp (CPU) | ✅ | ✅ | ✅ |
| llama.cpp (CUDA) | ✅ | ✅ | ❌ |
| llama.cpp (Metal) | ❌ | ❌ | ✅ |
| llama.cpp (ROCm) | ❌ | ✅ | ❌ |
| MCP stdio | ✅ | ✅ | ✅ |
| MCP HTTP | ✅ | ✅ | ✅ |
| OpenTelemetry | ✅ | ✅ | ✅ |

---

## Dependency License Audit

All core dependencies use permissive licenses:
- MIT: ollama, litellm, mcp, fastapi, uvicorn, httpx, pydantic, typer, rich, pytest, ruff, psutil, platformdirs
- Apache-2.0: opentelemetry-*, sqlalchemy, aiosqlite
- BSD-3: mypy

No GPL, LGPL, AGPL, or viral licenses in dependency tree.