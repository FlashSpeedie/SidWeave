# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-18

### Added
- Hardware profiling (CPU, GPU, memory, available inference backends)
- Model registry with Ollama and NVIDIA NIM provider support
- Model profiling (latency, throughput measurements)
- Adaptive routing engine with explainable decisions
  - Balanced, latency, cost, quality, and local-first strategies
  - Capability-based filtering (chat, code, reasoning, vision, tools, embedding)
  - Hardware compatibility scoring
  - Historical performance integration
- MCP verified execution engine
  - Precondition validation (file exists, custom checks)
  - Postcondition verification (result success, no error, custom checks)
  - Idempotency key support
  - Approval workflow for sensitive operations
  - Execution reconciliation for interrupted/ambiguous outcomes
- OpenAI-compatible API gateway
  - `/v1/models` - list available models
  - `/v1/chat/completions` - chat completions with routing
  - `/health` - component health status
  - `/v1/sidweave/*` - SidWeave-specific endpoints (routing, hardware, providers, profiling)
- CLI with commands:
  - `init` - initialize configuration and data directory
  - `doctor` - run diagnostics
  - `serve` - start runtime server
  - `status` - show runtime status
  - `models list|profile|stats` - model management
  - `route` - test routing decisions
  - `benchmark` - run baseline vs SidWeave comparison
  - `config` - show or edit configuration
- SQLite storage with full schema (7 tables)
- OpenTelemetry integration with custom metrics
- Benchmark runner with fair baseline/SidWeave comparison
- Configuration via environment variables and `.env` file
- Unit tests (91 passing), MyPy clean, Ruff clean

### Known Limitations
- SQLAlchemy/aiosqlite connection cleanup warnings (cosmetic, no functional impact)
- Provider model counts may differ between test and server processes if environment variables change after settings load
- Benchmark sample size small (3 tasks) - not statistically significant
- Some NVIDIA NIM models return 404 for chat completions (filtered in model registry)
- Local-first routing prefers Ollama even when cloud model selected (fallback works correctly)

## [Unreleased]

### Planned
- Learned routing policies from execution history
- LoRA/QLoRA experimentation framework
- Preference optimization (DPO, GRPO)
- Additional agent integrations
- Distributed inference support
- Improved connection pooling for SQLite