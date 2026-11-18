# Architecture Documentation

## Overview

SidWeave is an adaptive runtime middleware that sits between existing AI agents and model providers. It provides hardware-aware model routing, verified tool execution, and continuous learning from execution outcomes.

## Component Diagram

The main architecture diagram is in [`architecture.mmd`](architecture.mmd) (Mermaid format).

## Request Flow

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  AI Agent /     │────▶│  SidWeave API    │────▶│  Adaptive Routing   │
│  Application    │     │  (OpenAI compat) │     │  Engine             │
└─────────────────┘     └──────────────────┘     └─────────┬───────────┘
                                                            │
                    ┌───────────────────────────────────────┼───────────────┐
                    ▼                                       ▼               ▼
            ┌───────────────┐                     ┌─────────────────┐ ┌────────────┐
            │ Model Registry│                     │ Hardware Profiler│ │ Executions │
            │ - Ollama sync │                     │ - CPU/GPU/Memory │ │ History    │
            │ - NVIDIA sync │                     │ - Backends       │ └────────────┘
            │ - Capabilities│                     └─────────────────┘
            └───────┬───────┘
                    │
                    ▼
            ┌──────────────────┐     ┌────────────────────┐
            │ Provider Gateway │────▶│ Local: Ollama      │
            │ - Health checks  │     │ Cloud (optional):  │
            │ - Fallback logic │     │ - NVIDIA NIM       │
            │ - Dispatch       │     │ - OpenAI           │
            └──────────────────┘     │ - Anthropic        │
                                     │ - OpenRouter       │
                                     └────────────────────┘
```

## Data Flow

1. **Agent Request** → Agent sends OpenAI-compatible request to `POST /v1/chat/completions`
2. **Routing** → Routing Engine selects best model based on:
   - Task category (coding, reasoning, chat, etc.)
   - Required capabilities (tools, vision, code, etc.)
   - Hardware constraints (RAM, VRAM, backends)
   - Historical performance (latency, throughput, success rate)
   - Strategy weights (balanced, latency, cost, quality, local_first)
3. **Provider Dispatch** → Provider Gateway routes to selected provider with fallback
4. **Execution** → If tools involved, Execution Engine validates preconditions, executes, verifies postconditions
5. **Learning** → Outcomes recorded for future routing decisions

## Key Components

| Component | Responsibility | Key Files |
|-----------|----------------|-----------|
| API Gateway | OpenAI-compatible REST API | `src/sidweave/api/gateway.py` |
| Routing Engine | Model selection & scoring | `src/sidweave/routing/engine.py` |
| Model Registry | Discovery, profiling, persistence | `src/sidweave/models/registry.py` |
| Provider Gateway | Unified provider interface | `src/sidweave/providers/gateway.py` |
| Hardware Profiler | System capability detection | `src/sidweave/hardware/profiler.py` |
| Execution Engine | MCP tool verification | `src/sidweave/execution/engine.py` |
| Learning Engine | Outcome recording | `src/sidweave/learning/engine.py` |
| Storage | SQLite persistence | `src/sidweave/storage/__init__.py` |

## Optional vs Required

| Component | Required | Notes |
|-----------|----------|-------|
| Ollama | ✅ Yes | Local-first, no API key needed |
| NVIDIA NIM | ❌ No | Requires `NVIDIA_API_KEY` |
| OpenAI | ❌ No | Requires `OPENAI_API_KEY` |
| Anthropic | ❌ No | Requires `ANTHROPIC_API_KEY` |
| OpenRouter | ❌ No | Requires `OPENROUTER_API_KEY` |
| OpenTelemetry | ❌ No | Enabled by default, configurable |
| MCP Servers | ❌ No | Configured via `SIDWEAVE_EXECUTION_MCP_SERVERS` |