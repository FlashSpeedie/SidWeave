"""
OpenAI-compatible API gateway for SidWeave.
"""

import logging
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from sidweave.config import Settings, get_settings
from sidweave.core.runtime import SidWeaveRuntime, create_runtime
from sidweave.hardware import HardwareProfile
from sidweave.models import ModelCapability, ModelRegistry
from sidweave.providers import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ProviderGateway,
)
from sidweave.routing import RoutingConstraints, RoutingEngine, RoutingStrategy

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Global runtime instance
_runtime: SidWeaveRuntime | None = None


# Request/Response models (OpenAI-compatible)

class ModelInfo(BaseModel):
    """Model information."""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "sidweave"


class ModelListResponse(BaseModel):
    """Model list response."""
    object: str = "list"
    data: list[ModelInfo]


class ErrorResponse(BaseModel):
    """Error response."""
    error: dict[str, Any]


class ChatCompletionMessage(BaseModel):
    """Chat completion message."""
    role: str
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatCompletionChoice(BaseModel):
    """Chat completion choice."""
    index: int
    message: ChatCompletionMessage
    finish_reason: str | None = None


class ChatCompletionUsage(BaseModel):
    """Chat completion usage."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionChunk(BaseModel):
    """Chat completion streaming chunk."""
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[dict[str, Any]]


# API Gateway

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan handler."""
    global _runtime  # noqa: PLW0603

    settings = get_settings()
    logger.info("Starting runtime with settings: enable_cloud_providers=%s, ollama_host=%s",
                settings.provider.enable_cloud_providers, settings.provider.ollama_host)
    async with create_runtime(settings) as runtime:  # type: SidWeaveRuntime
        _runtime = runtime
        logger.info("SidWeave API gateway started")

        # Debug: Check provider gateway
        gateway = _runtime.get_component("provider_gateway")
        if gateway:
            logger.info("Provider gateway found, providers: %s", list(gateway.providers.keys()))
            for name, prov in gateway.providers.items():
                logger.info(
                    "  Provider %s: healthy=%s, models=%s",
                    name,
                    prov.is_healthy,
                    len(prov._models),
                )
        else:
            logger.warning("Provider gateway not found in runtime")

        yield

    if _runtime:
        await _runtime.stop()
    logger.info("SidWeave API gateway stopped")


app = FastAPI(
    title="SidWeave API",
    description="Adaptive runtime for AI agents - OpenAI-compatible API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    """List available models."""
    if not _runtime:
        raise HTTPException(503, "Runtime not initialized")

    model_registry = _runtime.get_component("model_registry")
    assert isinstance(model_registry, ModelRegistry)

    models = await model_registry.list_models()

    return ModelListResponse(
        data=[
            ModelInfo(
                id=m.name,
                owned_by=m.provider,
            )
            for m in models
        ]
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse | StreamingResponse:
    """OpenAI-compatible chat completions endpoint."""
    if not _runtime:
        raise HTTPException(503, "Runtime not initialized")

    provider_gateway = _runtime.get_component("provider_gateway")
    assert isinstance(provider_gateway, ProviderGateway)

    routing_engine = _runtime.get_component("routing_engine")
    assert isinstance(routing_engine, RoutingEngine)
    assert routing_engine is not None
    assert hasattr(routing_engine, "route")

    if not provider_gateway or not routing_engine:
        raise HTTPException(503, "Required components not available")

    # Convert messages
    messages = [
        ChatMessage(
            role=m.role,
            content=m.content or "",
            tool_calls=m.tool_calls,
            tool_call_id=m.tool_call_id,
        )
        for m in request.messages
    ]

    # Determine routing constraints
    constraints = RoutingConstraints(
        max_latency_ms=None,
        max_cost_per_1k_tokens=None,
        required_capabilities=[ModelCapability.CHAT],
        strategy=RoutingStrategy.BALANCED,
    )

    # If model specified, use it directly
    if request.model:
        # Check if model exists
        model_registry = _runtime.get_component("model_registry")
        assert isinstance(model_registry, ModelRegistry)
        model = await model_registry.get_model(request.model) if model_registry else None

        if model:
            # Use specified model
            provider_request = ChatCompletionRequest(
                model=request.model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=request.stream,
                tools=request.tools,
                tool_choice=request.tool_choice,
            )

            try:
                response = await provider_gateway.chat_completion(provider_request)
            except Exception as e:
                raise HTTPException(500, f"Provider error: {e}") from e
        else:
            raise HTTPException(404, f"Model not found: {request.model}")
    else:
        # Route to best model
        try:
            routing_result = await routing_engine.route(
                task_category="chat",
                required_capabilities=[ModelCapability.CHAT],
                constraints=constraints,
            )

            provider_request = ChatCompletionRequest(
                model=routing_result.selected_model.name,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=request.stream,
                tools=request.tools,
                tool_choice=request.tool_choice,
            )

            response = await provider_gateway.chat_completion(provider_request)
        except Exception as e:
            raise HTTPException(500, f"Routing/provider error: {e}") from e

    if request.stream:
        return StreamingResponse(
            _stream_response(response),
            media_type="text/event-stream",
        )

    # Convert to OpenAI format
    return ChatCompletionResponse(
        id=response.id,
        model=response.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(
                    role=choice["message"]["role"],
                    content=choice["message"].get("content"),
                    tool_calls=choice["message"].get("tool_calls"),
                ),
                finish_reason=choice.get("finish_reason"),
            ).model_dump()
            for choice in response.choices
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=response.usage.get("prompt_tokens", 0) if response.usage else 0,
            completion_tokens=response.usage.get("completion_tokens", 0) if response.usage else 0,
            total_tokens=response.usage.get("total_tokens", 0) if response.usage else 0,
        ).model_dump() if response.usage else None,
    )


async def _stream_response(response: ChatCompletionResponse) -> AsyncIterator[str]:
    """Convert provider response to streaming format."""
    # For now, simulate streaming by yielding the full response
    # Real implementation would stream from provider
    chunk = ChatCompletionChunk(
        id=response.id,
        model=response.model,
        choices=[
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": (
                        response.choices[0]
                        .get("message", {})
                        .get("content", "")
                        if response.choices
                        else ""
                    ),
                },
                "finish_reason": "stop",
            }
        ],
    )
    yield f"data: {chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    if not _runtime:
        return {"status": "starting"}

    status = _runtime.get_status()
    all_healthy = all(s.healthy for s in status.values())

    return {
        "status": "healthy" if all_healthy else "degraded",
        "components": {
            name: {
                "healthy": s.healthy,
                "message": s.message,
            }
            for name, s in status.items()
        },
    }


@app.get("/status")
async def get_status() -> dict[str, Any]:
    """Get detailed runtime status."""
    if not _runtime:
        raise HTTPException(503, "Runtime not initialized")

    return _runtime.get_status()


# Additional endpoints for SidWeave-specific features

@app.post("/v1/sidweave/route")
async def route_request(
    task_category: str,
    required_capabilities: list[str] | None = None,
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Route a request using SidWeave's adaptive routing."""
    if not _runtime:
        raise HTTPException(503, "Runtime not initialized")

    routing_engine = _runtime.get_component("routing_engine")
    assert isinstance(routing_engine, RoutingEngine)
    assert routing_engine is not None
    assert hasattr(routing_engine, "route")

    if not routing_engine:
        raise HTTPException(503, "Routing engine not available")

    caps = [ModelCapability(c) for c in (required_capabilities or [])]

    routing_constraints = RoutingConstraints(**(constraints or {}))

    result = await routing_engine.route(
        task_category=task_category,
        required_capabilities=caps,
        constraints=routing_constraints,
    )

    return {
        "selected_model": result.selected_model.name,
        "selected_provider": result.selected_provider.name,
        "fallback_models": [m.name for m in result.fallback_models],
        "rationale": result.rationale,
        "decision_time_ms": result.decision_time_ms,
    }


@app.get("/v1/sidweave/models/{model_name}/stats")
async def get_model_stats(model_name: str) -> dict[str, Any]:
    """Get model performance statistics."""
    if not _runtime:
        raise HTTPException(503, "Runtime not initialized")

    model_registry = _runtime.get_component("model_registry")
    assert isinstance(model_registry, ModelRegistry)
    assert model_registry is not None
    assert hasattr(model_registry, "get_model_stats")

    stats: dict[str, Any] = await model_registry.get_model_stats(model_name)
    return stats


@app.post("/v1/sidweave/models/{model_name}/profile")
async def profile_model(
    model_name: str, test_types: list[str] | None = None
) -> list[dict[str, Any]]:
    """Profile a model."""
    if not _runtime:
        raise HTTPException(503, "Runtime not initialized")

    model_registry = _runtime.get_component("model_registry")
    assert isinstance(model_registry, ModelRegistry)

    results = await model_registry.profile_model(model_name, test_types)
    return [r.model_dump() for r in results]


@app.get("/v1/sidweave/hardware")
async def get_hardware_profile() -> dict[str, Any]:
    """Get current hardware profile."""
    if not _runtime:
        raise HTTPException(503, "Runtime not initialized")

    profile = _runtime.get_component("hardware_profile")
    assert isinstance(profile, HardwareProfile)

    if not profile:
        raise HTTPException(503, "Hardware profile not available")

    return profile.to_dict()


@app.get("/v1/sidweave/providers")
async def list_providers() -> list[dict[str, Any]]:
    """List all providers and their status."""
    if not _runtime:
        raise HTTPException(503, "Runtime not initialized")

    provider_gateway = _runtime.get_component("provider_gateway")
    assert isinstance(provider_gateway, ProviderGateway)

    if not provider_gateway:
        raise HTTPException(503, "Provider gateway not available")

    providers = await provider_gateway.list_providers()
    return [p.__dict__ for p in providers]


@app.post("/v1/sidweave/providers/health")
async def check_providers() -> dict[str, bool]:
    """Check health of all providers."""
    if not _runtime:
        raise HTTPException(503, "Runtime not initialized")

    provider_gateway = _runtime.get_component("provider_gateway")
    assert isinstance(provider_gateway, ProviderGateway)

    if not provider_gateway:
        raise HTTPException(503, "Provider gateway not available")

    return await provider_gateway.health_check()


# Error handlers

@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "type": "http_error",
                "code": exc.status_code,
            }
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(
    _request: Request, exc: Exception
) -> JSONResponse:
    logger.error("Unhandled exception", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": str(exc),
                "type": "internal_error",
                "code": 500,
            }
        },
    )


def create_app(_settings: Settings | None = None) -> FastAPI:
    """Create the FastAPI application."""
    # The app is already created with lifespan
    return app
