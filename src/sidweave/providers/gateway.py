"""
Provider gateway for unified model access.
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import httpx
import litellm
import ollama

from sidweave.config import Settings

logger = logging.getLogger(__name__)


class ProviderType(StrEnum):
    """Supported provider types."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    NVIDIA_NIM = "nvidia_nim"
    OPENROUTER = "openrouter"
    LITELLM = "litellm"  # Generic LiteLLM fallback



class ModelCapability(StrEnum):
    """Supported model capabilities."""
    CHAT = "chat"
    COMPLETION = "completion"
    TOOLS = "tools"
    VISION = "vision"
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"
    CODE = "code"
    EMBEDDING = "embedding"
    AUDIO = "audio"


@dataclass
class Provider:
    """Provider information."""
    name: str
    type: ProviderType
    base_url: str | None = None
    models: list[str] = field(default_factory=list)
    healthy: bool = False
    last_check: float = 0
    latency_ms: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatMessage:
    """Chat message."""
    role: str
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


@dataclass
class ChatCompletionRequest:
    """Chat completion request."""
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatCompletionResponse:
    """Chat completion response."""
    id: str
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, Any] | None = None
    created: int = field(default_factory=lambda: int(time.time()))
    provider: str = ""
    latency_ms: float = 0


class BaseProvider(ABC):
    """Abstract base class for model providers."""

    def __init__(
        self, settings: Settings, name: str, provider_type: ProviderType
    ):
        self.settings = settings
        self.name = name
        self.type = provider_type
        self._healthy = False
        self._last_check: float = 0.0
        self._latency_ms: float | None = None
        self._error: str | None = None
        self._models: list[str] = []

    @property
    def is_healthy(self) -> bool:
        return self._healthy

    @property
    def latency_ms(self) -> float | None:
        return self._latency_ms

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def models(self) -> list[str]:
        return self._models

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider."""

    @abstractmethod
    async def list_models(self) -> list[str]:
        """List available models."""

    @abstractmethod
    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Execute chat completion."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Check provider health."""

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown provider."""

    def to_provider_info(self) -> Provider:
        """Convert to Provider info."""
        return Provider(
            name=self.name,
            type=self.type,
            base_url=self.base_url if hasattr(self, "base_url") else None,
            models=self._models,
            healthy=self._healthy,
            last_check=self._last_check,
            latency_ms=self._latency_ms,
            error=self._error,
        )


class OllamaProvider(BaseProvider):
    """Ollama local model provider."""

    def __init__(
        self, settings: Settings
    ):
        super().__init__(settings, "ollama", ProviderType.OLLAMA)
        self.client: ollama.AsyncClient | None = None
        self._models: list[str] = []

    async def initialize(self) -> None:
        """Initialize Ollama client."""
        self.client = ollama.AsyncClient(
            host=self.settings.provider.ollama_host,
            timeout=10.0,  # 10 second timeout for all requests
        )
        await self.health_check()
        if self._healthy:
            models = await self.list_models()
            self._models = models

    async def list_models(self) -> list[str]:
        """List available Ollama models."""
        if not self.client:
            return []
        try:
            response = await self.client.list()
            if hasattr(response, "models"):
                models_list = response.models
            elif isinstance(response, dict):
                models_list = response.get("models", [])
            else:
                models_list = []
            models = []
            for m in models_list:
                if hasattr(m, "model"):
                    name = m.model
                elif isinstance(m, dict):
                    name = m.get("name", "")
                else:
                    name = ""
                if name:
                    models.append(name)
            self._models = models
        except Exception as e:
            logger.warning("Failed to list Ollama models: %s", e)
            return []
        else:
            return self._models


    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Execute chat completion via Ollama."""
        if not self.client:
            raise RuntimeError("Ollama client not initialized")  # noqa: TRY003

        start = time.perf_counter()

        # Convert messages to Ollama format
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        options = {
            "temperature": request.temperature,
        }
        if request.max_tokens:
            options["num_predict"] = request.max_tokens

        try:
            if request.stream:
                # For streaming, we'd need a different approach
                # For now, fall back to non-streaming
                pass

            response = await self.client.chat(
                model=request.model,
                messages=messages,
                options=options,
            )

            elapsed = (time.perf_counter() - start) * 1000

            content = response.get("message", {}).get("content", "")

            return ChatCompletionResponse(
                id=f"ollama-{int(time.time() * 1000)}",
                model=request.model,
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }],
                usage={
                    "prompt_tokens": response.get("prompt_eval_count", 0),
                    "completion_tokens": response.get("eval_count", 0),
                    "total_tokens": (
                        response.get("prompt_eval_count", 0)
                        + response.get("eval_count", 0)
                    ),
                },
                provider=self.name,
                latency_ms=elapsed,
            )
        except Exception as e:
            raise RuntimeError(f"Ollama completion failed: {e}") from e  # noqa: TRY003

    async def health_check(self) -> bool:
        """Check Ollama health."""
        if not self.client:
            self._healthy = False
            self._error = "Client not initialized"
            return False

        start = time.perf_counter()
        try:
            # Simple health check - list models
            await self.client.list()
            self._latency_ms = (time.perf_counter() - start) * 1000
            self._healthy = True
            self._error = None
            self._last_check = time.time()
        except Exception as e:
            self._latency_ms = (time.perf_counter() - start) * 1000
            self._healthy = False
            self._error = str(e)
            self._last_check = time.time()
            return False
        else:
            return True

    async def shutdown(self) -> None:
        """Shutdown Ollama client."""
        self.client = None


class LiteLLMProvider(BaseProvider):
    """Generic LiteLLM provider for OpenAI-compatible APIs."""

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        settings: Settings,
        name: str,
        provider_type: ProviderType,
        base_url: str | None = None,
        api_key: str | None = None,
        model_prefix: str = "",
    ):
        super().__init__(settings, name, provider_type)
        self.base_url = base_url
        self.api_key = api_key
        self.model_prefix = model_prefix
        self._models: list[str] = []
        self._client: httpx.AsyncClient | None = None

    async def initialize(self) -> None:
        """Initialize HTTP client."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self._client = httpx.AsyncClient(
            base_url=self.base_url or "",
            headers=headers,
            timeout=30.0,
        )
        await self.health_check()
        if self._healthy:
            await self.list_models()

    async def list_models(self) -> list[str]:
        """List available models via LiteLLM or direct API."""
        # Try LiteLLM first
        try:
            models = litellm.get_valid_models()
            # Filter by provider type if possible
            filtered = [m for m in models if self._matches_provider(m)]
            if filtered:
                self._models = filtered
                return filtered
        except Exception:
            pass

        # Fallback: try direct API call
        if self._client and self.base_url:
            try:
                response = await self._client.get("/models")
                HTTP_OK = 200
                if response.status_code == HTTP_OK:
                    data = response.json()
                    models = [m.get("id", "") for m in data.get("data", []) if m.get("id")]
                    self._models = models
                    return models
            except Exception:
                pass

        return self._models

    def _matches_provider(self, model: str) -> bool:
        """Check if model matches this provider."""
        model_lower = model.lower()
        if self.type == ProviderType.OPENAI:
            openai_prefixes = (
                "gpt-", "o1-", "text-", "davinci",
                "curie", "babbage", "ada"
            )
            return model_lower.startswith(openai_prefixes)
        if self.type == ProviderType.ANTHROPIC:
            return "claude" in model_lower
        if self.type == ProviderType.NVIDIA_NIM:
            return "nvidia" in model_lower or "nemotron" in model_lower
        if self.type == ProviderType.OPENROUTER:
            return "/" in model  # OpenRouter models are org/model format
        return True

    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Execute chat completion via LiteLLM."""
        start = time.perf_counter()

        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        # Prepare LiteLLM parameters
        kwargs = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": request.stream,
            "tools": request.tools,
            "tool_choice": request.tool_choice,
            **request.extra,
        }

        # Add provider-specific parameters
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.api_key:
            kwargs["api_key"] = self.api_key

        try:
            response = await litellm.acompletion(**kwargs)

            elapsed = (time.perf_counter() - start) * 1000

            choice = response.choices[0]
            usage = response.usage

            return ChatCompletionResponse(
                id=response.id,
                model=response.model,
                choices=[{
                    "index": 0,
                    "message": {
                        "role": choice.message.role,
                        "content": choice.message.content,
                        "tool_calls": choice.message.tool_calls,
                    },
                    "finish_reason": choice.finish_reason,
                }],
                usage={
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "total_tokens": usage.total_tokens if usage else 0,
                },
                provider=self.name,
                latency_ms=elapsed,
            )
        except Exception as e:
            raise RuntimeError(f"{self.name} completion failed: {e}") from e  # noqa: TRY003

    async def health_check(self) -> bool:
        """Check provider health."""
        if not self._client:
            self._healthy = False
            self._error = "Client not initialized"
            return False

        start = time.perf_counter()
        try:
            # Try a simple models list call
            await self.list_models()
            self._latency_ms = (time.perf_counter() - start) * 1000
            self._healthy = True
            self._error = None
            self._last_check = time.time()
        except Exception as e:
            self._latency_ms = (time.perf_counter() - start) * 1000
            self._healthy = False
            self._error = str(e)
            self._last_check = time.time()
            return False
        else:
            return True

    async def shutdown(self) -> None:
        """Shutdown HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class ProviderGateway:
    """
    Unified gateway for all model providers.
    Manages provider lifecycle, health checks, and request routing with fallback.
    """

    def __init__(
        self, settings: Settings
    ):
        self.settings = settings
        self.providers: dict[str, BaseProvider] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all configured providers."""
        if self._initialized:
            return

        # Always add Ollama (local-first) - only if healthy
        ollama_provider = OllamaProvider(self.settings)
        await ollama_provider.initialize()
        if ollama_provider.is_healthy:
            self.providers["ollama"] = ollama_provider
        else:
            logger.warning("Ollama provider not healthy, skipping: %s", ollama_provider._error)

        # Add cloud providers if enabled
        if self.settings.provider.enable_cloud_providers:
            await self._initialize_cloud_providers()

        self._initialized = True
        logger.info(f"Provider gateway initialized with {len(self.providers)} providers")

    async def _initialize_cloud_providers(self) -> None:
        """Initialize cloud providers from configuration."""
        # Only initialize providers that have API keys configured

        # OpenAI via LiteLLM
        openai_key = self._get_env("OPENAI_API_KEY")
        if openai_key:
            openai_provider = LiteLLMProvider(
                self.settings,
                "openai",
                ProviderType.OPENAI,
                base_url="https://api.openai.com/v1",
                api_key=openai_key,
            )
            await openai_provider.initialize()
            if openai_provider.is_healthy:
                self.providers["openai"] = openai_provider

        # Anthropic via LiteLLM
        anthropic_key = self._get_env("ANTHROPIC_API_KEY")
        if anthropic_key:
            anthropic_provider = LiteLLMProvider(
                self.settings,
                "anthropic",
                ProviderType.ANTHROPIC,
                base_url="https://api.anthropic.com/v1",
                api_key=anthropic_key,
            )
            await anthropic_provider.initialize()
            if anthropic_provider.is_healthy:
                self.providers["anthropic"] = anthropic_provider

        # NVIDIA NIM
        nim_key = self._get_env("NVIDIA_API_KEY")
        if nim_key:
            nim_provider = LiteLLMProvider(
                self.settings,
                "nvidia_nim",
                ProviderType.NVIDIA_NIM,
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=nim_key,
            )
            await nim_provider.initialize()
            if nim_provider.is_healthy:
                self.providers["nvidia_nim"] = nim_provider

        # OpenRouter
        openrouter_key = self._get_env("OPENROUTER_API_KEY")
        if openrouter_key:
            openrouter_provider = LiteLLMProvider(
                self.settings,
                "openrouter",
                ProviderType.OPENROUTER,
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key,
            )
            await openrouter_provider.initialize()
            if openrouter_provider.is_healthy:
                self.providers["openrouter"] = openrouter_provider

        # Generic LiteLLM fallback for any other configured providers
        if self.settings.provider.litellm_base_url:
            master_key = (
                self.settings.provider.litellm_master_key.get_secret_value()
                if self.settings.provider.litellm_master_key
                else None
            )
            if master_key:
                litellm_provider = LiteLLMProvider(
                    self.settings,
                    "litellm",
                    ProviderType.LITELLM,
                    base_url=self.settings.provider.litellm_base_url,
                    api_key=master_key,
                )
                await litellm_provider.initialize()
                if litellm_provider.is_healthy:
                    self.providers["litellm"] = litellm_provider

    def _get_env(self, key: str) -> str | None:
        """Get environment variable."""
        return os.getenv(key)

    async def list_providers(self) -> list[Provider]:
        """List all providers with their status."""
        return [p.to_provider_info() for p in self.providers.values()]

    async def get_provider(self, name: str) -> BaseProvider | None:
        """Get provider by name."""
        return self.providers.get(name)

    async def list_all_models(self) -> dict[str, list[str]]:
        """List models from all providers."""
        result = {}
        for name, provider in self.providers.items():
            if provider.is_healthy:
                result[name] = await provider.list_models()
            else:
                result[name] = []
        return result

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        preferred_provider: str | None = None,
    ) -> ChatCompletionResponse:
        """
        Execute chat completion with automatic fallback.
        Args:
            request: Chat completion request
            preferred_provider: Optional preferred provider name
        Returns:
            Chat completion response
        Raises:
            RuntimeError: If all providers fail
        """
        # Determine provider order
        provider_order = self._get_provider_order(preferred_provider)

        last_error = None
        for provider_name in provider_order:
            provider = self.providers.get(provider_name)
            if not provider or not provider.is_healthy:
                continue

            try:
                logger.debug(f"Trying provider: {provider_name} for model: {request.model}")
                response = await provider.chat_completion(request)
                logger.info(
                    f"Completion successful via {provider_name} "
                    f"({response.latency_ms:.1f}ms)"
                )
            except Exception as e:
                last_error = e
                logger.warning(f"Provider {provider_name} failed: {e}")
                # Check if we should fallback
                if not self.settings.provider.fallback_to_cloud and provider_name != "ollama":
                    break
                continue
            else:
                return response

        if last_error is None:
            raise RuntimeError("No healthy providers available")  # noqa: TRY003
        raise RuntimeError(f"All providers failed. Last error: {last_error}")  # noqa: TRY003

    def _get_provider_order(self, preferred: str | None) -> list[str]:
        """Get provider order for fallback."""
        if self.settings.provider.prefer_local:
            # Local first, then cloud
            order = ["ollama"]
            order.extend([p for p in self.providers if p != "ollama"])
        else:
            order = list(self.providers.keys())

        # Move preferred to front
        if preferred and preferred in order:
            order.remove(preferred)
            order.insert(0, preferred)

        return order

    async def health_check(self) -> dict[str, bool]:
        """Check health of all providers."""
        results = {}
        for name, provider in self.providers.items():
            results[name] = await provider.health_check()
        return results

    async def shutdown(self) -> None:
        """Shutdown all providers."""
        for provider in self.providers.values():
            try:
                await provider.shutdown()
            except Exception:
                logger.exception("Error shutting down provider %s", provider.name)
        self.providers.clear()
        self._initialized = False
