"""
Model registry for tracking available models and their capabilities.
"""

from typing import Any

import litellm
import ollama
from pydantic import BaseModel

from sidweave.config import Settings
from sidweave.hardware import HardwareProfile
from sidweave.providers import (
    ChatCompletionRequest,
    ChatMessage,
    ModelCapability,
    ProviderGateway,
)
from sidweave.storage import (
    HardwareProfileRepository,
    MeasurementRepository,
    ModelMeasurement,
    ModelRepository,
    get_session,
)


class ProfilingResult(BaseModel):
    """Result of model profiling."""
    model_name: str
    test_type: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float | None = None
    throughput_tps: float | None = None
    memory_usage_mb: float | None = None
    gpu_memory_usage_mb: float | None = None
    quality_score: float | None = None
    success: bool = True
    error: str | None = None
    configuration: dict[str, Any] | None = None




class ModelInfo(BaseModel):
    """Model information."""
    name: str
    provider: str
    provider_model_id: str
    capabilities: list[ModelCapability] = []
    context_length: int = 4096
    max_output_tokens: int | None = None
    min_ram_gb: float | None = None
    min_vram_gb: float | None = None
    recommended_backend: str | None = None
    quantization: str | None = None
    parameter_count: str | None = None
    is_local: bool = False
    is_available: bool = True
    metadata: dict[str, Any] = {}

class ModelRegistry:
    """Registry for managing model information and discovery."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._initialized = False
        self._models: dict[str, ModelInfo] = {}

    async def initialize(self) -> None:
        """Initialize model registry."""
        self._initialized = True
        await self._discover_models()

    async def _discover_models(self) -> None:
        """Discover available models from all providers."""
        # Discover Ollama models
        await self._discover_ollama()

        # Discover LiteLLM models
        await self._discover_litellm()

        # Discover NVIDIA NIM models
        await self._discover_nvidia_nim()

        await self._persist_models()

    async def _discover_ollama(self) -> None:
        """Discover models from Ollama."""
        try:
            client = ollama.Client(host=self.settings.provider.ollama_host)
            response = client.list()
            models_data: list[Any] = []
            if hasattr(response, "models"):
                models_data = list(response.models)
            elif isinstance(response, dict):
                models_data = response.get("models", [])

            for m in models_data:
                name = getattr(m, "model", m.get("name", ""))
                if name:
                    capabilities = self._estimate_capabilities(name, {})
                    model_info = ModelInfo(
                        name=name,
                        provider="ollama",
                        provider_model_id=name,
                        capabilities=capabilities,
                        context_length=4096,
                        is_local=True,
                        is_available=True,
                    )
                    self._models[name] = model_info
        except Exception:
            pass

    async def _discover_litellm(self) -> None:
        """Discover models from LiteLLM."""
        try:
            models = litellm.get_valid_models()
            for model in models:
                if model not in self._models:
                    capabilities = self._estimate_capabilities(model, {})
                    model_info = ModelInfo(
                        name=model,
                        provider="litellm",
                        provider_model_id=model,
                        capabilities=capabilities,
                        context_length=self._estimate_context_length(model),
                        is_local=False,
                        is_available=True,
                    )
                    self._models[model] = model_info
        except Exception:
            pass

    async def _discover_nvidia_nim(self) -> None:
        """Discover models from NVIDIA NIM."""
        # Models listed by NVIDIA NIM /models endpoint but not actually available for chat
        unavailable_models = {
            "nvidia/cosmos-reason2-8b",
            "bigcode/starcoder2-15b",
            "nvidia/ising-calibration-1.5-31b",
            "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
            "nvidia/llama-3.2-nv-embedqa-1b-v1",
            "nvidia/nemotron-3-embed-1b",
            "nvidia/nemotron-3.5-embed-1b",
            "nvidia/nemotron-3.5-embed-8b",
            "nvidia/nv-embedqa-mistral-7b-v2",
            "nvidia/nvclip",
            "nvidia/riva-translate-4b-instruct-v2",
            "snowflake/arctic-embed-l",
        }
        try:
            self.settings.provider.enable_cloud_providers = True
            gateway = ProviderGateway(self.settings)
            await gateway.initialize()
            if "nvidia_nim" in gateway.providers:
                provider = gateway.providers["nvidia_nim"]
                if provider.is_healthy:
                    models = await provider.list_models()
                    for model_name in models:
                        if model_name in unavailable_models:
                            continue
                        if model_name not in self._models:
                            capabilities = self._estimate_capabilities(model_name, {})
                            model_info = ModelInfo(
                                name=model_name,
                                provider="nvidia_nim",
                                provider_model_id=model_name,
                                capabilities=capabilities,
                                context_length=128000,
                                is_local=False,
                                is_available=True,
                            )
                            self._models[model_name] = model_info
            await gateway.shutdown()
        except Exception:
            pass

    async def _persist_models(self) -> None:
        """Persist discovered models to database."""
        async with get_session() as session:
            repo = ModelRepository(session)
            for model in self._models.values():
                await repo.upsert(
                    name=model.name,
                    provider=model.provider,
                    provider_model_id=model.provider_model_id,
                    capabilities=[c.value for c in model.capabilities],
                    context_length=model.context_length,
                    max_output_tokens=model.max_output_tokens,
                    min_ram_gb=model.min_ram_gb,
                    min_vram_gb=model.min_vram_gb,
                    recommended_backend=model.recommended_backend,
                    quantization=model.quantization,
                    parameter_count=model.parameter_count,
                    is_local=model.is_local,
                    is_available=model.is_available,
                    metadata=model.metadata,
                )

    def _estimate_capabilities(self, name: str, _details: dict[str, Any]) -> list[ModelCapability]:
        """Estimate model capabilities from name and details."""
        capabilities = [ModelCapability.CHAT, ModelCapability.COMPLETION]
        name_lower = name.lower()

        # Tool use
        if any(x in name_lower for x in ["hermes", "function", "tool", "command"]):
            capabilities.append(ModelCapability.TOOLS)

        # Vision
        if any(x in name_lower for x in ["vision", "llava", "bakllava", "moondream"]):
            capabilities.append(ModelCapability.VISION)

        # Code generation
        code_keywords = [
            "code", "coder", "starcoder", "wizardcoder",
            "deepseek-coder", "codegen", "codellama",
            "poolside", "laguna", "qwen-coder", "phind-coder",
            "magicoder", "octocoder", "codegeex", "codegeex2",
        ]
        if any(x in name_lower for x in code_keywords):
            capabilities.append(ModelCapability.CODE_GENERATION)

        # Reasoning
        reasoning_keywords = ["reason", "o1", "deepseek-r1", "qwq"]
        if any(x in name_lower for x in reasoning_keywords):
            capabilities.append(ModelCapability.REASONING)

        # Code
        code_keywords = ["code", "coder", "starcoder", "wizardcoder", "deepseek-coder"]
        if any(x in name_lower for x in code_keywords):
            capabilities.append(ModelCapability.CODE)

        # Embedding
        embed_keywords = ["embed", "bge", "e5", "nomic-embed"]
        if any(x in name_lower for x in embed_keywords):
            capabilities.append(ModelCapability.EMBEDDING)

        # Vision (duplicate check)
        if any(x in name_lower for x in ["vision", "llava", "bakllava", "moondream"]):
            capabilities.append(ModelCapability.VISION)

        # Audio
        if any(x in name_lower for x in ["audio", "whisper", "speech"]):
            capabilities.append(ModelCapability.AUDIO)

        return capabilities

    def _estimate_capabilities_from_litellm(
        self, model_info: dict[str, Any]
    ) -> list[ModelCapability]:
        """Estimate capabilities from LiteLLM model info."""
        capabilities = [ModelCapability.CHAT, ModelCapability.COMPLETION]
        if model_info.get("supports_function_calling", False):
            capabilities.append(ModelCapability.TOOLS)
        if model_info.get("supports_vision", False):
            capabilities.append(ModelCapability.VISION)
        if "embedding" in model_info.get("model_name", "").lower():
            capabilities.append(ModelCapability.EMBEDDING)
        return capabilities

    def _estimate_context_length(self, name: str) -> int:
        """Estimate context length from model details."""
        name_lower = name.lower()

        if "llama3.1" in name_lower or "llama-3.1" in name_lower:
            return 131072
        if "llama3" in name_lower or "llama-3" in name_lower:
            return 8192
        if "mistral" in name_lower or "qwen" in name_lower:
            return 32768
        if "gemma" in name_lower:
            return 8192
        if any(
            x in name_lower
            for x in ["llama2", "llama-2", "phi3", "phi-3", "command-r"]
        ):
            return 4096
        return 4096

    def _estimate_resource_requirements(
        self, size_bytes: int, _param_count: str, quant: str
    ) -> tuple[float | None, float | None]:
        """Estimate RAM and VRAM requirements."""
        quant_lower = quant.lower()
        size_gb = size_bytes / (1024**3)

        # RAM: model size + overhead (2-4GB)
        min_ram = size_gb + 4

        min_vram = size_gb + 2

        # Adjust for quantization
        if "q4" in quant_lower:
            min_ram *= 0.8
            min_vram *= 0.8

        min_ram *= 1.1
        min_vram *= 1.1

        if "q2" in quant_lower or "q3" in quant_lower:
            min_vram *= 0.6

        return round(min_ram, 1), round(min_vram, 1)

    async def list_models(
        self,
        provider: str | None = None,
        capability: ModelCapability | None = None,
        is_local: bool | None = None,
        available_only: bool = True,
        limit: int = 100,
    ) -> list[ModelInfo]:
        """List models in registry."""
        async with get_session() as session:
            repo = ModelRepository(session)

            _capability_str = capability.value if capability else None
            models = await repo.list_models(
                provider=provider,
                capability=_capability_str,
                is_local=is_local,
                available_only=available_only,
                limit=limit,
            )

        return [
            ModelInfo(
                name=m.name,
                provider=m.provider,
                provider_model_id=m.provider_model_id,
                capabilities=[ModelCapability(c) for c in m.capabilities],
                context_length=m.context_length,
                max_output_tokens=m.max_output_tokens,
                min_ram_gb=m.min_ram_gb,
                min_vram_gb=m.min_vram_gb,
                recommended_backend=m.recommended_backend,
                quantization=m.quantization,
                parameter_count=m.parameter_count,
                is_local=m.is_local,
                is_available=m.is_available,
                metadata=m.metadata if isinstance(m.metadata, dict) else {},
            )
            for m in models
        ]

    async def get_model(self, name: str) -> ModelInfo | None:
        async with get_session() as session:
            repo = ModelRepository(session)
            model = await repo.get_by_name(name)

            if not model:
                return None

            return ModelInfo(
                name=model.name,
                provider=model.provider,
                provider_model_id=model.provider_model_id,
                capabilities=[ModelCapability(c) for c in model.capabilities],
                context_length=model.context_length,
                max_output_tokens=model.max_output_tokens,
                min_ram_gb=model.min_ram_gb,
                min_vram_gb=model.min_vram_gb,
                recommended_backend=model.recommended_backend,
                quantization=model.quantization,
                parameter_count=model.parameter_count,
                is_local=model.is_local,
                is_available=model.is_available,
                metadata=model.metadata if isinstance(model.metadata, dict) else {},
            )

    async def register_model(self, model: ModelInfo) -> None:
        """Register a new model."""
        async with get_session() as session:
            repo = ModelRepository(session)
            await repo.get_or_create(
                name=model.name,
                provider=model.provider,
                provider_model_id=model.provider_model_id,
                capabilities=[c.value for c in model.capabilities],
                context_length=model.context_length,
                max_output_tokens=model.max_output_tokens,
                min_ram_gb=model.min_ram_gb,
                min_vram_gb=model.min_vram_gb,
                recommended_backend=model.recommended_backend,
                quantization=model.quantization,
                parameter_count=model.parameter_count,
                is_local=model.is_local,
                is_available=model.is_available,
                metadata=model.metadata,
            )
        self._models[model.name] = model

    async def profile_model(
        self,
        model_name: str,
        test_types: list[str] | None = None,
        hardware_profile: HardwareProfile | None = None,
    ) -> list[ProfilingResult]:
        """Profile a model with various tests.

        Args:
            model_name: Name of model to profile
            test_types: Types of tests to run (latency, throughput, memory)
            hardware_profile: Hardware profile to associate with measurements

        Returns:
            List of profiling results
        """

        test_types = test_types or ["latency", "throughput"]

        model = await self.get_model(model_name)
        if not model:
            # Model not in local database - create a temporary ModelInfo for profiling
            model = ModelInfo(
                name=model_name,
                provider="unknown",
                provider_model_id=model_name,
                capabilities=[ModelCapability.CHAT, ModelCapability.COMPLETION],
                is_local=False,
                is_available=True,
            )

        results = []
        for test_type in test_types:
            if test_type == "latency":
                result = await self._profile_latency(model)
            elif test_type == "throughput":
                result = await self._profile_throughput(model)
            elif test_type == "memory":
                result = await self._profile_memory(model)
            else:
                continue

            # Save measurement only if model exists in database
            if hasattr(model, "id") and model.id:
                await self._save_measurement(model_name, result, hardware_profile)
            results.append(result)

        return results

    async def _profile_latency(self, model: ModelInfo) -> ProfilingResult:
        test_prompts = [
            "Hello, how are you?",
            "Write a short poem about coding.",
        ]

        latencies = []
        for prompt in test_prompts:
            try:
                gateway = ProviderGateway(self.settings)
                await gateway.initialize()
                provider = gateway.providers.get(model.provider)
                if provider:
                    request = ChatCompletionRequest(
                        model=model.provider_model_id,
                        messages=[ChatMessage(role="user", content=prompt)],
                        max_tokens=50,
                    )
                    response = await provider.chat_completion(request)
                    elapsed = response.latency_ms / 1000.0
                    latencies.append(elapsed)
                await gateway.shutdown()
            except Exception as e:
                # Return a failed result but don't crash
                return ProfilingResult(
                    model_name=model.name,
                    test_type="latency",
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=0,
                    success=False,
                    error=f"Provider error: {e}",
                )

        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        if avg_latency == 0:
            return ProfilingResult(
                model_name=model.name,
                test_type="latency",
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=0,
                success=False,
                error="No successful latency measurements",
            )

        return ProfilingResult(
            model_name=model.name,
            test_type="latency",
            prompt_tokens=0,
            completion_tokens=50,
            latency_ms=avg_latency * 1000,
            success=True,
        )

    async def _profile_throughput(self, model: ModelInfo) -> ProfilingResult:
        """Profile model throughput (tokens per second)."""
        prompt = "Write a detailed explanation of machine learning. " * 10  # ~200 tokens

        try:
            gateway = ProviderGateway(self.settings)
            await gateway.initialize()
            provider = gateway.providers.get(model.provider)
            if provider:
                request = ChatCompletionRequest(
                    model=model.provider_model_id,
                    messages=[ChatMessage(role="user", content=prompt)],
                    max_tokens=500,
                    temperature=0.7,
                )
                response = await provider.chat_completion(request)
                elapsed = response.latency_ms / 1000.0
                tokens = response.usage.get("completion_tokens", 0) if response.usage else 0
                throughput = tokens / elapsed if elapsed > 0 else 0
            else:
                return ProfilingResult(
                    model_name=model.name,
                    test_type="throughput",
                    prompt_tokens=0,
                    completion_tokens=0,
                    success=False,
                    error="Provider not available",
                )
            await gateway.shutdown()
        except Exception as e:
            return ProfilingResult(
                model_name=model.name,
                test_type="throughput",
                prompt_tokens=0,
                completion_tokens=0,
                success=False,
                error=str(e),
            )

        return ProfilingResult(
            model_name=model.name,
            test_type="throughput",
            prompt_tokens=len(prompt.split()),
            completion_tokens=tokens,
            latency_ms=elapsed * 1000,
            throughput_tps=throughput,
            success=True,
        )

    async def _profile_memory(self, model: ModelInfo) -> ProfilingResult:
        """Profile memory usage (placeholder)."""
        return ProfilingResult(
            model_name=model.name,
            test_type="memory",
            prompt_tokens=0,
            completion_tokens=0,
            success=True,
            configuration={"note": "Memory profiling not fully implemented"},
        )

    async def _save_measurement(
        self,
        model_name: str,
        result: ProfilingResult,
        hardware_profile: HardwareProfile | None = None,
    ) -> None:
        """Save profiling measurement to database."""
        async with get_session() as session:
            meas_repo = MeasurementRepository(session)

            hardware_profile_id = None
            if hardware_profile:
                hw_repo = HardwareProfileRepository(session)
                hw_profile = await hw_repo.add_hardware_profile(hardware_profile)
                hardware_profile_id = hw_profile.id

            measurement = ModelMeasurement(
                model_name=model_name,
                test_type=result.test_type,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                latency_ms=result.latency_ms,
                throughput_tps=result.throughput_tps,
                memory_usage_mb=result.memory_usage_mb,
                gpu_memory_usage_mb=result.gpu_memory_usage_mb,
                quality_score=result.quality_score,
                success=result.success,
                error=result.error,
                hardware_profile_id=hardware_profile_id,
            )
            await meas_repo.add_measurement(measurement)

    async def get_model_stats(self, model_name: str) -> dict[str, Any]:
        """Get aggregated statistics for a model."""
        async with get_session() as session:
            meas_repo = MeasurementRepository(session)
            measurements = await meas_repo.get_by_model(model_name)

            if not measurements:
                return {"error": "No measurements found"}

            successful = [m for m in measurements if m.success]
            _latencies = [m.latency_ms for m in successful if m.latency_ms]
            throughputs = [m.throughput_tps for m in successful if m.throughput_tps]

            return {
                "model_name": model_name,
                "total_measurements": len(measurements),
                "successful_measurements": len(successful),
                "avg_latency_ms": (
                    sum(m.latency_ms for m in successful if m.latency_ms)
                    / len(successful)
                    if successful else 0
                ),
                "avg_throughput_tps": (
                    sum(t for t in throughputs) / len(throughputs)
                    if throughputs else 0
                ),
                "success_rate": len(successful) / len(measurements),
                "measurement_count": len(measurements),
            }


__all__ = [
    "ModelCapability",
    "ModelInfo",
    "ModelRegistry",
    "ProfilingResult",
]
