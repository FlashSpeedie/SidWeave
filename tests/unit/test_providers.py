"""
Unit tests for provider gateway.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sidweave.config import Settings
from sidweave.providers import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    LiteLLMProvider,
    OllamaProvider,
    Provider,
    ProviderGateway,
    ProviderType,
)


class TestProviderModels:
    """Test provider data models."""

    def test_provider(self):
        p = Provider(name="ollama", type=ProviderType.OLLAMA)
        assert p.name == "ollama"
        assert p.type == ProviderType.OLLAMA
        assert p.healthy is False

    def test_chat_message(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_chat_completion_request(self):
        req = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            temperature=0.7,
        )
        assert req.model == "test-model"
        assert req.temperature == 0.7

    def test_chat_completion_response(self):
        resp = ChatCompletionResponse(
            id="resp-123",
            model="test-model",
            choices=[{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        )
        assert resp.id == "resp-123"
        assert resp.choices[0]["message"]["content"] == "Hi"


class TestOllamaProvider:
    """Test OllamaProvider."""

    @pytest.fixture
    def settings(self):
        return Settings()

    @pytest.fixture
    def provider(self, settings):
        return OllamaProvider(settings)

    @pytest.mark.asyncio
    async def test_init(self, provider):
        assert provider.name == "ollama"
        assert provider.type == ProviderType.OLLAMA

    @patch("sidweave.providers.gateway.ollama.AsyncClient")
    @pytest.mark.asyncio
    async def test_initialize(self, mock_client_class, provider):
        mock_client = AsyncMock()
        mock_client.list.return_value = {"models": [{"name": "llama3.1:8b"}]}
        mock_client_class.return_value = mock_client

        await provider.initialize()

        assert provider._healthy is True
        assert provider._models == ["llama3.1:8b"]

    @patch("sidweave.providers.gateway.ollama.AsyncClient")
    @pytest.mark.asyncio
    async def test_list_models(self, mock_client_class, provider):
        mock_client = AsyncMock()
        mock_client.list.return_value = {"models": [{"name": "model1"}, {"name": "model2"}]}
        mock_client_class.return_value = mock_client

        await provider.initialize()
        models = await provider.list_models()

        assert models == ["model1", "model2"]

    @patch("sidweave.providers.gateway.ollama.AsyncClient")
    @pytest.mark.asyncio
    async def test_chat_completion(self, mock_client_class, provider):
        mock_client = AsyncMock()
        mock_client.list = AsyncMock(return_value={"models": []})
        mock_client.chat = AsyncMock(return_value={
            "message": {"content": "Hello there!"},
            "prompt_eval_count": 10,
            "eval_count": 20,
        })
        mock_client_class.return_value = mock_client

        await provider.initialize()

        request = ChatCompletionRequest(
            model="llama3.1:8b",
            messages=[ChatMessage(role="user", content="Hello")],
            temperature=0.7,
            max_tokens=100,
        )

        response = await provider.chat_completion(request)

        assert response.model == "llama3.1:8b"
        assert response.choices[0]["message"]["content"] == "Hello there!"
        assert response.provider == "ollama"
        assert response.latency_ms > 0

    @patch("sidweave.providers.gateway.ollama.AsyncClient")
    @pytest.mark.asyncio
    async def test_health_check(self, mock_client_class, provider):
        mock_client = AsyncMock()
        mock_client.list.return_value = {"models": []}
        mock_client_class.return_value = mock_client

        await provider.initialize()
        result = await provider.health_check()

        assert result is True
        assert provider._healthy is True
        assert provider._latency_ms is not None


class TestLiteLLMProvider:
    """Test LiteLLMProvider."""

    @pytest.fixture
    def settings(self):
        return Settings()

    @pytest.fixture
    def provider(self, settings):
        return LiteLLMProvider(
            settings,
            "openai",
            ProviderType.OPENAI,
            base_url="https://api.openai.com/v1",
            api_key="test-key",
        )

    @pytest.mark.asyncio
    async def test_init(self, provider):
        assert provider.name == "openai"
        assert provider.type == ProviderType.OPENAI

    def test_matches_provider(self, provider):
        assert provider._matches_provider("gpt-4o") is True
        assert provider._matches_provider("gpt-3.5-turbo") is True
        assert provider._matches_provider("claude-3") is False

    @pytest.mark.asyncio
    async def test_initialize(self, provider):
        with patch("sidweave.providers.gateway.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            with patch("sidweave.providers.gateway.litellm.get_valid_models", return_value=["gpt-4o"]):
                await provider.initialize()

        assert provider._healthy is True

    @pytest.mark.asyncio
    async def test_chat_completion(self, provider):
        with patch("sidweave.providers.gateway.litellm.acompletion") as mock_acompletion:
            mock_response = MagicMock()
            mock_response.id = "resp-123"
            mock_response.model = "gpt-4o"
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.role = "assistant"
            mock_response.choices[0].message.content = "Hello!"
            mock_response.choices[0].finish_reason = "stop"
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 20
            mock_response.usage.total_tokens = 30
            mock_acompletion.return_value = mock_response

            request = ChatCompletionRequest(
                model="gpt-4o",
                messages=[ChatMessage(role="user", content="Hello")],
            )

            response = await provider.chat_completion(request)

            assert response.model == "gpt-4o"
            assert response.choices[0]["message"]["content"] == "Hello!"
            assert response.provider == "openai"


class TestProviderGateway:
    """Test ProviderGateway."""

    @pytest.fixture
    def settings(self):
        return Settings()

    @pytest.fixture
    def gateway(self, settings):
        return ProviderGateway(settings)

    @patch("sidweave.providers.gateway.OllamaProvider")
    @pytest.mark.asyncio
    async def test_initialize(self, mock_ollama_class, gateway, settings, monkeypatch):
        mock_ollama = AsyncMock()
        mock_ollama.initialize = AsyncMock()
        mock_ollama.is_healthy = True
        mock_ollama.name = "ollama"
        mock_ollama_class.return_value = mock_ollama

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        settings.provider.enable_cloud_providers = True
        settings.provider.litellm_master_key = "test-key"

        await gateway.initialize()

        assert "ollama" in gateway.providers
        assert "openai" in gateway.providers

    @patch("sidweave.providers.gateway.OllamaProvider")
    @pytest.mark.asyncio
    async def test_list_providers(self, mock_ollama_class, gateway, settings):
        mock_ollama = MagicMock()
        mock_ollama.initialize = AsyncMock()
        mock_ollama.is_healthy = True
        mock_ollama.name = "ollama"
        mock_ollama.to_provider_info.return_value = Provider(
            name="ollama",
            type="ollama",
            healthy=True,
        )
        mock_ollama_class.return_value = mock_ollama

        await gateway.initialize()
        providers = await gateway.list_providers()

        assert len(providers) >= 1
        assert providers[0].name == "ollama"

    @patch("sidweave.providers.gateway.OllamaProvider")
    @patch("sidweave.providers.gateway.LiteLLMProvider")
    @pytest.mark.asyncio
    async def test_chat_completion_with_fallback(self, mock_litellm_class, mock_ollama_class, gateway, settings, monkeypatch):
        """Test fallback when primary provider fails."""
        mock_ollama = MagicMock()
        mock_ollama.initialize = AsyncMock()
        mock_ollama.is_healthy = True
        mock_ollama.name = "ollama"
        mock_ollama.chat_completion = AsyncMock(side_effect=Exception("Ollama down"))
        mock_ollama_class.return_value = mock_ollama

        mock_openai = MagicMock()
        mock_openai.initialize = AsyncMock()
        mock_openai.is_healthy = True
        mock_openai.name = "openai"
        mock_openai.chat_completion = AsyncMock(return_value=ChatCompletionResponse(
            id="resp-1",
            model="gpt-4o",
            choices=[{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
            provider="openai",
            latency_ms=100.0,
        ))
        mock_litellm_class.return_value = mock_openai

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        settings.provider.enable_cloud_providers = True
        await gateway.initialize()

        request = ChatCompletionRequest(
            model="auto",
            messages=[ChatMessage(role="user", content="Hello")],
        )

        response = await gateway.chat_completion(request)

        assert response.provider == "openai"
        assert response.choices[0]["message"]["content"] == "Hi"

    @patch("sidweave.providers.gateway.OllamaProvider")
    @pytest.mark.asyncio
    async def test_chat_completion_preferred_provider(self, mock_ollama_class, gateway):
        """Test using preferred provider."""
        mock_ollama = MagicMock()
        mock_ollama.initialize = AsyncMock()
        mock_ollama.is_healthy = True
        mock_ollama.name = "ollama"
        mock_ollama.chat_completion = AsyncMock(return_value=ChatCompletionResponse(
            id="resp-1",
            model="llama3.1:8b",
            choices=[{"index": 0, "message": {"role": "assistant", "content": "Ollama response"}, "finish_reason": "stop"}],
            provider="ollama",
            latency_ms=50.0,
        ))
        mock_ollama_class.return_value = mock_ollama

        await gateway.initialize()

        request = ChatCompletionRequest(
            model="llama3.1:8b",
            messages=[ChatMessage(role="user", content="Hello")],
        )

        response = await gateway.chat_completion(request, preferred_provider="ollama")

        assert response.provider == "ollama"
        assert response.choices[0]["message"]["content"] == "Ollama response"

    @pytest.mark.asyncio
    async def test_get_provider(self, gateway):
        with patch("sidweave.providers.gateway.OllamaProvider") as mock_ollama_class:
            mock_ollama = AsyncMock()
            mock_ollama.initialize = AsyncMock()
            mock_ollama.is_healthy = True
            mock_ollama.name = "ollama"
            mock_ollama_class.return_value = mock_ollama

            await gateway.initialize()

            provider = await gateway.get_provider("ollama")
            assert provider is not None
            assert provider.name == "ollama"

            provider = await gateway.get_provider("nonexistent")
            assert provider is None
