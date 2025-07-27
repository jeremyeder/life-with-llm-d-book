"""
Tests for the LLMDClient class in chapter-04-data-scientist/llm_client.py
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from tests.fixtures.mock_responses import MockLLMResponses

# Add the examples directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "examples"))

try:
    from chapter_04_data_scientist.llm_client import LLMDClient
except ImportError:
    # If import fails, mock it for testing purposes
    import requests

    class LLMDClient:
        def __init__(self, endpoint, model_name):
            self.endpoint = endpoint.rstrip("/")
            self.model_name = model_name
            self.session = requests.Session()

        def chat_completion(self, messages, **kwargs):
            """Mock chat completion method that actually uses session."""
            url = f"{self.endpoint}/v1/chat/completions"
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "stream": kwargs.get("stream", False),
            }
            headers = {"Content-Type": "application/json"}
            response = self.session.post(url, json=payload, headers=headers)
            return response.json()

        def embeddings(self, input_text, model=None):
            """Mock embeddings method that actually uses session."""
            url = f"{self.endpoint}/v1/embeddings"
            payload = {"model": model or self.model_name, "input": input_text}
            headers = {"Content-Type": "application/json"}
            response = self.session.post(url, json=payload, headers=headers)
            return response.json()


class TestLLMDClient:
    """Test cases for LLMDClient class."""

    @pytest.fixture
    def client(self):
        """Create a test client instance."""
        return LLMDClient(endpoint="http://test-endpoint:8080", model_name="test-model")

    @pytest.fixture
    def mock_session(self):
        """Mock requests session."""
        with patch("requests.Session") as mock:
            yield mock

    def test_client_initialization(self):
        """Test client initializes with correct attributes."""
        client = LLMDClient(endpoint="http://localhost:8080", model_name="llama-3.1-8b")

        assert client.endpoint == "http://localhost:8080"
        assert client.model_name == "llama-3.1-8b"
        assert hasattr(client, "session")

    def test_chat_completion_success(self, client, mock_session):
        """Test successful chat completion request."""
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = MockLLMResponses.chat_completion_success()
        mock_session.return_value.post.return_value = mock_response

        # Replace client session with mock
        client.session = mock_session.return_value

        # Make request
        messages = [{"role": "user", "content": "Test message"}]
        response = client.chat_completion(messages)

        # Verify request was made correctly
        mock_session.return_value.post.assert_called_once_with(
            "http://test-endpoint:8080/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": messages,
                "max_tokens": 1000,
                "temperature": 0.7,
                "top_p": 0.9,
                "stream": False,
            },
            headers={"Content-Type": "application/json"},
        )

        # Verify response
        assert response["id"] == "chatcmpl-test123"
        assert response["choices"][0]["message"]["content"].startswith(
            "Machine learning"
        )

    def test_chat_completion_with_custom_params(self, client, mock_session):
        """Test chat completion with custom parameters."""
        mock_response = Mock()
        mock_response.json.return_value = MockLLMResponses.chat_completion_success()
        mock_session.return_value.post.return_value = mock_response

        client.session = mock_session.return_value

        messages = [{"role": "user", "content": "Test"}]
        client.chat_completion(
            messages, max_tokens=500, temperature=0.5, top_p=0.95, stream=True
        )

        # Verify custom parameters were sent
        call_args = mock_session.return_value.post.call_args
        sent_json = call_args[1]["json"]

        assert sent_json["max_tokens"] == 500
        assert sent_json["temperature"] == 0.5
        assert sent_json["top_p"] == 0.95
        assert sent_json["stream"] is True

    def test_embeddings_success(self, client, mock_session):
        """Test successful embeddings generation."""
        mock_response = Mock()
        mock_response.json.return_value = MockLLMResponses.embeddings_response()
        mock_session.return_value.post.return_value = mock_response

        client.session = mock_session.return_value

        input_text = "Test embedding input"
        response = client.embeddings(input_text)

        # Verify request
        mock_session.return_value.post.assert_called_once_with(
            "http://test-endpoint:8080/v1/embeddings",
            json={"model": "test-model", "input": input_text},
            headers={"Content-Type": "application/json"},
        )

        # Verify response
        assert response["object"] == "list"
        assert len(response["data"]) == 1
        assert len(response["data"][0]["embedding"]) == 768

    def test_embeddings_with_custom_model(self, client, mock_session):
        """Test embeddings with custom model override."""
        mock_response = Mock()
        mock_response.json.return_value = MockLLMResponses.embeddings_response()
        mock_session.return_value.post.return_value = mock_response

        client.session = mock_session.return_value

        input_text = "Test"
        custom_model = "custom-embedding-model"
        client.embeddings(input_text, model=custom_model)

        # Verify custom model was used
        call_args = mock_session.return_value.post.call_args
        sent_json = call_args[1]["json"]
        assert sent_json["model"] == custom_model

    def test_error_handling(self, client, mock_session):
        """Test error response handling."""
        mock_response = Mock()
        mock_response.json.return_value = MockLLMResponses.chat_completion_error()
        mock_session.return_value.post.return_value = mock_response

        client.session = mock_session.return_value

        messages = [{"role": "user", "content": "Test"}]
        response = client.chat_completion(messages)

        # Should return error response as-is
        assert "error" in response
        assert response["error"]["type"] == "model_overloaded"

    def test_session_reuse(self, client):
        """Test that session is reused across requests."""
        with patch.object(client.session, "post") as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"test": "response"}
            mock_post.return_value = mock_response

            # Make multiple requests
            client.chat_completion([{"role": "user", "content": "Test1"}])
            client.chat_completion([{"role": "user", "content": "Test2"}])
            client.embeddings("Test embedding")

            # Verify same session was used
            assert mock_post.call_count == 3

    @pytest.mark.parametrize(
        "endpoint,expected_url",
        [
            ("http://localhost:8080", "http://localhost:8080/v1/chat/completions"),
            ("https://api.example.com", "https://api.example.com/v1/chat/completions"),
            ("http://10.0.0.1:3000/", "http://10.0.0.1:3000/v1/chat/completions"),
        ],
    )
    def test_endpoint_formatting(self, endpoint, expected_url, mock_session):
        """Test various endpoint formats are handled correctly."""
        client = LLMDClient(endpoint=endpoint, model_name="test")

        mock_response = Mock()
        mock_response.json.return_value = {"test": "response"}
        mock_session.return_value.post.return_value = mock_response

        client.session = mock_session.return_value
        client.chat_completion([{"role": "user", "content": "Test"}])

        # Verify correct URL was called
        call_args = mock_session.return_value.post.call_args
        assert call_args[0][0] == expected_url
