import os
import httpx
from unittest.mock import MagicMock
from typing import List

import pytest
from pathlib import Path
from pydantic import BaseModel
from llama_index.core.prompts import PromptTemplate
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import (
    ChatMessage,
    DocumentBlock,
    TextBlock,
    MessageRole,
    ChatResponse,
    CachePoint,
    CacheControl,
    ToolCallBlock,
)
from llama_index.core.base.llms.types import ThinkingBlock
from llama_index.core.tools import FunctionTool
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.anthropic.base import AnthropicChatResponse, AnthropicCompletionResponse
from llama_index.llms.anthropic.utils import messages_to_anthropic_messages


def test_text_inference_embedding_class():
    names_of_base_classes = [b.__name__ for b in Anthropic.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


@pytest.mark.skipif(
    os.getenv("ANTHROPIC_PROJECT_ID") is None,
    reason="Project ID not available to test Vertex AI integration",
)
def test_anthropic_through_vertex_ai():
    anthropic_llm = Anthropic(
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5@20250929"),
        region=os.getenv("ANTHROPIC_REGION", "europe-west1"),
        project_id=os.getenv("ANTHROPIC_PROJECT_ID"),
    )

    completion_response = anthropic_llm.complete("Give me a recipe for banana bread")

    try:
        assert isinstance(completion_response.text, str)
        print("Assertion passed for completion_response.text")
    except AssertionError:
        print(
            f"Assertion failed for completion_response.text: {completion_response.text}"
        )
        raise


@pytest.mark.skipif(
    os.getenv("ANTHROPIC_AWS_REGION") is None,
    reason="AWS region not available to test Bedrock integration",
)
def test_anthropic_through_bedrock():
    anthropic_llm = Anthropic(
        aws_region=os.getenv("ANTHROPIC_AWS_REGION", "us-east-1"),
        model=os.getenv("ANTHROPIC_MODEL", "anthropic.claude-sonnet-4-5-20250929-v1:0"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    completion_response = anthropic_llm.complete("Give me a recipe for banana bread")
    print("testing completion")
    try:
        assert isinstance(completion_response.text, str)
        print("Assertion passed for completion_response.text")
    except AssertionError:
        print(
            f"Assertion failed for completion_response.text: {completion_response.text}"
        )
        raise

    # Test streaming completion
    stream_resp = anthropic_llm.stream_complete(
        "Answer in 5 sentences or less. Paul Graham is "
    )
    full_response = ""
    for chunk in stream_resp:
        full_response += chunk.delta

    try:
        assert isinstance(full_response, str)
        print("Assertion passed: full_response is a string")
    except AssertionError:
        print(f"Assertion failed: full_response is not a string")
        print(f"Type of full_response: {type(full_response)}")
        print(f"Content of full_response: {full_response}")
        raise

    messages = [
        ChatMessage(
            role="system", content="You are a pirate with a colorful personality"
        ),
        ChatMessage(role="user", content="Tell me a story"),
    ]

    chat_response = anthropic_llm.chat(messages)
    print("testing chat")
    try:
        assert isinstance(chat_response.message.content, str)
        print("Assertion passed for chat_response")
    except AssertionError:
        print(f"Assertion failed for chat_response: {chat_response}")
        raise

    # Test streaming chat
    stream_chat_resp = anthropic_llm.stream_chat(messages)
    print("testing stream chat")
    full_response = ""
    for chunk in stream_chat_resp:
        full_response += chunk.delta

    try:
        assert isinstance(full_response, str)
        print("Assertion passed: full_response is a string")
    except AssertionError:
        print(f"Assertion failed: full_response is not a string")
        print(f"Type of full_response: {type(full_response)}")
        print(f"Content of full_response: {full_response}")
        raise


@pytest.mark.skipif(
    os.getenv("ANTHROPIC_AWS_REGION") is None,
    reason="AWS region not available to test Bedrock integration",
)
@pytest.mark.asyncio
async def test_anthropic_through_bedrock_async():
    # Note: this assumes you have AWS credentials configured.
    anthropic_llm = Anthropic(
        aws_region=os.getenv("ANTHROPIC_AWS_REGION", "us-east-1"),
        model=os.getenv("ANTHROPIC_MODEL", "anthropic.claude-sonnet-4-5-20250929-v1:0"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    # Test standard async completion
    standard_resp = await anthropic_llm.acomplete(
        "Answer in two sentences or less. Paul Graham is "
    )
    try:
        assert isinstance(standard_resp.text, str)
    except AssertionError:
        print(f"Assertion failed for standard_resp.text: {standard_resp.text}")
        raise

    # Test async streaming
    stream_resp = await anthropic_llm.astream_complete(
        "Answer in 5 sentences or less. Paul Graham is "
    )
    full_response = ""
    async for chunk in stream_resp:
        full_response += chunk.delta

    try:
        assert isinstance(full_response, str)
    except AssertionError:
        print(f"Assertion failed: full_response is not a string")
        print(f"Content of full_response: {full_response}")
        raise
    # Test async chat
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant"),
        ChatMessage(role="user", content="Tell me a short story about AI"),
    ]

    chat_resp = await anthropic_llm.achat(messages)
    try:
        assert isinstance(chat_resp.message.content, str)
    except AssertionError:
        print(f"Assertion failed for chat_resp: {chat_resp}")
        raise

    # Test async streaming chat
    stream_chat_resp = await anthropic_llm.astream_chat(messages)
    full_response = ""
    async for chunk in stream_chat_resp:
        full_response += chunk.delta

    try:
        assert isinstance(full_response, str)
    except AssertionError:
        print(f"Assertion failed: full_response is not a string")
        print(f"Content of full_response: {full_response}")
        raise


def test_anthropic_tokenizer():
    """Test that the Anthropic tokenizer properly implements the Tokenizer protocol."""
    # Create a mock Messages object that returns a predictable token count
    mock_messages = MagicMock()
    mock_messages.count_tokens.return_value.input_tokens = 5

    # Create a mock Beta object that returns our mock messages
    mock_beta = MagicMock()
    mock_beta.messages = mock_messages

    # Create a mock client that returns our mock beta
    mock_client = MagicMock()
    mock_client.beta = mock_beta

    # Create the Anthropic instance with our mock
    anthropic_llm = Anthropic(model="claude-sonnet-4-5-20250929")
    anthropic_llm._client = mock_client

    # Test that tokenizer implements the protocol
    tokenizer = anthropic_llm.tokenizer
    assert hasattr(tokenizer, "encode")

    # Test that encode returns a list of integers
    test_text = "Hello, world!"
    tokens = tokenizer.encode(test_text)
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)
    assert len(tokens) == 5  # Should match our mocked token count

    # Verify the mock was called correctly
    mock_messages.count_tokens.assert_called_once_with(
        messages=[{"role": "user", "content": test_text}],
        model="claude-sonnet-4-5-20250929",
    )


def test__prepare_chat_with_tools_empty():
    llm = Anthropic()
    retval = llm._prepare_chat_with_tools(tools=[])
    assert retval["tools"] == []


@pytest.fixture()
def pdf_url() -> str:
    return "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"


@pytest.mark.skipif(
    os.getenv("ANTHROPIC_API_KEY") is None,
    reason="Anthropic API key not available to test Anthropic integration",
)
def test_tool_required():
    llm = Anthropic(model="claude-sonnet-4-5-20250929")

    search_tool = FunctionTool.from_defaults(fn=search, name="search")

    # Test with tool_required=True
    response = llm.chat_with_tools(
        user_msg="What is the weather in Paris?",
        tools=[search_tool],
        tool_required=True,
    )
    assert isinstance(response, AnthropicChatResponse)
    assert (
        len(
            [
                block
                for block in response.message.blocks
                if isinstance(block, ToolCallBlock)
            ]
        )
        > 0
    )
    assert (
        any(
            block.tool_name == "search"
            for block in response.message.blocks
            if isinstance(block, ToolCallBlock)
        )
        > 0
    )

    # Test with tool_required=False
    response = llm.chat_with_tools(
        user_msg="Say hello!",
        tools=[search_tool],
        tool_required=False,
    )
    assert isinstance(response, AnthropicChatResponse)
    # Should not use tools for a simple greeting
    assert (
        len(
            [
                block
                for block in response.message.blocks
                if isinstance(block, ToolCallBlock)
            ]
        )
        == 0
    )

    # should not blow up with no tools (regression test)
    response = llm.chat_with_tools(
        user_msg="Say hello!",
        tools=[],
        tool_required=False,
    )
    assert isinstance(response, AnthropicChatResponse)
    assert (
        len(
            [
                block
                for block in response.message.blocks
                if isinstance(block, ToolCallBlock)
            ]
        )
        == 0
    )


@pytest.mark.skipif(
    os.getenv("ANTHROPIC_API_KEY") is None,
    reason="Anthropic API key not available to test Anthropic document uploading ",
)
def test_document_upload(tmp_path: Path, pdf_url: str) -> None:
    llm = Anthropic(model="claude-sonnet-4-5-20250929")
    pdf_path = tmp_path / "test.pdf"
    pdf_content = httpx.get(pdf_url).content
    pdf_path.write_bytes(pdf_content)
    msg = ChatMessage(
        role=MessageRole.USER,
        blocks=[
            DocumentBlock(path=pdf_path),
            TextBlock(text="What does the document contain?"),
        ],
    )
    messages = [msg]
    response = llm.chat(messages)
    assert isinstance(response, ChatResponse)


def test_map_tool_choice_to_anthropic():
    """Test that tool_required is correctly mapped to Anthropic's tool_choice parameter."""
    llm = Anthropic()

    # Test with tool_required=True
    tool_choice = llm._map_tool_choice_to_anthropic(
        tool_required=True, allow_parallel_tool_calls=False
    )
    assert tool_choice["type"] == "any"
    assert tool_choice["disable_parallel_tool_use"]

    # Test with tool_required=False
    tool_choice = llm._map_tool_choice_to_anthropic(
        tool_required=False, allow_parallel_tool_calls=False
    )
    assert tool_choice["type"] == "auto"
    assert tool_choice["disable_parallel_tool_use"]

    # Test with allow_parallel_tool_calls=True
    tool_choice = llm._map_tool_choice_to_anthropic(
        tool_required=True, allow_parallel_tool_calls=True
    )
    assert tool_choice["type"] == "any"
    assert not tool_choice["disable_parallel_tool_use"]


def search(query: str) -> str:
    """Search for information about a query."""
    return f"Results for {query}"


search_tool = FunctionTool.from_defaults(
    fn=search, name="search_tool", description="A tool for searching information"
)


def test_prepare_chat_with_tools_tool_required():
    """Test that tool_required is correctly passed to the API request when True."""
    llm = Anthropic()

    # Test with tool_required=True
    result = llm._prepare_chat_with_tools(tools=[search_tool], tool_required=True)

    assert result["tool_choice"]["type"] == "any"
    assert len(result["tools"]) == 1
    assert result["tools"][0]["name"] == "search_tool"


def test_prepare_chat_with_tools_tool_not_required():
    """Test that tool_required is correctly passed to the API request when False."""
    llm = Anthropic()

    # Test with tool_required=False (default)
    result = llm._prepare_chat_with_tools(
        tools=[search_tool],
    )

    assert result["tool_choice"]["type"] == "auto"
    assert len(result["tools"]) == 1
    assert result["tools"][0]["name"] == "search_tool"


def test_prepare_chat_with_no_tools_tool_not_required():
    """Test that tool_required is correctly passed to the API request when False."""
    llm = Anthropic()

    result = llm._prepare_chat_with_tools(tools=[])

    assert "tool_choice" not in result
    assert len(result["tools"]) == 0


def test_cache_point_to_cache_control() -> None:
    messages = [
        ChatMessage(role="system", blocks=[TextBlock(text="Hello1")]),
        ChatMessage(
            role="user",
            blocks=[
                TextBlock(text="Hello"),
                CachePoint(cache_control=CacheControl(type="ephemeral")),
            ],
        ),
    ]
    ant_messages, _ = messages_to_anthropic_messages(messages)
    assert ant_messages[0]["content"][-1]["cache_control"]["type"] == "ephemeral"
    assert ant_messages[0]["content"][-1]["cache_control"]["ttl"] == "5m"


def test_thinking_input():
    messages = [
        ChatMessage(
            role="assistant",
            blocks=[
                ThinkingBlock(content="Hello"),
                TextBlock(text="World"),
            ],
        ),
    ]
    ant_messages, _ = messages_to_anthropic_messages(messages)
    assert ant_messages[0]["role"] == "assistant"
    assert ant_messages[0]["content"][0]["type"] == "thinking"
    assert ant_messages[0]["content"][0]["thinking"] == "Hello"
    assert ant_messages[0]["content"][1]["type"] == "text"
    assert ant_messages[0]["content"][1]["text"] == "World"


@pytest.mark.skipif(
    os.getenv("ANTHROPIC_API_KEY") is None,
    reason="Anthropic API key not available to test Anthropic document uploading ",
)
def test_thinking():
    llm = Anthropic(
        model="claude-sonnet-4-0",
        # max_tokens must be greater than budget_tokens
        max_tokens=64000,
        # temperature must be 1.0 for thinking to work
        temperature=1.0,
        thinking_dict={"type": "enabled", "budget_tokens": 1600},
    )
    res = llm.chat(
        messages=[
            ChatMessage(
                content="Please solve the following equation for x: x^2+12x+7=0. Please think before providing a response."
            )
        ]
    )
    assert any(isinstance(block, ThinkingBlock) for block in res.message.blocks)
    assert (
        len(
            "".join(
                [
                    block.content or ""
                    for block in res.message.blocks
                    if isinstance(block, ThinkingBlock)
                ]
            )
        )
        > 0
    )


@pytest.mark.skipif(
    os.getenv("ANTHROPIC_API_KEY") is None,
    reason="Anthropic API key not available to test Anthropic document uploading ",
)
def test_thinking_with_structured_output():
    # Example from: https://docs.llamaindex.ai/en/stable/examples/llm/anthropic/#structured-prediction
    class MenuItem(BaseModel):
        """A menu item in a restaurant."""

        course_name: str
        is_vegetarian: bool

    class Restaurant(BaseModel):
        """A restaurant with name, city, and cuisine."""

        name: str
        city: str
        cuisine: str
        menu_items: List[MenuItem]

    llm = Anthropic(
        model="claude-sonnet-4-0",
        # max_tokens must be greater than budget_tokens
        max_tokens=64000,
        # temperature must be 1.0 for thinking to work
        temperature=1.0,
        thinking_dict={"type": "enabled", "budget_tokens": 1600},
    )
    prompt_tmpl = PromptTemplate("Generate a restaurant in a given city {city_name}")

    restaurant_obj = (
        llm.as_structured_llm(Restaurant)
        .complete(prompt_tmpl.format(city_name="Miami"))
        .raw
    )

    assert isinstance(restaurant_obj, Restaurant)


@pytest.mark.skipif(
    os.getenv("ANTHROPIC_API_KEY") is None,
    reason="Anthropic API key not available to test Anthropic document uploading ",
)
def test_thinking_with_tool_should_fail():
    class MenuItem(BaseModel):
        """A menu item in a restaurant."""

        course_name: str
        is_vegetarian: bool

    class Restaurant(BaseModel):
        """A restaurant with name, city, and cuisine."""

        name: str
        city: str
        cuisine: str
        menu_items: List[MenuItem]

    def generate_restaurant(restaurant: Restaurant) -> Restaurant:
        return restaurant

    llm = Anthropic(
        model="claude-sonnet-4-0",
        # max_tokens must be greater than budget_tokens
        max_tokens=64000,
        # temperature must be 1.0 for thinking to work
        temperature=1.0,
        thinking_dict={"type": "enabled", "budget_tokens": 1600},
    )

    # Raises an exception because Anthropic doesn't support tool choice when thinking is enabled
    with pytest.raises(Exception):
        llm.chat_with_tools(
            user_msg="Generate a restaurant in a given city Miami",
            tools=[generate_restaurant],
            tool_choice={"type": "any"},
        )


def test_messages_to_anthropic_messages_with_cache_idx_supported_model():
    """Test cache_idx handling with a model that supports prompt caching."""
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="System prompt"),
        ChatMessage(role=MessageRole.USER, content="User message 1"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Assistant response 1"),
        ChatMessage(role=MessageRole.USER, content="User message 2"),
    ]

    # Use a model that supports caching with cache_idx=2
    # This should cache messages[0] (SYSTEM), messages[1] (USER), messages[2] (ASSISTANT)
    anthropic_messages, system_prompt = messages_to_anthropic_messages(
        messages, cache_idx=2, model="claude-sonnet-4-5-20250929"
    )

    # cache_idx=2 means cache up to and including index 2 in original messages
    # anthropic_messages[0] = messages[1] (USER) - should have cache
    # anthropic_messages[1] = messages[2] (ASSISTANT) - should have cache
    # anthropic_messages[2] = messages[3] (USER) - should NOT have cache
    assert "cache_control" in anthropic_messages[0]["content"][0]
    assert anthropic_messages[0]["content"][0]["cache_control"]["type"] == "ephemeral"
    assert "cache_control" in anthropic_messages[1]["content"][0]
    assert anthropic_messages[1]["content"][0]["cache_control"]["type"] == "ephemeral"
    assert "cache_control" not in anthropic_messages[2]["content"][0]


def test_messages_to_anthropic_messages_with_cache_idx_unsupported_model():
    """Test cache_idx handling with a model that doesn't support prompt caching."""
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="System prompt"),
        ChatMessage(role=MessageRole.USER, content="User message 1"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Assistant response 1"),
    ]

    # Use a model that doesn't support caching
    anthropic_messages, system_prompt = messages_to_anthropic_messages(
        messages, cache_idx=1, model="claude-2.1"
    )

    # No messages should have cache_control when model doesn't support it
    for msg in anthropic_messages:
        assert "cache_control" not in msg["content"][0]


def test_messages_to_anthropic_messages_with_cache_idx_no_model():
    """Test cache_idx handling when no model is specified (should allow caching)."""
    messages = [
        ChatMessage(role=MessageRole.USER, content="User message 1"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Assistant response 1"),
    ]

    # No model specified - should include cache_control
    anthropic_messages, system_prompt = messages_to_anthropic_messages(
        messages, cache_idx=0, model=None
    )

    # First message should have cache_control when model is None
    assert "cache_control" in anthropic_messages[0]["content"][0]
    assert anthropic_messages[0]["content"][0]["cache_control"]["type"] == "ephemeral"


def test_prepare_chat_with_tools_caching_supported_model():
    """Test tool caching with a model that supports prompt caching."""
    llm = Anthropic(model="claude-sonnet-4-5-20250929")

    # Prepare tools with prompt caching enabled
    result = llm._prepare_chat_with_tools(
        tools=[search_tool],
        extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
    )

    # Should have cache_control on last tool
    assert len(result["tools"]) == 1
    assert "cache_control" in result["tools"][0]
    assert result["tools"][0]["cache_control"]["type"] == "ephemeral"


def test_prepare_chat_with_tools_caching_unsupported_model(caplog):
    """Test tool caching with a model that doesn't support prompt caching."""
    llm = Anthropic(model="claude-2.1")

    # Prepare tools with prompt caching enabled but unsupported model
    result = llm._prepare_chat_with_tools(
        tools=[search_tool],
        extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
    )

    # Should not have cache_control when model doesn't support it
    assert len(result["tools"]) == 1
    assert "cache_control" not in result["tools"][0]

    # Check that warning was logged
    assert "does not support prompt caching" in caplog.text
    assert "claude-2.1" in caplog.text


@pytest.mark.skipif(
    os.getenv("ANTHROPIC_API_KEY") is None,
    reason="Anthropic API key not available to test streaming metadata",
)
def test_stream_chat_usage_and_stop_reason():
    """
    Test that streaming captures usage metadata and stop_reason from RawMessageDeltaEvent.

    This addresses issue #20194 - Anthropic RawMessageDeltaEvent support.
    The streaming API should capture:
    - input_tokens and output_tokens from usage metadata
    - stop_reason (e.g., 'end_turn', 'max_tokens') to understand why streaming stopped
    """
    llm = Anthropic(model="claude-3-5-sonnet-latest")
    messages = [
        ChatMessage(role="user", content="Say hello in 3 words"),
    ]

    # Stream the response
    stream_resp = llm.stream_chat(messages)
    last_chunk = None
    for chunk in stream_resp:
        last_chunk = chunk

    # Verify we got a response
    assert last_chunk is not None
    assert isinstance(last_chunk, AnthropicChatResponse)

    # Check that usage metadata was captured
    usage = last_chunk.message.additional_kwargs.get("usage")
    assert usage is not None, (
        "Usage metadata should be captured from RawMessageDeltaEvent"
    )
    assert "input_tokens" in usage, "Usage should include input_tokens"
    assert "output_tokens" in usage, "Usage should include output_tokens"
    assert isinstance(usage["input_tokens"], int)
    assert isinstance(usage["output_tokens"], int)
    assert usage["input_tokens"] > 0, "Should have processed input tokens"
    assert usage["output_tokens"] > 0, "Should have generated output tokens"

    # Check that stop_reason was captured
    stop_reason = last_chunk.message.additional_kwargs.get("stop_reason")
    assert stop_reason is not None, (
        "stop_reason should be captured from RawMessageDeltaEvent"
    )
    # Typical stop reasons: "end_turn", "max_tokens", "stop_sequence", "tool_use"
    assert isinstance(stop_reason, str)
    print(f"Stop reason: {stop_reason}")
    print(f"Usage: {usage}")


@pytest.mark.skipif(
    os.getenv("ANTHROPIC_API_KEY") is None,
    reason="Anthropic API key not available to test async streaming metadata",
)
@pytest.mark.asyncio
async def test_astream_chat_usage_and_stop_reason():
    """
    Test that async streaming captures usage metadata and stop_reason.

    Async version of the streaming metadata test for issue #20194.
    """
    llm = Anthropic(model="claude-3-5-sonnet-latest")
    messages = [
        ChatMessage(role="user", content="Count to 5"),
    ]

    # Stream the response asynchronously
    stream_resp = await llm.astream_chat(messages)
    last_chunk = None
    async for chunk in stream_resp:
        last_chunk = chunk

    # Verify we got a response
    assert last_chunk is not None
    assert isinstance(last_chunk, AnthropicChatResponse)

    # Check that usage metadata was captured
    usage = last_chunk.message.additional_kwargs.get("usage")
    assert usage is not None, "Usage metadata should be captured in async streaming"
    assert "input_tokens" in usage
    assert "output_tokens" in usage
    assert isinstance(usage["input_tokens"], int)
    assert isinstance(usage["output_tokens"], int)
    assert usage["output_tokens"] > 0

    # Check that stop_reason was captured
    stop_reason = last_chunk.message.additional_kwargs.get("stop_reason")
    assert stop_reason is not None, "stop_reason should be captured in async streaming"
    assert isinstance(stop_reason, str)
    print(f"Async - Stop reason: {stop_reason}")
    print(f"Async - Usage: {usage}")




# =============================================================================
# Unit tests for uncovered code paths
# =============================================================================


def _create_real_text_block(text: str):
    """Create a real anthropic TextBlock."""
    from anthropic.types import TextBlock
    return TextBlock(text=text, type="text", citations=None)


def _create_real_tool_use_block(tool_name: str, tool_input: dict, tool_id: str = "test_tool_id"):
    """Create a real anthropic ToolUseBlock."""
    from anthropic.types import ToolUseBlock
    return ToolUseBlock(
        id=tool_id,
        input=tool_input,
        name=tool_name,
        type="tool_use",
    )


def _create_real_thinking_block(thinking_text: str):
    """Create a real anthropic ThinkingBlock."""
    from anthropic.types import ThinkingBlock
    return ThinkingBlock(
        thinking=thinking_text,
        type="thinking",
        signature="test_signature",
    )


def _create_mock_messages_response(content_blocks: list) -> MagicMock:
    """Create a mock response from anthropic client's messages.create()."""
    response = MagicMock()
    response.content = content_blocks
    response.id = "test_msg_id"
    response.model = "claude-sonnet-4-5-20250929"
    response.role = "assistant"
    response.stop_reason = "end_turn"
    response.stop_sequence = None
    response.type = "message"
    response.usage = MagicMock()
    response.usage.input_tokens = 10
    response.usage.output_tokens = 20
    return response


def test_chat_basic():
    """Test basic chat functionality with mocked client."""
    llm = Anthropic(model="claude-sonnet-4-5-20250929")

    # Create mock response with real anthropic TextBlock
    mock_response = _create_mock_messages_response(
        [_create_real_text_block("Hello! How can I help you?")]
    )

    # Mock the client
    llm._client = MagicMock()
    llm._client.messages.create.return_value = mock_response

    messages = [ChatMessage(role=MessageRole.USER, content="Hi")]
    response = llm.chat(messages)

    assert isinstance(response, AnthropicChatResponse)
    assert response.message.role == MessageRole.ASSISTANT
    assert response.message.content == "Hello! How can I help you?"
    assert response.citations == []
    llm._client.messages.create.assert_called_once()


def test_chat_with_tool_call():
    """Test chat with tool call response."""
    llm = Anthropic(model="claude-sonnet-4-5-20250929")

    mock_response = _create_mock_messages_response(
        [
            _create_real_text_block("Let me search for that."),
            _create_real_tool_use_block(
                "search_tool", {"query": "weather in Paris"}, "tool_call_1"
            ),
        ]
    )

    llm._client = MagicMock()
    llm._client.messages.create.return_value = mock_response

    messages = [ChatMessage(role=MessageRole.USER, content="What is the weather in Paris?")]
    response = llm.chat(messages)

    assert isinstance(response, AnthropicChatResponse)
    assert len(response.message.blocks) == 2
    tool_blocks = [b for b in response.message.blocks if isinstance(b, ToolCallBlock)]
    assert len(tool_blocks) == 1
    assert tool_blocks[0].tool_name == "search_tool"
    assert tool_blocks[0].tool_kwargs == {"query": "weather in Paris"}


def test_chat_with_thinking():
    """Test chat with thinking block."""
    llm = Anthropic(model="claude-sonnet-4-5-20250929")

    mock_response = _create_mock_messages_response(
        [
            _create_real_thinking_block("I need to think about this..."),
            _create_real_text_block("The answer is 42."),
        ]
    )

    llm._client = MagicMock()
    llm._client.messages.create.return_value = mock_response

    messages = [ChatMessage(role=MessageRole.USER, content="What is the meaning of life?")]
    response = llm.chat(messages)

    assert isinstance(response, AnthropicChatResponse)
    thinking_blocks = [b for b in response.message.blocks if hasattr(b, "content") and hasattr(b, "additional_information")]
    assert len(thinking_blocks) == 1
    assert thinking_blocks[0].content == "I need to think about this..."


def test_complete():
    """Test complete method."""
    llm = Anthropic(model="claude-sonnet-4-5-20250929")

    mock_response = _create_mock_messages_response(
        [_create_real_text_block("Paul Graham is a programmer, writer, and investor.")]
    )

    llm._client = MagicMock()
    llm._client.messages.create.return_value = mock_response

    response = llm.complete("Paul Graham is ")

    assert response.text == "Paul Graham is a programmer, writer, and investor."
    assert isinstance(response, AnthropicCompletionResponse)


def test_completion_response_from_chat_response():
    """Test _completion_response_from_chat_response."""
    llm = Anthropic(model="claude-sonnet-4-5-20250929")

    chat_response = AnthropicChatResponse(
        message=ChatMessage(role=MessageRole.ASSISTANT, content="Test response"),
        delta="Test response",
        citations=[],
        raw={},
    )

    completion_response = llm._completion_response_from_chat_response(chat_response)
    assert isinstance(completion_response, AnthropicCompletionResponse)
    assert completion_response.text == "Test response"
    assert completion_response.delta == "Test response"


def test_get_blocks_and_tool_calls_and_thinking():
    """Test _get_blocks_and_tool_calls_and_thinking with various block types."""
    llm = Anthropic(model="claude-sonnet-4-5-20250929")

    # Create a mock response with text, tool use, and thinking blocks
    mock_response = MagicMock()
    text_block = _create_real_text_block("Hello")
    tool_block = _create_real_tool_use_block("test_tool", {"key": "value"}, "tool_1")
    thinking_block = _create_real_thinking_block("Thinking...")

    mock_response.content = [text_block, tool_block, thinking_block]

    blocks, citations = llm._get_blocks_and_tool_calls_and_thinking(mock_response)

    assert len(blocks) == 3
    assert len(citations) == 0

    # Check text block
    text_blocks = [b for b in blocks if hasattr(b, "text")]
    assert len(text_blocks) == 1
    assert text_blocks[0].text == "Hello"

    # Check tool call block
    tool_blocks = [b for b in blocks if isinstance(b, ToolCallBlock)]
    assert len(tool_blocks) == 1
    assert tool_blocks[0].tool_name == "test_tool"
    assert tool_blocks[0].tool_kwargs == {"key": "value"}

    # Check thinking block
    thinking_blocks = [b for b in blocks if hasattr(b, "content") and hasattr(b, "additional_information")]
    assert len(thinking_blocks) == 1
    assert thinking_blocks[0].content == "Thinking..."


def test_get_tool_calls_from_response():
    """Test get_tool_calls_from_response."""
    llm = Anthropic(model="claude-sonnet-4-5-20250929")

    # Create a chat response with tool calls
    response = AnthropicChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            blocks=[
                ToolCallBlock(
                    tool_name="search_tool",
                    tool_kwargs={"query": "test"},
                    tool_call_id="call_1",
                ),
                ToolCallBlock(
                    tool_name="weather_tool",
                    tool_kwargs={"location": "Paris"},
                    tool_call_id="call_2",
                ),
            ],
        ),
    )

    tool_selections = llm.get_tool_calls_from_response(response)
    assert len(tool_selections) == 2
    assert tool_selections[0].tool_name == "search_tool"
    assert tool_selections[0].tool_kwargs == {"query": "test"}
    assert tool_selections[1].tool_name == "weather_tool"
    assert tool_selections[1].tool_kwargs == {"location": "Paris"}


def test_get_tool_calls_from_response_no_tools():
    """Test get_tool_calls_from_response with no tool calls and error_on_no_tool_call=False."""
    llm = Anthropic(model="claude-sonnet-4-5-20250929")

    response = AnthropicChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content="No tools here",
        ),
    )

    tool_selections = llm.get_tool_calls_from_response(response, error_on_no_tool_call=False)
    assert len(tool_selections) == 0


def test_get_tool_calls_from_response_no_tools_error():
    """Test get_tool_calls_from_response with no tool calls and error_on_no_tool_call=True."""
    llm = Anthropic(model="claude-sonnet-4-5-20250929")

    response = AnthropicChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content="No tools here",
        ),
    )

    with pytest.raises(ValueError, match="Expected at least one tool call"):
        llm.get_tool_calls_from_response(response, error_on_no_tool_call=True)


def test_validate_chat_with_tools_response():
    """Test _validate_chat_with_tools_response."""
    llm = Anthropic(model="claude-sonnet-4-5-20250929")

    response = AnthropicChatResponse(
        message=ChatMessage(role=MessageRole.ASSISTANT, content="Test"),
    )

    # Should return the response unchanged (no parallel tool calls to validate)
    validated = llm._validate_chat_with_tools_response(response, tools=[])
    assert validated is response


def test_model_kwargs():
    """Test _model_kwargs property."""
    llm = Anthropic(
        model="claude-sonnet-4-5-20250929",
        temperature=0.5,
        max_tokens=1000,
        additional_kwargs={"stop_sequences": ["\n\n"]},
    )

    kwargs = llm._model_kwargs
    assert kwargs["model"] == "claude-sonnet-4-5-20250929"
    assert kwargs["temperature"] == 0.5
    assert kwargs["max_tokens"] == 1000
    assert kwargs["stop_sequences"] == ["\n\n"]


def test_get_all_kwargs():
    """Test _get_all_kwargs."""
    llm = Anthropic(model="claude-sonnet-4-5-20250929")

    kwargs = llm._get_all_kwargs(temperature=0.8)
    assert kwargs["model"] == "claude-sonnet-4-5-20250929"
    assert kwargs["temperature"] == 0.8  # Overridden by passed kwargs
    assert kwargs["max_tokens"] == 512  # Default


def test_get_all_kwargs_with_thinking():
    """Test _get_all_kwargs with thinking_dict set."""
    llm = Anthropic(
        model="claude-sonnet-4-5-20250929",
        thinking_dict={"type": "enabled", "budget_tokens": 16000},
    )

    kwargs = llm._get_all_kwargs()
    assert "thinking" in kwargs
    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 16000}


def test_get_all_kwargs_with_tools():
    """Test _get_all_kwargs with tools set on the LLM."""
    llm = Anthropic(
        model="claude-sonnet-4-5-20250929",
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
    )

    kwargs = llm._get_all_kwargs()
    assert "tools" in kwargs
    assert len(kwargs["tools"]) == 1


def test_get_all_kwargs_with_mcp_servers():
    """Test _get_all_kwargs with mcp_servers set."""
    llm = Anthropic(
        model="claude-sonnet-4-5-20250929",
        mcp_servers=[{"type": "mcp_server", "name": "test_server"}],
    )

    kwargs = llm._get_all_kwargs()
    assert "mcp_servers" in kwargs
    assert "betas" in kwargs
    assert kwargs["betas"] == ["mcp-client-2025-04-04"]


def test_class_name():
    """Test class_name class method."""
    assert Anthropic.class_name() == "Anthropic_LLM"


def test_metadata():
    """Test metadata property."""
    llm = Anthropic(model="claude-sonnet-4-5-20250929")
    metadata = llm.metadata
    assert metadata.is_chat_model
    assert metadata.model_name == "claude-sonnet-4-5-20250929"
    assert metadata.num_output == 512


def test_tokenizer_property():
    """Test tokenizer property."""
    llm = Anthropic(model="claude-sonnet-4-5-20250929")
    llm._client = MagicMock()
    tokenizer = llm.tokenizer
    assert tokenizer is not None
    assert hasattr(tokenizer, "encode")


def test_map_tool_choice_to_anthropic_thinking_enabled():
    """Test _map_tool_choice_to_anthropic with thinking enabled."""
    llm = Anthropic(
        model="claude-sonnet-4-5-20250929",
        thinking_dict={"type": "enabled", "budget_tokens": 16000},
    )

    # When thinking is enabled, tool_required should not force "any" type
    tool_choice = llm._map_tool_choice_to_anthropic(
        tool_required=True, allow_parallel_tool_calls=False
    )
    assert tool_choice["type"] == "auto"  # Falls back to auto when thinking is enabled
    assert tool_choice["disable_parallel_tool_use"]


def test_init_with_thinking_sets_temperature():
    """Test that __init__ sets temperature to 1 when thinking is enabled."""
    llm = Anthropic(
        model="claude-sonnet-4-5-20250929",
        temperature=0.5,
        thinking_dict={"type": "enabled", "budget_tokens": 16000},
    )
    # Temperature should be forced to 1 when thinking is enabled
    assert llm.temperature == 1


def test_stream_chat_basic():
    """Test stream_chat with mocked streaming response."""
    llm = Anthropic(model="claude-sonnet-4-5-20250929")

    from anthropic.types import (
        ContentBlockStartEvent,
        ContentBlockDeltaEvent,
        ContentBlockStopEvent,
        MessageStartEvent,
        MessageDeltaEvent,
        MessageStopEvent,
        TextBlock as AnthropicTextBlock,
        TextDelta,
    )

    # Create mock events for streaming using real types where possible
    mock_start_event = MagicMock(spec=MessageStartEvent)
    mock_start_event.type = "message_start"
    mock_start_event.message = MagicMock()
    mock_start_event.message.role = "assistant"
    mock_start_event.message.content = []

    mock_content_start = MagicMock(spec=ContentBlockStartEvent)
    mock_content_start.type = "content_block_start"
    mock_content_start.index = 0
    mock_content_start.content_block = MagicMock(spec=AnthropicTextBlock)
    mock_content_start.content_block.type = "text"
    mock_content_start.content_block.text = ""

    mock_content_delta = MagicMock(spec=ContentBlockDeltaEvent)
    mock_content_delta.type = "content_block_delta"
    mock_content_delta.index = 0
    mock_content_delta.delta = MagicMock(spec=TextDelta)
    mock_content_delta.delta.text = "Hello"
    mock_content_delta.delta.type = "text_delta"

    mock_content_stop = MagicMock(spec=ContentBlockStopEvent)
    mock_content_stop.type = "content_block_stop"
    mock_content_stop.index = 0

    mock_message_delta = MagicMock(spec=MessageDeltaEvent)
    mock_message_delta.type = "message_delta"
    mock_message_delta.delta = MagicMock()
    mock_message_delta.delta.stop_reason = "end_turn"
    mock_message_delta.usage = MagicMock()
    mock_message_delta.usage.output_tokens = 20

    mock_message_stop = MagicMock(spec=MessageStopEvent)
    mock_message_stop.type = "message_stop"

    # Mock the client to return these events
    llm._client = MagicMock()
    llm._client.messages.create.return_value = iter([
        mock_start_event,
        mock_content_start,
        mock_content_delta,
        mock_content_stop,
        mock_message_delta,
        mock_message_stop,
    ])

    messages = [ChatMessage(role=MessageRole.USER, content="Say hello")]
    stream_gen = llm.stream_chat(messages)
    responses = list(stream_gen)

    # Should have at least one response with the delta
    assert len(responses) > 0
    # The last response should have the accumulated text
    last_response = responses[-1]
    assert isinstance(last_response, AnthropicChatResponse)


def test_astream_chat_basic():
    """Test astream_chat with mocked async streaming response."""
    import asyncio

    llm = Anthropic(model="claude-sonnet-4-5-20250929")

    from anthropic.types import (
        ContentBlockStartEvent,
        ContentBlockDeltaEvent,
        ContentBlockStopEvent,
        MessageStartEvent,
        MessageDeltaEvent,
        MessageStopEvent,
        TextBlock as AnthropicTextBlock,
        TextDelta,
    )

    mock_start_event = MagicMock(spec=MessageStartEvent)
    mock_start_event.type = "message_start"
    mock_start_event.message = MagicMock()
    mock_start_event.message.role = "assistant"
    mock_start_event.message.content = []

    mock_content_start = MagicMock(spec=ContentBlockStartEvent)
    mock_content_start.type = "content_block_start"
    mock_content_start.index = 0
    mock_content_start.content_block = MagicMock(spec=AnthropicTextBlock)
    mock_content_start.content_block.type = "text"
    mock_content_start.content_block.text = ""

    mock_content_delta = MagicMock(spec=ContentBlockDeltaEvent)
    mock_content_delta.type = "content_block_delta"
    mock_content_delta.index = 0
    mock_content_delta.delta = MagicMock(spec=TextDelta)
    mock_content_delta.delta.text = "Hello"
    mock_content_delta.delta.type = "text_delta"

    mock_content_stop = MagicMock(spec=ContentBlockStopEvent)
    mock_content_stop.type = "content_block_stop"
    mock_content_stop.index = 0

    mock_message_delta = MagicMock(spec=MessageDeltaEvent)
    mock_message_delta.type = "message_delta"
    mock_message_delta.delta = MagicMock()
    mock_message_delta.delta.stop_reason = "end_turn"
    mock_message_delta.usage = MagicMock()
    mock_message_delta.usage.output_tokens = 20

    mock_message_stop = MagicMock(spec=MessageStopEvent)
    mock_message_stop.type = "message_stop"

    async def mock_create(**kwargs):
        async def event_generator():
            yield mock_start_event
            yield mock_content_start
            yield mock_content_delta
            yield mock_content_stop
            yield mock_message_delta
            yield mock_message_stop
        return event_generator()

    # Mock the async client
    llm._aclient = MagicMock()
    llm._aclient.messages.create = mock_create

    messages = [ChatMessage(role=MessageRole.USER, content="Say hello")]

    async def run_test():
        stream_gen = await llm.astream_chat(messages)
        responses = []
        async for r in stream_gen:
            responses.append(r)
        assert len(responses) > 0
        last_response = responses[-1]
        assert isinstance(last_response, AnthropicChatResponse)

    asyncio.run(run_test())


def test_achat():
    """Test achat with mocked async client."""
    import asyncio

    llm = Anthropic(model="claude-sonnet-4-5-20250929")

    mock_response = _create_mock_messages_response(
        [_create_real_text_block("Async response")]
    )

    async def mock_create(**kwargs):
        return mock_response

    llm._aclient = MagicMock()
    llm._aclient.messages.create = mock_create

    messages = [ChatMessage(role=MessageRole.USER, content="Test")]

    async def run_test():
        response = await llm.achat(messages)
        assert isinstance(response, AnthropicChatResponse)
        assert response.message.content == "Async response"

    asyncio.run(run_test())


def test_acomplete():
    """Test acomplete with mocked async client."""
    import asyncio

    llm = Anthropic(model="claude-sonnet-4-5-20250929")

    mock_response = _create_mock_messages_response(
        [_create_real_text_block("Async completion response")]
    )

    async def mock_create(**kwargs):
        return mock_response

    llm._aclient = MagicMock()
    llm._aclient.messages.create = mock_create

    async def run_test():
        response = await llm.acomplete("Complete this")
        assert response.text == "Async completion response"

    asyncio.run(run_test())


def test_stream_complete():
    """Test stream_complete."""
    llm = Anthropic(model="claude-sonnet-4-5-20250929")

    from anthropic.types import (
        ContentBlockStartEvent,
        ContentBlockDeltaEvent,
        ContentBlockStopEvent,
        MessageStartEvent,
        MessageDeltaEvent,
        MessageStopEvent,
        TextBlock as AnthropicTextBlock,
        TextDelta,
    )

    mock_start_event = MagicMock(spec=MessageStartEvent)
    mock_start_event.type = "message_start"
    mock_start_event.message = MagicMock()
    mock_start_event.message.role = "assistant"
    mock_start_event.message.content = []

    mock_content_start = MagicMock(spec=ContentBlockStartEvent)
    mock_content_start.type = "content_block_start"
    mock_content_start.index = 0
    mock_content_start.content_block = MagicMock(spec=AnthropicTextBlock)
    mock_content_start.content_block.type = "text"
    mock_content_start.content_block.text = ""

    mock_content_delta = MagicMock(spec=ContentBlockDeltaEvent)
    mock_content_delta.type = "content_block_delta"
    mock_content_delta.index = 0
    mock_content_delta.delta = MagicMock(spec=TextDelta)
    mock_content_delta.delta.text = "Hello world"
    mock_content_delta.delta.type = "text_delta"

    mock_content_stop = MagicMock(spec=ContentBlockStopEvent)
    mock_content_stop.type = "content_block_stop"
    mock_content_stop.index = 0

    mock_message_delta = MagicMock(spec=MessageDeltaEvent)
    mock_message_delta.type = "message_delta"
    mock_message_delta.delta = MagicMock()
    mock_message_delta.delta.stop_reason = "end_turn"
    mock_message_delta.usage = MagicMock()
    mock_message_delta.usage.output_tokens = 20

    mock_message_stop = MagicMock(spec=MessageStopEvent)
    mock_message_stop.type = "message_stop"

    llm._client = MagicMock()
    llm._client.messages.create.return_value = iter([
        mock_start_event,
        mock_content_start,
        mock_content_delta,
        mock_content_stop,
        mock_message_delta,
        mock_message_stop,
    ])

    stream_gen = llm.stream_complete("Say hello")
    responses = list(stream_gen)
    assert len(responses) > 0
    last_response = responses[-1]
    assert isinstance(last_response, AnthropicCompletionResponse)


def test_astream_complete():
    """Test astream_complete with mocked async streaming."""
    import asyncio

    llm = Anthropic(model="claude-sonnet-4-5-20250929")

    from anthropic.types import (
        ContentBlockStartEvent,
        ContentBlockDeltaEvent,
        ContentBlockStopEvent,
        MessageStartEvent,
        MessageDeltaEvent,
        MessageStopEvent,
        TextBlock as AnthropicTextBlock,
        TextDelta,
    )

    mock_start_event = MagicMock(spec=MessageStartEvent)
    mock_start_event.type = "message_start"
    mock_start_event.message = MagicMock()
    mock_start_event.message.role = "assistant"
    mock_start_event.message.content = []

    mock_content_start = MagicMock(spec=ContentBlockStartEvent)
    mock_content_start.type = "content_block_start"
    mock_content_start.index = 0
    mock_content_start.content_block = MagicMock(spec=AnthropicTextBlock)
    mock_content_start.content_block.type = "text"
    mock_content_start.content_block.text = ""

    mock_content_delta = MagicMock(spec=ContentBlockDeltaEvent)
    mock_content_delta.type = "content_block_delta"
    mock_content_delta.index = 0
    mock_content_delta.delta = MagicMock(spec=TextDelta)
    mock_content_delta.delta.text = "Hello world"
    mock_content_delta.delta.type = "text_delta"

    mock_content_stop = MagicMock(spec=ContentBlockStopEvent)
    mock_content_stop.type = "content_block_stop"
    mock_content_stop.index = 0

    mock_message_delta = MagicMock(spec=MessageDeltaEvent)
    mock_message_delta.type = "message_delta"
    mock_message_delta.delta = MagicMock()
    mock_message_delta.delta.stop_reason = "end_turn"
    mock_message_delta.usage = MagicMock()
    mock_message_delta.usage.output_tokens = 20

    mock_message_stop = MagicMock(spec=MessageStopEvent)
    mock_message_stop.type = "message_stop"

    async def mock_create(**kwargs):
        async def event_generator():
            yield mock_start_event
            yield mock_content_start
            yield mock_content_delta
            yield mock_content_stop
            yield mock_message_delta
            yield mock_message_stop
        return event_generator()

    llm._aclient = MagicMock()
    llm._aclient.messages.create = mock_create

    async def run_test():
        stream_gen = await llm.astream_complete("Say hello")
        responses = []
        async for r in stream_gen:
            responses.append(r)
        assert len(responses) > 0
        last_response = responses[-1]
        assert isinstance(last_response, AnthropicCompletionResponse)

    asyncio.run(run_test())
