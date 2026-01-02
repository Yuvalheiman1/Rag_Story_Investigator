import pytest
from src.core.prompt_builder import PromptBuilder
from src.core.models import RetrievedContext


@pytest.fixture
def builder():
    """Default PromptBuilder instance."""
    return PromptBuilder()


@pytest.fixture
def simple_context():
    """A simple context with short text."""
    return RetrievedContext(
        text="<sender ref='marcus'/><receiver ref='alex'/><body>Bring USB</body>",
        source_message_ids=["msg_1"]
    )


@pytest.fixture
def empty_context():
    """Context with empty text."""
    return RetrievedContext(text="", source_message_ids=[])


@pytest.fixture
def long_context():
    """Context that exceeds max length when combined with other prompt parts."""
    return RetrievedContext(
        text="x" * 3000,
        source_message_ids=["msg_1"]
    )


# Basic Functionality Tests
def test_build_prompt_returns_string(builder, simple_context):
    result = builder.build_prompt("Who sent message?", simple_context)
    assert isinstance(result, str)


def test_build_prompt_includes_question(builder, simple_context):
    result = builder.build_prompt("Who sent USB?", simple_context)
    assert "Who sent USB?" in result


def test_build_prompt_includes_context_text(builder, simple_context):
    result = builder.build_prompt("Question?", simple_context)
    assert "Bring USB" in result


def test_build_prompt_includes_default_system_instructions(builder, simple_context):
    result = builder.build_prompt("Question?", simple_context)
    assert "AI Investigator" in result or "investigator" in result.lower()


def test_build_prompt_includes_custom_system_instructions(simple_context):
    builder = PromptBuilder(system_instructions="Custom instruction here")
    result = builder.build_prompt("Question?", simple_context)
    assert "Custom instruction here" in result


def test_build_prompt_question_appears_after_context(builder, simple_context):
    result = builder.build_prompt("Who sent USB?", simple_context)
    context_pos = result.find("Bring USB")
    question_pos = result.find("Who sent USB?")
    assert context_pos < question_pos


def test_build_prompt_with_empty_question(builder, simple_context):
    result = builder.build_prompt("", simple_context)
    assert isinstance(result, str)


def test_build_prompt_with_empty_context(builder, empty_context):
    result = builder.build_prompt("Question?", empty_context)
    assert "Question?" in result


# Length Validation Tests
def test_build_prompt_raises_error_when_exceeds_max_length(builder, long_context):
    with pytest.raises(ValueError, match="max length"):
        builder.build_prompt("Question?", long_context)


def test_build_prompt_with_custom_max_length_passes_when_under_limit(simple_context):
    builder = PromptBuilder(max_length=5000)
    result = builder.build_prompt("Question?", simple_context)
    assert isinstance(result, str)


def test_build_prompt_with_custom_max_length_fails_when_over_limit(simple_context):
    builder = PromptBuilder(max_length=50)
    with pytest.raises(ValueError):
        builder.build_prompt("Question?", simple_context)


def test_build_prompt_at_exact_max_length_succeeds():
    builder = PromptBuilder(system_instructions="Be helpful.", max_length=200)
    context = RetrievedContext(text="<body>Hi</body>", source_message_ids=["m1"])
    result = builder.build_prompt("Q?", context)
    assert len(result) <= 200


# Edge Cases - Special Characters
def test_build_prompt_with_unicode_in_question(builder, simple_context):
    result = builder.build_prompt("Who sent 🚀 emoji?", simple_context)
    assert "🚀" in result


def test_build_prompt_with_unicode_in_context(builder):
    context = RetrievedContext(
        text="<body>Meeting at café ☕</body>",
        source_message_ids=["msg_1"]
    )
    result = builder.build_prompt("Where?", context)
    assert "café" in result and "☕" in result


def test_build_prompt_with_newlines_in_context(builder):
    context = RetrievedContext(
        text="<body>Line 1\nLine 2\nLine 3</body>",
        source_message_ids=["msg_1"]
    )
    result = builder.build_prompt("Question?", context)
    assert "Line 1" in result and "Line 2" in result


def test_build_prompt_with_xml_special_chars_in_context(builder):
    context = RetrievedContext(
        text="<body>Cost: &lt;$100 &amp; &gt;$50</body>",
        source_message_ids=["msg_1"]
    )
    result = builder.build_prompt("Cost?", context)
    assert "&lt;" in result or "<" in result


# Edge Cases - Multiple Message IDs
def test_build_prompt_preserves_context_with_multiple_messages(builder):
    context = RetrievedContext(
        text="<msg1>First</msg1><msg2>Second</msg2>",
        source_message_ids=["msg_1", "msg_2"]
    )
    result = builder.build_prompt("Question?", context)
    assert "First" in result and "Second" in result


# Whitespace Handling
def test_build_prompt_with_leading_trailing_whitespace_in_question(builder, simple_context):
    result = builder.build_prompt("  Question?  ", simple_context)
    assert "Question?" in result


def test_build_prompt_does_not_add_excessive_whitespace(builder, simple_context):
    result = builder.build_prompt("Question?", simple_context)
    assert "\n\n\n\n" not in result


# System Instructions Edge Cases
def test_build_prompt_with_none_system_instructions_uses_default(simple_context):
    builder = PromptBuilder(system_instructions=None)
    result = builder.build_prompt("Question?", simple_context)
    assert len(result) > 0


def test_build_prompt_with_empty_system_instructions(simple_context):
    builder = PromptBuilder(system_instructions="")
    result = builder.build_prompt("Question?", simple_context)
    assert "Question?" in result


# Format Validation
def test_build_prompt_has_clear_sections(builder, simple_context):
    result = builder.build_prompt("Who sent?", simple_context)
    result_lower = result.lower()
    assert ("context" in result_lower or "story" in result_lower) and "question" in result_lower