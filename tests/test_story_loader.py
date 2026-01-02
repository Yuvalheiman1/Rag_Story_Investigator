import pytest
import xml.etree.ElementTree as ET
from pathlib import Path
from src.core.story_loader import parse_messages


@pytest.fixture
def xml_file(tmp_path):
    """Fixture that returns a function to create XML files."""
    def _create_xml(content):
        file = tmp_path / "test.xml"
        file.write_text(content, encoding="utf-8")
        return file
    return _create_xml


@pytest.fixture
def valid_message_xml():
    """Template for a valid single message."""
    return '''<story xmlns="urn:whodunit:sms:1">
        <message id="m1">
            <sender ref="alice"/>
            <receiver ref="bob"/>
            <body>Hello World</body>
        </message>
    </story>'''


# Success Cases
def test_parse_single_message_returns_one_result(xml_file, valid_message_xml):
    assert len(parse_messages(xml_file(valid_message_xml))) == 1


def test_parse_message_extracts_correct_id(xml_file, valid_message_xml):
    assert parse_messages(xml_file(valid_message_xml))[0].id == "m1"


def test_parse_message_extracts_correct_sender(xml_file, valid_message_xml):
    assert parse_messages(xml_file(valid_message_xml))[0].sender == "alice"


def test_parse_message_extracts_correct_receiver(xml_file, valid_message_xml):
    assert parse_messages(xml_file(valid_message_xml))[0].receiver == "bob"


def test_parse_message_extracts_correct_body(xml_file, valid_message_xml):
    assert parse_messages(xml_file(valid_message_xml))[0].body == "Hello World"


def test_parse_multiple_messages_returns_correct_count(xml_file):
    xml = '''<story xmlns="urn:whodunit:sms:1">
        <message id="m1"><sender ref="a"/><receiver ref="b"/><body>First</body></message>
        <message id="m2"><sender ref="c"/><receiver ref="d"/><body>Second</body></message>
        <message id="m3"><sender ref="e"/><receiver ref="f"/><body>Third</body></message>
    </story>'''
    assert len(parse_messages(xml_file(xml))) == 3


def test_parse_messages_preserves_order(xml_file):
    xml = '''<story xmlns="urn:whodunit:sms:1">
        <message id="first"><sender ref="a"/><receiver ref="b"/><body>1</body></message>
        <message id="second"><sender ref="a"/><receiver ref="b"/><body>2</body></message>
        <message id="third"><sender ref="a"/><receiver ref="b"/><body>3</body></message>
    </story>'''
    result = parse_messages(xml_file(xml))
    assert [m.id for m in result] == ["first", "second", "third"]


def test_parse_messages_across_chapters_returns_all(xml_file):
    xml = '''<story xmlns="urn:whodunit:sms:1">
        <chapter id="1">
            <message id="m1"><sender ref="a"/><receiver ref="b"/><body>Ch1</body></message>
        </chapter>
        <chapter id="2">
            <message id="m2"><sender ref="a"/><receiver ref="b"/><body>Ch2</body></message>
        </chapter>
    </story>'''
    assert len(parse_messages(xml_file(xml))) == 2


# Edge Cases - Special Characters
def test_parse_message_with_unicode_emoji(xml_file):
    xml = '''<story xmlns="urn:whodunit:sms:1">
        <message id="m1"><sender ref="a"/><receiver ref="b"/><body>Ferry ⛴️ at harbour</body></message>
    </story>'''
    assert parse_messages(xml_file(xml))[0].body == "Ferry ⛴️ at harbour"


def test_parse_message_with_special_xml_characters(xml_file):
    xml = '''<story xmlns="urn:whodunit:sms:1">
        <message id="m1"><sender ref="a"/><receiver ref="b"/><body>Cost: &lt;$100 &amp; &gt;$50</body></message>
    </story>'''
    assert parse_messages(xml_file(xml))[0].body == "Cost: <$100 & >$50"


def test_parse_message_with_multiline_body(xml_file):
    xml = '''<story xmlns="urn:whodunit:sms:1">
        <message id="m1"><sender ref="a"/><receiver ref="b"/><body>Line 1
Line 2
Line 3</body></message>
    </story>'''
    result = parse_messages(xml_file(xml))[0].body
    assert "Line 1" in result and "Line 2" in result and "Line 3" in result


# Edge Cases - Missing/Empty Data
def test_parse_message_with_empty_body_returns_empty_string(xml_file):
    xml = '''<story xmlns="urn:whodunit:sms:1">
        <message id="m1"><sender ref="a"/><receiver ref="b"/><body></body></message>
    </story>'''
    result = parse_messages(xml_file(xml))[0].body
    assert result == "" or result is None


def test_parse_message_with_whitespace_only_body(xml_file):
    xml = '''<story xmlns="urn:whodunit:sms:1">
        <message id="m1"><sender ref="a"/><receiver ref="b"/><body>   </body></message>
    </story>'''
    assert parse_messages(xml_file(xml))[0].body.strip() == ""


def test_parse_message_missing_sender_ref_attribute(xml_file):
    xml = '''<story xmlns="urn:whodunit:sms:1">
        <message id="m1"><sender/><receiver ref="b"/><body>Test</body></message>
    </story>'''
    result = parse_messages(xml_file(xml))[0].sender
    assert result is None or result == ""


def test_parse_message_missing_receiver_ref_attribute(xml_file):
    xml = '''<story xmlns="urn:whodunit:sms:1">
        <message id="m1"><sender ref="a"/><receiver/><body>Test</body></message>
    </story>'''
    result = parse_messages(xml_file(xml))[0].receiver
    assert result is None or result == ""


def test_parse_message_missing_id_attribute(xml_file):
    xml = '''<story xmlns="urn:whodunit:sms:1">
        <message><sender ref="a"/><receiver ref="b"/><body>Test</body></message>
    </story>'''
    result = parse_messages(xml_file(xml))[0].id
    assert result is None or result == ""


def test_parse_story_with_no_messages_returns_empty_list(xml_file):
    xml = '''<story xmlns="urn:whodunit:sms:1"></story>'''
    assert parse_messages(xml_file(xml)) == []


def test_parse_story_with_empty_chapter_returns_empty_list(xml_file):
    xml = '''<story xmlns="urn:whodunit:sms:1">
        <chapter id="1"></chapter>
    </story>'''
    assert parse_messages(xml_file(xml)) == []


# Error Cases
def test_parse_nonexistent_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        parse_messages(Path("nonexistent_file.xml"))


def test_parse_malformed_xml_raises_parse_error(xml_file):
    with pytest.raises(ET.ParseError):
        parse_messages(xml_file("<story><message>Unclosed tag"))


def test_parse_invalid_xml_structure_raises_parse_error(xml_file):
    with pytest.raises(ET.ParseError):
        parse_messages(xml_file("<story><message></story></message>"))


# Performance/Scale
def test_parse_large_number_of_messages(xml_file):
    messages = "".join([
        f'<message id="m{i}"><sender ref="s"/><receiver ref="r"/><body>Message {i}</body></message>'
        for i in range(1000)
    ])
    xml = f'<story xmlns="urn:whodunit:sms:1">{messages}</story>'
    assert len(parse_messages(xml_file(xml))) == 1000