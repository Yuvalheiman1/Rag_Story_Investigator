import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List
from src.core.models import Message

NS = {"ns": "urn:whodunit:sms:1"}


def parse_messages(file_path: Path) -> List[Message]:
    """
    Parse messages from an XML story file.
    
    Args:
        file_path: Path to the XML file containing the story
        
    Returns:
        List of Message objects in document order
        
    Raises:
        FileNotFoundError: If the file does not exist
        ET.ParseError: If the XML is malformed or invalid
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    messages = []
    
    # Find all message elements (both direct children and nested in chapters)
    for message_elem in root.findall(".//ns:message", NS):
        msg_id = message_elem.get("id", "")
        timestamp = message_elem.get("ts", "")
        # Extract sender ref
        sender_elem = message_elem.find("ns:sender", NS)
        sender = sender_elem.get("ref", "") if sender_elem is not None else ""
        # Extract receiver ref
        receiver_elem = message_elem.find("ns:receiver", NS)
        receiver = receiver_elem.get("ref", "") if receiver_elem is not None else ""
        # Extract body text
        body_elem = message_elem.find("ns:body", NS)
        body = body_elem.text if body_elem is not None and body_elem.text is not None else ""
        messages.append(Message(
            id=msg_id,
            sender=sender,
            receiver=receiver,
            body=body,
            timestamp=timestamp
        ))
    
    return messages

if __name__ == "__main__":
    messages = parse_messages(Path("data/story.xml"))
    for m in messages:
        print(m)