import xml.etree.ElementTree as ET
import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import html
import re

logger = logging.getLogger(__name__)


class XMLParser:
    """
    Parser for XML content containing episodes and messages.
    Supports both legacy episodes format and new messages format.
    """
    
    def __init__(self, xml_content: Optional[str] = None):
        self.xml_content = xml_content
        self.start_date = datetime(2025, 1, 1)  # Base date for day calculation
    
    def _clean_xml_content(self, content: str) -> str:
        """
        Clean and normalize XML content to ensure it can be parsed.
        """
        if not content:
            return content
        
        # Convert HTML entities to XML entities
        # &lt; becomes &lt; (already correct)
        # &gt; becomes &gt; (already correct)
        # &amp; becomes &amp; (already correct)
        # &quot; becomes &quot; (already correct)
        # &apos; becomes &apos; (already correct)
        
        # The HTML entities are already in the correct format for XML
        # We just need to ensure the XML structure is valid
        
        # Remove any BOM or leading whitespace
        content = content.strip()
        
        # Ensure XML declaration is at the start
        if content.startswith('<?xml'):
            # XML declaration is already at start
            pass
        elif content.startswith('<'):
            # No XML declaration, add one
            content = '<?xml version="1.0" encoding="UTF-8"?>\n' + content
        else:
            # Content doesn't start with XML, try to fix
            content = '<?xml version="1.0" encoding="UTF-8"?>\n' + content.lstrip()
        
        return content
    
    def _looks_like_html(self, content: str) -> bool:
        """
        Check if content looks like HTML instead of XML.
        """
        if not content:
            return False
        
        # Check for HTML-like patterns that indicate this is actually HTML, not XML with entities
        html_indicators = [
            '<html', '<head', '<body',  # HTML document structure
            '<div', '<span', '<p>',     # Common HTML elements
            '<script', '<style',         # HTML script/style tags
            '<meta', '<link',            # HTML meta tags
        ]
        
        # Don't treat HTML entities as HTML indicators - they're valid in XML
        # The _clean_xml_content method already converts them
        
        return any(indicator in content for indicator in html_indicators)
    
    def _load_default(self) -> Optional[str]:
        """
        Load the default episodes.xml file from disk.
        """
        try:
            default_path = "episodes.xml"
            if os.path.exists(default_path):
                with open(default_path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            logger.warning("Failed to load default episodes.xml: %s", e)
        return None

    def _parse_date_to_day_number(self, date_str: str) -> int:
        """Convert date string (YYYY-MM-DD) to day number relative to 2025-01-01"""
        try:
            # Parse the date
            message_date = datetime.strptime(date_str, "%Y-%m-%d")
            # Use 2025-01-01 as day 1
            start_date = datetime(2025, 1, 1)
            # Calculate days difference
            delta = message_date - start_date
            return delta.days + 1  # Day 1 is 2025-01-01
        except ValueError as e:
            logger.warning("Failed to parse date '%s': %s", date_str, e)
            return 1  # Default to day 1

    def parse_episodes(self) -> List[Dict[str, Any]]:
        """
        Parses the XML content and returns a list of episodes.
        
        Now supports two formats:
        1. Legacy format: <journey><episode><message>...</message></episode></journey>
        2. New format: <messages><message><content>...</content><date>...</date>...</message></messages>
        """
        content = self.xml_content or ""

        # Clean the XML content first
        content = self._clean_xml_content(content)

        # Try parsing the content directly - no fallback to episodes.xml
        try:
            root = ET.fromstring(content)
        except ET.ParseError as e:
            logger.error("Error parsing XML: %s. User input could not be parsed.", e)
            # Return empty list instead of falling back to episodes.xml
            return []

        episodes: List[Dict[str, Any]] = []
        
        # Check if this is the new messages format
        if root.tag == 'messages':
            logger.info("Parsing new messages format")
            # Create a single episode containing all messages
            episode = {
                "name": "User Messages",
                "duration": "variable",
                "context": "Messages from user input with specific dates and times",
                "messages": []
            }
            
            for msg_elem in root.findall('message'):
                content_elem = msg_elem.find('content')
                sender_elem = msg_elem.find('sender')
                date_elem = msg_elem.find('date')
                time_elem = msg_elem.find('time')
                
                content_text = content_elem.text if content_elem is not None else ""
                sender_text = sender_elem.text if sender_elem is not None else "Rohan"
                date_text = date_elem.text if date_elem is not None else "2025-01-01"
                time_text = time_elem.text if time_elem is not None else "09:00 AM"
                
                # Convert date to day number
                day_number = self._parse_date_to_day_number(date_text)
                
                message = {
                    "sender": sender_text,
                    "day": str(day_number),
                    "date": date_text,
                    "time": time_text,
                    "text": content_text
                }
                episode["messages"].append(message)
            
            episodes.append(episode)
            
        else:
            # Legacy format: parse episodes
            logger.info("Parsing legacy episodes format")
            for episode_elem in root.findall('episode'):
                context_elem = episode_elem.find('context')
                context_text = context_elem.text if context_elem is not None else ""
                episode = {
                    "name": episode_elem.get('name'),
                    "duration": episode_elem.get('duration'),
                    "context": context_text,
                    "messages": []
                }
                for msg_elem in episode_elem.findall('.//message'):
                    message = {
                        "sender": msg_elem.get('sender'),
                        "day": msg_elem.get('day'),
                        "text": msg_elem.text or ""
                    }
                    episode["messages"].append(message)
                episodes.append(episode)

        logger.info("Parsed %d episodes with %d total messages", 
                   len(episodes), 
                   sum(len(ep.get("messages", [])) for ep in episodes))
        return episodes
