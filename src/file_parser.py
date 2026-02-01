import os
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import xmltodict

def parse_txt(file_path):
    """Reads a text file and returns its content."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def parse_epub(file_path):
    """Extracts text from an EPUB file."""
    book = epub.read_epub(file_path)
    chapters = []
    
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_body_content(), 'html.parser')
            text = soup.get_text('\n').strip()
            if text:
                chapters.append(text)
                
    return '\n\n'.join(chapters)

def parse_fb2(file_path):
    """Extracts text from an FB2 file."""
    with open(file_path, 'rb') as f:
        xml_content = f.read()
        
    parsed = xmltodict.parse(xml_content)
    
    # FB2 structure can be complex, this is a basic extraction
    # Usually content is in FictionBook -> body -> section
    
    text_parts = []
    
    try:
        bodies = parsed.get('FictionBook', {}).get('body', [])
        if not isinstance(bodies, list):
            bodies = [bodies]
            
        for body in bodies:
            sections = body.get('section', [])
            if not isinstance(sections, list):
                sections = [sections]
                
            for section in sections:
                # Recursively extract text from section
                extract_text_from_section(section, text_parts)
                
    except Exception as e:
        print(f"Error parsing FB2: {e}")
        return ""
        
    return '\n\n'.join(text_parts)

def extract_text_from_section(section, text_parts):
    # This is a simplified recursive extractor
    if isinstance(section, str):
        text_parts.append(section)
        return
        
    if isinstance(section, dict):
        # Check for paragraph 'p'
        if 'p' in section:
            p = section['p']
            if isinstance(p, list):
                for para in p:
                    if isinstance(para, str):
                         text_parts.append(para)
                    elif isinstance(para, dict) and '#text' in para:
                        text_parts.append(para['#text'])
            elif isinstance(p, str):
                text_parts.append(p)
            elif isinstance(p, dict) and '#text' in p:
                 text_parts.append(p['#text'])
                 
        # Recurse into subsections if any (often nested sections are allowed)
        # Note: xmltodict structure might vary, this needs robustness
        if 'section' in section:
            subsections = section['section']
            if not isinstance(subsections, list):
                subsections = [subsections]
            for sub in subsections:
                extract_text_from_section(sub, text_parts)

def split_text_into_chunks(text, max_chars=400):
    """
    Splits text into chunks of approximately max_chars length,
    respecting sentence boundaries (. ? ! \n).
    """
    import re
    # Split by sentence delimiters or newlines, keeping the delimiters
    # The regex matches (. ? ! followed by space or end of string) or newlines
    tokens = re.split(r'([.?!]+(?:\s+|$)|[\n]+)', text)
    
    chunks = []
    current_chunk = ""
    
    for token in tokens:
        # If adding this token significantly exceeds max_chars, start a new chunk
        # But if the token itself is huge (e.g. valid code or weird text), we might have to split it hard
        # For now, let's just append and check size
        
        if len(current_chunk) + len(token) > max_chars and len(current_chunk) > 0:
            chunks.append(current_chunk.strip())
            current_chunk = token
        else:
            current_chunk += token
            
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return [c for c in chunks if c]

