# Enable annotations for better type hinting support in Python
from __future__ import annotations

# Import regex module for text splitting and manipulation
import re

# Import List type for type hinting the return value of functions
from typing import List

# Define the original Markdown splitting function (kept for reference but deprecated)
def split_markdown_into_chunks(md: str, target_chars: int = 1200, overlap: int = 120) -> List[str]:
    # Normalize line endings to Unix-style (\n) and remove leading/trailing whitespace
    text = re.sub(r"\r\n?", "\n", md).strip()
    
    # Split text at lines starting with Markdown headers, using positive lookahead
    parts = re.split(r"(?m)^(?=\s*#)", text)
    
    # Initialize list to store intermediate chunks
    chunks = []
    
    # Initialize buffer to accumulate text for each chunk
    buf = ""
    
    # Iterate over each part (header section or initial content)
    for part in parts:
        # Skip empty or whitespace-only parts
        if not part.strip():
            continue
            
        # If adding part to buffer stays within target size, append it
        if len(buf) + len(part) <= target_chars:
            # Add part with double newline separator if buffer isn't empty
            buf += ("\n\n" if buf else "") + part
        else:
            # Handle parts exceeding buffer space
            while len(part) > 0:
                # Calculate remaining space in buffer
                space = target_chars - len(buf)
                
                # If no space left, finalize current buffer
                if space <= 0:
                    # Append non-empty buffer to chunks
                    if buf:
                        chunks.append(buf)
                    # Start new buffer with up to target_chars from part
                    buf = part[:target_chars]
                    # Keep overlap characters for next chunk
                    part = part[target_chars - overlap:]
                else:
                    # Take as much of part as fits in buffer
                    take = part[:space]
                    # Add to buffer with separator if needed
                    buf += ("\n\n" if buf else "") + take
                    # Keep overlap for next chunk
                    part = part[space - overlap:]
                    # If buffer is full, append to chunks and reset
                    if len(buf) >= target_chars:
                        chunks.append(buf)
                        buf = ""
    
    # Append any remaining buffer content to chunks
    if buf:
        chunks.append(buf)
    
    # Initialize list for merged chunks
    merged = []
    
    # Merge small chunks to avoid overly short segments
    for ch in chunks:
        # If chunk is small and merged list exists, append to last chunk
        if merged and len(ch) < 200:
            merged[-1] += "\n\n" + ch
        else:
            # Add chunk as new entry in merged list
            merged.append(ch)
    
    # Return final list of merged chunks
    return merged

# Define a new recursive character text splitter (inspired by LangChain)
def recursive_character_split(text: str, chunk_size: int = 1200, chunk_overlap: int = 120, separators: List[str] = None) -> List[str]:
    # Set default separators if none provided (paragraphs, lines, words, characters)
    if separators is None:
        separators = ["\n\n", "\n", " ", ""]
    
    # Initialize list to store resulting chunks
    chunks = []
    
    # Define recursive splitting helper function
    def _split_recursive(current_text: str, sep_idx: int = 0):
        # If text fits within chunk size, add it to chunks
        if len(current_text) <= chunk_size:
            chunks.append(current_text)
            return
        
        # If no more separators, split by character with overlap
        if sep_idx >= len(separators):
            for i in range(0, len(current_text), chunk_size - chunk_overlap):
                # Extract chunk of size chunk_size
                chunk = current_text[i:i + chunk_size]
                # Add non-empty chunk to list
                if chunk:
                    chunks.append(chunk)
            return
        
        # Get current separator
        sep = separators[sep_idx]
        # Split text by separator, preserving separator in parts
        parts = re.split(f"({re.escape(sep)})", current_text) if sep else [current_text[i:i+chunk_size] for i in range(0, len(current_text), chunk_size - chunk_overlap)]
        
        # Initialize buffer for accumulating parts
        buf = ""
        # Iterate over parts
        for part in parts:
            # If part fits in buffer, add it
            if len(buf) + len(part) <= chunk_size:
                buf += part
            else:
                # If buffer has content, split it recursively with next separator
                if buf:
                    _split_recursive(buf, sep_idx + 1)
                # Start new buffer with current part
                buf = part
        # Process any remaining buffer content
        if buf:
            _split_recursive(buf, sep_idx + 1)
    
    # Start recursive splitting with initial text
    _split_recursive(text)
    # Return list of chunks
    return chunks