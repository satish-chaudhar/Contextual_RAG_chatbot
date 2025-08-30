from __future__ import annotations
import re # Import the regular expression module for text processing
from typing import List # Import List type for type hinting

# Define a function to split Markdown text into chunks
def split_markdown_into_chunks(md: str, target_chars: int = 1200, overlap: int = 120) -> List[str]:
    # Normalize line endings to Unix-style (\n) and strip leading/trailing whitespace
    text = re.sub(r"\r\n?", "\n", md).strip()
    
    # Split text at the start of lines beginning with a Markdown header (#)
    # Uses positive lookahead to include the header in the next part
    parts = re.split(r"(?m)^(?=\s*#)", text)
    
    # Initialize list to store intermediate chunks
    chunks = []
    
    # Initialize buffer to accumulate text before forming a chunk
    buf = ""
    
    # Iterate over each part (section starting with a header or initial content)
    for part in parts:
        # Skip empty or whitespace-only parts
        if not part.strip():
            continue
            
        # Check if adding the part to the buffer stays within target_chars
        if len(buf) + len(part) <= target_chars:
            # Add part to buffer, with double newline separator if buffer is not empty
            buf += ("\n\n" if buf else "") + part
        else:
            # Handle parts that exceed the remaining space in the buffer
            while len(part) > 0:
                # Calculate available space in the buffer
                space = target_chars - len(buf)
                
                # If no space left in buffer
                if space <= 0:
                    # If buffer is not empty, append it to chunks
                    if buf:
                        chunks.append(buf)
                    # Start new buffer with up to target_chars from part
                    buf = part[:target_chars]
                    # Keep overlap characters for the next chunk
                    part = part[target_chars - overlap:]
                else:
                    # Take as much of the part as fits in the remaining space
                    take = part[:space]
                    # Add to buffer with double newline if buffer is not empty
                    buf += ("\n\n" if buf else "") + take
                    # Keep overlap characters for the next chunk
                    part = part[space - overlap:]
                    # If buffer is now full or exceeds target_chars
                    if len(buf) >= target_chars:
                        chunks.append(buf)  # Append buffer to chunks
                        buf = ""  # Reset buffer
            
    # If buffer contains remaining text, append it to chunks
    if buf:
        chunks.append(buf)
    
    # Initialize list for merged chunks
    merged = []
    
    # Merge small chunks to avoid overly short segments
    for ch in chunks:
        # If merged list exists and chunk is small (< 200 characters)
        if merged and len(ch) < 200:
            # Append small chunk to the last merged chunk with a double newline
            merged[-1] += "\n\n" + ch
        else:
            # Add chunk as a new entry in merged list
            merged.append(ch)
    
    # Return the final list of merged chunks
    return merged