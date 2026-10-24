import os
import hashlib
from functools import lru_cache
from pypdf import PdfReader

@lru_cache(maxsize=128)
def _cached_extract_bytes(file_hash: str, pdf_bytes: bytes) -> str:
    import io
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text
    except Exception as e:
        return f"Error: {str(e)}"

def extract_text_from_pdf(pdf_path):
    """
    Extracts all text from a PDF file with MD5 hash-based LRU caching.
    """
    if not os.path.exists(pdf_path):
        return "Error: File not found."
    
    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        file_hash = hashlib.md5(pdf_bytes).hexdigest()
        return _cached_extract_bytes(file_hash, pdf_bytes)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # Test script
    print("PDF Processor Tool Loaded.")
