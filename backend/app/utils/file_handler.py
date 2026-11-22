import os
import uuid
import shutil
from pathlib import Path
from fastapi import UploadFile

STORAGE_BASE = Path(__file__).resolve().parent.parent.parent.parent / "storage"
UPLOAD_BASE_DIR = STORAGE_BASE / "uploads"
PAGE_IMAGES_DIR = STORAGE_BASE / "page_images"
REPORTS_DIR = STORAGE_BASE / "reports"

UPLOAD_BASE_DIR.mkdir(parents=True, exist_ok=True)
PAGE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_STUDENT_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".txt"}
ALLOWED_REFERENCE_EXTENSIONS = {".txt", ".pdf", ".docx"}

def validate_file_extension(filename: str, allowed_extensions: set) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in allowed_extensions

async def save_uploaded_file(file: UploadFile, subfolder: str = "general") -> dict:
    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix.lower()
    dest_dir = UPLOAD_BASE_DIR / subfolder
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{file_id}{ext}"
    
    content = await file.read()
    size = len(content)
    with open(dest_path, "wb") as f:
        f.write(content)
        
    return {
        "file_id": file_id,
        "original_filename": file.filename,
        "stored_path": str(dest_path),
        "extension": ext,
        "size_bytes": size,
        "size_formatted": format_file_size(size)
    }

def format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
