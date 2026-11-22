from fastapi import APIRouter, UploadFile, File, HTTPException
import logging
from pathlib import Path
from ..utils.file_handler import save_uploaded_file, ALLOWED_STUDENT_EXTENSIONS, validate_file_extension
from ..services.ocr_service import process_student_document

router = APIRouter(prefix="/api/upload", tags=["student"])
logger = logging.getLogger(__name__)

@router.post("/student")
async def upload_student_sheet(file: UploadFile = File(...)):
    """
    Accepts student answer sheet (PDF, PNG, JPG).
    Saves the file, converts pages to images, runs OCR (Pipeline 1),
    and returns page-wise extracted text and question answers.
    """
    if not validate_file_extension(file.filename, ALLOWED_STUDENT_EXTENSIONS):
        ext = Path(file.filename).suffix.lower()
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{ext}'. Allowed types: {list(ALLOWED_STUDENT_EXTENSIONS)}"
        )
    
    try:
        saved_info = await save_uploaded_file(file, subfolder="students")
        result = process_student_document(
            file_path=saved_info["stored_path"], 
            file_id=saved_info["file_id"],
            original_filename=file.filename
        )
        result["filename"] = file.filename
        result["file_id"] = saved_info["file_id"]
        result["size_formatted"] = saved_info["size_formatted"]
        return result
    except Exception as e:
        logger.error(f"Error processing student sheet: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process student answer sheet: {str(e)}")
