from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os
import logging

from ..services.evaluation_service import (
    run_full_evaluation, 
    get_evaluation, 
    update_manual_override, 
    get_all_evaluations
)
from ..utils.report_generator import generate_pdf_report, generate_excel_report

router = APIRouter(prefix="/api", tags=["evaluation"])
logger = logging.getLogger(__name__)

class EvaluateRequest(BaseModel):
    student_file_id: Optional[str] = None
    reference_file_id: Optional[str] = None
    extracted_answers: Dict[str, str]
    qa_bank: Any
    pages: Optional[List[Dict[str, Any]]] = None

class OverrideRequest(BaseModel):
    question_number: int
    awarded_marks: float
    remarks: Optional[str] = ""
    updated_answer: Optional[str] = None

@router.post("/evaluate")
async def evaluate_answers(req: EvaluateRequest):
    """
    Receives extracted student answers and structured reference QA bank.
    Executes the SBERT + DeBERTa NLI + Rubric evaluation engine.
    Returns question-wise scores and in-depth performance analysis.
    """
    try:
        # Convert student answers dict
        student_answers = req.extracted_answers
        qa_bank = req.qa_bank
                
        result = run_full_evaluation(
            extracted_answers=student_answers,
            qa_bank=qa_bank,
            student_file_id=req.student_file_id,
            reference_file_id=req.reference_file_id,
            pages=req.pages
        )
        return result
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Evaluation engine failed: {str(e)}")

@router.get("/results/{evaluation_id}")
async def get_evaluation_results(evaluation_id: str):
    """
    Retrieves the complete evaluation report by ID.
    """
    eval_data = get_evaluation(evaluation_id)
    if not eval_data:
        raise HTTPException(status_code=404, detail=f"Evaluation with ID '{evaluation_id}' not found.")
    return eval_data

@router.put("/results/{evaluation_id}/override")
async def override_score(evaluation_id: str, req: OverrideRequest):
    """
    Allows evaluator to manually update marks, provide feedback/remarks,
    or edit extracted text. Recalculates total scores automatically.
    """
    updated_eval = update_manual_override(
        evaluation_id=evaluation_id,
        question_number=req.question_number,
        awarded_marks=req.awarded_marks,
        remarks=req.remarks,
        updated_answer=req.updated_answer
    )
    if not updated_eval:
        raise HTTPException(status_code=404, detail=f"Evaluation or question not found.")
    return updated_eval

@router.get("/export/{evaluation_id}/pdf")
async def export_pdf(evaluation_id: str):
    """
    Generates and downloads a clean, professional PDF report.
    """
    eval_data = get_evaluation(evaluation_id)
    if not eval_data:
        raise HTTPException(status_code=404, detail=f"Evaluation with ID '{evaluation_id}' not found.")
    
    try:
        pdf_path = generate_pdf_report(eval_data)
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=500, detail="PDF generation failed.")
        return FileResponse(
            pdf_path, 
            media_type="application/pdf", 
            filename=f"IntelliPlace_Report_{evaluation_id[:8]}.pdf"
        )
    except Exception as e:
        logger.error(f"PDF export error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to export PDF: {str(e)}")

@router.get("/export/{evaluation_id}/excel")
async def export_excel(evaluation_id: str):
    """
    Generates and downloads an Excel spreadsheet report.
    """
    eval_data = get_evaluation(evaluation_id)
    if not eval_data:
        raise HTTPException(status_code=404, detail=f"Evaluation with ID '{evaluation_id}' not found.")
    
    try:
        excel_path = generate_excel_report(eval_data)
        if not os.path.exists(excel_path):
            raise HTTPException(status_code=500, detail="Excel generation failed.")
        return FileResponse(
            excel_path,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=f"IntelliPlace_Report_{evaluation_id[:8]}.xlsx"
        )
    except Exception as e:
        logger.error(f"Excel export error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to export Excel: {str(e)}")

@router.get("/history")
async def list_evaluations():
    """
    Returns summary list of all evaluated sessions.
    """
    return get_all_evaluations()
