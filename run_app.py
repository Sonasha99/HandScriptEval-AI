"""
IntelliPlace AI Answer Sheet Evaluation System
Full-Stack Application Launcher
"""

import sys
import os
import uvicorn

if __name__ == "__main__":
    print("=" * 70)
    print("   HANDSCRIPTEVAL AI ANSWER SHEET EVALUATION SYSTEM (v2.0)")
    print("=" * 70)
    print("  🎨 UI Palette    : Strict Monochromatic (Black & White)")
    print("  📑 Pipeline 1   : Student Sheet OCR (Qwen2.5-VL / PyMuPDF Layout)")
    print("  📋 Pipeline 2   : Reference Answer Key & Rubric Extraction")
    print("  🧠 Evaluation   : SBERT (all-mpnet-base-v2) + DeBERTa NLI + Rubrics")
    print("  🚀 Web UI       : http://localhost:8000")
    print("  📚 API Docs     : http://localhost:8000/docs")
    print("=" * 70)
    
    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
