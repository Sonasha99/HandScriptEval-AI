import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .routes import student, reference, evaluation

# Initialize FastAPI application
app = FastAPI(
    title="HandScriptEval AI",
    description="Automated handwritten answer sheet evaluation powered by OCR and intelligent assessment.",
    version="2.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage directories
BASE_STORAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "storage")
PAGE_IMAGES_DIR = os.path.join(BASE_STORAGE_DIR, "page_images")
REPORTS_DIR = os.path.join(BASE_STORAGE_DIR, "reports")
FRONTEND_DIST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend", "ui", "dist")

os.makedirs(PAGE_IMAGES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Mount Static Files for images and reports
app.mount("/api/static/page_images", StaticFiles(directory=PAGE_IMAGES_DIR), name="page_images")
app.mount("/storage/page_images", StaticFiles(directory=PAGE_IMAGES_DIR), name="storage_page_images")
app.mount("/api/static/reports", StaticFiles(directory=REPORTS_DIR), name="reports")

# Include Routers
app.include_router(student.router)
app.include_router(reference.router)
app.include_router(evaluation.router)

@app.on_event("startup")
def startup_event():
    """Warm up evaluation models in background thread for zero-latency first evaluations."""
    import threading
    from .services.evaluation_service import get_models
    threading.Thread(target=get_models, daemon=True).start()

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "system": "HandScriptEval AI"}

# Mount Frontend production build if available
if os.path.exists(FRONTEND_DIST_DIR):
    assets_dir = os.path.join(FRONTEND_DIST_DIR, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="static_assets")
    
    @app.get("/")
    async def serve_root():
        index_file = os.path.join(FRONTEND_DIST_DIR, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"status": "online", "message": "Frontend build not found"}

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        # Don't intercept API, documentation, or static paths
        if any(full_path.startswith(prefix) for prefix in ["api/", "docs", "openapi.json", "storage/", "assets/"]):
            return None
        index_file = os.path.join(FRONTEND_DIST_DIR, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"status": "online"}
else:
    @app.get("/")
    def root():
        return {
            "system": "IntelliPlace AI Answer Sheet Evaluation System",
            "status": "online",
            "version": "2.0.0",
            "docs_url": "/docs"
        }
