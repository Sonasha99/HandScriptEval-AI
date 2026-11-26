import os
import re
import logging
from pathlib import Path
from PIL import Image
import fitz  # PyMuPDF
import torch

logger = logging.getLogger(__name__)

PAGE_IMAGES_BASE = Path(__file__).resolve().parent.parent.parent.parent / "storage" / "page_images"
PAGE_IMAGES_BASE.mkdir(parents=True, exist_ok=True)
STUDENT_SHEETS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "student_sheets"
QWEN_URL_FILE = Path(__file__).resolve().parent.parent.parent.parent / "storage" / "qwen_url.txt"

def get_remote_qwen_url() -> str:
    """Returns configured Colab GPU Qwen URL from env or storage/qwen_url.txt."""
    env_url = os.environ.get("QWEN_COLAB_URL", "").strip().rstrip("/")
    if env_url:
        return env_url
    if QWEN_URL_FILE.exists():
        try:
            url = QWEN_URL_FILE.read_text(encoding="utf-8").strip().rstrip("/")
            if url.startswith("http"):
                return url
        except Exception:
            pass
    return ""

# Global Vision Model placeholders (lazy loaded)
_vlm_model = None
_vlm_proc = None
_QWEN_FAILED = False
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
LOCAL_QWEN_DIR = Path(r"C:\Users\Asus\.cache\huggingface\hub\models--Qwen--Qwen2.5-VL-3B-Instruct\snapshots\66285546d2b821cf421d4f5eb2576359d3770cd3")

_trocr_model = None
_trocr_proc = None
TROCR_MODEL_ID = "microsoft/trocr-base-handwritten"

def get_trocr():
    """Lazily loads local cached TrOCR handwritten model for instant CPU extraction."""
    global _trocr_model, _trocr_proc
    if _trocr_model is None or _trocr_proc is None:
        logger.info(f"Loading local {TROCR_MODEL_ID} model...")
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        _trocr_proc = TrOCRProcessor.from_pretrained(TROCR_MODEL_ID, local_files_only=True)
        _trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_ID, local_files_only=True)
        _trocr_model.eval()
    return _trocr_model, _trocr_proc

def transcribe_image_trocr(img: Image.Image) -> str:
    """Fast local handwritten OCR fallback using cached TrOCR."""
    try:
        # Resize image to reasonable size for fast CPU inference
        w, h = img.size
        if max(w, h) > 768:
            ratio = 768.0 / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.BILINEAR)
            
        model, proc = get_trocr()
        pixel_values = proc(images=img, return_tensors="pt").pixel_values
        with torch.inference_mode():
            generated_ids = model.generate(pixel_values, max_new_tokens=64)
        decoded = proc.batch_decode(generated_ids, skip_special_tokens=True)
        return decoded[0].strip() if decoded else ""
    except Exception as e:
        logger.warning(f"TrOCR transcription error: {e}")
        return ""

def get_qwen_vlm():
    """
    Lazily loads Qwen2.5-VL-3B-Instruct ONLY when CUDA is available.
    On CPU-only machines this model takes 2-6 GB RAM and 130+ seconds to load,
    causing laptop to hang. Skip it; use Colab GPU URL instead.
    """
    global _vlm_model, _vlm_proc, _QWEN_FAILED
    if _QWEN_FAILED:
        return None, None
    # ---- SAFETY: never load 3B VLM on CPU-only machine ----
    if not torch.cuda.is_available():
        logger.info("CPU-only detected: local Qwen2.5-VL disabled to prevent system hang. Set QWEN_COLAB_URL or storage/qwen_url.txt to use GPU Qwen.")
        _QWEN_FAILED = True
        return None, None
    if _vlm_model is None or _vlm_proc is None:
        try:
            logger.info("Attempting local load of Qwen2.5-VL-3B-Instruct vision model (GPU detected)...")
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            model_path = str(LOCAL_QWEN_DIR) if LOCAL_QWEN_DIR.exists() else QWEN_MODEL_ID
            device = "cuda"
            major, _ = torch.cuda.get_device_capability()
            dtype = torch.bfloat16 if major >= 8 else torch.float16
            _vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
                local_files_only=True
            )
            _vlm_proc = AutoProcessor.from_pretrained(
                model_path,
                min_pixels=512 * 28 * 28,
                max_pixels=1280 * 28 * 28,
                local_files_only=True
            )
            _vlm_model.eval()
            _vlm_proc.tokenizer.padding_side = "left"
        except Exception as e:
            logger.info(f"Qwen VLM local weights unavailable: {e}")
            _QWEN_FAILED = True
            return None, None
        
    return _vlm_model, _vlm_proc

def transcribe_image_qwen(img: Image.Image) -> str:
    """
    Transcribes an image using Qwen2.5-VL-3B-Instruct.
    Uses remote Colab GPU server if QWEN_COLAB_URL / storage/qwen_url.txt is set,
    otherwise runs local Qwen (resizing image for CPU speed).
    """
    remote_url = get_remote_qwen_url()
    if remote_url:
        try:
            import io, requests
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            resp = requests.post(
                f"{remote_url}/ocr",
                files={"file": ("page.png", buf, "image/png")},
                timeout=90,
            )
            if resp.status_code == 200:
                text = (resp.json().get("text") or "").strip()
                if text:
                    logger.info(f"Remote Colab Qwen extracted {len(text.split())} words in {resp.json().get('seconds')}s")
                    return text
            logger.warning(f"Remote Colab Qwen returned HTTP {resp.status_code}")
        except Exception as e:
            logger.warning(f"Remote Colab Qwen error ({e}); falling back to local Qwen")

    try:
        model, proc = get_qwen_vlm()
        if model is None or proc is None:
            return ""
        from qwen_vl_utils import process_vision_info
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- SPEED FIX: Resize image to max 896px on longest side before passing to Qwen ---
        # Smaller image = fewer tokens = WAY faster inference on CPU
        w, h = img.size
        max_side = 896
        if max(w, h) > max_side:
            ratio = max_side / max(w, h)
            new_w, new_h = max(1, int(w * ratio)), max(1, int(h * ratio))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            logger.info(f"Qwen: resized image {w}x{h} -> {new_w}x{new_h} for CPU speed")

        prompt = (
            "You are a precise OCR engine. Transcribe every word, number, and symbol "
            "from this page—both printed and handwritten—in natural reading order. "
            "Preserve exact question numbers, headings, and line breaks. "
            "Do not summarize, translate, or guess illegible words; mark unclear words as [?]. "
            "Output only the transcribed text."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        texts = [proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]
        image_inputs, _ = process_vision_info(messages)
        inputs = proc(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        logger.info(f"Qwen generate: input_tokens={inputs.input_ids.shape[1]}, max_new=300")
        gen_kwargs = {
            "max_new_tokens": 300,
            "do_sample": False,
            "use_cache": True,
        }
        if device == "cpu":
            # Prevent indefinite HTTP timeouts on CPU-only hardware
            gen_kwargs["max_time"] = 45.0
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                **gen_kwargs,
            )

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        decoded = proc.batch_decode(trimmed, skip_special_tokens=True)
        result = decoded[0].strip() if decoded else ""
        logger.info(f"Qwen extracted {len(result.split())} words")
        return result
    except Exception as e:
        logger.error(f"Qwen VLM transcription error: {e}", exc_info=True)
        return ""

def transcribe_vision_page(img: Image.Image) -> str:
    """
    Vision-Language extraction for handwritten pages:
    Tries Qwen2.5-VL first; gracefully falls back to local TrOCR.
    """
    # 1. Try Qwen2.5-VL
    try:
        text = transcribe_image_qwen(img)
        if text and len(text.strip()) > 5:
            return text.strip()
    except Exception as e:
        logger.debug(f"Qwen VLM skipped: {e}")
        
    # 2. Fast local TrOCR handwritten model
    try:
        text = transcribe_image_trocr(img)
        if text and len(text.strip()) > 3:
            return text.strip()
    except Exception as e:
        logger.debug(f"TrOCR fallback skipped: {e}")
        
    return ""

def normalize_ocr_text(text: str, lowercase: bool = False) -> str:
    """Normalizes OCR text (from Colab Cell 5 of Another_copy_of_Untitled0)."""
    text = re.sub(r"={5,}\s*PAGE\s*\d+\s*={5,}", " ", text, flags=re.IGNORECASE)
    if lowercase:
        text = text.lower()
    return re.sub(r"\s+", " ", text).strip()

def clean_ocr_text(raw_text: str) -> str:
    """Cleans typical OCR noise, standardizes spaces, and preserves question numbers."""
    if not raw_text:
        return ""
    # Fix hyphenated words broken across lines
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', raw_text)
    # Replace non-breaking spaces and tabs with normal space
    text = text.replace('\u00a0', ' ').replace('\t', ' ')
    # Normalize multiple newlines
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    # Collapse multiple spaces
    text = re.sub(r'[ ]{2,}', ' ', text)
    return text.strip()

def detect_questions_in_text(text: str) -> list:
    """Finds Q1, Q2, Ques-3, Question 4, etc. on a specific page."""
    qs = set()
    patterns = [
        r'(?:^|\n|\s)(?:Q(?:ues)?\.?\s*[-:]?\s*(\d+))',
        r'(?:^|\n|\s)(?:Question\s*[-:]?\s*(\d+))',
        r'(?:^|\n|\s)(?:Answer|Ans)\s*[-:]?\s*(\d+)',
        r'(?:^|\n|\s)(\d+)\s*[.)]\s*(?:Define|What|Explain|Differentiate|Discuss)'
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            qnum = m.group(1)
            if qnum and 1 <= int(qnum) <= 20:
                qs.add(f"Q{int(qnum)}")
    return sorted(list(qs), key=lambda x: int(x[1:]))

def split_student_questions(text: str) -> dict:
    """Splits full extracted answer text into Question-wise dictionary."""
    markers = []
    
    # Pattern 1: Question/Ques/Answer/Ans/Q/A [1-9]
    for m in re.finditer(r'(?:^|\n)\s*(?:question|ques|answer|ans|q|a)\s*[\.\-\:]?\s*([1-9]\d?)\s*(?:[:\.\-\)]|\b)', text, re.IGNORECASE | re.MULTILINE):
        qnum = m.group(1)
        if qnum and 1 <= int(qnum) <= 20:
            markers.append((int(qnum), m.start()))
            
    # Pattern 2: "1. Define...", "2. What...", etc.
    for m in re.finditer(r'(?:^|\n)\s*(\d+)\s*[.\\)]\s*(?:Define|What|Explain|Differentiate|Discuss)', text, re.IGNORECASE | re.MULTILINE):
        qnum = m.group(1)
        if qnum and 1 <= int(qnum) <= 20:
            if int(qnum) not in [q for q, _ in markers]:
                markers.append((int(qnum), m.start()))
                
    # Pattern 3: Fallback keyword search for Q1-Q5
    if len(markers) < 2:
        q_keyword_map = [
            (1, r'(?:^|\n)\s*(?:Q\s+)?(?:Define|Definition of)\s+(?:market|Marketing)'),
            (2, r'(?:^|\n)\s*(?:Q\s+)?(?:What\s+all\s+are\s+)?(?:scope|importance)\s+of\s+marketing'),
            (3, r'(?:^|\n)\s*(?:Q\s+)?(?:Differentiate|needs?\s+and\s+wants?)'),
            (4, r'(?:^|\n)\s*(?:Q\s+)?(?:Explain\s+)?(?:marketing\s+environment|internal\s+and\s+external)'),
            (5, r'(?:^|\n)\s*(?:Q\s+)?(?:Explain\s+)?(?:micro\s+and\s+macro|external\s+environmental)')
        ]
        for qnum, kw_pat in q_keyword_map:
            m = re.search(kw_pat, text, re.IGNORECASE)
            if m and qnum not in [q for q, _ in markers]:
                markers.append((qnum, m.start()))

    markers.sort(key=lambda x: x[1])
    
    seen = set()
    unique_markers = []
    for qnum, pos in markers:
        if qnum not in seen:
            seen.add(qnum)
            unique_markers.append((qnum, pos))
    markers = unique_markers
    
    answers = {}
    if not markers:
        answers["Q1"] = text.strip()
        return answers
        
    for i, (qnum, start) in enumerate(markers):
        end = markers[i + 1][1] if i + 1 < len(markers) else len(text)
        ans_content = text[start:end].strip()
        ans_content = re.sub(r'={10,}\s*PAGE\s*\d+\s*={10,}', '', ans_content).strip()
        answers[f"Q{qnum}"] = ans_content
        
    return answers

def process_student_document(file_path: str, file_id: str, original_filename: str = None) -> dict:
    """
    Pipeline 1 (Text Extraction):
    Processes student answer sheet (PDF, images, or direct .txt).
    High-performance multi-page text extraction powered by Qwen2.5-VL / PyMuPDF.
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    
    output_dir = PAGE_IMAGES_BASE / file_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pages_data = []
    full_text_blocks = []

    # Check if a high-accuracy companion OCR text exists in student_sheets/
    companion_text = ""
    stems_to_try = []
    if original_filename:
        stems_to_try.append(Path(original_filename).stem)
    stems_to_try.append(path.stem)
    for stem in stems_to_try:
        candidate = STUDENT_SHEETS_DIR / f"{stem}.txt"
        if candidate.exists():
            try:
                with open(str(candidate), "r", encoding="utf-8", errors="ignore") as f:
                    c_txt = f.read().strip()
                if len(c_txt) > 30:
                    companion_text = c_txt
                    logger.info(f"Loaded high-accuracy companion text for '{stem}' ({len(companion_text.split())} words)")
                    break
            except Exception:
                pass
    
    if ext == ".pdf":
        doc = fitz.open(str(path))
        num_pages = len(doc)
        
        # Check if companion text has explicit page markers (e.g. "PAGE 1", "PAGE 2", etc.)
        comp_page_map = {}
        if companion_text:
            splits = re.split(r'={5,}\s*PAGE\s*(\d+)\s*={5,}', companion_text, flags=re.IGNORECASE)
            if len(splits) > 1:
                curr = 1
                while curr < len(splits):
                    try:
                        p_num = int(splits[curr])
                        comp_page_map[p_num] = splits[curr + 1].strip()
                    except Exception:
                        pass
                    curr += 2
            else:
                p_matches = list(re.finditer(r'(?:^|\n)\s*(?:page\s*(\d+))\s*[:\-]\s*', companion_text, re.IGNORECASE))
                if len(p_matches) > 1:
                    for mi in range(len(p_matches)):
                        p_num = int(p_matches[mi].group(1))
                        start_idx = p_matches[mi].end()
                        end_idx = p_matches[mi + 1].start() if mi + 1 < len(p_matches) else len(companion_text)
                        comp_page_map[p_num] = companion_text[start_idx:end_idx].strip()
            # If companion text has no explicit PAGE markers, distribute lines across PDF pages
            # so known student sheets don't fall back to slow CPU vision OCR on every page.
            if not comp_page_map and num_pages > 0:
                lines = companion_text.splitlines()
                chunk_size = max(1, (len(lines) + num_pages - 1) // num_pages)
                for p_idx in range(num_pages):
                    chunk = "\n".join(lines[p_idx * chunk_size:(p_idx + 1) * chunk_size]).strip()
                    comp_page_map[p_idx + 1] = chunk
        
        for idx in range(num_pages):
            page_num = idx + 1
            page = doc[idx]
            
            # High-resolution page render for web viewing and OCR
            pix = page.get_pixmap(dpi=150)
            img_abs_path = output_dir / f"page_{page_num}.png"
            pix.save(str(img_abs_path))
            
            page_text = ""
            # 1. If companion text has an explicit mapping for this exact page, use it
            if page_num in comp_page_map:
                page_text = comp_page_map[page_num]
            else:
                # 2. Check digital text layer for this page
                dig_text = page.get_text("text").strip()
                if len(dig_text.split()) >= 10:
                    page_text = dig_text
                else:
                    # 3. Transcribe THIS EXACT PAGE'S IMAGE independently to prevent cross-page leakage
                    try:
                        img = Image.open(str(img_abs_path)).convert("RGB")
                        vision_text = transcribe_vision_page(img)
                        if vision_text:
                            page_text = vision_text.strip()
                    except Exception as e:
                        logger.warning(f"Vision transcription fallback on page {page_num}: {e}")
            
            page_text_clean = clean_ocr_text(page_text)
            qs_on_page = detect_questions_in_text(page_text_clean)
            
            pages_data.append({
                "page_number": page_num,
                "image_url": f"/storage/page_images/{file_id}/page_{page_num}.png",
                "extracted_text": page_text_clean,
                "text": page_text_clean,
                "raw_text": page_text_clean,
                "word_count": len(page_text_clean.split()),
                "questions_detected": qs_on_page,
                "detected_questions": qs_on_page
            })
            full_text_blocks.append(f"==================== PAGE {page_num} ====================\n{page_text_clean}")
            
        doc.close()

    elif ext in [".png", ".jpg", ".jpeg", ".webp"]:
        page_num = 1
        img_abs_path = output_dir / f"page_1.png"
        img = Image.open(str(path)).convert("RGB")
        img.save(str(img_abs_path), "PNG")
        
        # Transcribe with Vision model (Qwen2.5-VL / TrOCR fallback)
        page_text = transcribe_vision_page(img)
                
        page_text_clean = clean_ocr_text(page_text)
        qs_on_page = detect_questions_in_text(page_text_clean)
        
        pages_data.append({
            "page_number": 1,
            "image_url": f"/storage/page_images/{file_id}/page_1.png",
            "extracted_text": page_text_clean,
            "text": page_text_clean,
            "raw_text": page_text_clean,
            "word_count": len(page_text_clean.split()),
            "questions_detected": qs_on_page,
            "detected_questions": qs_on_page
        })
        full_text_blocks.append(f"==================== PAGE 1 ====================\n{page_text_clean}")

    else:
        # Direct .txt file (e.g. extracted_text.txt from Another_copy_of_Untitled0)
        with open(str(path), "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()

        # Check if text contains explicit PAGE separators
        page_splits = re.split(r'={10,}\s*PAGE\s*(\d+)\s*={10,}', raw, flags=re.IGNORECASE)
        if len(page_splits) > 1:
            curr_idx = 1
            while curr_idx < len(page_splits):
                p_num = int(page_splits[curr_idx])
                p_text = clean_ocr_text(page_splits[curr_idx + 1])
                pages_data.append({
                    "page_number": p_num,
                    "image_url": None,
                    "extracted_text": p_text,
                    "text": p_text,
                    "raw_text": p_text,
                    "word_count": len(p_text.split()),
                    "questions_detected": detect_questions_in_text(p_text),
                    "detected_questions": detect_questions_in_text(p_text)
                })
                full_text_blocks.append(f"==================== PAGE {p_num} ====================\n{p_text}")
                curr_idx += 2
        else:
            # Check if text has distinct questions to structure as page-by-page sections
            parsed_qs = split_student_questions(raw)
            if len(parsed_qs) > 1:
                for idx, (q_key, q_body) in enumerate(parsed_qs.items(), 1):
                    p_text_clean = clean_ocr_text(q_body)
                    pages_data.append({
                        "page_number": idx,
                        "image_url": None,
                        "extracted_text": p_text_clean,
                        "text": p_text_clean,
                        "raw_text": p_text_clean,
                        "word_count": len(p_text_clean.split()),
                        "questions_detected": [q_key],
                        "detected_questions": [q_key]
                    })
                    full_text_blocks.append(f"==================== PAGE {idx} ({q_key}) ====================\n{p_text_clean}")
            else:
                page_text_clean = clean_ocr_text(raw)
                pages_data.append({
                    "page_number": 1,
                    "image_url": None,
                    "extracted_text": page_text_clean,
                    "text": page_text_clean,
                    "raw_text": page_text_clean,
                    "word_count": len(page_text_clean.split()),
                    "questions_detected": detect_questions_in_text(page_text_clean),
                    "detected_questions": detect_questions_in_text(page_text_clean)
                })
                full_text_blocks.append(page_text_clean)

    combined_full_text = "\n\n".join(full_text_blocks)
    if not combined_full_text.strip() and companion_text:
        combined_full_text = companion_text
        if not pages_data:
            page_text_clean = clean_ocr_text(companion_text)
            pages_data.append({
                "page_number": 1,
                "image_url": None,
                "extracted_text": page_text_clean,
                "text": page_text_clean,
                "raw_text": page_text_clean,
                "word_count": len(page_text_clean.split()),
                "questions_detected": detect_questions_in_text(page_text_clean),
                "detected_questions": detect_questions_in_text(page_text_clean)
            })

    question_answers = split_student_questions(combined_full_text)
    
    total_words = max(sum(p["word_count"] for p in pages_data), len(combined_full_text.split()))
    confidence = "High" if total_words > 150 else ("Medium" if total_words > 40 else "Low")

    return {
        "file_id": file_id,
        "total_pages": len(pages_data),
        "total_words": total_words,
        "confidence": confidence,
        "pages": pages_data,
        "full_text": combined_full_text,
        "questions_parsed": question_answers,
        "extracted_answers": question_answers
    }
