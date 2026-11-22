import os
import re
import uuid
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Optimize CPU threads for PyTorch tensor math
torch.set_num_threads(min(8, os.cpu_count() or 4))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model placeholders
_bi_encoder = None
_nli_tokenizer = None
_nli_model = None

# Fast in-memory caches to speed up evaluation
_NLI_CACHE: Dict[str, float] = {}
_REF_DOC_CACHE: Dict[str, np.ndarray] = {}
_REF_POINTS_CACHE: Dict[str, np.ndarray] = {}

# Learned feature weights from Colab Model (Cell 3.5 & Cell 10)
LEARNED_WEIGHTS = {
    "Analytical": {
        "sim": 0.274176,
        "ce": 0.689740,
        "ce_sq": 0.001180,
        "concept": 0.029231,
        "length": 0.005672
    },
    "Factual": {
        "sim": 0.000000,
        "ce": 0.474802,
        "ce_sq": 0.493302,
        "concept": 0.031896,
        "length": 0.000000
    },
    "Global": {
        "sim": 0.319729,
        "ce": 0.626837,
        "ce_sq": 0.013396,
        "concept": 0.039204,
        "length": 0.000834
    }
}

# In-memory storage for evaluations
EVALUATIONS_DB: Dict[str, Any] = {}

def get_models():
    """Lazily loads and caches the bi-encoder and NLI models."""
    global _bi_encoder, _nli_tokenizer, _nli_model
    if _bi_encoder is None:
        logger.info(f"Loading SentenceTransformer (all-mpnet-base-v2) on {DEVICE}...")
        from sentence_transformers import SentenceTransformer
        try:
            _bi_encoder = SentenceTransformer("all-mpnet-base-v2", device=DEVICE, local_files_only=True)
        except Exception:
            _bi_encoder = SentenceTransformer("all-mpnet-base-v2", device=DEVICE)
        
    if _nli_tokenizer is None or _nli_model is None:
        logger.info(f"Loading DeBERTa NLI on {DEVICE}...")
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        nli_name = "cross-encoder/nli-deberta-v3-small"
        try:
            _nli_tokenizer = AutoTokenizer.from_pretrained(nli_name, local_files_only=True)
            _nli_model = AutoModelForSequenceClassification.from_pretrained(
                nli_name,
                low_cpu_mem_usage=True,
                local_files_only=True
            ).to(DEVICE)
        except Exception:
            _nli_tokenizer = AutoTokenizer.from_pretrained(nli_name)
            _nli_model = AutoModelForSequenceClassification.from_pretrained(
                nli_name,
                low_cpu_mem_usage=True
            ).to(DEVICE)
        _nli_model.eval()
        
    return _bi_encoder, _nli_tokenizer, _nli_model

def classify_question_type(q_text: str) -> str:
    """Classifies question as Factual or Analytical (from Colab Cell 10)."""
    q = str(q_text).lower()
    factual_triggers = ["define", "what is", "list", "state", "name", "meaning", "give"]
    return "Factual" if (any(w in q for w in factual_triggers) and len(q.split()) <= 14) else "Analytical"

def strip_marks_tags(text: str) -> str:
    """Removes [X marks] tags and leading bullet/number markers."""
    t = re.sub(r'\[[^\]]*marks?[^\]]*\]', '', str(text), flags=re.IGNORECASE)
    t = re.sub(r'^[\d+\.\-*•\s]+', '', t.strip())
    return re.sub(r'\s+', ' ', t).strip()

def split_into_logical_points(text: str) -> List[str]:
    """Splits answer or reference text into logical sub-points."""
    parts = re.split(r'(?:^|\n)\s*(?=\d+\.\s+)', str(text).strip())
    cleaned = []
    for p in parts:
        p_flat = re.sub(r'\s+', ' ', p).strip()
        if len(p_flat) > 15:
            cleaned.append(p_flat)
    return cleaned if cleaned else [re.sub(r'\s+', ' ', str(text)).strip()]

def get_doc_embedding(text: str, bi_encoder, chunk_words: int = 180, overlap: int = 40) -> np.ndarray:
    """
    Chunk-based document embedding (180 words, 40 word overlap)
    averages and normalizes chunks (from Colab Cell 10).
    """
    words = text.split()
    if len(words) <= chunk_words:
        return bi_encoder.encode(text, convert_to_numpy=True)
    chunks = [" ".join(words[i:i + chunk_words]) for i in range(0, len(words), chunk_words - overlap)]
    embs = bi_encoder.encode(chunks, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    doc_emb = np.mean(embs, axis=0)
    norm = np.linalg.norm(doc_emb)
    return (doc_emb / norm) if norm > 0 else doc_emb

def check_contradiction(premise: str, hypothesis: str, tokenizer, model) -> float:
    """Computes contradiction probability using DeBERTa NLI with caching."""
    cache_key = f"{premise[:75]}|||{hypothesis[:75]}"
    if cache_key in _NLI_CACHE:
        return _NLI_CACHE[cache_key]
        
    try:
        inputs = tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(DEVICE)
        with torch.inference_mode():
            logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)[0]
        # Label 2 = Contradiction
        contra_val = float(probs[2].item())
        _NLI_CACHE[cache_key] = contra_val
        return contra_val
    except Exception:
        return 0.0

def compute_length_score(stu_text: str, ref_text: str) -> float:
    """Calculates length score normalized to 85% of reference word count."""
    stu_words = len(stu_text.split())
    ref_words = max(25, len(ref_text.split()))
    target_words = 0.85 * ref_words
    if stu_words < 10:
        return 0.0
    return float(np.clip(stu_words / target_words, 0.0, 1.0))

def split_student_answers_by_question(student_text: str, qa_bank: list, bi_encoder) -> Dict[int, str]:
    """
    Matches student answers to questions using regex boundaries
    with semantic paragraph alignment fallback (from Colab Cell 10).
    """
    total_questions = len(qa_bank)
    pattern = r'(?:^|\n)\s*(?:question|ques|answer|ans|q|a)\s*[\.\-\:]?\s*([1-9]\d?)\s*(?:[:\.\-\)]|\b)'
    matches = list(re.finditer(pattern, student_text, re.IGNORECASE))

    valid_matches = []
    expected_next = 1
    for m in matches:
        q_num = int(m.group(1))
        if 1 <= q_num <= total_questions and q_num >= expected_next:
            valid_matches.append(m)
            expected_next = q_num + 1

    q_map = {}
    if len(valid_matches) >= 2:
        for idx, m in enumerate(valid_matches):
            q_num = int(m.group(1))
            start = m.end()
            end = valid_matches[idx + 1].start() if idx + 1 < len(valid_matches) else len(student_text)
            chunk = student_text[start:end].strip()

            ans_split = re.split(
                r'(?:^|\n)\s*(?:reference\s*answer|model\s*answer|answer|ans)\s*[:\-]?\s*',
                chunk, maxsplit=1, flags=re.IGNORECASE
            )
            if len(ans_split) == 2 and len(ans_split[1].strip()) > 15:
                chunk = ans_split[1].strip()
            else:
                lines = chunk.split("\n", 1)
                if len(lines) == 2 and lines[0].strip().endswith("?"):
                    chunk = lines[1].strip()

            q_map[q_num] = chunk

    # Semantic paragraph alignment for any missing questions
    missing_qs = [i for i in range(1, total_questions + 1) if not q_map.get(i, "").strip()]
    if missing_qs:
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', student_text) if len(p.strip().split()) >= 8]
        if paragraphs:
            para_embs = bi_encoder.encode(paragraphs, convert_to_numpy=True)
            ref_texts = [qa.get("question", "") + " " + qa.get("reference_answer", "") for qa in qa_bank]
            ref_embs = bi_encoder.encode(ref_texts, convert_to_numpy=True)
            sim_matrix = cosine_similarity(para_embs, ref_embs)
            assigned = {i: [] for i in range(1, total_questions + 1)}
            for p_idx, para in enumerate(paragraphs):
                best_q = int(np.argmax(sim_matrix[p_idx])) + 1
                assigned[best_q].append(para)
            for q_idx in missing_qs:
                if assigned[q_idx]:
                    q_map[q_idx] = "\n\n".join(assigned[q_idx]).strip()

    return q_map

def evaluate_single_question_v2(
    qa: dict,
    stu_ans: str,
    bi_encoder,
    nli_tokenizer,
    nli_model
) -> dict:
    """
    Evaluates one question using the Balanced Dynamic Weighted Grading Engine
    from Colab Notebook CELL 10.
    """
    q_id = qa.get("question_id", f"Q{qa.get('question_number', 1)}")
    q_text = qa.get("question", qa.get("question_text", ""))
    ref_ans_raw = qa.get("reference_answer", "")
    ref_ans_clean = strip_marks_tags(ref_ans_raw)
    rubric_table = qa.get("rubric_table", [])
    total_q_marks = float(qa.get("max_marks", sum(r["marks"] for r in rubric_table) if rubric_table else 10.0))

    stu_ans = (stu_ans or "").strip()

    # Unattempted check
    if not stu_ans or len(stu_ans.split()) < 5:
        missing_comps = [r.get("component", r.get("criteria", "Point")) for r in rubric_table] if rubric_table else ["Complete Answer"]
        return {
            "question_id": q_id,
            "question_number": qa.get("question_number", int(re.sub(r'\D', '', q_id) or 1)),
            "question_text": q_text,
            "reference_answer": ref_ans_raw,
            "extracted_student_answer": stu_ans or "[No answer detected in student sheet]",
            "max_marks": total_q_marks,
            "marks_awarded": 0.0,
            "percentage": 0.0,
            "verdict": "Not Attempted",
            "status_badge": "incorrect",
            "explanation": "No substantial student answer was detected for this question.",
            "rubric_breakdown": [
                {
                    "component": r.get("component", "Item"),
                    "criteria": r.get("criteria", ""),
                    "awarded": 0.0,
                    "max": r.get("marks", 2.0),
                    "pct": 0.0,
                    "status": "LOW"
                } for r in rubric_table
            ],
            "concept_matches": [],
            "missing_key_concepts": missing_comps,
            "incorrect_statements": [],
            "suggestions_for_improvement": ["Review the topic and provide a structured, complete answer."],
            "dynamic_weights": {"sim": 0.35, "concept": 0.50, "length": 0.15},
            "metrics": {
                "similarity_pct": 0.0,
                "concept_pct": 0.0,
                "length_pct": 0.0,
                "question_type": classify_question_type(q_text)
            },
            "needs_manual_review": False,
            "confidence": "High"
        }

    # 1. Question Classification & Dynamic Weights
    q_cat = classify_question_type(q_text)
    raw_w = LEARNED_WEIGHTS.get(q_cat, LEARNED_WEIGHTS["Global"])
    w_sem_raw = float(raw_w.get("sim", 0.35)) + 0.5 * float(raw_w.get("ce", 0.15))
    w_con_raw = max(0.45, float(raw_w.get("concept", 0.45)))
    w_len_raw = max(0.15, float(raw_w.get("length", 0.15)))
    w_tot = w_sem_raw + w_con_raw + w_len_raw
    dyn_w = {
        "sim": round(w_sem_raw / w_tot, 3),
        "concept": round(w_con_raw / w_tot, 3),
        "length": round(w_len_raw / w_tot, 3)
    }

    # 2. Length Score
    length_ratio = compute_length_score(stu_ans, ref_ans_clean)

    # 3. Document Semantic Similarity (Capped by length)
    stu_doc_emb = get_doc_embedding(stu_ans, bi_encoder)
    if ref_ans_clean not in _REF_DOC_CACHE:
        _REF_DOC_CACHE[ref_ans_clean] = get_doc_embedding(ref_ans_clean, bi_encoder)
    ref_doc_emb = _REF_DOC_CACHE[ref_ans_clean]
    raw_doc_sim = float(np.clip(cosine_similarity([stu_doc_emb], [ref_doc_emb])[0][0], 0.0, 1.0))
    similarity_ratio = float(np.clip((raw_doc_sim - 0.40) / (0.85 - 0.40), 0.0, 1.0))
    similarity_ratio = similarity_ratio * (0.40 + 0.60 * length_ratio)

    # 4. Strict Pointwise Concept Matching
    raw_ref_points = split_into_logical_points(ref_ans_raw)
    ref_points = []
    for line in raw_ref_points:
        m_pts = re.search(r'\[\s*(\d+(?:\.\d+)?)\s*marks?\s*\]', line, re.IGNORECASE)
        pt_weight = float(m_pts.group(1)) if m_pts else 1.0
        clean_pt = strip_marks_tags(line)
        if len(clean_pt) > 10:
            ref_points.append({"text": clean_pt, "weight": pt_weight})

    if not ref_points:
        ref_points = [{"text": ref_ans_clean, "weight": total_q_marks}]

    stu_logical = [strip_marks_tags(s) for s in split_into_logical_points(stu_ans) if len(s.split()) >= 4]
    stu_sents = [strip_marks_tags(s) for s in re.split(r'(?<=[.!?])\s+', stu_ans) if len(s.strip().split()) >= 4]
    stu_segments = list(dict.fromkeys(stu_logical + stu_sents)) or [stu_ans]

    stu_embs = bi_encoder.encode(stu_segments, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    
    ref_cache_key = "|||".join(p["text"] for p in ref_points)
    if ref_cache_key not in _REF_POINTS_CACHE:
        _REF_POINTS_CACHE[ref_cache_key] = bi_encoder.encode([p["text"] for p in ref_points], batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    ref_embs = _REF_POINTS_CACHE[ref_cache_key]

    sim_matrix = cosine_similarity(ref_embs, stu_embs)
    point_scores = []
    used_segments = set()
    incorrect_statements = []
    concept_matches = []

    for r_idx, rp in enumerate(ref_points):
        sims = sim_matrix[r_idx].copy()
        for u_idx in used_segments:
            sims[u_idx] *= 0.15  # Strict penalty so one sentence cannot score across multiple rubric points

        best_s_idx = int(np.argmax(sims))
        raw_sim = float(sims[best_s_idx])
        best_sent = stu_segments[best_s_idx]
        used_segments.add(best_s_idx)

        # Realistic MPNet calibration: remapped so partial answers don't get 90%
        pt_ratio = float(np.clip((raw_sim - 0.48) / (0.86 - 0.48), 0.0, 1.0)) ** 1.15

        # Check contradiction via DeBERTa NLI if similarity is borderline
        if 0.45 <= raw_sim < 0.75:
            contra_prob = check_contradiction(best_sent, rp["text"], nli_tokenizer, nli_model)
            if contra_prob > 0.75:
                pt_ratio *= 0.40
                incorrect_statements.append(f"Student statement '{best_sent[:90]}...' contradicts key criterion '{rp['text'][:50]}...'.")

        point_scores.append(pt_ratio)
        match_pct = round(pt_ratio * 100, 1)
        concept_matches.append({
            "concept_name": " ".join(rp["text"].split()[:5]) + "...",
            "match_pct": f"{match_pct}%",
            "best_sentence": best_sent[:120],
            "status": "full" if match_pct >= 75 else ("partial" if match_pct >= 40 else "missed")
        })

    weights_arr = np.array([p["weight"] for p in ref_points], dtype=float)
    scores_arr = np.array(point_scores, dtype=float)
    concept_ratio = float(np.sum(weights_arr * scores_arr) / max(1e-6, np.sum(weights_arr)))

    # 5. Rubric Table Breakdown with Calibrated Human-Grade Alignment
    rubric_breakdown = []
    missing_key_concepts = []

    length_gate = 0.45 + 0.55 * length_ratio

    if rubric_table:
        for r_item in rubric_table:
            q_emb = bi_encoder.encode(f"{r_item['component']} {r_item['criteria']}", convert_to_numpy=True)
            rel = np.clip(cosine_similarity([q_emb], ref_embs)[0], 0.0, 1.0)
            top_mask = rel >= (np.max(rel) * 0.65)
            comp_concept_ratio = float(np.average(scores_arr[top_mask], weights=rel[top_mask])) if np.any(top_mask) else concept_ratio

            if comp_concept_ratio < 0.25:
                blended_ratio = comp_concept_ratio * 0.75 * length_gate
            else:
                # 80% driven by actual rubric concept match, 20% by overall semantic similarity
                base_ratio = (0.80 * comp_concept_ratio) + (0.20 * similarity_ratio)
                blended_ratio = base_ratio * length_gate

            blended_ratio = float(np.clip(blended_ratio, 0.0, 1.0))
            awarded = round(blended_ratio * r_item["marks"], 2)
            pct_val = round(blended_ratio * 100, 1)
            status_tag = "FULL" if pct_val >= 80 else ("PARTIAL" if pct_val >= 45 else "LOW")

            rubric_breakdown.append({
                "component": r_item["component"],
                "criteria": r_item.get("criteria", r_item["component"]),
                "awarded": awarded,
                "max": r_item["marks"],
                "pct": pct_val,
                "status": status_tag
            })
            if pct_val < 50:
                missing_key_concepts.append(r_item["component"])

        marks_obtained = round(sum(x["awarded"] for x in rubric_breakdown), 2)
    else:
        scale_factor = total_q_marks / max(1e-6, float(np.sum(weights_arr)))
        for rp, sc in zip(ref_points, scores_arr):
            max_pt_m = round(rp["weight"] * scale_factor, 2)
            if sc < 0.25:
                blended_ratio = sc * 0.75 * length_gate
            else:
                base_ratio = (0.80 * sc) + (0.20 * similarity_ratio)
                blended_ratio = base_ratio * length_gate
            blended_ratio = float(np.clip(blended_ratio, 0.0, 1.0))
            aw_pt_m = round(blended_ratio * max_pt_m, 2)
            pct_val = round(blended_ratio * 100, 1)
            label = " ".join(rp["text"].split()[:5]) + "..."
            status_tag = "FULL" if pct_val >= 80 else ("PARTIAL" if pct_val >= 45 else "LOW")

            rubric_breakdown.append({
                "component": label,
                "criteria": label,
                "awarded": aw_pt_m,
                "max": max_pt_m,
                "pct": pct_val,
                "status": status_tag
            })
            if pct_val < 50:
                missing_key_concepts.append(label)

        marks_obtained = round(sum(x["awarded"] for x in rubric_breakdown), 2)

    marks_obtained = min(total_q_marks, marks_obtained)
    q_pct = round((marks_obtained / max(1.0, total_q_marks)) * 100, 1)

    # Verdict & Status Badge
    if q_pct >= 85:
        verdict = "Excellent"
        status_badge = "correct"
        explanation = f"Outstanding answer with thorough rubric coverage ({q_pct}%). Key terminology and conceptual explanations are accurate."
    elif q_pct >= 70:
        verdict = "Good"
        status_badge = "correct"
        explanation = f"Strong grasp of the core concepts ({q_pct}%). Few minor details could be elaborated further."
    elif q_pct >= 50:
        verdict = "Partial"
        status_badge = "partially_correct"
        explanation = f"Moderate understanding demonstrated ({q_pct}%). Incomplete rubric coverage on: {', '.join(missing_key_concepts[:2]) if missing_key_concepts else 'sub-points'}."
    else:
        verdict = "Weak"
        status_badge = "incorrect"
        explanation = f"Significant conceptual gaps or brevity ({q_pct}%). Missed essential criteria: {', '.join(missing_key_concepts[:2]) if missing_key_concepts else 'primary concepts'}."

    suggestions = []
    if missing_key_concepts:
        suggestions.append(f"Add detailed explanation for: {', '.join(missing_key_concepts[:3])}.")
    if length_ratio < 0.70:
        suggestions.append("Expand the response with more analytical depth and structured bullet points.")
    if incorrect_statements:
        suggestions.append("Review contradictive assertions highlighted in the assessment breakdown.")
    if not suggestions:
        suggestions.append("Excellent answer quality. Maintain this depth and clarity!")

    needs_review = (q_pct < 45 and len(stu_ans.split()) > 80) or len(incorrect_statements) > 0
    confidence = "High" if len(stu_segments) >= 2 and similarity_ratio > 0.35 else "Medium"

    return {
        "question_id": q_id,
        "question_number": qa.get("question_number", int(re.sub(r'\D', '', q_id) or 1)),
        "question_text": q_text,
        "reference_answer": ref_ans_raw,
        "extracted_student_answer": stu_ans,
        "max_marks": total_q_marks,
        "marks_awarded": marks_obtained,
        "percentage": q_pct,
        "verdict": verdict,
        "status_badge": status_badge,
        "explanation": explanation,
        "rubric_breakdown": rubric_breakdown,
        "concept_matches": concept_matches,
        "missing_key_concepts": missing_key_concepts,
        "incorrect_statements": incorrect_statements,
        "suggestions_for_improvement": suggestions,
        "dynamic_weights": dyn_w,
        "metrics": {
            "similarity_pct": round(similarity_ratio * 100, 1),
            "concept_pct": round(concept_ratio * 100, 1),
            "length_pct": round(length_ratio * 100, 1),
            "question_type": q_cat
        },
        "needs_manual_review": needs_review,
        "confidence": confidence
    }

def run_evaluation_pipeline(
    student_data: dict,
    reference_data: dict
) -> dict:
    """
    Executes the complete evaluation comparing student answers
    against reference answer key & rubrics.
    """
    evaluation_id = str(uuid.uuid4())
    bi_encoder, nli_tokenizer, nli_model = get_models()

    student_questions = student_data.get("questions_parsed", {}) or student_data.get("extracted_answers", {})
    ref_questions = reference_data.get("questions", [])

    # If student answers were provided as a single unparsed block or missing question keys
    if not student_questions or len(student_questions) == 0:
        full_text = student_data.get("full_text", "")
        if full_text:
            parsed_by_num = split_student_answers_by_question(full_text, ref_questions, bi_encoder)
            student_questions = {f"Q{k}": v for k, v in parsed_by_num.items()}

    evaluated_questions = []
    total_obtained = 0.0
    total_max = 0.0
    manual_review_count = 0

    for idx, q_item in enumerate(ref_questions, 1):
        q_id = q_item.get("question_id", f"Q{idx}")
        q_num = q_item.get("question_number", idx)

        # Match student answer
        stu_ans = student_questions.get(q_id) or student_questions.get(f"Q{q_num}") or student_questions.get(str(q_num), "")

        result = evaluate_single_question_v2(
            qa=q_item,
            stu_ans=stu_ans,
            bi_encoder=bi_encoder,
            nli_tokenizer=nli_tokenizer,
            nli_model=nli_model
        )
        evaluated_questions.append(result)

        total_obtained += result["marks_awarded"]
        total_max += result["max_marks"]
        if result["needs_manual_review"]:
            manual_review_count += 1

    overall_pct = round((total_obtained / max(1.0, total_max)) * 100, 1)

    # Grade determination (matching Colab Cell 10)
    if overall_pct >= 85:
        overall_grade = "A+ (Outstanding)"
    elif overall_pct >= 75:
        overall_grade = "A (Very Good)"
    elif overall_pct >= 60:
        overall_grade = "B (Good)"
    elif overall_pct >= 50:
        overall_grade = "C (Average)"
    else:
        overall_grade = "F (Needs Improvement)"

    report = {
        "evaluation_id": evaluation_id,
        "timestamp": str(uuid.uuid1()),
        "student_file_id": student_data.get("file_id"),
        "reference_file_id": reference_data.get("file_id"),
        "total_marks_obtained": round(total_obtained, 2),
        "total_max_marks": round(total_max, 2),
        "overall_percentage": overall_pct,
        "total_questions_evaluated": len(evaluated_questions),
        "questions_requiring_manual_review": manual_review_count,
        "overall_grade": overall_grade,
        "questions": evaluated_questions,
        "summary": {
            "total_obtained": round(total_obtained, 2),
            "total_max": round(total_max, 2),
            "percentage": overall_pct,
            "total_questions": len(evaluated_questions),
            "review_flags_count": manual_review_count,
            "grade": overall_grade
        },
        "question_evaluations": evaluated_questions,
        "pages": student_data.get("pages", []),
        "student_metadata": {
            "total_pages": student_data.get("total_pages", len(student_data.get("pages", []))),
            "total_words": student_data.get("total_words", 0),
            "ocr_confidence": student_data.get("confidence", "High")
        }
    }

    EVALUATIONS_DB[evaluation_id] = report
    return report

def run_full_evaluation(
    extracted_answers: dict,
    qa_bank: Any,
    student_file_id: str = None,
    reference_file_id: str = None,
    pages: list = None
) -> dict:
    """Entry point for /api/evaluate endpoint."""
    student_data = {
        "file_id": student_file_id,
        "questions_parsed": extracted_answers,
        "pages": pages or []
    }

    if isinstance(qa_bank, dict):
        if "questions" in qa_bank:
            ref_questions = qa_bank["questions"]
        else:
            ref_questions = list(qa_bank.values())
    elif isinstance(qa_bank, list):
        ref_questions = qa_bank
    else:
        ref_questions = []

    reference_data = {
        "file_id": reference_file_id,
        "questions": ref_questions
    }

    return run_evaluation_pipeline(student_data, reference_data)

def get_evaluation(evaluation_id: str) -> Optional[dict]:
    return EVALUATIONS_DB.get(evaluation_id)

def get_all_evaluations() -> List[dict]:
    return list(EVALUATIONS_DB.values())

def update_manual_override(
    evaluation_id: str,
    question_number: int,
    awarded_marks: float,
    remarks: str = "",
    updated_answer: str = None
) -> Optional[dict]:
    """Allows manual mark overrides and recalculates total scores."""
    report = EVALUATIONS_DB.get(evaluation_id)
    if not report:
        return None

    for q in report["questions"]:
        if q["question_number"] == question_number or q["question_id"] == f"Q{question_number}":
            q["marks_awarded"] = float(awarded_marks)
            q["percentage"] = round((awarded_marks / max(1.0, q["max_marks"])) * 100, 1)
            q["needs_manual_review"] = False
            q["status_badge"] = "correct" if q["percentage"] >= 70 else ("partially_correct" if q["percentage"] >= 40 else "incorrect")
            q["verdict"] = "Manual Override"
            if remarks:
                q["explanation"] = f"[Evaluator Override]: {remarks}"
            if updated_answer is not None:
                q["extracted_student_answer"] = updated_answer
            break

    # Recalculate grand totals
    total_obtained = sum(q["marks_awarded"] for q in report["questions"])
    total_max = sum(q["max_marks"] for q in report["questions"])
    overall_pct = round((total_obtained / max(1.0, total_max)) * 100, 1)

    if overall_pct >= 85:
        overall_grade = "A+ (Outstanding)"
    elif overall_pct >= 75:
        overall_grade = "A (Very Good)"
    elif overall_pct >= 60:
        overall_grade = "B (Good)"
    elif overall_pct >= 50:
        overall_grade = "C (Average)"
    else:
        overall_grade = "F (Needs Improvement)"

    report["total_marks_obtained"] = round(total_obtained, 2)
    report["overall_percentage"] = overall_pct
    report["overall_grade"] = overall_grade
    report["summary"]["total_obtained"] = round(total_obtained, 2)
    report["summary"]["percentage"] = overall_pct
    report["summary"]["grade"] = overall_grade
    report["summary"]["review_flags_count"] = sum(1 for q in report["questions"] if q["needs_manual_review"])

    return report
