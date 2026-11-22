import os
import re
from pathlib import Path
import fitz  # PyMuPDF
import docx

def extract_reference_text(file_path: str) -> str:
    """Extract raw text from PDF, DOCX, or TXT reference key files."""
    path = Path(file_path)
    ext = path.suffix.lower()
    
    if ext == ".pdf":
        doc = fitz.open(str(path))
        pages_text = []
        for page in doc:
            t = page.get_text("text") or ""
            pages_text.append(t)
        doc.close()
        return "\n".join(pages_text).strip()
    elif ext == ".docx":
        doc = docx.Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs).strip()
    elif ext == ".txt":
        with open(str(path), "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    else:
        raise ValueError(f"Unsupported reference key file extension: {ext}")

def parse_marking_scheme(scheme_text: str) -> list:
    """
    Parses structured rubric criteria and marks from Marking Scheme text
    matching CELL 9 of FIntelliPlace_Answer_Grading_Google_Colab.
    """
    rubric = []
    if not scheme_text:
        return rubric

    lines = scheme_text.splitlines()
    patterns = [
        # Definition - 2 marks
        r"^\s*(?:\d+[\.\)]\s*)?(.+?)\s*[-:]\s*(\d+(?:\.\d+)?)\s*marks?\s*$",
        # 2 marks - Definition
        r"^\s*(?:\d+[\.\)]\s*)?(\d+(?:\.\d+)?)\s*marks?\s*[-:]\s*(.+?)\s*$",
        # Definition (2 marks)
        r"^\s*(?:\d+[\.\)]\s*)?(.+?)\s*\(\s*(\d+(?:\.\d+)?)\s*marks?\s*\)\s*$"
    ]

    for line in lines:
        line = line.strip()
        if not line or re.match(r"(?i)^MARKING\s+SCHEME", line):
            continue

        for pattern_index, pattern in enumerate(patterns):
            match = re.match(pattern, line, re.IGNORECASE)
            if not match:
                continue

            if pattern_index == 1:
                marks = float(match.group(1))
                criteria = match.group(2).strip()
            else:
                criteria = match.group(1).strip()
                marks = float(match.group(2))

            criteria = re.sub(r"^[\d\.\)\-\s]+", "", criteria).strip()
            if criteria and marks > 0:
                rubric.append({
                    "component": criteria,
                    "criteria": criteria,
                    "marks": marks
                })
            break

    return rubric

def extract_concepts_from_text(answer_text: str, marking_scheme: str) -> list:
    """Extracts 3-5 structured rubric concepts with balanced weights."""
    concepts = []
    
    # 1. Try parsing marking scheme with regex
    if marking_scheme:
        rubric_table = parse_marking_scheme(marking_scheme)
        if rubric_table:
            total_m = sum(r["marks"] for r in rubric_table)
            for r in rubric_table:
                concepts.append({
                    "concept_name": r["component"],
                    "reference_explanation": r["criteria"],
                    "weight": round(r["marks"] / max(1.0, total_m), 3),
                    "marks": r["marks"]
                })
            return concepts

        # Bullet points fallback
        lines = marking_scheme.split('\n')
        for line in lines:
            line = line.strip()
            m = re.search(r'^[•\-\*\d\.\)]+\s*(.+?)(?:[:\-–]\s*(\d+(?:\.\d+)?)\s*(?:marks?|%))?$', line, re.IGNORECASE)
            if m and len(m.group(1).strip()) > 5:
                c_name = m.group(1).strip()
                c_weight = float(m.group(2)) if m.group(2) else 1.0
                concepts.append({
                    "concept_name": c_name,
                    "reference_explanation": c_name,
                    "weight": c_weight,
                    "marks": c_weight
                })
                
    # 2. Extract paragraphs / key sentences fallback
    if not concepts:
        paragraphs = [p.strip() for p in answer_text.split('\n\n') if len(p.strip()) > 25]
        if len(paragraphs) >= 2:
            for p in paragraphs[:5]:
                first_sentence = p.split('.')[0].strip()
                concepts.append({
                    "concept_name": first_sentence[:60],
                    "reference_explanation": p[:250],
                    "weight": 1.0,
                    "marks": 2.0
                })
        else:
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer_text) if len(s.strip()) > 15]
            for s in sentences[:4]:
                concepts.append({
                    "concept_name": s[:50],
                    "reference_explanation": s,
                    "weight": 1.0,
                    "marks": 2.5
                })
                
    if not concepts:
        concepts = [{
            "concept_name": "Core Technical Explanation",
            "reference_explanation": answer_text[:200],
            "weight": 1.0,
            "marks": 10.0
        }]
        
    total_w = sum(c["weight"] for c in concepts)
    if total_w > 0:
        for c in concepts:
            c["weight"] = round(c["weight"] / total_w, 3)
            
    return concepts

def parse_reference_qa_bank(raw_text: str) -> list:
    """
    Parses questions, reference model answers, and marking criteria.
    Supports formats:
    - Standard Assignment format: Question X: ... REFERENCE ANSWER: ... MARKING SCHEME: ...
    - Q1. / Q2. formats
    - Question X formats
    """
    # Clean document headers/footers
    cleaned_text = re.sub(r'Marketing Assignment - 01 \| Reference Answer Key & Rubric\s*Page \d+ of \d+', '', raw_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'Page \d+ of \d+', '', cleaned_text, flags=re.IGNORECASE)
    
    qa_bank = []
    
    # Check if format has "Question X:" sections
    sections = re.split(r'(?=(?:^|\n)\s*Question\s+\d+\s*:?)', cleaned_text, flags=re.IGNORECASE)
    valid_sections = [s.strip() for s in sections if re.match(r'^Question\s+\d+\s*:?', s.strip(), re.IGNORECASE)]
    
    if len(valid_sections) >= 1:
        for idx, sec in enumerate(valid_sections, 1):
            q_match = re.search(r'^(Question\s+\d+\s*:?.*?)(?=\n\s*(?:REFERENCE|MODEL)?\s*ANSWER\b|$)', sec, re.DOTALL | re.IGNORECASE)
            a_match = re.search(r'(?:REFERENCE|MODEL)\s*ANSWER\s*:?\s*(.*?)(?=\n\s*MARKING\s+SCHEME\b|$)', sec, re.DOTALL | re.IGNORECASE)
            m_match = re.search(r'MARKING\s+SCHEME\s*:?\s*(.*)$', sec, re.DOTALL | re.IGNORECASE)
            
            q_text = " ".join(q_match.group(1).split()) if q_match else f"Question {idx}"
            ref_ans = a_match.group(1).strip() if a_match else sec
            marking_scheme = m_match.group(1).strip() if m_match else ""
            
            # Extract structured rubric table (matching Colab Cell 9)
            rubric_table = parse_marking_scheme(marking_scheme)
            concepts = extract_concepts_from_text(ref_ans, marking_scheme)
            
            # Determine max marks
            if rubric_table:
                max_marks = float(sum(r["marks"] for r in rubric_table))
            else:
                max_marks = 10.0
                
            q_lower = q_text.lower()
            if any(k in q_lower for k in ["define", "what is", "state", "list", "name", "meaning"]):
                q_type = "Factual"
            else:
                q_type = "Analytical"
                
            qa_bank.append({
                "question_id": f"Q{idx}",
                "question_number": idx,
                "question_type": q_type,
                "question": q_text,
                "question_text": q_text,
                "reference_answer": ref_ans,
                "marking_scheme": marking_scheme,
                "max_marks": max_marks,
                "rubric_table": rubric_table,
                "rubric": concepts
            })
    else:
        # Fallback to generic Q1, Q2 splitting
        generic_markers = list(re.finditer(r'(?:^|\n)\s*(?:Q(?:ues)?\.?\s*[-:)]?\s*(\d+)|(\d+)\s*[.)]\s*(?:Define|What|Explain|Differentiate|Discuss))', cleaned_text, re.IGNORECASE))
        if generic_markers:
            for i, m in enumerate(generic_markers):
                qnum = int(m.group(1) or m.group(2))
                start = m.start()
                end = generic_markers[i + 1].start() if i + 1 < len(generic_markers) else len(cleaned_text)
                block = cleaned_text[start:end].strip()
                
                lines = block.split('\n', 1)
                q_text = lines[0].strip()
                ref_ans = lines[1].strip() if len(lines) > 1 else lines[0].strip()
                rubric_table = parse_marking_scheme(ref_ans)
                concepts = extract_concepts_from_text(ref_ans, "")
                
                max_marks = float(sum(r["marks"] for r in rubric_table)) if rubric_table else 10.0
                q_lower = q_text.lower()
                q_type = "Factual" if any(k in q_lower for k in ["define", "what is", "list"]) else "Analytical"
                
                qa_bank.append({
                    "question_id": f"Q{qnum}",
                    "question_number": qnum,
                    "question_type": q_type,
                    "question": q_text,
                    "question_text": q_text,
                    "reference_answer": ref_ans,
                    "marking_scheme": "",
                    "max_marks": max_marks,
                    "rubric_table": rubric_table,
                    "rubric": concepts
                })
        else:
            # Single question document
            rubric_table = parse_marking_scheme(cleaned_text)
            concepts = extract_concepts_from_text(cleaned_text, "")
            max_marks = float(sum(r["marks"] for r in rubric_table)) if rubric_table else 10.0
            
            qa_bank.append({
                "question_id": "Q1",
                "question_number": 1,
                "question_type": "Analytical",
                "question": "Assignment Reference Question",
                "question_text": "Assignment Reference Question",
                "reference_answer": cleaned_text,
                "marking_scheme": "",
                "max_marks": max_marks,
                "rubric_table": rubric_table,
                "rubric": concepts
            })
            
    return qa_bank

def process_reference_document(file_path: str, file_id: str) -> dict:
    """Pipeline 2: Processes reference answer key file and prepares QA Bank."""
    raw_text = extract_reference_text(file_path)
    qa_bank = parse_reference_qa_bank(raw_text)
    
    total_max_marks = sum(q.get("max_marks", 10.0) for q in qa_bank)
    
    return {
        "file_id": file_id,
        "total_questions": len(qa_bank),
        "total_max_marks": total_max_marks,
        "questions": qa_bank,
        "qa_bank": qa_bank,
        "raw_text_length": len(raw_text)
    }
