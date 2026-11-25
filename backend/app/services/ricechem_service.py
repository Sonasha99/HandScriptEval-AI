import os
import logging
from pathlib import Path
from typing import Optional
import torch

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MODEL = str(_REPO_ROOT / "models" / "ricechem_ce_final")

MODEL_ID = os.getenv("RICECHEM_MODEL", _DEFAULT_MODEL)
ENABLED = os.getenv("USE_RICECHEM", "1") == "1"
try:
    ALPHA = min(1.0, max(0.0, float(os.getenv("RICECHEM_ALPHA", "1.0"))))
except ValueError:
    ALPHA = 1.0

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_tok = None
_model = None
_failed = False


def _load():
    global _tok, _model, _failed
    if _failed or not ENABLED:
        return None, None
    if _model is None:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            _tok = AutoTokenizer.from_pretrained(MODEL_ID)
            _model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
            _model.to(_DEVICE).eval()
            logger.info("RiceChem model loaded from %s on %s", MODEL_ID, _DEVICE)
        except Exception as exc:
            logger.warning("RiceChem model unavailable (%s). Using the existing engine only.", exc)
            _failed = True
            return None, None
    return _tok, _model


def is_available() -> bool:
    tok, model = _load()
    return model is not None


def criterion_probability(question: str, criterion: str, answer: str,
                          win: int = 150, step: int = 115) -> Optional[float]:
    tok, model = _load()
    if model is None or not (answer or "").strip():
        return None
    words = answer.split()
    chunks = [" ".join(words[i:i + win]) for i in range(0, max(1, len(words)), step)]
    first = f"{question} | {criterion}"
    enc = tok([first] * len(chunks), chunks, truncation="only_second",
              max_length=512, padding=True, return_tensors="pt").to(_DEVICE)
    with torch.inference_mode():
        probs = torch.softmax(model(**enc).logits, dim=-1)[:, 1]
    return float(probs.max().item())


def blend(old_ratio: float, prob: Optional[float]) -> float:
    if prob is None:
        return old_ratio
    return ALPHA * old_ratio + (1.0 - ALPHA) * prob
