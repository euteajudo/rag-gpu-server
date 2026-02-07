# -*- coding: utf-8 -*-
"""
Fixtures para golden set tests.

Carrega PDFs e ground truth YAMLs do diretório fixtures/ e ground_truth/.
"""

import os
import pytest
import yaml

GOLDEN_SET_DIR = os.path.dirname(__file__)
FIXTURES_DIR = os.path.join(GOLDEN_SET_DIR, "fixtures")
GROUND_TRUTH_DIR = os.path.join(GOLDEN_SET_DIR, "ground_truth")


def load_ground_truth(yaml_name: str) -> dict:
    """Carrega um arquivo YAML de ground truth."""
    path = os.path.join(GROUND_TRUTH_DIR, yaml_name)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_pdf(pdf_name: str) -> bytes:
    """Carrega um PDF fixture."""
    path = os.path.join(FIXTURES_DIR, pdf_name)
    if not os.path.exists(path):
        pytest.skip(f"PDF fixture não encontrado: {pdf_name}")
    with open(path, "rb") as f:
        return f.read()


def get_available_ground_truths() -> list[str]:
    """Retorna lista de arquivos YAML de ground truth disponíveis."""
    if not os.path.exists(GROUND_TRUTH_DIR):
        return []
    return [
        f for f in os.listdir(GROUND_TRUTH_DIR)
        if f.endswith(".yaml") or f.endswith(".yml")
    ]


def compute_error_metrics(chunks: list, truth: dict) -> dict:
    """
    Computa métricas de erro comparando chunks com ground truth.

    Args:
        chunks: Lista de ProcessedChunk (ou dicts com span_id, device_type, text, etc.)
        truth: Ground truth dict carregado do YAML

    Returns:
        Dict com:
        - precision: encontrados no ground truth / total encontrados
        - recall: ground truth encontrados / total esperados
        - offset_rate: com offsets válidos / total evidence
        - text_match_rate: snippet contém text_prefix / total
        - details: dict com listas de matched/missed/extra
    """
    expected_devices = truth.get("expected_devices", [])
    expected_span_ids = {d["span_id"] for d in expected_devices}
    expected_by_span = {d["span_id"]: d for d in expected_devices}

    # Chunks como span_ids
    chunk_span_ids = set()
    chunk_by_span = {}
    for c in chunks:
        sid = c.span_id if hasattr(c, "span_id") else c.get("span_id", "")
        chunk_span_ids.add(sid)
        chunk_by_span[sid] = c

    # Precision & Recall
    matched = expected_span_ids & chunk_span_ids
    missed = expected_span_ids - chunk_span_ids
    extra = chunk_span_ids - expected_span_ids

    total_expected = len(expected_span_ids) or 1
    total_found = len(chunk_span_ids) or 1

    precision = len(matched) / total_found
    recall = len(matched) / total_expected

    # Offset rate (evidence chunks with valid offsets)
    evidence_types = {"article", "paragraph", "inciso", "alinea"}
    evidence_count = 0
    valid_offset_count = 0
    for c in chunks:
        dt = c.device_type if hasattr(c, "device_type") else c.get("device_type", "")
        if dt in evidence_types:
            evidence_count += 1
            cs = c.canonical_start if hasattr(c, "canonical_start") else c.get("canonical_start", -1)
            ce = c.canonical_end if hasattr(c, "canonical_end") else c.get("canonical_end", -1)
            if cs >= 0 and ce > cs:
                valid_offset_count += 1

    offset_rate = valid_offset_count / max(evidence_count, 1)

    # Text match rate
    text_match_count = 0
    for span_id in matched:
        expected = expected_by_span[span_id]
        chunk = chunk_by_span.get(span_id)
        if not chunk:
            continue
        text = chunk.text if hasattr(chunk, "text") else chunk.get("text", "")
        prefix = expected.get("text_prefix", "")
        if prefix and prefix in text:
            text_match_count += 1

    text_match_rate = text_match_count / max(len(matched), 1)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "offset_rate": round(offset_rate, 4),
        "text_match_rate": round(text_match_rate, 4),
        "details": {
            "matched": sorted(matched),
            "missed": sorted(missed),
            "extra": sorted(extra),
            "evidence_count": evidence_count,
            "valid_offset_count": valid_offset_count,
        },
    }
