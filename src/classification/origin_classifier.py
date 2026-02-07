"""
OriginClassifier — Classificador de proveniência de chunks.

Detecta quando chunks pertencem a outra norma (ex: artigos do Código Penal
inseridos pela Lei 14.133/2021) usando score híbrido + máquina de estados.

Custo GPU: zero (regex + heurística).
Dependências: re (stdlib).
"""

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Thresholds ──────────────────────────────────────────────────────────
T_ENTER = 0.60
T_EXIT = 0.40
TTL_CHUNKS = 50
CTX_WINDOW = 800

# ── Pesos de entrada (E1-E7) ───────────────────────────────────────────
W_TRIGGER_PHRASE = 0.40
W_QUOTE_OPEN = 0.20
W_OUT_OF_SEQUENCE = 0.50
W_CHAPTER_IN_QUOTES = 0.40
W_TARGET_REF = 0.30
W_TARGET_NAME = 0.20
W_ANNEX_HEADER = 0.15

# ── Pesos de saída (S1-S4) ─────────────────────────────────────────────
W_NR_MARKER = 0.70
W_QUOTE_CLOSE_RESUME = 0.50
W_RESUME_SEQUENCE = 0.30
W_NEW_TRIGGER = 0.40

# ── Frases gatilho (E1) ────────────────────────────────────────────────
TRIGGER_PHRASES = [
    "passa a vigorar acrescido do seguinte",
    "passa a vigorar com a seguinte redação",
    "passam a vigorar com a seguinte redação",
    "fica acrescido do seguinte",
    "fica acrescida do seguinte",
    "dá-se a seguinte redação",
    "a seguinte redação ao",
    "passa a vigorar acrescido de",
    "com a redação dada por",
    "na redação da",
]

# ── Regexes de referência (seção 11.1) ─────────────────────────────────
REFERENCE_PATTERNS = [
    # "da Lei nº 13.105, de 16 de março de 2015 (Código de Processo Civil)"
    re.compile(r'da Lei n[ºo°]\s*([\d.]+),?\s*de\s+(.+?)\s*\((.+?)\)', re.IGNORECASE),
    # "da Lei nº 8.987, de 13 de fevereiro de 1995,"
    re.compile(r'da Lei n[ºo°]\s*([\d.]+),?\s*de\s+(.+?)[.,;]', re.IGNORECASE),
    # "do Decreto-Lei nº 2.848, de 7 de dezembro de 1940 (Código Penal)"
    re.compile(r'do Decreto-Lei n[ºo°]\s*([\d.]+),?\s*de\s+(.+?)\s*\((.+?)\)', re.IGNORECASE),
    # "do Decreto nº 10.024"
    re.compile(r'do Decreto n[ºo°]\s*([\d.]+)', re.IGNORECASE),
    # "da Lei Complementar nº 123"
    re.compile(r'da Lei Complementar n[ºo°]\s*([\d.]+)', re.IGNORECASE),
    # "da Medida Provisória nº 1.047"
    re.compile(r'da Medida Provis[óo]ria n[ºo°]\s*([\d.]+)', re.IGNORECASE),
]

# ── Nomes conhecidos (seção 11.2) ──────────────────────────────────────
KNOWN_REFERENCES: dict[str, dict] = {
    "2.848":  {"tipo": "DL",  "id": "DL-2848-1940",   "nome": "Código Penal"},
    "13.105": {"tipo": "LEI", "id": "LEI-13105-2015",  "nome": "Código de Processo Civil"},
    "8.987":  {"tipo": "LEI", "id": "LEI-8987-1995",   "nome": "Lei de Concessões"},
    "11.079": {"tipo": "LEI", "id": "LEI-11079-2004",  "nome": "Lei de PPPs"},
    "8.666":  {"tipo": "LEI", "id": "LEI-8666-1993",   "nome": "Lei de Licitações (revogada)"},
    "10.520": {"tipo": "LEI", "id": "LEI-10520-2002",  "nome": "Lei do Pregão (revogada)"},
    "12.462": {"tipo": "LEI", "id": "LEI-12462-2011",  "nome": "RDC"},
    "13.303": {"tipo": "LEI", "id": "LEI-13303-2016",  "nome": "Lei das Estatais"},
    "12.232": {"tipo": "LEI", "id": "LEI-12232-2010",  "nome": "Lei de Publicidade Institucional"},
    "11.107": {"tipo": "LEI", "id": "LEI-11107-2005",  "nome": "Lei dos Consórcios Públicos"},
    "10.406": {"tipo": "LEI", "id": "LEI-10406-2002",  "nome": "Código Civil"},
    "5.172":  {"tipo": "LEI", "id": "LEI-5172-1966",   "nome": "Código Tributário Nacional"},
    "9.784":  {"tipo": "LEI", "id": "LEI-9784-1999",   "nome": "Lei do Processo Administrativo"},
}

# ── Regex helpers ───────────────────────────────────────────────────────
_RE_ART_NUM = re.compile(r'Art\.\s*([\d.]+(?:-[A-Z])?)', re.IGNORECASE)
_RE_QUOTE_OPEN = re.compile(r'["\u201C]\s*[A-Z]')
_RE_CHAPTER_IN_QUOTES = re.compile(r'["\u201C](CAP[ÍI]TULO|Se[çc][ãa]o|T[ÍI]TULO|LIVRO|PARTE)', re.IGNORECASE)
_RE_ANNEX_HEADER = re.compile(r'^\s*ANEXO\s+[IVXLCDM\d]+', re.IGNORECASE | re.MULTILINE)
_RE_NR_MARKER = re.compile(r'["\u201D]\s*\(NR\)')
_RE_QUOTE_CLOSE = re.compile(r'["\u201D]\s*$', re.MULTILINE)
_RE_TARGET_NAME = re.compile(r'\(([^)]{3,60})\)')


# ── Dataclass ───────────────────────────────────────────────────────────

@dataclass
class ClassifierState:
    mode: str = "SELF"
    zone_target_id: str = ""
    zone_target_name: str = ""
    zone_entry_chunk_id: str = ""
    zone_host_article: str = ""
    zone_chunk_count: int = 0
    zone_enter_score: float = 0.0
    zone_reasons: list[str] = field(default_factory=list)


# ── Funções ─────────────────────────────────────────────────────────────

def get_context(canonical_text: str, start: int, end: int) -> tuple[str, str]:
    """Retorna janela de contexto (ctx_before, ctx_after) ao redor do chunk."""
    if start < 0 or end < 0:
        return ("", "")
    ctx_before = canonical_text[max(0, start - CTX_WINDOW):start]
    ctx_after = canonical_text[end:min(len(canonical_text), end + CTX_WINDOW)]
    return ctx_before, ctx_after


def _parse_article_number(text: str) -> float | None:
    """Extrai o número de artigo de um texto (ex: 'Art. 337-E' → 337.0)."""
    m = _RE_ART_NUM.search(text)
    if not m:
        return None
    raw = m.group(1)
    # Remove pontos de milhar e sufixo de letra (337-E → 337)
    num_str = raw.split("-")[0].replace(".", "")
    # Handle ordinal markers: "1º", "2º"
    num_str = num_str.replace("º", "").replace("°", "")
    try:
        return float(num_str)
    except ValueError:
        return None


def _extract_host_article_numbers(chunks: list, host_doc_id: str) -> float:
    """Encontra o maior artigo da lei hospedeira visto até agora (em modo SELF)."""
    max_num = 0.0
    for chunk in chunks:
        if getattr(chunk, "origin_type", "self") == "self":
            num = _parse_article_number(chunk.text)
            if num is not None and num > max_num:
                max_num = num
    return max_num


def compute_enter_score(
    text: str,
    ctx_before: str,
    ctx_after: str,
    host_doc_id: str,
    max_host_article: float = 0.0,
) -> tuple[float, list[str]]:
    """Calcula score de entrada com features E1-E7. Retorna (score, reasons)."""
    score = 0.0
    reasons: list[str] = []
    search_area = ctx_before + text

    # E1 — Trigger phrases
    search_lower = search_area.lower()
    for phrase in TRIGGER_PHRASES:
        if phrase in search_lower:
            score += W_TRIGGER_PHRASE
            reasons.append("trigger_phrase")
            break

    # E2 — Aspas abrindo
    quote_zone = ctx_before[-200:] + text[:200] if ctx_before else text[:200]
    if _RE_QUOTE_OPEN.search(quote_zone):
        score += W_QUOTE_OPEN
        reasons.append("quote_open")

    # E3 — Out-of-sequence
    art_num = _parse_article_number(text)
    if art_num is not None and max_host_article > 0:
        if art_num >= max_host_article * 2 or art_num >= max_host_article + 100:
            score += W_OUT_OF_SEQUENCE
            reasons.append("out_of_sequence")

    # E4 — Chapter in quotes
    if _RE_CHAPTER_IN_QUOTES.search(text):
        score += W_CHAPTER_IN_QUOTES
        reasons.append("chapter_in_quotes")

    # E5 — Target reference
    for pat in REFERENCE_PATTERNS:
        if pat.search(search_area):
            score += W_TARGET_REF
            reasons.append("target_ref")
            break

    # E6 — Target name (nome legível entre parênteses após referência)
    for pat in REFERENCE_PATTERNS:
        m = pat.search(search_area)
        if m and m.lastindex and m.lastindex >= 3:
            score += W_TARGET_NAME
            reasons.append("target_name")
            break

    # E7 — Annex header
    if _RE_ANNEX_HEADER.search(text):
        score += W_ANNEX_HEADER
        reasons.append("annex_header")

    return (score, reasons)


def compute_exit_score(
    text: str,
    ctx_before: str,
    ctx_after: str,
    host_doc_id: str,
) -> tuple[float, list[str]]:
    """Calcula score de saída com features S1-S4. Retorna (score, reasons)."""
    score = 0.0
    reasons: list[str] = []
    exit_zone = text + ctx_after[:200]

    # Detect complementary signals first (needed by S1 discount logic)
    has_quote_close = _RE_QUOTE_CLOSE.search(text) or _RE_QUOTE_CLOSE.search(ctx_after[:100])
    after_art = _parse_article_number(ctx_after[:400])
    after_lower = ctx_after[:400].lower()
    has_new_trigger = any(phrase in after_lower for phrase in TRIGGER_PHRASES)
    # Complementary = evidence that the transcription block truly ended:
    # host-law article resumption, OR a new trigger opening another zone.
    # Note: _RE_NR_MARKER already requires closing quotes in its pattern,
    # so quote_close is NOT an independent complement for NR.
    has_complementary = (after_art is not None) or has_new_trigger

    # S1 — NR marker (with discount if isolated)
    # When a law amends multiple devices of the same target in sequence,
    # each device may have its own (NR). Without a complementary signal
    # (host-law article resumption or new trigger), a lone (NR) should
    # NOT close the zone — the transcription block is still open.
    if _RE_NR_MARKER.search(exit_zone):
        if has_complementary:
            score += W_NR_MARKER
            reasons.append("nr_marker")
        else:
            score += W_NR_MARKER - 0.35  # 0.35 — below T_EXIT alone
            reasons.append("nr_marker_isolated")

    # S2 — Quote close + resume (only if S1 didn't already fire with full weight)
    if has_quote_close and "nr_marker" not in reasons:
        if after_art is not None:
            score += W_QUOTE_CLOSE_RESUME
            reasons.append("quote_close_resume")

    # S3 — Resume sequence (without quotes, only if S1/S2 didn't fire)
    if "quote_close_resume" not in reasons and "nr_marker" not in reasons:
        if after_art is not None:
            score += W_RESUME_SEQUENCE
            reasons.append("resume_sequence")

    # S4 — New trigger phrase (after_lower already computed above)
    for phrase in TRIGGER_PHRASES:
        if phrase in after_lower:
            score += W_NEW_TRIGGER
            reasons.append("new_trigger")
            break

    return (score, reasons)


def resolve_reference(
    text: str,
    ctx_before: str,
    ctx_after: str,
) -> tuple[str, str]:
    """Extrai referência normativa (ref_id, ref_name). Retorna ('', '') se não encontrar."""
    search_area = ctx_before + text

    for pat in REFERENCE_PATTERNS:
        m = pat.search(search_area)
        if not m:
            continue

        num = m.group(1)  # ex: "2.848"

        # Check KNOWN_REFERENCES
        if num in KNOWN_REFERENCES:
            known = KNOWN_REFERENCES[num]
            return (known["id"], known["nome"])

        # Build canonical ID from pattern
        # Determine tipo from pattern text
        pattern_str = pat.pattern
        if "Decreto-Lei" in pattern_str:
            tipo = "DL"
        elif "Lei Complementar" in pattern_str:
            tipo = "LC"
        elif "Medida Provis" in pattern_str:
            tipo = "MP"
        elif "Decreto" in pattern_str:
            tipo = "DEC"
        else:
            tipo = "LEI"

        # Clean number for ID: "13.105" → "13105"
        num_clean = num.replace(".", "")

        # Try to extract year from match (group 2 often contains date text)
        year = ""
        if m.lastindex and m.lastindex >= 2:
            date_text = m.group(2)
            year_match = re.search(r'(\d{4})', date_text)
            if year_match:
                year = year_match.group(1)

        ref_id = f"{tipo}-{num_clean}"
        if year:
            ref_id = f"{tipo}-{num_clean}-{year}"

        # Try to extract name from parentheses (group 3)
        ref_name = ""
        if m.lastindex and m.lastindex >= 3:
            ref_name = m.group(3)

        return (ref_id, ref_name)

    return ("", "")


def compute_confidence(state: "ClassifierState") -> str:
    """Calcula confidence baseada na riqueza de evidências da zona."""
    # TTL forced close always returns low
    if "ttl_forced_close" in state.zone_reasons:
        return "low"

    evidence_score = 0.0

    if state.zone_target_id:
        evidence_score += 0.4
    if state.zone_target_name:
        evidence_score += 0.2
    if state.zone_enter_score >= 0.80:
        evidence_score += 0.3
    elif state.zone_enter_score >= 0.60:
        evidence_score += 0.1
    if len(state.zone_reasons) >= 3:
        evidence_score += 0.1

    if evidence_score >= 0.7:
        return "high"
    elif evidence_score >= 0.4:
        return "medium"
    else:
        return "low"


def _format_reasons(reasons: list[str]) -> str:
    """Formata lista de reasons em string única, deduplicada."""
    seen: set[str] = set()
    unique: list[str] = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return " + ".join(unique)


def assign_origin(chunk: object, state: "ClassifierState", reasons: list[str]) -> None:
    """Preenche os 6 campos origin_* no chunk."""
    if state.mode == "EXTERNAL":
        chunk.origin_type = "external"
        chunk.is_external_material = True
        chunk.origin_reference = state.zone_target_id
        chunk.origin_reference_name = state.zone_target_name
        chunk.origin_confidence = compute_confidence(state)
        chunk.origin_reason = _format_reasons(state.zone_reasons + reasons)
    else:
        chunk.origin_type = "self"
        chunk.is_external_material = False
        chunk.origin_reference = ""
        chunk.origin_reference_name = ""
        chunk.origin_confidence = "high"
        chunk.origin_reason = ""


def classify_document(
    chunks: list,
    canonical_text: str,
    host_doc_id: str,
) -> list:
    """
    Classifica chunks por proveniência usando score + máquina de estados.

    Processa chunks em ordem canônica (canonical_start crescente).
    Retorna chunks com campos origin_* preenchidos.
    """
    if not chunks:
        return chunks

    state = ClassifierState()
    zones_detected: list[dict] = []
    forced_closes = 0

    # Ordena por posição canônica
    chunks_sorted = sorted(chunks, key=lambda c: getattr(c, "canonical_start", -1))

    # Calcula max artigo hospedeiro (para E3 out-of-sequence)
    max_host_article = 0.0

    for chunk in chunks_sorted:
        c_start = getattr(chunk, "canonical_start", -1)
        c_end = getattr(chunk, "canonical_end", -1)
        text = getattr(chunk, "text", "")
        chunk_id = getattr(chunk, "chunk_id", "")

        # Chunks sem offsets recebem default self
        if c_start < 0 or c_end < 0:
            assign_origin(chunk, ClassifierState(), [])
            continue

        ctx_before, ctx_after = get_context(canonical_text, c_start, c_end)

        # Track max host article in SELF mode
        if state.mode == "SELF":
            art_num = _parse_article_number(text)
            if art_num is not None and art_num > max_host_article:
                max_host_article = art_num

        # Calcular scores
        enter_score, enter_reasons = compute_enter_score(
            text, ctx_before, ctx_after, host_doc_id, max_host_article
        )
        exit_score, exit_reasons = compute_exit_score(
            text, ctx_before, ctx_after, host_doc_id
        )

        # Resolver referência (independente do estado)
        ref_id, ref_name = resolve_reference(text, ctx_before, ctx_after)

        # Transições
        if state.mode == "SELF":
            if enter_score >= T_ENTER:
                # ABRE zona
                state.mode = "EXTERNAL"
                state.zone_target_id = ref_id
                state.zone_target_name = ref_name
                state.zone_entry_chunk_id = chunk_id
                state.zone_chunk_count = 0
                state.zone_enter_score = enter_score
                state.zone_reasons = list(enter_reasons)

        elif state.mode == "EXTERNAL":
            state.zone_chunk_count += 1

            # Resolução tardia de referência
            if ref_id and not state.zone_target_id:
                state.zone_target_id = ref_id
                state.zone_target_name = ref_name

            if exit_score >= T_EXIT:
                # FECHA zona — este chunk é o último da zona
                zones_detected.append({
                    "entry_chunk": state.zone_entry_chunk_id,
                    "exit_chunk": chunk_id,
                    "target": state.zone_target_id,
                    "chunks": state.zone_chunk_count,
                    "close": "normal",
                })
                assign_origin(chunk, state, exit_reasons)
                state = ClassifierState()
                continue

            if state.zone_chunk_count >= TTL_CHUNKS:
                # FORCE CLOSE
                forced_closes += 1
                state.zone_reasons.append("ttl_forced_close")
                zones_detected.append({
                    "entry_chunk": state.zone_entry_chunk_id,
                    "exit_chunk": chunk_id,
                    "target": state.zone_target_id,
                    "chunks": state.zone_chunk_count,
                    "close": "ttl",
                })
                logger.warning(
                    f"[{host_doc_id}] OriginClassifier TTL forced close after "
                    f"{state.zone_chunk_count} chunks (entry={state.zone_entry_chunk_id})"
                )
                assign_origin(chunk, state, ["ttl_forced_close"])
                state = ClassifierState()
                continue

        # Atribuir campos
        assign_origin(
            chunk, state,
            enter_reasons if state.mode == "EXTERNAL" else [],
        )

    # Se terminou em zona aberta, forçar close
    if state.mode == "EXTERNAL":
        forced_closes += 1
        zones_detected.append({
            "entry_chunk": state.zone_entry_chunk_id,
            "exit_chunk": "(end-of-document)",
            "target": state.zone_target_id,
            "chunks": state.zone_chunk_count,
            "close": "eof",
        })
        logger.warning(
            f"[{host_doc_id}] OriginClassifier zone still open at end of document "
            f"(entry={state.zone_entry_chunk_id}, chunks={state.zone_chunk_count})"
        )

    # Log resumo
    external_count = sum(1 for c in chunks_sorted if getattr(c, "origin_type", "self") == "external")
    if zones_detected:
        logger.info(
            f"[{host_doc_id}] OriginClassifier: {len(zones_detected)} zones, "
            f"{external_count}/{len(chunks_sorted)} external, "
            f"{forced_closes} forced closes"
        )
    else:
        logger.debug(f"[{host_doc_id}] OriginClassifier: no external zones detected")

    return chunks_sorted
