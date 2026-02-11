"""
Camada 2 — Classificação Regex de blocos PyMuPDF em dispositivos legais.
Versão corrigida: ordem de prioridade DISPOSITIVOS > FILTROS.

Origem: app de testes PyMuPDF (validado com IN 65/2021 e IN 58/2022)
Integrado ao rag-gpu-server como src/extraction/regex_classifier.py
"""

import re
from dataclasses import dataclass, field

# ============================================================
# ClassifiedDevice — output para o pipeline
# ============================================================

@dataclass
class ClassifiedDevice:
    """Dispositivo legal classificado pelo regex, com offsets nativos do PyMuPDF."""
    device_type: str           # "article", "paragraph", "inciso", "alinea"
    span_id: str               # "ART-005", "PAR-005-1", "INC-005-1", "ALI-005-1-a"
    parent_span_id: str        # "" para artigos, "ART-005" para filhos
    children_span_ids: list    # ["PAR-005-1", "PAR-005-2"]
    text: str                  # texto completo do bloco
    text_preview: str          # primeiros 120 chars
    identifier: str            # "Art. 5º", "§ 1º", "I", "a"
    article_number: int        # 5 (extraído do span_id)
    hierarchy_depth: int       # 0=artigo, 1=§/inciso, 2=inciso sob §, 3=alínea
    char_start: int            # offset global no canonical_text (NATIVO)
    char_end: int              # offset global no canonical_text (NATIVO)
    page_number: int           # página (1-indexed)
    bbox: list = field(default_factory=list)  # [x0, y0, x1, y1] PDF points

# ============================================================
# Regex patterns
# ============================================================

RE_ARTICLE = re.compile(
    r"^\s*Art\.\s*(\d+(?:\.\d+)*)"
    r"[º°o]?"
    r"(-[A-Za-z]+)?"
    r"[\w-]*"
    r"[.\s]",
    re.IGNORECASE,
)

RE_PARAGRAPH = re.compile(
    r"^\s*("
    r"§\s*(\d+)[º°o]?\.?\s"
    r"|Par[aá]grafo\s+[uú]nico"
    r")",
    re.IGNORECASE,
)

ROMAN_NUMERALS = [
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
    "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX",
    "XXI", "XXII", "XXIII", "XXIV", "XXV", "XXVI", "XXVII", "XXVIII", "XXIX", "XXX",
    "XXXI", "XXXII", "XXXIII", "XXXIV", "XXXV", "XXXVI", "XXXVII", "XXXVIII", "XXXIX", "XL",
    "XLI", "XLII", "XLIII", "XLIV", "XLV", "XLVI", "XLVII", "XLVIII", "XLIX", "L",
    "LI", "LII", "LIII", "LIV", "LV", "LVI", "LVII", "LVIII", "LIX", "LX",
    "LXI", "LXII", "LXIII", "LXIV", "LXV", "LXVI", "LXVII", "LXVIII", "LXIX", "LXX",
    "LXXI", "LXXII", "LXXIII", "LXXIV", "LXXV", "LXXVI", "LXXVII", "LXXVIII", "LXXIX", "LXXX",
]
_romanos_sorted = sorted(ROMAN_NUMERALS, key=len, reverse=True)
_romanos_pattern = "|".join(_romanos_sorted)
RE_INCISO = re.compile(rf"^\s*({_romanos_pattern})\s*[-–—]\s?")

RE_ALINEA = re.compile(r"^\s*([a-z])\)\s")

RE_CHAPTER = re.compile(r"^\s*CAP[ÍI]TULO\s", re.IGNORECASE)

RE_ALL_CAPS = re.compile(r"^[A-ZÁÉÍÓÚÂÊÔÃÕÇ\s\-–—/,.:;()]+$")

RE_LEGAL_MARKER = re.compile(
    r"^\s*(Art\.|§|" + _romanos_pattern + r"\s*[-–—]|[a-z]\)|\d+\.)",
    re.IGNORECASE,
)

ROMAN_TO_INT = {r: i + 1 for i, r in enumerate(ROMAN_NUMERALS)}

# ============================================================
# Metadata detection
# ============================================================

METADATA_KEYWORDS = [
    "DIÁRIO OFICIAL DA UNIÃO",
    "Publicado em:",
    "Imprensa Nacional",
    "https://",
    "http://",
]
METADATA_FONTS = {"ArialMT", "Arial-BoldMT"}


def _get_first_span(block):
    lines = block.get("lines", [])
    if lines:
        spans = lines[0].get("spans", [])
        if spans:
            return spans[0]
    return None


def _is_metadata(block):
    text = block["text"]
    span = _get_first_span(block)
    if span and span.get("font", "") in METADATA_FONTS:
        return True
    for kw in METADATA_KEYWORDS:
        if kw in text:
            return True
    if re.match(r"^\d{2}/\d{2}/\d{4},?\s+\d{2}:\d{2}", text):
        return True
    return False


# ============================================================
# Preâmbulo detection
# ============================================================

PREAMBULO_PATTERNS = [
    re.compile(r"^\s*O\s+(SECRETÁRI[OA]|MINISTRO|PRESIDENTE|DIRETOR)", re.IGNORECASE),
    re.compile(r"^\s*(RESOLVE|CONSIDERANDO)", re.IGNORECASE),
    re.compile(r"no uso d[ea]s?\s+atribuiç", re.IGNORECASE),
    re.compile(r"resolve\s*:", re.IGNORECASE),
]

RE_NOME_NORMA = re.compile(
    r"^\s*(INSTRUÇÃO NORMATIVA|DECRETO|LEI\s+(COMPLEMENTAR\s+)?N[º°]|PORTARIA|RESOLUÇÃO)",
    re.IGNORECASE,
)

RE_EMENTA = re.compile(
    r"^\s*(Dispõe|Altera|Regulamenta|Estabelece|Institui)\s+",
    re.IGNORECASE,
)

RE_ORGAO = re.compile(r"^\s*[OÓ]rg[aã]o\s*:", re.IGNORECASE)


def _is_preambulo(text):
    for pat in PREAMBULO_PATTERNS:
        if pat.search(text):
            return True
    return False


# ============================================================
# Cabecalho / subtítulo detection
# ============================================================

def _is_cabecalho(block):
    text = block["text"].strip()
    span = _get_first_span(block)
    if RE_CHAPTER.match(text):
        return True
    if RE_ALL_CAPS.match(text) and len(text) < 120:
        return True
    if RE_LEGAL_MARKER.match(text):
        return False
    if len(text) < 80:
        if span and bool(span.get("flags", 0) & 16):
            return True
        if len(text) < 60 and text.count(".") <= 1 and text.count(",") <= 1:
            if text[0].isupper() and len(text.split()) <= 10:
                return True
    return False


# ============================================================
# Classificador principal
# ============================================================

def classify_block(block):
    """
    Classifica um bloco. Retorna (device_type, identifier, reason).
    ORDEM: 1.Metadata -> 2.Dispositivos -> 3.Filtros editoriais -> 4.Nao classificado
    """
    text = block["text"].strip()
    if not text:
        return "metadata", None, "Bloco vazio"

    # PASSO 1: Metadata do DOU
    if _is_metadata(block):
        return "metadata", None, "Font/keyword DOU"

    # PASSO 2: Dispositivos normativos (PRIORIDADE MAXIMA)
    m = RE_ARTICLE.match(text)
    if m:
        num_str = m.group(1)  # "337", "5", "1.048"
        suffix = m.group(2) or ""  # "-E" or ""
        num_int = int(num_str.replace(".", ""))
        if suffix:
            identifier = f"Art. {num_int}{suffix}"
        else:
            identifier = f"Art. {num_int}º"
        return "article", identifier, identifier

    m = RE_PARAGRAPH.match(text)
    if m:
        if m.group(2):
            num = int(m.group(2))
            return "paragraph", f"§ {num}º", f"§ {num}º"
        else:
            return "paragraph", "Parágrafo único", "Parágrafo único"

    m = RE_INCISO.match(text)
    if m:
        roman = m.group(1)
        return "inciso", roman, f"Inciso {roman}"

    m = RE_ALINEA.match(text)
    if m:
        letter = m.group(1)
        return "alinea", letter, f"Alínea {letter}"

    # PASSO 3: Filtros editoriais
    if RE_ORGAO.match(text):
        return "metadata", None, "Órgão emissor"
    if RE_NOME_NORMA.match(text):
        return "cabecalho", None, "Nome da norma"
    if _is_preambulo(text):
        return "preambulo", None, "Preâmbulo"
    if RE_EMENTA.match(text):
        return "preambulo", None, "Ementa"
    if _is_cabecalho(block):
        return "cabecalho", None, "Texto título/cabeçalho"

    # PASSO 4: Nao classificado
    return "nao_classificado", None, "Sem match"


# ============================================================
# Helpers para span_id
# ============================================================

def _extract_article_number(identifier):
    m = re.search(r"(\d+)", identifier or "")
    return int(m.group(1)) if m else 0

def _extract_article_parts(identifier):
    """Extrai numero e sufixo do identifier de artigo.
    'Art. 337-E' → (337, '-E')
    'Art. 5º' → (5, '')
    'Art. 1.048' → (1048, '')
    'Art. 6-A' → (6, '-A')
    """
    if not identifier:
        return 0, ""
    m = re.search(r"(\d+(?:\.\d+)*)(?:[º°o])?(-[A-Za-z]+)?", identifier)
    if not m:
        return 0, ""
    num = int(m.group(1).replace(".", ""))
    suffix = m.group(2) or ""
    return num, suffix

def _extract_paragraph_number(identifier):
    if identifier and ("único" in identifier.lower() or "unico" in identifier.lower()):
        return 0
    m = re.search(r"(\d+)", identifier or "")
    return int(m.group(1)) if m else 0

def _extract_article_number_from_span_id(span_id):
    """Extrai o numero do artigo de qualquer span_id: ART-005 -> 5, INC-003-2 -> 3"""
    m = re.match(r"[A-Z]+-(\d+)", span_id or "")
    return int(m.group(1)) if m else 0

def _build_span_id(device_type, identifier, parent_chain):
    if device_type == "article":
        num, suffix = _extract_article_parts(identifier)
        return f"ART-{num:03d}{suffix}"
    art_suffix = parent_chain.get("article_suffix", "")
    if device_type == "paragraph":
        art_num = parent_chain.get("article_num", 0)
        par_num = _extract_paragraph_number(identifier)
        return f"PAR-{art_num:03d}{art_suffix}-{par_num}"
    if device_type == "inciso":
        art_num = parent_chain.get("article_num", 0)
        par_num = parent_chain.get("paragraph_num", None)
        roman = identifier or ""
        inc_num = ROMAN_TO_INT.get(roman, 0)
        if par_num is not None:
            return f"INC-{art_num:03d}{art_suffix}-{par_num}-{inc_num}"
        else:
            return f"INC-{art_num:03d}{art_suffix}-{inc_num}"
    if device_type == "alinea":
        art_num = parent_chain.get("article_num", 0)
        par_num = parent_chain.get("paragraph_num", None)
        inc_num = parent_chain.get("inciso_num", None)
        letter = identifier or ""
        parts = [f"ALI-{art_num:03d}{art_suffix}"]
        if par_num is not None:
            parts.append(str(par_num))
        if inc_num is not None:
            parts.append(str(inc_num))
        parts.append(letter)
        return "-".join(parts)
    return None


# ============================================================
# Classificacao do documento inteiro
# ============================================================

def classify_document(pages):
    """
    Classifica todos os blocos do documento (3 passes).
    Input: lista de dicts de paginas com blocos PyMuPDF.
    Output: dict com 'devices', 'filtered', 'unclassified', 'stats'.
    """
    # Flatten
    all_blocks = []
    for page in pages:
        for block in page["blocks"]:
            all_blocks.append({**block, "page_number": page["page_number"]})
    all_blocks.sort(key=lambda b: b["char_start"])

    # Pass 1: classify
    classified = []
    for block in all_blocks:
        device_type, identifier, reason = classify_block(block)
        classified.append({
            "block": block, "device_type": device_type,
            "identifier": identifier, "reason": reason,
        })

    # Pass 2: hierarchy
    current_article = None
    current_paragraph = None
    current_inciso = None
    devices = []
    filtered = []
    unclassified = []

    for item in classified:
        block = item["block"]
        dtype = item["device_type"]
        ident = item["identifier"]
        reason = item["reason"]

        if dtype in ("metadata", "cabecalho", "preambulo"):
            filtered.append({
                "block_index": block["block_index"],
                "page_number": block["page_number"],
                "filter_type": dtype,
                "reason": reason,
                "text_preview": block["text"][:80],
            })
            continue

        if dtype == "nao_classificado":
            unclassified.append({
                "block_index": block["block_index"],
                "page_number": block["page_number"],
                "reason": reason,
                "text_preview": block["text"][:80],
            })
            continue

        parent_span_id = None
        parent_chain = {}
        hierarchy_depth = 0

        if dtype == "article":
            art_num, art_suffix = _extract_article_parts(ident)
            parent_chain = {"article_num": art_num, "article_suffix": art_suffix}
            current_article = {"span_id": None, "num": art_num, "suffix": art_suffix}
            current_paragraph = None
            current_inciso = None
            hierarchy_depth = 0
        elif dtype == "paragraph":
            par_num = _extract_paragraph_number(ident)
            if current_article:
                parent_span_id = current_article["span_id"]
                parent_chain = {"article_num": current_article["num"], "article_suffix": current_article.get("suffix", ""), "paragraph_num": par_num}
            else:
                parent_chain = {"article_num": 0, "article_suffix": "", "paragraph_num": par_num}
            current_paragraph = {"span_id": None, "num": par_num}
            current_inciso = None
            hierarchy_depth = 1
        elif dtype == "inciso":
            roman = ident or ""
            inc_num = ROMAN_TO_INT.get(roman, 0)
            if current_paragraph:
                parent_span_id = current_paragraph["span_id"]
                parent_chain = {
                    "article_num": current_article["num"] if current_article else 0,
                    "article_suffix": current_article.get("suffix", "") if current_article else "",
                    "paragraph_num": current_paragraph["num"],
                    "inciso_num": inc_num,
                }
                hierarchy_depth = 2
            elif current_article:
                parent_span_id = current_article["span_id"]
                parent_chain = {"article_num": current_article["num"], "article_suffix": current_article.get("suffix", ""), "inciso_num": inc_num}
                hierarchy_depth = 1
            else:
                parent_chain = {"article_num": 0, "article_suffix": "", "inciso_num": inc_num}
            current_inciso = {"span_id": None, "num": inc_num}
        elif dtype == "alinea":
            if current_inciso:
                parent_span_id = current_inciso["span_id"]
                parent_chain = {
                    "article_num": current_article["num"] if current_article else 0,
                    "article_suffix": current_article.get("suffix", "") if current_article else "",
                    "paragraph_num": current_paragraph["num"] if current_paragraph else None,
                    "inciso_num": current_inciso["num"],
                }
                hierarchy_depth = 3 if current_paragraph else 2
            else:
                parent_chain = {"article_num": current_article["num"] if current_article else 0, "article_suffix": current_article.get("suffix", "") if current_article else ""}
                hierarchy_depth = 1

        span_id = _build_span_id(dtype, ident, parent_chain)

        if dtype == "article" and current_article:
            current_article["span_id"] = span_id
        elif dtype == "paragraph" and current_paragraph:
            current_paragraph["span_id"] = span_id
        elif dtype == "inciso" and current_inciso:
            current_inciso["span_id"] = span_id

        devices.append({
            "block_index": block["block_index"],
            "page_number": block["page_number"],
            "device_type": dtype,
            "identifier": ident,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "hierarchy_depth": hierarchy_depth,
            "text_preview": block["text"][:120],
            "full_text": block["text"],
            "char_start": block["char_start"],
            "char_end": block["char_end"],
            "bbox": block["bbox"],
            "children_span_ids": [],
        })

    # Pass 3: children
    span_id_map = {d["span_id"]: d for d in devices}
    for device in devices:
        parent_id = device["parent_span_id"]
        if parent_id and parent_id in span_id_map:
            span_id_map[parent_id]["children_span_ids"].append(device["span_id"])

    # Stats
    by_device_type = {}
    for d in devices:
        by_device_type[d["device_type"]] = by_device_type.get(d["device_type"], 0) + 1
    by_filter_type = {}
    for f in filtered:
        by_filter_type[f["filter_type"]] = by_filter_type.get(f["filter_type"], 0) + 1

    return {
        "devices": devices,
        "filtered": filtered,
        "unclassified": unclassified,
        "stats": {
            "total_blocks": len(all_blocks),
            "devices": len(devices),
            "filtered": len(filtered),
            "unclassified": len(unclassified),
            "by_device_type": by_device_type,
            "by_filter_type": by_filter_type,
            "max_hierarchy_depth": max((d["hierarchy_depth"] for d in devices), default=0),
        },
    }


# ============================================================
# Interface para o pipeline de producao
# ============================================================

def classify_to_devices(pages_data) -> list[ClassifiedDevice]:
    """
    Interface principal para o pipeline.py.
    Chama classify_document() e converte para List[ClassifiedDevice].
    """
    result = classify_document(pages_data)
    devices = []
    for d in result["devices"]:
        devices.append(ClassifiedDevice(
            device_type=d["device_type"],
            span_id=d["span_id"],
            parent_span_id=d["parent_span_id"] or "",
            children_span_ids=d.get("children_span_ids", []),
            text=d["full_text"],
            text_preview=d["text_preview"],
            identifier=d["identifier"] or "",
            article_number=_extract_article_number_from_span_id(d["span_id"]),
            hierarchy_depth=d["hierarchy_depth"],
            char_start=d["char_start"],
            char_end=d["char_end"],
            page_number=d["page_number"],
            bbox=d["bbox"],
        ))
    return devices
