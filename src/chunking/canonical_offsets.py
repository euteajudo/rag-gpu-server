# -*- coding: utf-8 -*-
"""
Canonical Offsets - Extração de offsets do ParsedDocument (PR13).

Este módulo extrai os offsets (start_pos, end_pos) de cada span
do ParsedDocument, permitindo slicing puro sem find().

Princípio PR13:
==============
    Quando canonical_hash == hash_atual E start/end >= 0:
        → usa slicing puro: canonical_text[start:end]
    Caso contrário:
        → fallback best-effort via find()

Uso:
====
    from chunking.canonical_offsets import extract_offsets_from_parsed_doc

    # Após SpanParser
    parsed_doc = span_parser.parse(markdown)

    # Extrai offsets
    offsets_map, canonical_hash = extract_offsets_from_parsed_doc(parsed_doc)

    # Passa para ChunkMaterializer
    materializer = ChunkMaterializer(
        document_id=...,
        offsets_map=offsets_map,
        canonical_hash=canonical_hash,
    )
"""

import hashlib
import logging
import unicodedata
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


def normalize_canonical_text(text: str) -> str:
    """
    Normaliza texto canônico para garantir determinismo.

    Regras aplicadas (em ordem):
    1. Unicode NFC normalization
    2. Normaliza line endings para LF (\\n)
    3. Remove trailing whitespace de cada linha
    4. Garante exatamente um \\n no final

    IMPORTANTE: Esta função DEVE ser usada sempre que o canonical
    é construído ou comparado, para garantir byte-a-byte igualdade.

    Args:
        text: Texto a normalizar

    Returns:
        Texto normalizado (determinístico)
    """
    if not text:
        return ""

    # 1. Unicode NFC (Canonical Decomposition, followed by Canonical Composition)
    text = unicodedata.normalize("NFC", text)

    # 2. Normaliza line endings: CRLF -> LF, CR -> LF
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Remove trailing whitespace de cada linha
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    # 4. Garante exatamente um \n no final (se houver conteúdo)
    text = text.rstrip("\n")
    if text:
        text += "\n"

    return text


def compute_canonical_hash(canonical_text: str) -> str:
    """
    Computa hash SHA256 do texto canônico.

    O hash é usado para detectar mismatch entre offsets armazenados
    e o canonical atual (anti-mismatch).

    Args:
        canonical_text: Texto canônico (já normalizado)

    Returns:
        Hash SHA256 como string hexadecimal (64 chars)
    """
    return hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()


def extract_offsets_from_parsed_doc(
    parsed_doc,  # ParsedDocument
) -> Tuple[Dict[str, Tuple[int, int]], str]:
    """
    Extrai offsets de todos os spans de um ParsedDocument.

    Cada span do ParsedDocument tem start_pos e end_pos que indicam
    a posição no texto fonte. Esta função coleta esses offsets
    e computa o hash do source_text para validação anti-mismatch.

    Args:
        parsed_doc: ParsedDocument com spans parseados

    Returns:
        Tupla (offsets_map, canonical_hash) onde:
        - offsets_map: dict[span_id, (start_pos, end_pos)]
        - canonical_hash: SHA256 do source_text normalizado
    """
    offsets_map: Dict[str, Tuple[int, int]] = {}

    # Extrai offsets de cada span
    for span in parsed_doc.spans:
        # Verifica se o span tem offsets válidos
        # NOTA: Não usar "or -1" pois start_pos=0 é válido (início do documento)
        start = getattr(span, 'start_pos', -1)
        if start is None:
            start = -1
        end = getattr(span, 'end_pos', -1)
        if end is None:
            end = -1

        if start >= 0 and end > start:
            offsets_map[span.span_id] = (start, end)
        else:
            # Log span sem offsets válidos (não crítico)
            logger.debug(f"Span {span.span_id} sem offsets válidos: start={start}, end={end}")

    # Computa hash do source_text normalizado
    source_text = getattr(parsed_doc, 'source_text', '') or ''
    normalized_text = normalize_canonical_text(source_text)
    canonical_hash = compute_canonical_hash(normalized_text) if normalized_text else ""

    logger.info(
        f"Extraídos offsets de {len(offsets_map)} spans. "
        f"Hash: {canonical_hash[:16]}..."
    )

    return offsets_map, canonical_hash


def validate_offsets_hash(
    stored_hash: str,
    current_canonical_text: str,
) -> bool:
    """
    Valida se o hash armazenado confere com o texto canônico atual.

    Se o hash não confere, significa que o canonical_text mudou
    e os offsets armazenados são inválidos (devem usar fallback find()).

    Args:
        stored_hash: Hash armazenado no chunk
        current_canonical_text: Texto canônico atual (será normalizado)

    Returns:
        True se hash confere (offsets válidos), False caso contrário
    """
    if not stored_hash:
        return False

    normalized = normalize_canonical_text(current_canonical_text)
    current_hash = compute_canonical_hash(normalized)

    return stored_hash == current_hash


def extract_snippet_by_offsets(
    canonical_text: str,
    start: int,
    end: int,
    stored_hash: str,
) -> Tuple[str, bool]:
    """
    Extrai snippet usando offsets (zero fallback find).

    Esta é a função principal do PR13: usa slicing puro quando
    os offsets são válidos e o hash confere.

    Args:
        canonical_text: Texto canônico completo
        start: Offset início
        end: Offset fim
        stored_hash: Hash armazenado para validação

    Returns:
        Tupla (snippet, used_offsets) onde:
        - snippet: Texto extraído
        - used_offsets: True se usou slicing puro, False se fallback
    """
    # Verifica se pode usar slicing puro
    if start >= 0 and end > start and stored_hash:
        if validate_offsets_hash(stored_hash, canonical_text):
            # PR13: slicing puro (zero find)
            snippet = canonical_text[start:end]
            return snippet, True
        else:
            logger.warning(
                f"Hash mismatch: offsets inválidos. "
                f"stored_hash={stored_hash[:16]}..."
            )

    # Fallback: não pode usar offsets
    return "", False


# =============================================================================
# PR13 STRICT: Resolução determinística de offsets para filhos
# =============================================================================

class OffsetResolutionError(Exception):
    """Erro na resolução de offsets (não encontrado ou ambíguo)."""

    def __init__(
        self,
        message: str,
        document_id: str = "",
        span_id: str = "",
        device_type: str = "",
        reason: str = "",
    ):
        self.document_id = document_id
        self.span_id = span_id
        self.device_type = device_type
        self.reason = reason
        super().__init__(message)

    def __str__(self):
        return (
            f"OffsetResolutionError: {self.args[0]} "
            f"[document_id={self.document_id}, span_id={self.span_id}, "
            f"device_type={self.device_type}, reason={self.reason}]"
        )


def resolve_child_offsets(
    canonical_text: str,
    parent_start: int,
    parent_end: int,
    chunk_text: str,
    document_id: str = "",
    span_id: str = "",
    device_type: str = "",
) -> tuple[int, int]:
    """
    Resolve offsets de um chunk filho dentro do range do pai.

    Busca determinística: chunk_text DEVE aparecer exatamente UMA VEZ
    dentro do range [parent_start:parent_end] do canonical_text.

    Args:
        canonical_text: Texto canônico completo (normalizado)
        parent_start: Offset início do pai no canonical_text
        parent_end: Offset fim do pai no canonical_text
        chunk_text: Texto do chunk filho a localizar
        document_id: ID do documento (para logs)
        span_id: ID do span (para logs)
        device_type: Tipo do dispositivo (para logs)

    Returns:
        Tupla (absolute_start, absolute_end) com offsets absolutos

    Raises:
        OffsetResolutionError: Se chunk_text não encontrado ou ambíguo
    """
    if not chunk_text or not chunk_text.strip():
        raise OffsetResolutionError(
            f"chunk_text vazio para {span_id}",
            document_id=document_id,
            span_id=span_id,
            device_type=device_type,
            reason="EMPTY_TEXT",
        )

    # Valida range do pai
    if parent_start < 0 or parent_end <= parent_start:
        raise OffsetResolutionError(
            f"Range do pai inválido: [{parent_start}:{parent_end}]",
            document_id=document_id,
            span_id=span_id,
            device_type=device_type,
            reason="INVALID_PARENT_RANGE",
        )

    # Extrai texto do pai
    parent_text = canonical_text[parent_start:parent_end]

    # Normaliza chunk_text para busca (remove whitespace extra nas bordas)
    search_text = chunk_text.strip()

    # Busca todas as ocorrências dentro do pai
    occurrences = []
    search_start = 0
    while True:
        pos = parent_text.find(search_text, search_start)
        if pos == -1:
            break
        occurrences.append(pos)
        search_start = pos + 1

    # Validação: exatamente UMA ocorrência
    if len(occurrences) == 0:
        # Tenta busca com texto simplificado (sem múltiplos espaços)
        simplified_search = " ".join(search_text.split())
        simplified_parent = " ".join(parent_text.split())

        if simplified_search in simplified_parent:
            reason = "NOT_FOUND_WHITESPACE_MISMATCH"
            hint = "Texto existe mas com whitespace diferente"
        else:
            reason = "NOT_FOUND"
            hint = "Texto não existe no range do pai"

        # Log detalhado para debug
        logger.error(
            f"Offset NOT_FOUND debug: span_id={span_id}, "
            f"parent_range=[{parent_start}:{parent_end}] ({parent_end - parent_start} chars), "
            f"search_text[:80]={repr(search_text[:80])}"
        )
        # Tenta encontrar substring similar
        if len(search_text) > 20:
            prefix = search_text[:20]
            if prefix in parent_text:
                pos = parent_text.find(prefix)
                logger.error(
                    f"  HINT: Prefixo '{prefix}' encontrado em pos={pos}. "
                    f"Contexto: ...{repr(parent_text[max(0,pos-10):pos+50])}..."
                )
            else:
                logger.error(f"  HINT: Prefixo '{prefix}' NÃO encontrado no parent_text")

        raise OffsetResolutionError(
            f"Chunk '{span_id}' não encontrado no range do pai. {hint}. "
            f"chunk_text[0:50]='{search_text[:50]}...'",
            document_id=document_id,
            span_id=span_id,
            device_type=device_type,
            reason=reason,
        )

    if len(occurrences) > 1:
        raise OffsetResolutionError(
            f"Chunk '{span_id}' é AMBÍGUO: {len(occurrences)} ocorrências no range do pai. "
            f"chunk_text[0:50]='{search_text[:50]}...'",
            document_id=document_id,
            span_id=span_id,
            device_type=device_type,
            reason="AMBIGUOUS_MULTIPLE_MATCHES",
        )

    # Exatamente uma ocorrência: calcula offsets absolutos
    relative_start = occurrences[0]
    absolute_start = parent_start + relative_start
    absolute_end = absolute_start + len(search_text)

    logger.debug(
        f"Offset resolvido: {span_id} [{absolute_start}:{absolute_end}] "
        f"(relativo ao pai: {relative_start})"
    )

    return absolute_start, absolute_end


def resolve_offsets_recursive(
    canonical_text: str,
    canonical_hash: str,
    article_text: str,
    article_start: int,
    article_end: int,
    children: list[dict],
    document_id: str = "",
) -> dict[str, tuple[int, int, str]]:
    """
    Resolve offsets para todos os filhos de um artigo recursivamente.

    Esta função resolve offsets para parágrafos, incisos e alíneas,
    garantindo que cada um seja encontrado exatamente uma vez dentro
    do range do seu pai.

    Args:
        canonical_text: Texto canônico completo (normalizado)
        canonical_hash: Hash SHA256 do canonical_text
        article_text: Texto do artigo (para validação)
        article_start: Offset início do artigo
        article_end: Offset fim do artigo
        children: Lista de dicts com {span_id, device_type, text, parent_span_id}
        document_id: ID do documento (para logs)

    Returns:
        Dict span_id -> (start, end, hash) para todos os filhos

    Raises:
        OffsetResolutionError: Se qualquer filho não puder ser resolvido
    """
    offsets_map: dict[str, tuple[int, int, str]] = {}

    # Primeiro, adiciona o artigo
    offsets_map[f"article_root"] = (article_start, article_end, canonical_hash)

    # Organiza filhos por parent
    children_by_parent: dict[str, list[dict]] = {}
    for child in children:
        parent_id = child.get("parent_span_id", "article_root")
        if parent_id not in children_by_parent:
            children_by_parent[parent_id] = []
        children_by_parent[parent_id].append(child)

    # Processa em ordem: primeiro parágrafos (diretos do artigo),
    # depois incisos (podem estar sob parágrafos ou artigo),
    # depois alíneas (sob incisos)
    def process_children(parent_id: str, parent_start: int, parent_end: int):
        if parent_id not in children_by_parent:
            return

        for child in children_by_parent[parent_id]:
            span_id = child["span_id"]
            device_type = child["device_type"]
            text = child["text"]

            # Resolve offset do filho dentro do pai
            child_start, child_end = resolve_child_offsets(
                canonical_text=canonical_text,
                parent_start=parent_start,
                parent_end=parent_end,
                chunk_text=text,
                document_id=document_id,
                span_id=span_id,
                device_type=device_type,
            )

            offsets_map[span_id] = (child_start, child_end, canonical_hash)

            # Processa filhos deste filho recursivamente
            process_children(span_id, child_start, child_end)

    # Inicia processamento a partir do artigo
    process_children("article_root", article_start, article_end)

    return offsets_map
