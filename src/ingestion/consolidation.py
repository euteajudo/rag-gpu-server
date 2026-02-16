"""
Dual Ingestion: consolida artigos com filhos em chunks @FULL.

Regras:
- Artigo com >= MIN_CHILDREN filhos diretos -> consolida
- Texto = caput + \n + filho1 + \n + filho2 + ...
- Se texto > MAX_CHARS -> split em @FULL-P1, @FULL-P2 com caput repetido
- Chunks @FULL tem device_type="article_consolidated", is_consolidated=True
"""

import logging

from .models import IngestRequest, ProcessedChunk

logger = logging.getLogger(__name__)

MIN_CHILDREN = 3          # Minimo de filhos diretos para consolidar
MAX_CHARS = 2000          # ~500 tokens (4 chars/token)
OVERLAP_CHILDREN = 1      # Filhos de overlap entre partes


def should_consolidate(article_node_id: str, all_chunks: list[ProcessedChunk]) -> list[ProcessedChunk]:
    """
    Retorna filhos diretos do artigo se count >= MIN_CHILDREN.

    Args:
        article_node_id: node_id do artigo (ex: leis:LEI-14133-2021#ART-033)
        all_chunks: Todos os chunks do documento

    Returns:
        Lista de filhos diretos se elegivel, lista vazia caso contrario
    """
    children = [
        c for c in all_chunks
        if c.parent_node_id == article_node_id
    ]
    if len(children) >= MIN_CHILDREN:
        return children
    return []


def _sort_children(children: list[ProcessedChunk]) -> list[ProcessedChunk]:
    """Ordena filhos por span_id para preservar ordem estrutural."""
    return sorted(children, key=lambda c: c.span_id)


def build_consolidated_text(
    article_chunk: ProcessedChunk,
    children_sorted: list[ProcessedChunk],
) -> str:
    """
    Concatena caput + filhos em texto unico.

    Args:
        article_chunk: Chunk do artigo (caput)
        children_sorted: Filhos ordenados por span_id

    Returns:
        Texto consolidado
    """
    parts = [article_chunk.text]
    for child in children_sorted:
        if child.text:
            parts.append(child.text)
    return "\n".join(parts)


def compute_child_offsets(
    consolidated_text: str,
    children_sorted: list[ProcessedChunk],
) -> list[dict]:
    """
    Calcula offsets de cada filho dentro do texto consolidado.

    Returns:
        Lista de dicts: [{"node_id": "...", "start": N, "end": M}, ...]
    """
    offsets = []
    search_start = 0
    for child in children_sorted:
        if not child.text:
            continue
        idx = consolidated_text.find(child.text, search_start)
        if idx >= 0:
            offsets.append({
                "node_id": child.node_id,
                "start": idx,
                "end": idx + len(child.text),
            })
            search_start = idx + len(child.text)
    return offsets


def split_consolidated(
    consolidated_text: str,
    children_sorted: list[ProcessedChunk],
    article_chunk: ProcessedChunk,
) -> list[dict]:
    """
    Split texto consolidado se > MAX_CHARS.

    Cada parte comeca com o caput. Split por filho inteiro (nunca corta no meio).
    Overlap: ultimo filho da parte N = primeiro filho da parte N+1.

    Returns:
        Lista de dicts: [{"text": "...", "part_index": 1, "part_total": N, "children": [...]}, ...]
    """
    if len(consolidated_text) <= MAX_CHARS:
        return [{
            "text": consolidated_text,
            "part_index": 1,
            "part_total": 1,
            "children": children_sorted,
        }]

    caput = article_chunk.text
    parts = []
    current_children: list[ProcessedChunk] = []
    current_text = caput

    for i, child in enumerate(children_sorted):
        candidate = current_text + "\n" + child.text

        if len(candidate) > MAX_CHARS and current_children:
            # Fecha parte atual
            parts.append({
                "text": current_text,
                "children": list(current_children),
            })
            # Nova parte comeca com caput + overlap
            current_text = caput
            current_children = []
            # Overlap: readiciona ultimo(s) filho(s) da parte anterior
            overlap_start = max(0, i - OVERLAP_CHILDREN)
            for j in range(overlap_start, i):
                overlap_child = children_sorted[j]
                current_text += "\n" + overlap_child.text
                current_children.append(overlap_child)

        current_text += "\n" + child.text
        current_children.append(child)

    # Ultima parte
    if current_children:
        parts.append({
            "text": current_text,
            "children": list(current_children),
        })

    # Numera partes
    total = len(parts)
    for idx, part in enumerate(parts):
        part["part_index"] = idx + 1
        part["part_total"] = total

    return parts


def generate_consolidated_chunks(
    chunks: list[ProcessedChunk],
    request: IngestRequest,
) -> list[ProcessedChunk]:
    """
    Funcao principal: gera chunks consolidados @FULL para artigos elegiveis.

    NÃO altera chunks granulares existentes.

    Args:
        chunks: Lista de todos os chunks do documento
        request: Request de ingestão

    Returns:
        Lista de chunks @FULL (novos, para serem adicionados)
    """
    consolidated_chunks: list[ProcessedChunk] = []

    # Filtra artigos
    articles = [c for c in chunks if c.device_type == "article"]

    for article in articles:
        children = should_consolidate(article.node_id, chunks)
        if not children:
            continue

        children_sorted = _sort_children(children)

        # Build texto consolidado
        full_text = build_consolidated_text(article, children_sorted)

        # Split se necessario
        parts = split_consolidated(full_text, children_sorted, article)

        for part in parts:
            part_text = part["text"]
            part_children = part["children"]
            part_index = part["part_index"]
            part_total = part["part_total"]

            # Node ID
            if part_total == 1:
                suffix = "@FULL"
            else:
                suffix = f"@FULL-P{part_index}"

            chunk_id = f"{request.document_id}#{article.span_id}{suffix}"
            node_id = f"leis:{chunk_id}"

            child_node_ids = [c.node_id for c in part_children]
            child_offsets = compute_child_offsets(part_text, part_children)

            pc = ProcessedChunk(
                node_id=node_id,
                chunk_id=chunk_id,
                parent_node_id="",  # article level
                span_id=article.span_id,
                device_type="article_consolidated",
                chunk_level="article",
                text=part_text,
                parent_text="",
                retrieval_text=part_text,  # ja e concatenacao completa
                document_id=request.document_id,
                tipo_documento=request.tipo_documento,
                numero=request.numero,
                ano=request.ano,
                article_number=article.article_number,
                citations=[],  # @FULL nao tem citations proprias
                # Offsets nao mapeiam 1:1 para canonical
                canonical_start=-1,
                canonical_end=-1,
                canonical_hash="",
                # Coordenadas do artigo-pai
                page_number=article.page_number,
                bbox=article.bbox,
                bbox_img=[],
                img_width=0,
                img_height=0,
                confidence=article.confidence,
                is_cross_page=False,
                bbox_spans=[],
                # Split
                part_index=part_index,
                part_total=part_total,
                # Dual Ingestion
                is_consolidated=True,
                child_node_ids=child_node_ids,
                child_offsets=child_offsets,
                # Origin: herda do artigo
                origin_type=article.origin_type,
                origin_confidence=article.origin_confidence,
                origin_reference=article.origin_reference,
                origin_reference_name=article.origin_reference_name,
                is_external_material=article.is_external_material,
                origin_reason=article.origin_reason,
                # Acordao fields (vazios para LEIs)
                colegiado=article.colegiado,
                processo=article.processo,
                relator=article.relator,
                data_sessao=article.data_sessao,
                unidade_tecnica=article.unidade_tecnica,
            )

            consolidated_chunks.append(pc)

    if consolidated_chunks:
        logger.info(
            f"[{request.document_id}] Dual Ingestion: "
            f"{len(consolidated_chunks)} chunks @FULL gerados "
            f"de {len(articles)} artigos"
        )

    return consolidated_chunks
