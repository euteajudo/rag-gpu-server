"""
AcordaoChunker — Chunking por seções com overlap para acórdãos do TCU.

Converte a lista detalhada de AcordaoDevice (~103 devices) em ~15-25 chunks
baseados em seções (EMENTA, RELATÓRIO, VOTO, ACÓRDÃO) com 20% de overlap
entre partes consecutivas de uma mesma seção.

Elimina o problema de PKs duplicadas no Milvus quando o VOTO transcreve
parágrafos de acórdãos citados.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class ParsedSection:
    """Seção extraída do acórdão, pronta para chunking."""
    section_type: str          # "ementa", "relatorio", "voto", "acordao"
    text: str                  # canonical_text[canonical_start:canonical_end]
    canonical_start: int
    canonical_end: int
    paragraphs: list = field(default_factory=list)
    # Each paragraph: {"num": str, "text": str, "start": int, "end": int}
    # start/end are absolute offsets in canonical_text


@dataclass
class AcordaoChunk:
    """Chunk de seção com overlap, pronto para conversão em ProcessedChunk."""
    span_id: str               # "SEC-EMENTA", "SEC-VOTO-P01", "SEC-RELATORIO-P03"
    section_type: str          # "ementa", "relatorio", "voto", "acordao"
    authority_level: str       # "metadado", "opinativo", "fundamentacao", "vinculante"
    text: str                  # texto do chunk
    retrieval_text: str        # [CONTEXTO: ...] + text
    canonical_start: int       # offset absoluto no canonical_text
    canonical_end: int         # offset absoluto
    page_number: int           # estimativa
    part_number: int           # 1-indexed
    total_parts: int           # total de partes desta seção
    section_path: str          # "RELATÓRIO", "VOTO", "ACÓRDÃO", "EMENTA"


# =============================================================================
# Constants
# =============================================================================

AUTHORITY_MAP = {
    "ementa": "metadado",
    "relatorio": "opinativo",
    "voto": "fundamentacao",
    "acordao": "vinculante",
}

SECTION_PATH_MAP = {
    "ementa": "EMENTA",
    "relatorio": "RELATÓRIO",
    "voto": "VOTO",
    "acordao": "ACÓRDÃO",
}

SPAN_PREFIX_MAP = {
    "ementa": "SEC-EMENTA",
    "relatorio": "SEC-RELATORIO",
    "voto": "SEC-VOTO",
    "acordao": "SEC-ACORDAO",
}

# Regex to find SUMÁRIO in canonical text
RE_SUMARIO = re.compile(
    r'SUM[AÁ]RIO\s*:\s*(.+?)(?:\n\n|\n(?=RELAT[OÓ]RIO))',
    re.DOTALL | re.IGNORECASE,
)


# =============================================================================
# build_sections — AcordaoDevice list → ParsedSection list
# =============================================================================

def build_sections(
    devices: list,
    canonical_text: str,
    header_metadata: dict,
) -> List[ParsedSection]:
    """
    Converte List[AcordaoDevice] → List[ParsedSection].

    1. EMENTA: from SUMÁRIO in canonical_text (pre-RELATÓRIO)
    2. RELATÓRIO: SEC-RELATORIO device, paragraphs from PAR-RELATORIO-* devices
    3. VOTO: SEC-VOTO device, paragraphs from PAR-VOTO-* devices
    4. ACÓRDÃO: SEC-ACORDAO device, paragraphs from ITEM-* devices
    """
    sections: List[ParsedSection] = []

    # Index devices by span_id prefix
    device_by_span = {d.span_id: d for d in devices}

    # 1. EMENTA — try to extract from canonical_text before first primary section
    ementa_section = _build_ementa_section(
        canonical_text, devices, header_metadata,
    )
    if ementa_section:
        sections.append(ementa_section)

    # 2-4. Primary sections: RELATÓRIO, VOTO, ACÓRDÃO
    for sec_span_id, sec_type, para_prefix in [
        ("SEC-RELATORIO", "relatorio", "PAR-RELATORIO"),
        ("SEC-VOTO", "voto", "PAR-VOTO"),
        ("SEC-ACORDAO", "acordao", "ITEM"),
    ]:
        if sec_span_id not in device_by_span:
            continue

        sec_device = device_by_span[sec_span_id]
        text = sec_device.text or ""
        if not text:
            continue

        # Collect child paragraphs/items
        paragraphs = []
        for d in devices:
            if d.span_id.startswith(para_prefix + "-") and d.device_type in ("paragraph", "item_dispositivo"):
                paragraphs.append({
                    "num": d.identifier,
                    "text": d.text or "",
                    "start": d.char_start,
                    "end": d.char_end,
                })

        # Sort paragraphs by offset
        paragraphs.sort(key=lambda p: p["start"])

        sections.append(ParsedSection(
            section_type=sec_type,
            text=text,
            canonical_start=sec_device.char_start,
            canonical_end=sec_device.char_end,
            paragraphs=paragraphs,
        ))

    logger.info(
        f"build_sections: {len(sections)} seções "
        f"({', '.join(s.section_type for s in sections)})"
    )
    return sections


def _build_ementa_section(
    canonical_text: str,
    devices: list,
    header_metadata: dict,
) -> Optional[ParsedSection]:
    """Extract EMENTA/SUMÁRIO section from canonical_text."""
    # Find the earliest primary section to limit search area
    first_section_start = len(canonical_text)
    for d in devices:
        if d.span_id in ("SEC-RELATORIO", "SEC-VOTO", "SEC-ACORDAO"):
            first_section_start = min(first_section_start, d.char_start)

    search_area = canonical_text[:first_section_start]
    m = RE_SUMARIO.search(search_area)
    if m:
        # Include the "SUMÁRIO:" label in the text
        text = canonical_text[m.start():m.end()].strip()
        return ParsedSection(
            section_type="ementa",
            text=text,
            canonical_start=m.start(),
            canonical_end=m.start() + len(text),
            paragraphs=[],
        )

    # Fallback: use sumario from header_metadata
    sumario = header_metadata.get("sumario", "")
    if sumario:
        idx = canonical_text.find(sumario[:60])  # Find by prefix
        if idx >= 0:
            end = idx + len(sumario)
            actual_text = canonical_text[idx:end]
            return ParsedSection(
                section_type="ementa",
                text=actual_text,
                canonical_start=idx,
                canonical_end=end,
                paragraphs=[],
            )

    return None


# =============================================================================
# AcordaoChunker
# =============================================================================

class AcordaoChunker:
    """Divide seções de acórdão em chunks com overlap por parágrafos."""

    def __init__(
        self,
        max_chunk_chars: int = 4000,
        overlap_ratio: float = 0.20,
        min_overlap_chars: int = 200,
        max_overlap_chars: int = 1200,
    ):
        self.max_chunk_chars = max_chunk_chars
        self.overlap_ratio = overlap_ratio
        self.min_overlap_chars = min_overlap_chars
        self.max_overlap_chars = max_overlap_chars

    def chunk(
        self,
        sections: List[ParsedSection],
        document_id: str,
        canonical_hash: str,
        metadata: dict,
    ) -> List[AcordaoChunk]:
        """Divide seções em chunks com overlap. Returns ~15-25 AcordaoChunk."""
        all_chunks: List[AcordaoChunk] = []

        for section in sections:
            sec_type = section.section_type
            authority = AUTHORITY_MAP.get(sec_type, "opinativo")
            section_path = SECTION_PATH_MAP.get(sec_type, sec_type.upper())
            span_prefix = SPAN_PREFIX_MAP.get(sec_type, f"SEC-{sec_type.upper()}")

            # Split section into parts
            parts = self._split_section(section)
            total_parts = len(parts)

            for i, (text, abs_start, abs_end) in enumerate(parts):
                part_number = i + 1

                # Build span_id
                if total_parts == 1:
                    span_id = span_prefix
                else:
                    span_id = f"{span_prefix}-P{part_number:02d}"

                # Estimate page number (rough: ~3500 chars per page)
                page_number = max(1, (abs_start // 3500) + 1)

                retrieval_text = self._build_retrieval_text(
                    text, sec_type, part_number, total_parts, metadata,
                )

                all_chunks.append(AcordaoChunk(
                    span_id=span_id,
                    section_type=sec_type,
                    authority_level=authority,
                    text=text,
                    retrieval_text=retrieval_text,
                    canonical_start=abs_start,
                    canonical_end=abs_end,
                    page_number=page_number,
                    part_number=part_number,
                    total_parts=total_parts,
                    section_path=section_path,
                ))

        logger.info(
            f"AcordaoChunker: {len(all_chunks)} chunks gerados de "
            f"{len(sections)} seções"
        )
        return all_chunks

    def _split_section(
        self, section: ParsedSection,
    ) -> List[tuple]:
        """
        Divide seção em partes por fronteira de parágrafo.
        Returns [(text, abs_start, abs_end), ...]

        Se seção < max_chunk_chars, retorna chunk único.
        Overlap é por parágrafos completos (nunca corta frase).
        """
        text = section.text
        if len(text) <= self.max_chunk_chars:
            return [(text, section.canonical_start, section.canonical_end)]

        paragraphs = section.paragraphs
        if not paragraphs:
            # No parsed paragraphs — split by double newline
            return self._split_by_natural_breaks(section)

        # Split by paragraph boundaries with overlap
        return self._split_by_paragraphs(section)

    def _split_by_paragraphs(
        self, section: ParsedSection,
    ) -> List[tuple]:
        """Split using paragraph boundaries with overlap."""
        paragraphs = section.paragraphs
        parts: List[tuple] = []

        # Track which paragraphs go into each chunk
        chunk_start_idx = 0  # index into paragraphs list

        while chunk_start_idx < len(paragraphs):
            # Accumulate paragraphs until we exceed max_chunk_chars
            chunk_end_idx = chunk_start_idx
            accumulated_len = 0

            while chunk_end_idx < len(paragraphs):
                para = paragraphs[chunk_end_idx]
                para_len = para["end"] - para["start"]
                if accumulated_len + para_len > self.max_chunk_chars and chunk_end_idx > chunk_start_idx:
                    break
                accumulated_len += para_len
                chunk_end_idx += 1

            # Build chunk text from canonical offsets
            abs_start = paragraphs[chunk_start_idx]["start"]
            abs_end = paragraphs[chunk_end_idx - 1]["end"]

            # For the first chunk, extend to section start to capture section header
            if not parts:
                abs_start = section.canonical_start

            # For the last chunk-range, extend to section end
            if chunk_end_idx >= len(paragraphs):
                abs_end = section.canonical_end

            chunk_text = section.text[abs_start - section.canonical_start:abs_end - section.canonical_start]
            parts.append((chunk_text, abs_start, abs_end))

            # Calculate overlap: go back N paragraphs to cover ~20% of current chunk
            target_overlap = int(len(chunk_text) * self.overlap_ratio)
            target_overlap = max(self.min_overlap_chars, min(self.max_overlap_chars, target_overlap))

            # Find how many paragraphs from the end of this chunk to cover overlap
            overlap_chars = 0
            overlap_start_idx = chunk_end_idx
            while overlap_start_idx > chunk_start_idx:
                overlap_start_idx -= 1
                para = paragraphs[overlap_start_idx]
                overlap_chars += para["end"] - para["start"]
                if overlap_chars >= target_overlap:
                    break

            # If no paragraph fits overlap, use the last paragraph
            if overlap_start_idx >= chunk_end_idx:
                overlap_start_idx = max(chunk_start_idx, chunk_end_idx - 1)

            # Next chunk starts from overlap point
            if chunk_end_idx >= len(paragraphs):
                break  # done
            chunk_start_idx = overlap_start_idx

        # Absorb runt final chunk (< 20% of max_chunk_chars)
        runt_threshold = int(self.max_chunk_chars * 0.20)
        if len(parts) > 1:
            last_text, last_start, last_end = parts[-1]
            if len(last_text) < runt_threshold:
                # Merge into previous chunk
                prev_text, prev_start, prev_end = parts[-2]
                merged_text = section.text[prev_start - section.canonical_start:last_end - section.canonical_start]
                parts[-2] = (merged_text, prev_start, last_end)
                parts.pop()

        return parts

    def _split_by_natural_breaks(
        self, section: ParsedSection,
    ) -> List[tuple]:
        """Split by double newline when no parsed paragraphs are available."""
        text = section.text
        raw_paras = text.split("\n\n")

        if len(raw_paras) <= 1:
            # No natural breaks — single chunk
            return [(text, section.canonical_start, section.canonical_end)]

        # Build paragraph-like entries with offsets
        offset = 0
        para_entries = []
        for rp in raw_paras:
            start = text.find(rp, offset)
            if start == -1:
                start = offset
            end = start + len(rp)
            if rp.strip():
                para_entries.append({
                    "num": "",
                    "text": rp,
                    "start": section.canonical_start + start,
                    "end": section.canonical_start + end,
                })
            offset = end

        if not para_entries:
            return [(text, section.canonical_start, section.canonical_end)]

        # Reuse paragraph-based splitting
        synth_section = ParsedSection(
            section_type=section.section_type,
            text=section.text,
            canonical_start=section.canonical_start,
            canonical_end=section.canonical_end,
            paragraphs=para_entries,
        )
        return self._split_by_paragraphs(synth_section)

    def _build_retrieval_text(
        self,
        chunk_text: str,
        section_type: str,
        part: int,
        total: int,
        metadata: dict,
    ) -> str:
        """Constrói retrieval_text com prefixo [CONTEXTO: ...]."""
        numero = metadata.get("numero", "")
        ano = metadata.get("ano", "")
        colegiado = metadata.get("colegiado", "")
        relator = metadata.get("relator", "")

        ref = f"Acórdão {numero}/{ano}-{colegiado}" if numero else "Acórdão"

        if section_type == "ementa":
            prefix = f"[CONTEXTO: {ref} | Ementa]"
        elif section_type == "relatorio":
            if total > 1:
                prefix = f"[CONTEXTO: {ref} | Relatório | parte {part} de {total}]"
            else:
                prefix = f"[CONTEXTO: {ref} | Relatório]"
        elif section_type == "voto":
            rel_part = f" do Min. {relator}" if relator else ""
            if total > 1:
                prefix = f"[CONTEXTO: {ref} | Voto{rel_part} | parte {part} de {total}]"
            else:
                prefix = f"[CONTEXTO: {ref} | Voto{rel_part}]"
        elif section_type == "acordao":
            prefix = f"[CONTEXTO: {ref} | Dispositivo (decisão vinculante)]"
        else:
            prefix = f"[CONTEXTO: {ref} | {section_type}]"

        return f"{prefix}\n{chunk_text}"
