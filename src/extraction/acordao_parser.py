"""
AcordaoParser — Parser regex para acórdãos do TCU.

Extrai estrutura hierárquica: seções primárias (RELATÓRIO, VOTO, ACÓRDÃO),
subseções dinâmicas do RELATÓRIO, parágrafos numerados, itens do dispositivo.

Opera sobre canonical_text (não sobre blocos PyMuPDF).
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# AcordaoDevice — output dataclass
# =============================================================================

@dataclass
class AcordaoDevice:
    """Dispositivo de acórdão classificado pelo parser."""
    device_type: str           # "section", "paragraph", "item_dispositivo"
    span_id: str               # "SEC-RELATORIO", "PAR-VOTO-7", "ITEM-9.4.1"
    parent_span_id: str        # "" para seções primárias
    children_span_ids: list = field(default_factory=list)
    text: str = ""
    text_preview: str = ""
    identifier: str = ""       # "RELATÓRIO", "7", "9.4.1"
    section_type: str = ""     # "relatorio", "voto", "acordao"
    authority_level: str = ""  # "opinativo", "fundamentacao", "vinculante"
    section_path: str = ""     # "RELATÓRIO > EXAME TÉCNICO > I.3.1"
    hierarchy_depth: int = 0   # 0=primária, 1=subseção, 2=sub-subseção, etc
    char_start: int = 0
    char_end: int = 0
    page_number: int = 1
    bbox: list = field(default_factory=list)


# =============================================================================
# Regex patterns
# =============================================================================

# Seções primárias
RE_RELATORIO = re.compile(r'^\s*RELAT[OÓ]RIO\s*$', re.MULTILINE)
RE_VOTO = re.compile(r'^\s*VOTO\s*$', re.MULTILINE)
RE_ACORDAO_SECTION = re.compile(
    r'^\s*AC[OÓ]RD[AÃ]O\s+(?:N[°ºo.]?\s*)?\d+',
    re.MULTILINE | re.IGNORECASE,
)

# Subseções do relatório
RE_HEADING_SUB = re.compile(r'^([IVX]+\.\d+(?:\.\d+)*)\.\s+(.+)', re.MULTILINE)
RE_HEADING_ROMAN = re.compile(r'^([IVX]+)\.\s+(.+)', re.MULTILINE)
RE_HEADING_UPPER = re.compile(
    r'^([A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][A-ZÁÀÂÃÉÊÍÓÔÕÚÇ\s]{3,})$', re.MULTILINE,
)

# Parágrafos numerados (1., 2., 25.) — NÃO seguido de dígito (evita confundir com itens 9.1)
RE_NUMBERED_PARA = re.compile(r'^(\d{1,3})\.\s+', re.MULTILINE)

# Itens do dispositivo (9.1., 9.4.1.)
RE_ITEM = re.compile(r'^(\d+\.\d+(?:\.\d+)*)\.\s+', re.MULTILINE)


def _strip_with_offsets(text: str, base_offset: int):
    """Strip text and return (stripped_text, adjusted_start, adjusted_end).

    Garante que canonical_text[start:end] == stripped_text.
    """
    lead = len(text) - len(text.lstrip())
    stripped = text.strip()
    start = base_offset + lead
    end = start + len(stripped)
    return stripped, start, end


def _slugify(text: str) -> str:
    """Converte heading para slug de span_id: 'EXAME TÉCNICO' → 'EXAME-TECNICO'."""
    import unicodedata
    nfkd = unicodedata.normalize('NFKD', text.strip())
    ascii_text = nfkd.encode('ASCII', 'ignore').decode('ASCII')
    slug = re.sub(r'[^A-Z0-9]+', '-', ascii_text.upper()).strip('-')
    return slug


# =============================================================================
# AcordaoParser
# =============================================================================

class AcordaoParser:
    """Parser de acórdãos do TCU: detecta seções, parágrafos e itens."""

    def parse(
        self,
        canonical_text: str,
        page_boundaries: list,
    ) -> List[AcordaoDevice]:
        """
        Pipeline completo: detecta seções → parágrafos → itens → hierarquia.

        Args:
            canonical_text: Texto integral normalizado do acórdão.
            page_boundaries: Lista de (char_start, char_end) por página (1-indexed).
                             page_boundaries[0] = (start, end) da página 1.

        Returns:
            List[AcordaoDevice] com hierarquia completa.
        """
        devices: List[AcordaoDevice] = []

        # 1. Detecta seções primárias
        sections = self._detect_primary_sections(canonical_text)

        if not sections:
            logger.warning("AcordaoParser: nenhuma seção primária encontrada")
            return devices

        # 2. Processa cada seção
        for sec in sections:
            sec_type = sec["type"]
            sec_start = sec["start"]
            sec_end = sec["end"]
            sec_text = canonical_text[sec_start:sec_end]

            if sec_type == "RELATORIO":
                sec_devices = self._parse_relatorio(
                    sec_text, sec_start, page_boundaries,
                )
            elif sec_type == "VOTO":
                sec_devices = self._parse_voto(
                    sec_text, sec_start, page_boundaries,
                )
            elif sec_type == "ACORDAO":
                sec_devices = self._parse_acordao_items(
                    sec_text, sec_start, page_boundaries,
                )
            else:
                continue

            devices.extend(sec_devices)

        # 3. Deduplica span_ids (citações de acórdãos anteriores no VOTO
        #    geram parágrafos com números repetidos: PAR-VOTO-13, PAR-VOTO-13-2)
        devices = self._deduplicate_span_ids(devices)

        # 4. Hierarquia
        devices = self._build_hierarchy(devices)

        logger.info(
            f"AcordaoParser: {len(devices)} devices extraídos "
            f"({sum(1 for d in devices if d.device_type == 'section')} sections, "
            f"{sum(1 for d in devices if d.device_type == 'paragraph')} paragraphs, "
            f"{sum(1 for d in devices if d.device_type == 'item_dispositivo')} items)"
        )

        return devices

    def _detect_primary_sections(self, text: str) -> list:
        """
        Encontra RELATÓRIO, VOTO, ACÓRDÃO no texto.

        Returns:
            Lista de dicts: [{type, start, end, marker_end}, ...] ordenados por posição.
        """
        markers = []

        for m in RE_RELATORIO.finditer(text):
            markers.append({"type": "RELATORIO", "marker_start": m.start(), "marker_end": m.end()})

        for m in RE_VOTO.finditer(text):
            markers.append({"type": "VOTO", "marker_start": m.start(), "marker_end": m.end()})

        for m in RE_ACORDAO_SECTION.finditer(text):
            markers.append({"type": "ACORDAO", "marker_start": m.start(), "marker_end": m.end()})

        if not markers:
            return []

        # Ordena por posição
        markers.sort(key=lambda m: m["marker_start"])

        # Deduplica: pega a ÚLTIMA ocorrência de ACORDAO (primeira é header),
        # e a PRIMEIRA de RELATÓRIO e VOTO.
        unique_markers = []
        by_type: Dict[str, list] = {}
        for m in markers:
            by_type.setdefault(m["type"], []).append(m)

        # Ordem canônica: RELATÓRIO, VOTO, ACÓRDÃO
        for stype in ("RELATORIO", "VOTO", "ACORDAO"):
            if stype in by_type:
                if stype == "ACORDAO" and len(by_type[stype]) > 1:
                    # Última ocorrência é a seção real, primeira é o header
                    unique_markers.append(by_type[stype][-1])
                else:
                    unique_markers.append(by_type[stype][0])

        markers = sorted(unique_markers, key=lambda m: m["marker_start"])

        # Define limites de cada seção
        sections = []
        for i, marker in enumerate(markers):
            start = marker["marker_start"]
            if i + 1 < len(markers):
                end = markers[i + 1]["marker_start"]
            else:
                end = len(text)
            sections.append({
                "type": marker["type"],
                "start": start,
                "end": end,
                "marker_end": marker["marker_end"],
            })

        return sections

    def _parse_relatorio(
        self,
        text: str,
        base_offset: int,
        page_boundaries: list,
    ) -> List[AcordaoDevice]:
        """Seções dinâmicas (headings) + parágrafos numerados do RELATÓRIO."""
        devices: List[AcordaoDevice] = []

        # Seção primária RELATÓRIO
        sec_stripped, sec_cs, sec_ce = _strip_with_offsets(text, base_offset)
        sec_device = AcordaoDevice(
            device_type="section",
            span_id="SEC-RELATORIO",
            parent_span_id="",
            text=sec_stripped,
            text_preview=sec_stripped[:120],
            identifier="RELATÓRIO",
            section_type="relatorio",
            authority_level="opinativo",
            section_path="RELATÓRIO",
            hierarchy_depth=0,
            char_start=sec_cs,
            char_end=sec_ce,
            page_number=self._page_for_offset(sec_cs, page_boundaries),
        )
        devices.append(sec_device)

        # Detecta subseções
        subsections = self._detect_subsections(text)

        if subsections:
            used_sub_ids = set()
            for sub in subsections:
                sub_start = sub["start"]
                sub_end = sub["end"]
                raw_sub = text[sub_start:sub_end]
                sub_text, abs_start, abs_end = _strip_with_offsets(
                    raw_sub, base_offset + sub_start,
                )
                heading_name = sub["heading"]
                slug = _slugify(heading_name)

                # Span ID baseado no tipo de heading
                if sub.get("numbering"):
                    # Numeração romana preservada: I.3.1
                    base_span_id = f"SEC-RELATORIO-{sub['numbering']}"
                else:
                    base_span_id = f"SEC-RELATORIO-{slug}"

                # Deduplica span_id (ex: DESPACHO DO RELATOR aparece múltiplas vezes)
                span_id = base_span_id
                counter = 2
                while span_id in used_sub_ids:
                    span_id = f"{base_span_id}-{counter}"
                    counter += 1
                used_sub_ids.add(span_id)

                sub_device = AcordaoDevice(
                    device_type="section",
                    span_id=span_id,
                    parent_span_id="SEC-RELATORIO",
                    text=sub_text,
                    text_preview=sub_text[:120],
                    identifier=heading_name,
                    section_type="relatorio",
                    authority_level="opinativo",
                    section_path=f"RELATÓRIO > {heading_name}",
                    hierarchy_depth=sub.get("depth", 1),
                    char_start=abs_start,
                    char_end=abs_end,
                    page_number=self._page_for_offset(abs_start, page_boundaries),
                )
                devices.append(sub_device)

                # Parágrafos dentro da subseção
                sub_text_for_paras = text[sub_start:sub_end]
                paras = self._extract_numbered_paragraphs(
                    sub_text_for_paras, base_offset + sub_start, "relatorio",
                    section_path=f"RELATÓRIO > {heading_name}",
                    parent_span_id=span_id,
                    page_boundaries=page_boundaries,
                )
                devices.extend(paras)
        else:
            # Sem subseções: parágrafos diretos do RELATÓRIO
            paras = self._extract_numbered_paragraphs(
                text, base_offset, "relatorio",
                section_path="RELATÓRIO",
                parent_span_id="SEC-RELATORIO",
                page_boundaries=page_boundaries,
            )
            devices.extend(paras)

        return devices

    def _parse_voto(
        self,
        text: str,
        base_offset: int,
        page_boundaries: list,
    ) -> List[AcordaoDevice]:
        """Parágrafos numerados do VOTO."""
        devices: List[AcordaoDevice] = []

        # Seção primária VOTO
        sec_stripped, sec_cs, sec_ce = _strip_with_offsets(text, base_offset)
        sec_device = AcordaoDevice(
            device_type="section",
            span_id="SEC-VOTO",
            parent_span_id="",
            text=sec_stripped,
            text_preview=sec_stripped[:120],
            identifier="VOTO",
            section_type="voto",
            authority_level="fundamentacao",
            section_path="VOTO",
            hierarchy_depth=0,
            char_start=sec_cs,
            char_end=sec_ce,
            page_number=self._page_for_offset(sec_cs, page_boundaries),
        )
        devices.append(sec_device)

        # Parágrafos numerados
        paras = self._extract_numbered_paragraphs(
            text, base_offset, "voto",
            section_path="VOTO",
            parent_span_id="SEC-VOTO",
            page_boundaries=page_boundaries,
        )
        devices.extend(paras)

        return devices

    def _parse_acordao_items(
        self,
        text: str,
        base_offset: int,
        page_boundaries: list,
    ) -> List[AcordaoDevice]:
        """Itens decimais do ACÓRDÃO: 9.1, 9.4.1, etc."""
        devices: List[AcordaoDevice] = []

        # Seção primária ACÓRDÃO
        sec_stripped, sec_cs, sec_ce = _strip_with_offsets(text, base_offset)
        sec_device = AcordaoDevice(
            device_type="section",
            span_id="SEC-ACORDAO",
            parent_span_id="",
            text=sec_stripped,
            text_preview=sec_stripped[:120],
            identifier="ACÓRDÃO",
            section_type="acordao",
            authority_level="vinculante",
            section_path="ACÓRDÃO",
            hierarchy_depth=0,
            char_start=sec_cs,
            char_end=sec_ce,
            page_number=self._page_for_offset(sec_cs, page_boundaries),
        )
        devices.append(sec_device)

        # Encontra itens decimais
        matches = list(RE_ITEM.finditer(text))
        if not matches:
            return devices

        for i, m in enumerate(matches):
            item_number = m.group(1)  # "9.1", "9.4.1"
            item_start = m.start()
            if i + 1 < len(matches):
                item_end = matches[i + 1].start()
            else:
                item_end = len(text)

            raw_item = text[item_start:item_end]
            item_text, abs_start, abs_end = _strip_with_offsets(
                raw_item, base_offset + item_start,
            )

            # Hierarquia: 9.4.1 → pai é 9.4
            parts = item_number.split(".")
            if len(parts) > 2:
                parent_number = ".".join(parts[:-1])
                parent_span_id = f"ITEM-{parent_number}"
            else:
                parent_span_id = "SEC-ACORDAO"

            depth = len(parts) - 1  # 9.1 → depth 1, 9.4.1 → depth 2

            device = AcordaoDevice(
                device_type="item_dispositivo",
                span_id=f"ITEM-{item_number}",
                parent_span_id=parent_span_id,
                text=item_text,
                text_preview=item_text[:120],
                identifier=item_number,
                section_type="acordao",
                authority_level="vinculante",
                section_path=f"ACÓRDÃO > {item_number}",
                hierarchy_depth=depth,
                char_start=abs_start,
                char_end=abs_end,
                page_number=self._page_for_offset(abs_start, page_boundaries),
            )
            devices.append(device)

        return devices

    def _detect_subsections(self, text: str) -> list:
        """
        Detecta headings dentro do RELATÓRIO.

        Returns:
            Lista de dicts: [{heading, start, end, numbering, depth}, ...]
        """
        headings = []

        # Prioridade: RE_HEADING_SUB > RE_HEADING_ROMAN > RE_HEADING_UPPER
        # Coletamos todos e depois resolvemos conflitos

        for m in RE_HEADING_SUB.finditer(text):
            numbering = m.group(1)
            heading_text = m.group(2).strip()
            depth = numbering.count(".") + 1  # I.3.1 → depth 3
            headings.append({
                "heading": f"{numbering}. {heading_text}",
                "numbering": numbering,
                "start": m.start(),
                "priority": 1,
                "depth": depth,
            })

        for m in RE_HEADING_ROMAN.finditer(text):
            numbering = m.group(1)
            heading_text = m.group(2).strip()
            # Pula se já capturado como sub-heading (I.3.1 inicia com I.)
            if any(
                h["start"] == m.start()
                for h in headings
            ):
                continue
            headings.append({
                "heading": f"{numbering}. {heading_text}",
                "numbering": numbering,
                "start": m.start(),
                "priority": 2,
                "depth": 1,
            })

        for m in RE_HEADING_UPPER.finditer(text):
            heading_text = m.group(1).strip()
            # Pula headings muito curtos ou que são marcadores de seção primária
            if len(heading_text) < 4:
                continue
            if re.match(r'^(RELAT[OÓ]RIO|VOTO|AC[OÓ]RD[AÃ]O)\s*$', heading_text, re.IGNORECASE):
                continue
            # Pula se já capturado por outro pattern
            if any(abs(h["start"] - m.start()) < 5 for h in headings):
                continue
            headings.append({
                "heading": heading_text,
                "numbering": None,
                "start": m.start(),
                "priority": 3,
                "depth": 1,
            })

        if not headings:
            return []

        # Ordena por posição
        headings.sort(key=lambda h: h["start"])

        # Define limites
        for i in range(len(headings)):
            if i + 1 < len(headings):
                headings[i]["end"] = headings[i + 1]["start"]
            else:
                headings[i]["end"] = len(text)

        return headings

    def _extract_numbered_paragraphs(
        self,
        text: str,
        base_offset: int,
        section_type: str,
        section_path: str,
        parent_span_id: str,
        page_boundaries: list,
    ) -> List[AcordaoDevice]:
        """
        Extrai parágrafos numerados (1., 2., 3., ...) de um trecho de texto.

        Cuidado: não confundir com itens decimais (9.1. tem dois números).
        Filtra: parágrafo deve ter número seguido de '. ' (ponto + espaço),
        e o número NÃO deve ter ponto antes dele (evita capturar 9.1, 9.4.1).
        """
        authority_map = {
            "relatorio": "opinativo",
            "voto": "fundamentacao",
            "acordao": "vinculante",
        }
        authority = authority_map.get(section_type, "opinativo")

        # Prefixo do span_id
        prefix_map = {
            "relatorio": "PAR-RELATORIO",
            "voto": "PAR-VOTO",
            "acordao": "PAR-ACORDAO",
        }
        prefix = prefix_map.get(section_type, "PAR")

        paragraphs: List[AcordaoDevice] = []

        # Encontra parágrafos numerados, filtrando itens decimais
        matches = []
        for m in RE_NUMBERED_PARA.finditer(text):
            num = m.group(1)
            pos = m.start()
            # Verifica que não é item decimal: antes do número não pode ter '.'
            # (ex: "9.1. " → skip, "1. " → ok)
            if pos > 0 and text[pos - 1] == '.':
                continue
            # Verifica que o char antes do match (ignorando whitespace) não é dígito+ponto
            pre_text = text[:pos].rstrip()
            if pre_text and pre_text[-1] == '.':
                # Pode ser item decimal — verifica se antes do ponto é dígito
                if len(pre_text) >= 2 and pre_text[-2].isdigit():
                    continue
            matches.append((m, num))

        for i, (m, num) in enumerate(matches):
            para_start = m.start()
            if i + 1 < len(matches):
                para_end = matches[i + 1][0].start()
            else:
                para_end = len(text)

            raw_para = text[para_start:para_end]
            para_text, abs_start, abs_end = _strip_with_offsets(
                raw_para, base_offset + para_start,
            )
            if not para_text:
                continue

            device = AcordaoDevice(
                device_type="paragraph",
                span_id=f"{prefix}-{num}",
                parent_span_id=parent_span_id,
                text=para_text,
                text_preview=para_text[:120],
                identifier=num,
                section_type=section_type,
                authority_level=authority,
                section_path=f"{section_path} > § {num}",
                hierarchy_depth=2,
                char_start=abs_start,
                char_end=abs_end,
                page_number=self._page_for_offset(abs_start, page_boundaries),
            )
            paragraphs.append(device)

        return paragraphs

    def _page_for_offset(self, offset: int, page_boundaries: list) -> int:
        """Retorna page_number (1-indexed) para um offset no canonical_text."""
        if not page_boundaries:
            return 1
        for i, (start, end) in enumerate(page_boundaries):
            if start <= offset < end:
                return i + 1
        # Se offset está além da última boundary, retorna última página
        return len(page_boundaries)

    def _deduplicate_span_ids(self, devices: List[AcordaoDevice]) -> List[AcordaoDevice]:
        """
        Garante unicidade de span_ids, adicionando sufixo -2, -3 para duplicatas.

        Necessário porque o VOTO frequentemente cita acórdãos anteriores verbatim,
        e os parágrafos citados têm a mesma numeração dos originais.
        """
        seen: Dict[str, int] = {}
        dupes_found = 0
        for device in devices:
            sid = device.span_id
            if sid in seen:
                seen[sid] += 1
                new_sid = f"{sid}-{seen[sid]}"
                device.span_id = new_sid
                dupes_found += 1
            else:
                seen[sid] = 1
        if dupes_found:
            logger.info(f"AcordaoParser: {dupes_found} span_ids deduplicados")
        return devices

    def _build_hierarchy(self, devices: List[AcordaoDevice]) -> List[AcordaoDevice]:
        """Atribui parent_span_id e children_span_ids."""
        device_by_span = {d.span_id: d for d in devices}

        for device in devices:
            parent_id = device.parent_span_id
            if parent_id and parent_id in device_by_span:
                parent = device_by_span[parent_id]
                if device.span_id not in parent.children_span_ids:
                    parent.children_span_ids.append(device.span_id)

        return devices
