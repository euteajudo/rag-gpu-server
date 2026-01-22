"""
AcordaoSpanParser - Parser Deterministico para Acordaos do TCU.

Este modulo implementa um parser regex-first que identifica a estrutura
de acordaos do Tribunal de Contas da Uniao (TCU).

Estrutura de um Acordao TCU:
============================

    CABEÇALHO
    ├── Processo: TC 002.019/2024-8
    ├── Código eletrônico: AC-2724-47/25-P
    ├── Grupo/Classe: GRUPO II – CLASSE VII – Plenário
    ├── Relator: Ministro Benjamin Zymler
    └── Unidade Técnica: SecexAdministração

    SUMÁRIO
    └── Resumo do acordao (1 bloco)

    RELATÓRIO
    ├── REL-001: Paragrafo 1
    ├── REL-002: Paragrafo 2
    └── ...

    VOTO
    ├── VOTO-001: Paragrafo 1
    ├── VOTO-002: Paragrafo 2
    └── ...

    ACÓRDÃO
    ├── ACORDAO: Bloco principal "ACORDAM os Ministros..."
    └── Deliberações
        ├── ACORDAO-9-1: Item 9.1
        ├── ACORDAO-9-2: Item 9.2
        └── ...

Formato de Span IDs:
===================

| Bloco       | Formato         | Exemplo       |
|-------------|-----------------|---------------|
| Sumário     | SUMARIO         | SUMARIO       |
| Relatório   | REL-{nnn}       | REL-001       |
| Voto        | VOTO-{nnn}      | VOTO-001      |
| Acórdão     | ACORDAO         | ACORDAO       |
| Deliberação | ACORDAO-{X}-{Y} | ACORDAO-9-1   |

Exemplo de Uso:
==============

    ```python
    from parsing.acordao_span_parser import AcordaoSpanParser

    parser = AcordaoSpanParser()
    acordao = parser.parse(markdown_text)

    print(f"Acordao: {acordao.acordao_id}")
    print(f"Relator: {acordao.metadata.relator}")
    print(f"Total de spans: {len(acordao.spans)}")

    for span in acordao.relatorios:
        print(f"  {span.span_id}: {span.text[:50]}...")
    ```

@author: VectorGov
@version: 1.0.0
@since: 22/01/2025
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple

from .acordao_models import (
    AcordaoSpan,
    AcordaoSpanType,
    AcordaoMetadata,
    ParsedAcordao,
    normalize_acordao_id,
    parse_colegiado,
)

logger = logging.getLogger(__name__)


@dataclass
class AcordaoParserConfig:
    """Configuracao do parser de acordaos."""
    normalize_whitespace: bool = True
    extract_subparagrafos: bool = True  # Paragrafos como D, E, F dentro do relatorio
    min_paragrafo_chars: int = 20  # Minimo de caracteres para considerar paragrafo


class AcordaoSpanParser:
    """
    Parser deterministico para acordaos do TCU.

    Usa regex para identificar a estrutura do acordao,
    gerando spans com IDs unicos.
    """

    # =========================================================================
    # REGEX PATTERNS
    # =========================================================================

    # Secoes principais - Suporta formato Docling (## RELATÓRIO) e formato texto puro
    PATTERN_SUMARIO = re.compile(
        r'(?:^|\n)(?:#*\s*)?(?:SUM[ÁA]RIO|SUMARIO)[:\s]*(.+?)(?=\n(?:#*\s*)?(?:RELAT[ÓO]RIO|RELATORIO)|$)',
        re.IGNORECASE | re.DOTALL
    )

    PATTERN_RELATORIO = re.compile(
        r'(?:^|\n)(?:#*\s*)?(?:RELAT[ÓO]RIO|RELATORIO)[:\s]*\n(.+?)(?=\n(?:#*\s*)?(?:VOTO|É O RELAT[ÓO]RIO)|$)',
        re.IGNORECASE | re.DOTALL
    )

    PATTERN_VOTO = re.compile(
        r'(?:^|\n)(?:#*\s*)?VOTO[:\s]*\n(.+?)(?=\n(?:#*\s*)?AC[ÓO]RD[ÃA]O|\nACORDAM|$)',
        re.IGNORECASE | re.DOTALL
    )

    PATTERN_ACORDAO = re.compile(
        r'(?:^|\n)(?:#*\s*)?(AC[ÓO]RD[ÃA]O|ACORDAM)(.+?)(?=\n\d+\.\s+|$)',
        re.IGNORECASE | re.DOTALL
    )

    # Paragrafos numerados - Suporta formato bullet "- 1." e formato normal "1."
    PATTERN_PARAGRAFO_NUM = re.compile(
        r'(?:^|\n)\s*(?:-\s*)?(\d+)\.\s+(.+?)(?=\n\s*(?:-\s*)?\d+\.\s|\n(?:#*\s*)?[A-Z]\.\s|\n(?:#*\s*)?[A-Z]\d?\.\s|$)',
        re.DOTALL
    )

    # Subparagrafos alfabeticos: "D.", "E.", "## D.", "## F.1" etc.
    PATTERN_SUBPARAGRAFO = re.compile(
        r'(?:^|\n)(?:#*\s*)?([A-Z])(?:\.\d*)?\.\s+(.+?)(?=\n(?:#*\s*)?[A-Z](?:\.\d*)?\.\s|\n\s*(?:-\s*)?\d+\.\s|$)',
        re.DOTALL
    )

    # Deliberacoes: "9.1.", "- 9.1.", "9.2.", etc.
    PATTERN_DELIBERACAO = re.compile(
        r'(?:^|\n)\s*(?:-\s*)?(9)\.(\d+)[.:]?\s+(.+?)(?=\n\s*(?:-\s*)?9\.\d+|\n\s*\d+\.\s+(?:Ata|Data|Código)|$)',
        re.DOTALL
    )

    # Metadados do cabecalho
    PATTERN_PROCESSO = re.compile(r'(?:TC|Processo)[:\s]+(\d{3}\.\d{3}/\d{4}-\d+)', re.IGNORECASE)
    PATTERN_CODIGO = re.compile(r'C[óo]digo\s+eletr[ôo]nico[:\s]+([A-Z0-9\-/]+)', re.IGNORECASE)
    PATTERN_RELATOR = re.compile(r'(?:Relator|Ministro[a]?\s+Relator[a]?)[:\s]+(?:Ministro\s+)?([^\n]+)', re.IGNORECASE)
    PATTERN_DATA = re.compile(r'(?:Data\s+da\s+[Ss]ess[ãa]o|Sess[ãa]o\s+de)[:\s]+(\d{1,2}/\d{1,2}/\d{4})', re.IGNORECASE)
    PATTERN_UNIDADE = re.compile(r'(?:Unidade\s+[Tt][ée]cnica|UNIDADE)[:\s]+([^\n]+)', re.IGNORECASE)
    PATTERN_GRUPO_CLASSE = re.compile(r'(GRUPO\s+[IVX]+)[^–-]*(CLASSE\s+[IVX]+)', re.IGNORECASE)
    PATTERN_INTERESSADO = re.compile(r'(?:Interessado|INTERESSADO)[:\s]+([^\n]+)', re.IGNORECASE)

    def __init__(self, config: Optional[AcordaoParserConfig] = None):
        """Inicializa o parser."""
        self.config = config or AcordaoParserConfig()

    def parse(self, markdown: str) -> ParsedAcordao:
        """
        Parseia markdown de acordao e retorna documento com spans.

        Args:
            markdown: Texto em markdown (output do Docling)

        Returns:
            ParsedAcordao com todos os spans indexados
        """
        doc = ParsedAcordao(source_text=markdown)

        # Normaliza whitespace
        if self.config.normalize_whitespace:
            markdown = self._normalize_whitespace(markdown)

        # 1. Extrai metadados do cabecalho
        self._extract_metadata(markdown, doc)

        # 2. Extrai SUMARIO
        self._extract_sumario(markdown, doc)

        # 3. Extrai RELATORIO
        self._extract_relatorio(markdown, doc)

        # 4. Extrai VOTO
        self._extract_voto(markdown, doc)

        # 5. Extrai ACORDAO e deliberacoes
        self._extract_acordao(markdown, doc)

        logger.info(
            f"Parsed acordao {doc.acordao_id}: {len(doc.spans)} spans "
            f"({len(doc.relatorios)} REL, {len(doc.votos)} VOTO, "
            f"{len(doc.deliberacoes)} deliberacoes)"
        )

        return doc

    def _normalize_whitespace(self, text: str) -> str:
        """Normaliza espacos em branco."""
        # Remove espacos multiplos (mantem quebras de linha)
        text = re.sub(r'[^\S\n]+', ' ', text)
        # Remove linhas em branco multiplas
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _extract_metadata(self, markdown: str, doc: ParsedAcordao):
        """Extrai metadados do cabecalho do acordao."""
        # Pega apenas o inicio do documento (antes do SUMARIO)
        sumario_match = self.PATTERN_SUMARIO.search(markdown)
        header_end = sumario_match.start() if sumario_match else min(5000, len(markdown))
        header = markdown[:header_end]

        # Processo
        processo_match = self.PATTERN_PROCESSO.search(header)
        if processo_match:
            doc.metadata.processo = processo_match.group(1)

        # Codigo eletronico - busca no documento INTEIRO (pode estar no final, na secao ACORDAO)
        codigo_match = self.PATTERN_CODIGO.search(markdown)  # Busca em TODO o documento
        if codigo_match:
            doc.metadata.codigo_eletronico = codigo_match.group(1)
            # Normaliza para acordao_id
            normalized = normalize_acordao_id(codigo_match.group(1))
            if normalized:
                parts = normalized.split('-')
                if len(parts) >= 4:
                    doc.metadata.numero = int(parts[1])
                    doc.metadata.ano = int(parts[2])
                    doc.metadata.colegiado = parts[3]

        # Relator
        relator_match = self.PATTERN_RELATOR.search(header)
        if relator_match:
            doc.metadata.relator = relator_match.group(1).strip()

        # Data da sessao - busca no documento inteiro (pode estar no final)
        data_match = self.PATTERN_DATA.search(markdown)
        if data_match:
            doc.metadata.data_sessao = data_match.group(1)

        # Unidade tecnica
        unidade_match = self.PATTERN_UNIDADE.search(header)
        if unidade_match:
            doc.metadata.unidade_tecnica = unidade_match.group(1).strip()

        # Grupo e Classe
        grupo_match = self.PATTERN_GRUPO_CLASSE.search(header)
        if grupo_match:
            doc.metadata.grupo = grupo_match.group(1).strip()
            doc.metadata.classe = grupo_match.group(2).strip()

        # Interessado
        interessado_match = self.PATTERN_INTERESSADO.search(header)
        if interessado_match:
            doc.metadata.interessado = interessado_match.group(1).strip()

        # Fallback: tenta extrair do titulo "ACÓRDÃO Nº X/YYYY - TCU - Plenário"
        if not doc.metadata.numero:
            # Busca no titulo do acordao (formato Docling: ## ACÓRDÃO Nº 2724/2025 - TCU - Plenário)
            titulo_match = re.search(
                r'AC[ÓO]RD[ÃA]O\s+N[°ºo]?\s*(\d+)/(\d{4})\s*-\s*TCU\s*-\s*(Plen[áa]rio|1[ªa]\s*C[âa]mara|2[ªa]\s*C[âa]mara)',
                markdown, re.IGNORECASE
            )
            if titulo_match:
                doc.metadata.numero = int(titulo_match.group(1))
                doc.metadata.ano = int(titulo_match.group(2))
                colegiado_text = titulo_match.group(3).upper()
                if 'PLEN' in colegiado_text:
                    doc.metadata.colegiado = 'P'
                elif '1' in colegiado_text:
                    doc.metadata.colegiado = '1C'
                elif '2' in colegiado_text:
                    doc.metadata.colegiado = '2C'

        # Fallback adicional se ainda nao achou
        if not doc.metadata.numero:
            # Busca "Acórdão nº 2724" ou similar
            num_match = re.search(r'Ac[óo]rd[ãa]o\s+(?:n[°ºo]?\s*)?(\d+)', header, re.IGNORECASE)
            if num_match:
                doc.metadata.numero = int(num_match.group(1))

            # Busca ano na data ou no texto
            ano_match = re.search(r'/(\d{4})', header)
            if ano_match:
                doc.metadata.ano = int(ano_match.group(1))

            # Busca colegiado
            if 'PLEN' in header.upper() or 'Plenário' in header:
                doc.metadata.colegiado = 'P'
            elif '1ª' in header or 'PRIMEIRA' in header.upper():
                doc.metadata.colegiado = '1C'
            elif '2ª' in header or 'SEGUNDA' in header.upper():
                doc.metadata.colegiado = '2C'

    def _extract_sumario(self, markdown: str, doc: ParsedAcordao):
        """Extrai bloco do sumario."""
        match = self.PATTERN_SUMARIO.search(markdown)
        if match:
            text = match.group(1).strip()
            # Remove quebras de linha extras e normaliza
            text = re.sub(r'\n+', ' ', text).strip()
            if text:
                span = AcordaoSpan(
                    span_id="SUMARIO",
                    span_type=AcordaoSpanType.SUMARIO,
                    text=text,
                    start_pos=match.start(),
                    end_pos=match.end(),
                )
                doc.add_span(span)
        else:
            # Tenta formato inline: "SUMÁRIO: texto..."
            inline_match = re.search(
                r'(?:^|\n)SUM[ÁA]RIO[:\s]+([^\n]+)',
                markdown, re.IGNORECASE
            )
            if inline_match:
                text = inline_match.group(1).strip()
                if text:
                    span = AcordaoSpan(
                        span_id="SUMARIO",
                        span_type=AcordaoSpanType.SUMARIO,
                        text=text,
                        start_pos=inline_match.start(),
                        end_pos=inline_match.end(),
                    )
                    doc.add_span(span)

    def _extract_relatorio(self, markdown: str, doc: ParsedAcordao):
        """Extrai paragrafos do relatorio."""
        match = self.PATTERN_RELATORIO.search(markdown)
        if not match:
            return

        relatorio_text = match.group(1)
        relatorio_start = match.start()

        # Extrai paragrafos numerados
        paragrafos = self._extract_paragrafos_numerados(
            relatorio_text,
            prefix="REL",
            span_type=AcordaoSpanType.RELATORIO,
            base_pos=relatorio_start,
        )

        for span in paragrafos:
            doc.add_span(span)

    def _extract_voto(self, markdown: str, doc: ParsedAcordao):
        """Extrai paragrafos do voto."""
        match = self.PATTERN_VOTO.search(markdown)
        if not match:
            return

        voto_text = match.group(1)
        voto_start = match.start()

        # Extrai paragrafos numerados
        paragrafos = self._extract_paragrafos_numerados(
            voto_text,
            prefix="VOTO",
            span_type=AcordaoSpanType.VOTO,
            base_pos=voto_start,
        )

        for span in paragrafos:
            doc.add_span(span)

    def _extract_acordao(self, markdown: str, doc: ParsedAcordao):
        """Extrai bloco do acordao e deliberacoes."""
        # Encontra inicio do ACORDAO
        match = self.PATTERN_ACORDAO.search(markdown)
        if not match:
            # Tenta encontrar "ACORDAM os Ministros"
            acordam_match = re.search(r'\bACORDAM\s+os\s+Ministros', markdown, re.IGNORECASE)
            if acordam_match:
                # Pega o texto ate a primeira deliberacao
                start = acordam_match.start()
                delib_match = re.search(r'\n\s*9\.1', markdown[start:])
                if delib_match:
                    end = start + delib_match.start()
                else:
                    end = len(markdown)

                acordao_text = markdown[start:end].strip()
            else:
                return
        else:
            acordao_text = match.group(0).strip()

        # Cria span do ACORDAO principal
        span = AcordaoSpan(
            span_id="ACORDAO",
            span_type=AcordaoSpanType.ACORDAO,
            text=acordao_text[:1000],  # Limita tamanho
            start_pos=match.start() if match else 0,
            end_pos=match.end() if match else 0,
        )
        doc.add_span(span)

        # Extrai deliberacoes (9.1, 9.2, etc.)
        self._extract_deliberacoes(markdown, doc)

    def _extract_deliberacoes(self, markdown: str, doc: ParsedAcordao):
        """Extrai deliberacoes do acordao (itens 9.1, 9.2, etc.)."""
        for match in self.PATTERN_DELIBERACAO.finditer(markdown):
            major = match.group(1)  # 9
            minor = match.group(2)  # 1, 2, 3...
            text = match.group(3).strip()

            if len(text) < self.config.min_paragrafo_chars:
                continue

            span = AcordaoSpan(
                span_id=f"ACORDAO-{major}-{minor}",
                span_type=AcordaoSpanType.DELIBERACAO,
                text=f"{major}.{minor}. {text}",
                identifier=f"{major}.{minor}",
                parent_id="ACORDAO",  # Deliberacoes sao filhas do ACORDAO
                start_pos=match.start(),
                end_pos=match.end(),
            )
            doc.add_span(span)

    def _extract_paragrafos_numerados(
        self,
        text: str,
        prefix: str,
        span_type: AcordaoSpanType,
        base_pos: int = 0,
    ) -> List[AcordaoSpan]:
        """
        Extrai paragrafos numerados (1., 2., 3., etc.).

        Args:
            text: Texto da secao
            prefix: Prefixo do span_id (REL, VOTO)
            span_type: Tipo do span
            base_pos: Posicao base no documento original

        Returns:
            Lista de spans
        """
        spans = []
        counter = 0  # Contador sequencial para garantir IDs unicos

        for match in self.PATTERN_PARAGRAFO_NUM.finditer(text):
            numero = match.group(1)
            content = match.group(2).strip()

            if len(content) < self.config.min_paragrafo_chars:
                continue

            counter += 1
            span = AcordaoSpan(
                span_id=f"{prefix}-{counter:03d}",  # Usa contador sequencial, nao numero do paragrafo
                span_type=span_type,
                text=f"{numero}. {content}",
                identifier=numero,  # Mantem numero original no identifier
                start_pos=base_pos + match.start(),
                end_pos=base_pos + match.end(),
            )
            spans.append(span)

        return spans

    def parse_to_annotated(self, markdown: str) -> str:
        """
        Parseia e retorna markdown anotado com span_ids.

        Returns:
            Markdown com cada span prefixado por [SPAN_ID]
        """
        doc = self.parse(markdown)
        return doc.to_annotated_markdown()


# =============================================================================
# FUNCOES AUXILIARES
# =============================================================================

def extract_acordao_id_from_pdf(markdown: str) -> Optional[str]:
    """
    Extrai e normaliza acordao_id de um markdown de acordao.

    Args:
        markdown: Texto em markdown do acordao

    Returns:
        acordao_id normalizado ou None
    """
    # Busca codigo eletronico
    pattern = re.compile(r'C[óo]digo\s+eletr[ôo]nico[:\s]+([A-Z0-9\-/]+)', re.IGNORECASE)
    match = pattern.search(markdown)
    if match:
        return normalize_acordao_id(match.group(1))

    # Busca alternativa: "Acórdão nº X/YYYY" ou "AC-X-Y/YY-Z"
    alt_pattern = re.compile(r'AC[–-](\d+)[–-]\d+/(\d+)[–-]([A-Z0-9]+)')
    alt_match = alt_pattern.search(markdown)
    if alt_match:
        return normalize_acordao_id(alt_match.group(0))

    return None


def classify_deliberacao_type(text: str) -> str:
    """
    Classifica o tipo de deliberacao baseado no texto.

    Returns:
        Tipo: determinacao, recomendacao, ciencia, arquivamento, etc.
    """
    text_lower = text.lower()

    if 'determinar' in text_lower or 'determine' in text_lower:
        return 'determinacao'
    if 'recomendar' in text_lower or 'recomende' in text_lower:
        return 'recomendacao'
    if 'ciência' in text_lower or 'dar ciencia' in text_lower:
        return 'ciencia'
    if 'arquiv' in text_lower:
        return 'arquivamento'
    if 'aplicar' in text_lower and 'multa' in text_lower:
        return 'multa'
    if 'julgar' in text_lower and 'regular' in text_lower:
        return 'julgamento_regular'
    if 'julgar' in text_lower and 'irregular' in text_lower:
        return 'julgamento_irregular'
    if 'autorizar' in text_lower:
        return 'autorizacao'

    return 'outro'
