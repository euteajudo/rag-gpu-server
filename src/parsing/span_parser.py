"""
SpanParser - Parser Determinístico para Documentos Legais Brasileiros.

Este módulo implementa um parser regex-first que identifica a estrutura
hierárquica de documentos legais de forma determinística. O LLM NUNCA
descobre estrutura - apenas classifica ou enriquece os spans já identificados
pelo parser.

Arquitetura Anti-Alucinação:
===========================

    Markdown (Docling)
           │
           ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         SpanParser                                  │
    │                    (100% Determinístico)                            │
    │                                                                     │
    │  Markdown ──► [Regex Patterns] ──► Spans com IDs únicos             │
    │                                                                     │
    │  Não usa LLM para descobrir estrutura.                              │
    │  Padrões regex capturam TODA a hierarquia do documento.             │
    │  IDs gerados são determinísticos e verificáveis.                    │
    └─────────────────────────────────────────────────────────────────────┘
           │
           ▼
    ParsedDocument com spans indexados
           │
           ├── spans: List[Span] - todos os spans do documento
           ├── articles: List[Span] - apenas artigos
           ├── capitulos: List[Span] - apenas capítulos
           └── _span_index: Dict[str, Span] - lookup por ID

Hierarquia de Documentos Legais Brasileiros:
============================================

    CAPÍTULO (CAP-I, CAP-II, CAP-III)
        │
        └── Seção (SE[ÇC][ÃA]O I, II...)
              │
              └── Subseção (opcional)
                    │
                    └── Artigo (ART-001, ART-002)
                          │
                          ├── Caput (texto principal do artigo)
                          │
                          ├── Parágrafo (PAR-001-1, PAR-001-UNICO)
                          │     │
                          │     └── Inciso (dentro do parágrafo)
                          │
                          └── Inciso (INC-001-I, INC-001-II)
                                │
                                └── Alínea (ALI-001-I-a)
                                      │
                                      └── Item (1), 2)... - raro)

Padrões Regex Implementados:
===========================

| Elemento   | Pattern Base                              | Exemplo Match        |
|------------|-------------------------------------------|----------------------|
| CAPÍTULO   | `^(?:CAPÍTULO|CAP\\.?)\\s+([IVXLC]+)`     | "CAPÍTULO I", "CAP. II"|
| Seção      | `^SE[ÇC][ÃA]O\\s+([IVXLC]+)`              | "Seção I", "SEÇÃO II"|
| Artigo     | `^[-*]?\\s*Art\\.?\\s*(\\d+)[°ºo]?`       | "Art. 1º", "- Art. 10"|
| Parágrafo  | `^[-*]?\\s*(?:§\\s*(\\d+)[°ºo]?|Par...)`  | "§ 1º", "Parágrafo único"|
| Inciso     | `^[-*]?\\s*([IVXLC]+)\\s*[-–—]`           | "I -", "II –", "III -"|
| Alínea     | `^[-*]?\\s*([a-z])\\)`                    | "a)", "b)", "c)"     |
| Item       | `^[-*]?\\s*(\\d+)\\)`                     | "1)", "2)", "3)"     |

Formato de Span IDs:
===================

| Tipo      | Formato              | Exemplo                      |
|-----------|----------------------|------------------------------|
| Capítulo  | CAP-{romano}         | CAP-I, CAP-II, CAP-III       |
| Artigo    | ART-{nnn}            | ART-001, ART-012, ART-100    |
| Parágrafo | PAR-{art}-{n}        | PAR-001-1, PAR-001-UNICO     |
| Inciso    | INC-{art}-{romano}   | INC-001-I, INC-001-IV        |
| Alínea    | ALI-{art}-{inc}-{l}  | ALI-001-I-a, ALI-001-II-b    |
| Cabeçalho | HDR-{seq}            | HDR-001                      |

Fluxo de Parsing:
================

    1. parse(markdown)
           │
           ├── _normalize_whitespace()     # Limpa espaços extras
           │
           ├── _extract_header()           # HDR-001 (ementa, órgão)
           │       │
           │       └── _parse_header_metadata() # Extrai tipo, número, data
           │
           ├── _extract_capitulos()        # CAP-I, CAP-II...
           │
           ├── _extract_artigos()          # ART-001, ART-002...
           │       │
           │       └── _find_parent_capitulo() # Vincula artigo ao capítulo
           │
           └── Para cada artigo:
                   │
                   └── _extract_article_children()
                           │
                           ├── _extract_paragrafos() # PAR-001-1...
                           │       │
                           │       └── _extract_incisos() # INC-001-I_2 (dentro de §)
                           │
                           └── _extract_incisos()   # INC-001-I (do caput)
                                   │
                                   └── _extract_alineas() # ALI-001-I-a

Desambiguação de Incisos:
========================

Quando o mesmo numeral romano aparece em múltiplos contextos (ex: inciso I
no caput e no §2), adiciona-se sufixo sequencial:

    Art. 5º ...
        I - texto do inciso I do caput     → INC-005-I
        II - texto do inciso II            → INC-005-II

    § 2º ...
        I - texto do inciso I do §2        → INC-005-I_2  (sufixo _2)
        II - texto do inciso II do §2      → INC-005-II_2

O sufixo é apenas um DESAMBIGUADOR, não indica qual parágrafo é o parent.
O parent é determinado pela posição no texto e armazenado em span.parent_id.

Exemplo de Uso:
==============

    ```python
    from parsing import SpanParser, ParserConfig

    # Parser com configuração padrão
    parser = SpanParser()

    # Parseia markdown extraído pelo Docling
    doc = parser.parse(markdown_text)

    print(f"Total de spans: {len(doc.spans)}")
    print(f"Artigos: {len(doc.articles)}")
    print(f"Capítulos: {len(doc.capitulos)}")

    # Itera sobre artigos
    for article in doc.articles:
        print(f"{article.span_id}: {article.text[:50]}...")

        # Obtém filhos do artigo
        for child in doc.get_children(article.span_id):
            print(f"  {child.span_id}: {child.text[:30]}...")

    # Gera markdown anotado para o LLM
    annotated = doc.to_annotated_markdown()
    # [ART-001] Art. 1º Para fins do disposto...
    # [PAR-001-1] § 1º O termo X significa...
    # [INC-001-I] I - definição de...
    ```

Configuração:
============

    ```python
    config = ParserConfig(
        include_headers=True,       # Extrai cabeçalho (ementa)
        include_texto_livre=False,  # Ignora texto entre estruturas
        normalize_whitespace=True,  # Remove espaços duplicados
        extract_titles=True,        # Extrai títulos de seções
    )

    parser = SpanParser(config)
    ```

Funções Auxiliares:
==================

- roman_to_int(roman: str) -> int: Converte "IV" → 4
- int_to_roman(num: int) -> str: Converte 4 → "IV"

Módulos Relacionados:
====================

- span_models.py: Estruturas de dados Span, SpanType, ParsedDocument
- article_orchestrator.py: Extração LLM por artigo usando spans
- page_spans.py: Coordenadas PDF para citações visuais

@author: Equipe VectorGov
@version: 1.0.0
@since: 23/12/2024
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional

from .span_models import Span, SpanType, ParsedDocument

logger = logging.getLogger(__name__)


@dataclass
class ParserConfig:
    """Configuração do parser."""

    # Padrões podem ser customizados por tipo de documento
    include_headers: bool = True
    include_texto_livre: bool = False  # Texto entre estruturas
    normalize_whitespace: bool = True
    extract_titles: bool = True  # Títulos de artigos/seções


class SpanParser:
    """
    Parser determinístico para documentos legais brasileiros.

    Usa regex para identificar a estrutura hierárquica do documento,
    gerando spans com IDs únicos que podem ser referenciados pelo LLM.

    Usage:
        parser = SpanParser()
        doc = parser.parse(markdown_text)

        for span in doc.articles:
            print(f"{span.span_id}: {span.text[:50]}...")

        # Markdown anotado para o LLM
        annotated = doc.to_annotated_markdown()
    """

    # =========================================================================
    # REGEX PATTERNS - Estrutura Legal Brasileira
    # =========================================================================

    # Capítulo: "CAPÍTULO I", "CAPITULO II", "CAP. III"
    PATTERN_CAPITULO = re.compile(
        r'^(?:CAP[ÍI]TULO|CAP\.?)\s+([IVXLC]+)\b[^\n]*',
        re.IGNORECASE | re.MULTILINE
    )

    # Seção: "Seção I", "SEÇÃO II"
    PATTERN_SECAO = re.compile(
        r'^SE[ÇC][ÃA]O\s+([IVXLC]+)\b[^\n]*',
        re.IGNORECASE | re.MULTILINE
    )

    # Subseção: "Subseção I", "SUBSEÇÃO II"
    PATTERN_SUBSECAO = re.compile(
        r'^SUBSE[ÇC][ÃA]O\s+([IVXLC]+)\b[^\n]*',
        re.IGNORECASE | re.MULTILINE
    )

    # Lookahead comum para estruturas de nível superior
    # Usado para parar captura em: Capítulo, Seção, Subseção
    ESTRUTURA_SUPERIOR = r'(?:CAP[ÍI]TULO|SE[ÇC][ÃA]O|SUBSE[ÇC][ÃA]O)'

    # Artigo: "Art. 1º", "Art. 2o", "- Art. 10", "11. Art. 56" (com prefixo numérico do Docling)
    # Captura: grupo 1 = número, resto = conteúdo (até próximo artigo ou estrutura superior)
    # NÃO para nos incisos/parágrafos - eles serão extraídos depois
    # NOTA: Prefixo numérico (ex: "11.") é ignorado na captura - aparece quando Docling
    #       interpreta artigos como itens de lista numerada
    # NOTA 2: Captura opcional de sufixo letra (A-Z) para Art. 337-E, 337-F, etc.
    #         Grupo 1: número do artigo (ex: "337")
    #         Grupo 2: sufixo letra opcional (ex: "E", "F", ou None) - DEVE ser seguido de ponto
    #         Grupo 3: conteúdo do artigo
    #         O sufixo só é capturado se: "337º E." ou "337-E." (letra + ponto obrigatório)
    PATTERN_ARTIGO = re.compile(
        rf'^(?:\d+\.\s*)?[-*]?\s*Art\.?\s*(\d+)[°ºo]?\s*[-]?\s*([A-Z](?=\.))?\s*[-.]?\s*(.+?)(?=\n(?:\d+\.\s*)?[-*]?\s*Art\.?\s*\d+[°ºo]?(?:\s|[-.])|^{ESTRUTURA_SUPERIOR}|\Z)',
        re.IGNORECASE | re.MULTILINE | re.DOTALL
    )

    # Parágrafo: "§ 1º", "§ 2o", "Parágrafo único", "- § 3º", "12. § 1º" (com prefixo numérico)
    # NÃO para nos incisos/alíneas - eles serão extraídos depois
    PATTERN_PARAGRAFO = re.compile(
        rf'^(?:\d+\.\s*)?[-*]?\s*(?:§\s*(\d+)[°ºo]?|[Pp]ar[áa]grafo\s+[úu]nico)\s*[-.]?\s*(.+?)(?=\n(?:\d+\.\s*)?[-*]?\s*§\s*\d+|\n(?:\d+\.\s*)?[-*]?\s*Art\.?\s*\d+[°ºo]?|^{ESTRUTURA_SUPERIOR}|\Z)',
        re.IGNORECASE | re.MULTILINE | re.DOTALL
    )

    # Numerais romanos: I-C (1-100)
    # Estrutura: (dezenas opcionais)(unidades opcionais)
    # Dezenas: XC(90), L?X{0,3}(50-89), XL(40), ou vazio(1-9)
    # Unidades: IX(9), IV(4), V?I{0,3}(0-3,5-8)
    ROMAN_NUMERALS = r'(?:(?:XC|L?X{0,3}|XL)(?:IX|IV|V?I{0,3})|(?:IX|IV|V?I{0,3}))'

    # Inciso: "- I  -", "- II –", "III -", "12. I -" (com prefixo numérico do Docling)
    # Formato Docling varia: "- I  -  texto" ou "III - texto" (com ou sem bullet)
    # NÃO para nas alíneas - elas serão extraídas depois
    PATTERN_INCISO = re.compile(
        rf'^(?:\d+\.\s*)?[-*]?\s*({ROMAN_NUMERALS})\s*[-–—]\s*(.+?)(?=\n(?:\d+\.\s*)?[-*]?\s*{ROMAN_NUMERALS}\s*[-–—]|\n(?:\d+\.\s*)?[-*]?\s*§\s*\d+|\n(?:\d+\.\s*)?[-*]?\s*Art\.?\s*\d+[°ºo]?|\n{ESTRUTURA_SUPERIOR}|\Z)',
        re.MULTILINE | re.DOTALL
    )

    # Alínea: "a)", "b)", "c)", "12. a)" (com prefixo numérico do Docling)
    PATTERN_ALINEA = re.compile(
        rf'^(?:\d+\.\s*)?[-*]?\s*([a-z])\)\s*(.+?)(?=\n(?:\d+\.\s*)?[-*]?\s*[a-z]\)|^(?:\d+\.\s*)?[-*]?\s*{ROMAN_NUMERALS}\s*[-–]|^(?:\d+\.\s*)?[-*]?\s*§|^(?:\d+\.\s*)?[-*]?\s*Art\.?\s*\d+|^{ESTRUTURA_SUPERIOR}|\Z)',
        re.MULTILINE | re.DOTALL
    )

    # Item numérico: "1)", "2)", "3)" (dentro de alíneas, raro)
    PATTERN_ITEM = re.compile(
        r'^[-*]?\s*(\d+)\)\s*(.+?)(?=\n[-*]?\s*\d+\)|^[-*]?\s*[a-z]\)|^[-*]?\s*(?:I{1,3}|IV|VI{0,3})\s*[-–]|\Z)',
        re.MULTILINE | re.DOTALL
    )

    def __init__(self, config: Optional[ParserConfig] = None):
        """Inicializa o parser."""
        self.config = config or ParserConfig()

    def parse(self, markdown: str) -> ParsedDocument:
        """
        Parseia markdown e retorna documento com spans identificados.

        Args:
            markdown: Texto em markdown (output do Docling)

        Returns:
            ParsedDocument com todos os spans indexados
        """
        doc = ParsedDocument(source_text=markdown)

        # Normaliza whitespace se configurado
        if self.config.normalize_whitespace:
            markdown = self._normalize_whitespace(markdown)

        # 1. Extrai metadados do cabeçalho
        self._extract_header(markdown, doc)

        # 2. Extrai capítulos
        self._extract_capitulos(markdown, doc)

        # 3. Extrai artigos (principal)
        self._extract_artigos(markdown, doc)

        # 4. Para cada artigo, extrai subdivisões (passa markdown para offsets PR13)
        for span in list(doc.articles):
            self._extract_article_children(span, markdown, doc)

        logger.info(
            f"Parsed document: {len(doc.spans)} spans, "
            f"{len(doc.articles)} articles, "
            f"{len(doc.capitulos)} chapters"
        )

        return doc

    def _normalize_whitespace(self, text: str) -> str:
        """Normaliza espaços em branco."""
        # Remove espaços múltiplos (mantém quebras de linha)
        text = re.sub(r'[^\S\n]+', ' ', text)
        # Remove linhas em branco múltiplas
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _extract_header(self, markdown: str, doc: ParsedDocument):
        """Extrai cabeçalho do documento."""
        if not self.config.include_headers:
            return

        # Encontra primeiro capítulo ou artigo
        first_cap = self.PATTERN_CAPITULO.search(markdown)
        first_art = self.PATTERN_ARTIGO.search(markdown)

        end_pos = len(markdown)
        if first_cap:
            end_pos = min(end_pos, first_cap.start())
        if first_art:
            end_pos = min(end_pos, first_art.start())

        if end_pos > 100:  # Só se tiver conteúdo significativo
            header_text = markdown[:end_pos].strip()
            if header_text:
                span = Span(
                    span_id="HDR-001",
                    span_type=SpanType.HEADER,
                    text=header_text,
                    start_pos=0,
                    end_pos=end_pos,
                )
                doc.add_span(span)

                # Extrai metadados do header
                self._parse_header_metadata(header_text, doc)

    def _parse_header_metadata(self, header: str, doc: ParsedDocument):
        """Extrai metadados do cabeçalho."""
        # Tipo de documento
        tipo_match = re.search(
            r'(LEI|DECRETO|INSTRU[ÇC][ÃA]O NORMATIVA|PORTARIA|RESOLU[ÇC][ÃA]O)',
            header,
            re.IGNORECASE
        )
        if tipo_match:
            doc.metadata["document_type"] = tipo_match.group(1).upper()

        # Número
        num_match = re.search(r'N[°ºo]?\s*(\d+)', header, re.IGNORECASE)
        if num_match:
            doc.metadata["number"] = num_match.group(1)

        # Data
        data_match = re.search(
            r'(\d{1,2})\s+(?:DE\s+)?(\w+)\s+(?:DE\s+)?(\d{4})',
            header,
            re.IGNORECASE
        )
        if data_match:
            doc.metadata["date_raw"] = data_match.group(0)

    def _extract_capitulos(self, markdown: str, doc: ParsedDocument):
        """Extrai capítulos do documento."""
        for match in self.PATTERN_CAPITULO.finditer(markdown):
            numero = match.group(1)
            text = match.group(0).strip()

            # Busca título na próxima linha
            end_pos = match.end()
            next_newline = markdown.find('\n', end_pos)
            if next_newline != -1:
                # Próxima linha pode ser o título
                next_line_end = markdown.find('\n', next_newline + 1)
                if next_line_end == -1:
                    next_line_end = len(markdown)
                next_line = markdown[next_newline:next_line_end].strip()

                # Se não começa com Art. ou outro padrão, é título
                if next_line and not re.match(r'^[-*]?\s*(Art\.|§|[IVXLC]+\s*[-–]|[a-z]\))', next_line, re.IGNORECASE):
                    text = f"{text}\n{next_line}"
                    end_pos = next_line_end

            span = Span(
                span_id=f"CAP-{numero}",
                span_type=SpanType.CAPITULO,
                text=text,
                identifier=numero,
                start_pos=match.start(),
                end_pos=end_pos,
            )
            doc.add_span(span)

    def _extract_artigos(self, markdown: str, doc: ParsedDocument):
        """Extrai artigos do documento.

        PR13: O end_pos de cada artigo vai até o início do próximo artigo
        (ou fim do documento), para incluir todos os filhos no range.
        """
        # Coleta todos os matches primeiro para calcular end_pos correto
        matches = list(self.PATTERN_ARTIGO.finditer(markdown))

        for i, match in enumerate(matches):
            numero = match.group(1)
            sufixo_letra = match.group(2)  # Captura letra opcional (E, F, G para Art. 337-E, etc.)
            content = match.group(3).strip() if match.group(3) else ""

            # Limpa conteúdo (remove subdivisões que serão extraídas depois)
            content_lines = content.split('\n')
            main_content = []
            for line in content_lines:
                # Para no primeiro inciso, parágrafo, ou alínea
                if re.match(r'^[-*]?\s*(§|[IVXLC]+\s*[-–]|[a-z]\))', line.strip(), re.IGNORECASE):
                    break
                # Remove prefixo numérico de lista do Docling (ex: "12. " antes de "I -")
                clean_line = re.sub(r'^\d+\.\s*', '', line)
                main_content.append(clean_line)

            text = '\n'.join(main_content).strip()

            # Encontra capítulo pai
            parent_id = self._find_parent_capitulo(match.start(), doc)

            # PR13: end_pos = início do próximo artigo ou fim do documento
            if i + 1 < len(matches):
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(markdown)

            # Gera span_id e identifier com sufixo letra se presente (ex: ART-337-E)
            if sufixo_letra:
                span_id = f"ART-{numero.zfill(3)}-{sufixo_letra.upper()}"
                identifier = f"{numero}-{sufixo_letra.upper()}"
                text_prefix = f"Art. {numero}º {sufixo_letra.upper()}."
            else:
                span_id = f"ART-{numero.zfill(3)}"
                identifier = numero
                text_prefix = f"Art. {numero}º"

            span = Span(
                span_id=span_id,
                span_type=SpanType.ARTIGO,
                text=f"{text_prefix} {text}" if text else text_prefix,
                identifier=identifier,
                parent_id=parent_id,
                start_pos=match.start(),
                end_pos=end_pos,
                metadata={"full_match": match.group(0)},
            )
            doc.add_span(span)

    def _find_parent_capitulo(self, position: int, doc: ParsedDocument) -> Optional[str]:
        """Encontra capítulo que contém a posição."""
        parent = None
        for cap in doc.capitulos:
            if cap.start_pos < position:
                parent = cap.span_id
        return parent

    def _extract_article_children(self, article: Span, markdown: str, doc: ParsedDocument):
        """Extrai parágrafos, incisos e alíneas de um artigo.

        PR13: Usa markdown[start_pos:end_pos] para obter o texto do artigo
        e calcula offsets absolutos para todos os filhos.
        """
        # PR13: Usa slice do markdown baseado nos offsets do artigo
        full_text = markdown[article.start_pos:article.end_pos]
        base_offset = article.start_pos

        if not full_text:
            return

        art_num = article.identifier.zfill(3)

        # Encontra onde começam os parágrafos (se existirem)
        first_par = self.PATTERN_PARAGRAFO.search(full_text)
        if first_par:
            caput_text = full_text[:first_par.start()]
            paragrafos_text = full_text[first_par.start():]
            paragrafos_base = base_offset + first_par.start()
        else:
            caput_text = full_text
            paragrafos_text = ""
            paragrafos_base = 0

        # Extrai incisos do caput (antes dos parágrafos) com offsets absolutos
        self._extract_incisos(caput_text, base_offset, art_num, article.span_id, doc)

        # Extrai parágrafos (que por sua vez extraem seus próprios incisos)
        if paragrafos_text:
            self._extract_paragrafos(paragrafos_text, paragrafos_base, art_num, article.span_id, doc)

    def _extract_paragrafos(
        self,
        text: str,
        base_offset: int,
        art_num: str,
        parent_id: str,
        doc: ParsedDocument
    ):
        """Extrai parágrafos de um artigo.

        PR13: Calcula offsets absolutos usando base_offset.
        """
        for match in self.PATTERN_PARAGRAFO.finditer(text):
            numero = match.group(1)
            content = match.group(2).strip() if match.group(2) else ""

            # Determina identificador
            if numero:
                identifier = numero
                span_id = f"PAR-{art_num}-{numero}"
            else:
                identifier = "único"
                span_id = f"PAR-{art_num}-UNICO"

            # Limpa conteúdo (para no primeiro inciso ou alínea)
            content_lines = content.split('\n')
            main_content = []
            for line in content_lines:
                # Remove prefixo numérico do Docling antes de verificar
                clean_line = re.sub(r'^\d+\.\s*', '', line)
                if re.match(r'^[-*]?\s*([IVXLC]+\s*[-–]|[a-z]\))', clean_line.strip()):
                    break
                main_content.append(clean_line)

            clean_content = '\n'.join(main_content).strip()

            span = Span(
                span_id=span_id,
                span_type=SpanType.PARAGRAFO,
                text=f"§ {identifier}º {clean_content}" if identifier != "único" else f"Parágrafo único. {clean_content}",
                identifier=identifier,
                parent_id=parent_id,
                start_pos=base_offset + match.start(),
                end_pos=base_offset + match.end(),
                metadata={"full_match": match.group(0)},
            )
            doc.add_span(span)

            # Extrai incisos dentro do parágrafo (parent_id vincula ao parágrafo)
            # PR13: Passa offset absoluto do parágrafo para incisos
            inciso_base = base_offset + match.start()
            self._extract_incisos(match.group(0), inciso_base, art_num, span_id, doc)

    def _extract_incisos(
        self,
        text: str,
        base_offset: int,
        art_num: str,
        parent_id: str,
        doc: ParsedDocument
    ):
        """Extrai incisos de um artigo ou parágrafo.

        PR13: Calcula offsets absolutos usando base_offset.
        """
        for match in self.PATTERN_INCISO.finditer(text):
            romano = match.group(1)
            content = match.group(2).strip() if match.group(2) else ""

            # Limpa conteúdo (para na primeira alínea)
            content_lines = content.split('\n')
            main_content = []
            for line in content_lines:
                # Remove prefixo numérico do Docling antes de verificar
                clean_line = re.sub(r'^\d+\.\s*', '', line)
                if re.match(r'^[-*]?\s*[a-z]\)', clean_line.strip()):
                    break
                main_content.append(clean_line)

            clean_content = '\n'.join(main_content).strip()

            # ID base: INC-{art}-{romano}
            base_id = f"INC-{art_num}-{romano}"
            span_id = base_id

            # Se ID já existe, adiciona sufixo para desambiguar
            suffix = 2
            while doc.get_span(span_id) is not None:
                span_id = f"{base_id}_{suffix}"
                suffix += 1

            span = Span(
                span_id=span_id,
                span_type=SpanType.INCISO,
                text=f"{romano} - {clean_content}",
                identifier=romano,
                parent_id=parent_id,
                start_pos=base_offset + match.start(),
                end_pos=base_offset + match.end(),
                metadata={"full_match": match.group(0)},
            )
            doc.add_span(span)

            # Extrai alíneas dentro do inciso
            # PR13: Passa offset absoluto do inciso para alíneas
            alinea_base = base_offset + match.start()
            self._extract_alineas(match.group(0), alinea_base, art_num, romano, span_id, doc)

    def _extract_alineas(
        self,
        text: str,
        base_offset: int,
        art_num: str,
        inciso: str,
        parent_id: str,
        doc: ParsedDocument
    ):
        """Extrai alíneas de um inciso.

        PR13: Calcula offsets absolutos usando base_offset.
        """
        for match in self.PATTERN_ALINEA.finditer(text):
            letra = match.group(1)
            content = match.group(2).strip() if match.group(2) else ""

            # Remove prefixos numéricos do Docling do conteúdo
            content = re.sub(r'^\d+\.\s*', '', content)

            span_id = f"ALI-{art_num}-{inciso}-{letra}"

            span = Span(
                span_id=span_id,
                span_type=SpanType.ALINEA,
                text=f"{letra}) {content}",
                identifier=letra,
                parent_id=parent_id,
                start_pos=base_offset + match.start(),
                end_pos=base_offset + match.end(),
            )
            doc.add_span(span)

    def parse_to_annotated(self, markdown: str) -> str:
        """
        Parseia e retorna markdown anotado com span_ids.

        Este é o formato que será enviado ao LLM para classificação.
        O LLM só precisa selecionar IDs, não gerar texto.

        Returns:
            Markdown com cada linha prefixada por [SPAN_ID]
        """
        doc = self.parse(markdown)
        return doc.to_annotated_markdown()


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def roman_to_int(roman: str) -> int:
    """Converte numeral romano para inteiro."""
    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100}
    result = 0
    prev = 0
    for char in reversed(roman.upper()):
        curr = values.get(char, 0)
        if curr < prev:
            result -= curr
        else:
            result += curr
        prev = curr
    return result


def int_to_roman(num: int) -> str:
    """Converte inteiro para numeral romano."""
    val = [100, 90, 50, 40, 10, 9, 5, 4, 1]
    syms = ['C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
    result = ''
    for i, v in enumerate(val):
        while num >= v:
            result += syms[i]
            num -= v
    return result
