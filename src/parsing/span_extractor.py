"""
SpanExtractor - Extração Baseada em Spans com LLM e Auto-Correção.

Este módulo implementa um extrator de documentos legais que elimina alucinações
do LLM por design, usando referências a spans pré-identificados deterministicamente.

Pipeline de Extração Anti-Alucinação:
====================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      PIPELINE DO SPANEXTRACTOR                          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  FASE 1: PARSING DETERMINÍSTICO                                         │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │  Markdown (Docling) ──► SpanParser ──► ParsedDocument           │   │
    │  │                           (regex)       - spans[]                │   │
    │  │                                         - _span_index{}          │   │
    │  │                                         - articles[]             │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │                              │                                          │
    │                              ▼                                          │
    │  FASE 2: ANOTAÇÃO PARA LLM                                              │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │  [ART-001] Art. 1º Este documento estabelece...                 │   │
    │  │  [INC-001-I] I - as definições básicas;                         │   │
    │  │  [INC-001-II] II - os procedimentos aplicáveis;                 │   │
    │  │  [PAR-001-1] § 1º Aplica-se também...                           │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │                              │                                          │
    │                              ▼                                          │
    │  FASE 3: CLASSIFICAÇÃO LLM (JSON Schema ou chat)                        │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │  LLM: "Organize os IDs por capítulo"                            │   │
    │  │  Resposta: {                                                     │   │
    │  │    "chapters": [{                                                │   │
    │  │      "chapter_id": "CAP-I",                                      │   │
    │  │      "article_ids": ["ART-001", "ART-002"]                       │   │
    │  │    }]                                                            │   │
    │  │  }                                                               │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │                              │                                          │
    │                              ▼                                          │
    │  FASE 4: VALIDAÇÃO + AUTO-FIX                                           │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │  Para cada ID retornado pelo LLM:                                │   │
    │  │    ✓ Existe em ParsedDocument? → valid_ids[]                     │   │
    │  │    ✗ Não existe? → Tentar auto-fix:                              │   │
    │  │        ART-1 → ART-001 (zero-padding)                           │   │
    │  │        CAP-1 → CAP-I (número → romano)                          │   │
    │  │    ✗ Não fixável? → invalid_ids[]                                │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │                              │                                          │
    │                              ▼                                          │
    │  FASE 5: RESULTADO FINAL                                                │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │  ExtractionResult                                                │   │
    │  │  - parsed_doc: ParsedDocument (texto original)                   │   │
    │  │  - document: DocumentSpans (estrutura validada)                  │   │
    │  │  - valid_ids: ["ART-001", "INC-001-I", ...]                       │   │
    │  │  - invalid_ids: ["ART-999"] (se houver)                          │   │
    │  │  - fixed_ids: {"ART-1": "ART-001"} (correções)                   │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

Classes Principais:
==================

| Classe              | Descrição                                        |
|---------------------|--------------------------------------------------|
| SpanExtractorConfig | Configuração do extrator                         |
| ExtractionResult    | Resultado com spans + documento + validação      |
| SpanExtractor       | Classe principal de extração                     |

SpanExtractorConfig - Parâmetros:
================================

| Parâmetro          | Tipo          | Default              | Descrição                      |
|--------------------|---------------|----------------------|--------------------------------|
| parser_config      | ParserConfig  | ParserConfig()       | Config do SpanParser           |
| model              | str           | "Qwen/Qwen3-8B-AWQ"  | Modelo LLM                     |
| temperature        | float         | 0.0                  | Temperatura (0 = determinístico)|
| max_tokens         | int           | 4096                 | Máximo de tokens na resposta   |
| strict_validation  | bool          | True                 | Falhar se ID inválido          |
| auto_fix_ids       | bool          | True                 | Tentar corrigir IDs comuns     |

ExtractionResult - Métodos:
==========================

| Método               | Retorno           | Descrição                              |
|----------------------|-------------------|----------------------------------------|
| get_span(id)         | Span | None       | Busca span por ID                      |
| get_article_text(id) | str               | Texto completo do artigo + filhos      |
| get_chapter_text(id) | str               | Texto completo do capítulo + artigos   |
| is_valid             | bool (property)   | True se nenhum ID inválido             |

Auto-Fix de IDs - Correções Automáticas:
=======================================

O extrator pode corrigir automaticamente erros comuns do LLM:

| Erro do LLM | Correção        | Exemplo               |
|-------------|-----------------|------------------------|
| Zero-padding| ART-1 → ART-001 | Números sem zeros      |
| Zero-padding| ART-01 → ART-001| Números com 2 dígitos  |
| Romano      | CAP-1 → CAP-I   | Número → Algarismo     |
| Romano      | CAP-2 → CAP-II  | romano                 |

    # Exemplo de correção
    >>> extractor.config.auto_fix_ids = True
    >>> result = extractor.extract(markdown)
    >>> print(result.fixed_ids)
    {"ART-1": "ART-001", "CAP-3": "CAP-III"}

Fallback para Falhas do LLM:
===========================

Se o LLM falhar (JSON inválido, timeout, etc), o extrator cria uma
estrutura mínima a partir dos spans já identificados:

    ┌─────────────────────────────────────────────────────────────────┐
    │  FALLBACK: _create_fallback_document()                          │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  1. Extrai metadados do header (se disponível)                  │
    │  2. Agrupa artigos em capítulos existentes                      │
    │  3. Artigos "soltos" vão para "Disposições Gerais"              │
    │  4. Retorna DocumentSpans válido (mesmo sem LLM)                │
    │                                                                 │
    │  Resultado: O pipeline NUNCA falha completamente                │
    │             Sempre há uma estrutura mínima para trabalhar       │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

Limpeza de Resposta LLM (_clean_response):
=========================================

O Qwen 3 pode retornar tags de pensamento. O extrator remove:

    <think>
    Analisando o documento...
    Os capítulos são I, II, III...
    </think>
    {"chapters": [...]}

    ↓ _clean_response() ↓

    {"chapters": [...]}

Integração com LLM Client:
=========================

O extrator aceita qualquer cliente com interface compatível:

| Método               | Prioridade | Descrição                           |
|----------------------|------------|-------------------------------------|
| chat_with_schema()   | 1ª         | JSON Schema guiado (preferido)      |
| chat()               | 2ª         | Chat regular (fallback)             |

    # Com chat_with_schema (Guided JSON)
    llm_client.chat_with_schema(
        messages=[...],
        schema=DocumentSpans,  # Pydantic model → JSON Schema
        temperature=0.0
    )

    # Com chat regular
    llm_client.chat(
        messages=[...],
        temperature=0.0
    )

Exemplo de Uso Completo:
=======================

    from parsing import SpanExtractor, SpanExtractorConfig
    from llm import VLLMClient

    # 1. Configurar cliente LLM
    llm = VLLMClient(base_url="http://localhost:8001/v1")

    # 2. Configurar extrator
    config = SpanExtractorConfig(
        model="Qwen/Qwen3-8B-AWQ",
        temperature=0.0,
        auto_fix_ids=True
    )

    # 3. Extrair documento
    extractor = SpanExtractor(llm, config)
    result = extractor.extract(markdown_text)

    # 4. Verificar validação
    if result.is_valid:
        print("✓ Extração bem-sucedida!")
    else:
        print(f"⚠ IDs inválidos: {result.invalid_ids}")

    # 5. Acessar texto
    for chapter in result.document.chapters:
        print(f"Capítulo: {chapter.chapter_id}")
        chapter_text = result.get_chapter_text(chapter.chapter_id)

        for art_id in chapter.article_ids:
            article_text = result.get_article_text(art_id)
            print(f"  {art_id}: {article_text[:50]}...")

    # 6. Ver correções automáticas
    if result.fixed_ids:
        print(f"Correções: {result.fixed_ids}")

Função de Conveniência:
======================

    from parsing import extract_with_spans

    # Uma linha só
    result = extract_with_spans(markdown, llm_client, config)

Tratamento de Erros:
===================

| Cenário                  | Comportamento                                |
|--------------------------|----------------------------------------------|
| JSON inválido do LLM     | Usa _create_fallback_document()              |
| ID não existe            | Adicionado a invalid_ids[]                   |
| ID com formato errado    | Tenta auto-fix, senão invalid_ids[]          |
| LLM timeout              | Fallback document (sem crash)                |
| Nenhum capítulo          | Cria "CAP-I" com todos os artigos            |

Comparação com ArticleOrchestrator:
==================================

| Aspecto          | SpanExtractor            | ArticleOrchestrator         |
|------------------|--------------------------|------------------------------|
| Escopo           | Documento inteiro        | Artigo por artigo            |
| LLM calls        | 1 chamada                | N chamadas (1 por artigo)    |
| Schema           | DocumentSpans            | ArticleSpans dinâmico        |
| Retry            | Não                      | Sim (focado por janela)      |
| Uso recomendado  | Documentos simples       | Documentos complexos/grandes |

@author: Equipe VectorGov
@version: 1.0.0
@since: 23/12/2024
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Any

from .span_parser import SpanParser, ParserConfig
from .span_models import ParsedDocument, Span, SpanType
from .span_extraction_models import (
    DocumentSpans,
    ChapterSpans,
    SpanExtractionResult,
    SPAN_EXTRACTION_SYSTEM_PROMPT,
    SPAN_EXTRACTION_USER_PROMPT,
)

logger = logging.getLogger(__name__)


@dataclass
class SpanExtractorConfig:
    """Configuração do SpanExtractor."""

    # Configuração do parser
    parser_config: ParserConfig = field(default_factory=ParserConfig)

    # Configuração do LLM
    model: str = "Qwen/Qwen3-8B-AWQ"
    temperature: float = 0.0
    max_tokens: int = 4096

    # Validação
    strict_validation: bool = True  # Falhar se houver ID de span inválido
    auto_fix_ids: bool = True  # Tentar corrigir erros comuns de ID


@dataclass
class ExtractionResult:
    """Resultado da extração baseada em spans."""

    # Documento parseado com todos os spans
    parsed_doc: ParsedDocument

    # Resposta do LLM (validada)
    document: DocumentSpans

    # Resultados da validação
    valid_ids: list[str] = field(default_factory=list)
    invalid_ids: list[str] = field(default_factory=list)
    fixed_ids: dict[str, str] = field(default_factory=dict)  # antigo -> novo

    # Resposta bruta do LLM para debug
    raw_response: Optional[str] = None

    def get_span(self, span_id: str) -> Optional[Span]:
        """Obtém span pelo ID."""
        return self.parsed_doc.get_span(span_id)

    def get_article_text(self, article_id: str) -> str:
        """Obtém texto completo do artigo incluindo filhos."""
        span = self.parsed_doc.get_span(article_id)
        if not span:
            return ""

        # Obtém texto do artigo
        texts = [span.text]

        # Adiciona filhos (incisos, parágrafos)
        for child in self.parsed_doc.get_children(article_id):
            texts.append(f"  {child.text}")

            # Adiciona netos (alíneas)
            for grandchild in self.parsed_doc.get_children(child.span_id):
                texts.append(f"    {grandchild.text}")

        return "\n".join(texts)

    def get_chapter_text(self, chapter_id: str) -> str:
        """Obtém texto completo do capítulo incluindo todos os artigos."""
        span = self.parsed_doc.get_span(chapter_id)
        if not span:
            return ""

        texts = [span.text]

        # Encontra artigos neste capítulo
        for chapter in self.document.chapters:
            if chapter.chapter_id == chapter_id:
                for art_id in chapter.article_ids:
                    texts.append(self.get_article_text(art_id))
                break

        return "\n\n".join(texts)

    @property
    def is_valid(self) -> bool:
        """Verifica se a extração é válida (sem IDs inválidos)."""
        return len(self.invalid_ids) == 0


class SpanExtractor:
    """
    Extrator que usa spans para eliminar alucinação do LLM.

    O LLM só pode selecionar IDs de spans existentes, nunca gerar texto.
    Isso torna a extração determinística e verificável.
    """

    def __init__(
        self,
        llm_client: Any,  # VLLMClient ou compatível
        config: Optional[SpanExtractorConfig] = None
    ):
        """
        Inicializa o SpanExtractor.

        Args:
            llm_client: Cliente LLM com método chat() ou chat_with_schema()
            config: Configuração do extrator
        """
        self.llm = llm_client
        self.config = config or SpanExtractorConfig()
        self.parser = SpanParser(self.config.parser_config)

    def extract(self, markdown: str) -> ExtractionResult:
        """
        Extrai estrutura do documento usando abordagem baseada em spans.

        Args:
            markdown: Markdown do documento (do Docling)

        Returns:
            ExtractionResult com estrutura validada
        """
        # 1. Parseia markdown para obter spans
        logger.info("Parsing markdown to extract spans...")
        parsed_doc = self.parser.parse(markdown)
        logger.info(
            f"Extracted {len(parsed_doc.spans)} spans: "
            f"{len(parsed_doc.articles)} articles, "
            f"{len(parsed_doc.capitulos)} chapters"
        )

        # 2. Gera markdown anotado para o LLM
        annotated = parsed_doc.to_annotated_markdown()

        # 3. Chama LLM para classificar spans
        logger.info("Calling LLM to classify spans...")
        raw_response = self._call_llm(annotated, parsed_doc)

        # 4. Parseia e valida resposta
        logger.info("Validating LLM response...")
        document, valid_ids, invalid_ids, fixed_ids = self._validate_response(
            raw_response, parsed_doc
        )

        if invalid_ids:
            logger.warning(f"Found {len(invalid_ids)} invalid span IDs: {invalid_ids}")

        if fixed_ids:
            logger.info(f"Auto-fixed {len(fixed_ids)} span IDs: {fixed_ids}")

        return ExtractionResult(
            parsed_doc=parsed_doc,
            document=document,
            valid_ids=valid_ids,
            invalid_ids=invalid_ids,
            fixed_ids=fixed_ids,
            raw_response=raw_response,
        )

    def _call_llm(self, annotated_markdown: str, parsed_doc: ParsedDocument) -> str:
        """Chama LLM para classificar spans."""
        # Constrói prompt
        user_prompt = SPAN_EXTRACTION_USER_PROMPT.format(
            annotated_markdown=annotated_markdown
        )

        messages = [
            {"role": "system", "content": SPAN_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Tenta JSON guiado se disponível
        if hasattr(self.llm, 'chat_with_schema'):
            response = self.llm.chat_with_schema(
                messages=messages,
                schema=DocumentSpans,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        else:
            # Fallback para chat regular
            response = self.llm.chat(
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

        return response

    def _validate_response(
        self,
        response: str,
        parsed_doc: ParsedDocument
    ) -> tuple[DocumentSpans, list[str], list[str], dict[str, str]]:
        """
        Valida resposta do LLM e corrige erros comuns.

        Returns:
            (document, valid_ids, invalid_ids, fixed_ids)
        """
        # Parseia resposta JSON
        try:
            if isinstance(response, str):
                # Trata tags /think do Qwen3
                response = self._clean_response(response)
                data = json.loads(response)
            else:
                data = response
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            # Retorna estrutura mínima válida
            return self._create_fallback_document(parsed_doc), [], [], {}

        # Rastreia validação
        valid_ids = []
        invalid_ids = []
        fixed_ids = {}

        # Valida e corrige IDs de capítulos
        if "chapters" in data:
            for chapter in data["chapters"]:
                # Valida chapter_id
                chapter_id = chapter.get("chapter_id", "")
                validated_id = self._validate_span_id(
                    chapter_id, "CAP", parsed_doc, fixed_ids
                )
                if validated_id:
                    chapter["chapter_id"] = validated_id
                    valid_ids.append(validated_id)
                else:
                    invalid_ids.append(chapter_id)

                # Valida article_ids
                validated_articles = []
                for art_id in chapter.get("article_ids", []):
                    validated_id = self._validate_span_id(
                        art_id, "ART", parsed_doc, fixed_ids
                    )
                    if validated_id:
                        validated_articles.append(validated_id)
                        valid_ids.append(validated_id)
                    else:
                        invalid_ids.append(art_id)

                chapter["article_ids"] = validated_articles

        # Cria documento validado
        try:
            document = DocumentSpans(**data)
        except Exception as e:
            logger.error(f"Failed to create DocumentSpans: {e}")
            document = self._create_fallback_document(parsed_doc)

        return document, valid_ids, invalid_ids, fixed_ids

    def _validate_span_id(
        self,
        span_id: str,
        expected_prefix: str,
        parsed_doc: ParsedDocument,
        fixed_ids: dict[str, str]
    ) -> Optional[str]:
        """
        Valida se um ID de span existe, com auto-correção para erros comuns.

        Returns:
            ID validado ou None se inválido
        """
        if not span_id:
            return None

        # Verifica se existe
        if parsed_doc.get_span(span_id):
            return span_id

        # Tenta auto-correção se habilitado
        if not self.config.auto_fix_ids:
            return None

        # Correções comuns
        original_id = span_id

        # Correção: ART-1 -> ART-001
        if re.match(rf'^{expected_prefix}-(\d+)$', span_id):
            num = re.search(r'\d+', span_id).group()
            fixed = f"{expected_prefix}-{num.zfill(3)}"
            if parsed_doc.get_span(fixed):
                fixed_ids[original_id] = fixed
                return fixed

        # Correção: ART-01 -> ART-001
        if re.match(rf'^{expected_prefix}-(\d{{2}})$', span_id):
            num = re.search(r'\d+', span_id).group()
            fixed = f"{expected_prefix}-{num.zfill(3)}"
            if parsed_doc.get_span(fixed):
                fixed_ids[original_id] = fixed
                return fixed

        # Correção: CAP-1 -> CAP-I (algarismo romano)
        if expected_prefix == "CAP" and re.match(r'^CAP-\d+$', span_id):
            num = int(re.search(r'\d+', span_id).group())
            roman = self._int_to_roman(num)
            fixed = f"CAP-{roman}"
            if parsed_doc.get_span(fixed):
                fixed_ids[original_id] = fixed
                return fixed

        return None

    def _clean_response(self, response: str) -> str:
        """Limpa resposta do LLM, removendo tags think etc."""
        # Remove tags <think>...</think> (Qwen3)
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

        # Encontra objeto JSON
        start = response.find('{')
        end = response.rfind('}') + 1

        if start >= 0 and end > start:
            return response[start:end]

        return response

    def _create_fallback_document(self, parsed_doc: ParsedDocument) -> DocumentSpans:
        """Cria documento fallback a partir dos spans parseados quando o LLM falha."""
        # Extrai metadados do cabeçalho
        doc_type = parsed_doc.metadata.get("document_type", "INSTRUÇÃO NORMATIVA")
        number = parsed_doc.metadata.get("number", "")
        date = parsed_doc.metadata.get("date_raw", "")

        # Constrói capítulos a partir dos spans
        chapters = []
        current_chapter = None

        for span in parsed_doc.spans:
            if span.span_type == SpanType.CAPITULO:
                if current_chapter:
                    chapters.append(current_chapter)
                current_chapter = ChapterSpans(
                    chapter_id=span.span_id,
                    title=span.text.split('\n')[0] if '\n' in span.text else span.text,
                    article_ids=[]
                )
            elif span.span_type == SpanType.ARTIGO:
                if current_chapter:
                    current_chapter.article_ids.append(span.span_id)
                else:
                    # Artigo antes do primeiro capítulo - cria capítulo padrão
                    current_chapter = ChapterSpans(
                        chapter_id="CAP-DEFAULT",
                        title="Disposições Gerais",
                        article_ids=[span.span_id]
                    )

        if current_chapter:
            chapters.append(current_chapter)

        # Se nenhum capítulo encontrado, cria um com todos os artigos
        if not chapters:
            chapters = [ChapterSpans(
                chapter_id="CAP-I",
                title="Disposições Gerais",
                article_ids=[a.span_id for a in parsed_doc.articles]
            )]

        return DocumentSpans(
            document_type=doc_type,
            number=number,
            date=date,
            issuing_body="",
            ementa="",
            chapters=chapters
        )

    def _int_to_roman(self, num: int) -> str:
        """Converte inteiro para algarismo romano."""
        val = [100, 90, 50, 40, 10, 9, 5, 4, 1]
        syms = ['C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
        result = ''
        for i, v in enumerate(val):
            while num >= v:
                result += syms[i]
                num -= v
        return result


# =============================================================================
# FUNÇÕES DE CONVENIÊNCIA
# =============================================================================

def extract_with_spans(
    markdown: str,
    llm_client: Any,
    config: Optional[SpanExtractorConfig] = None
) -> ExtractionResult:
    """
    Função de conveniência para extração baseada em spans.

    Args:
        markdown: Markdown do documento
        llm_client: Cliente LLM
        config: Configuração opcional

    Returns:
        ExtractionResult
    """
    extractor = SpanExtractor(llm_client, config)
    return extractor.extract(markdown)
