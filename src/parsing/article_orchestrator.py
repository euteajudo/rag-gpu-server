"""
Article Orchestrator - Extração LLM por Artigo com Hierarquia Completa.

Este módulo implementa a extração guiada por LLM de documentos legais,
processando artigo por artigo com validação de cobertura e retry focado.

Arquitetura Anti-Alucinação:
===========================

    O orquestrador usa múltiplas camadas de proteção contra alucinação:

    ┌─────────────────────────────────────────────────────────────────────┐
    │                   CAMADAS DE PROTEÇÃO                               │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │  1. SCHEMA ENUM DINÂMICO                                            │
    │     ┌─────────────────────────────────────────────────────────────┐ │
    │     │ Para cada artigo, gera JSON Schema com:                     │ │
    │     │ - paragrafo_ids: enum com IDs permitidos do parser         │ │
    │     │ - inciso_ids: enum com IDs permitidos do parser            │ │
    │     │                                                             │ │
    │     │ LLM só pode escolher IDs que EXISTEM no documento!          │ │
    │     └─────────────────────────────────────────────────────────────┘ │
    │                                                                     │
    │  2. VALIDAÇÃO DE COBERTURA                                          │
    │     ┌─────────────────────────────────────────────────────────────┐ │
    │     │ Parser detectou: 3 PAR, 5 INC                               │ │
    │     │ LLM retornou:    2 PAR, 4 INC                               │ │
    │     │ Cobertura:       67% PAR, 80% INC → SUSPECT                 │ │
    │     └─────────────────────────────────────────────────────────────┘ │
    │                                                                     │
    │  3. RETRY FOCADO POR JANELA                                         │
    │     ┌─────────────────────────────────────────────────────────────┐ │
    │     │ Se cobertura PAR < 100%: retry focado em parágrafos         │ │
    │     │ Se cobertura INC < 100%: retry focado em incisos            │ │
    │     │ Max 2 retries (1 para PAR, 1 para INC)                      │ │
    │     └─────────────────────────────────────────────────────────────┘ │
    │                                                                     │
    │  4. VALIDAÇÃO DE IDs                                                │
    │     ┌─────────────────────────────────────────────────────────────┐ │
    │     │ - ID deve existir no ParsedDocument                         │ │
    │     │ - Detecta duplicatas                                        │ │
    │     │ - Valida consistência parent-child                          │ │
    │     └─────────────────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────────────────┘

Fluxo de Processamento por Artigo:
=================================

    Para cada artigo no documento:

    ┌──────────────────────────────────────────────────────────────────────┐
    │                    FLUXO DE EXTRAÇÃO                                 │
    ├──────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  1. CURTO-CIRCUITO                                                   │
    │     ┌──────────────────────────────────────────────────────────────┐│
    │     │ if artigo.filhos == 0:                                       ││
    │     │     return chunk_simples  # Não chama LLM!                   ││
    │     └──────────────────────────────────────────────────────────────┘│
    │                          │                                          │
    │                          ▼ (tem filhos)                             │
    │  2. GERA DOCUMENTO ANOTADO                                          │
    │     ┌──────────────────────────────────────────────────────────────┐│
    │     │ [ART-005] Art. 5º O estudo técnico preliminar...             ││
    │     │ [PAR-005-1] § 1º O ETP deverá conter...                      ││
    │     │ [INC-005-I] I - descrição da necessidade;                    ││
    │     │ [INC-005-II] II - estimativa das quantidades;                ││
    │     └──────────────────────────────────────────────────────────────┘│
    │                          │                                          │
    │                          ▼                                          │
    │  3. EXTRAI IDs PERMITIDOS (do parser)                               │
    │     ┌──────────────────────────────────────────────────────────────┐│
    │     │ allowed_paragrafos = ["PAR-005-1"]                           ││
    │     │ allowed_incisos = ["INC-005-I", "INC-005-II", ...]           ││
    │     └──────────────────────────────────────────────────────────────┘│
    │                          │                                          │
    │                          ▼                                          │
    │  4. CHAMA LLM COM SCHEMA DINÂMICO                                   │
    │     ┌──────────────────────────────────────────────────────────────┐│
    │     │ Schema JSON: {                                               ││
    │     │   "article_id": "ART-005",                                   ││
    │     │   "paragrafo_ids": {"enum": ["PAR-005-1"]},                  ││
    │     │   "inciso_ids": {"enum": ["INC-005-I", "INC-005-II"]}        ││
    │     │ }                                                            ││
    │     │ → LLM só pode escolher desses IDs!                           ││
    │     └──────────────────────────────────────────────────────────────┘│
    │                          │                                          │
    │                          ▼                                          │
    │  5. VALIDA RESPOSTA                                                 │
    │     ┌──────────────────────────────────────────────────────────────┐│
    │     │ - IDs existem? → valid_ids / invalid_ids                     ││
    │     │ - Duplicatas?  → INVALID                                     ││
    │     │ - Cobertura?   → VALID / SUSPECT                             ││
    │     └──────────────────────────────────────────────────────────────┘│
    │                          │                                          │
    │                          ▼ (se cobertura < 100%)                    │
    │  6. RETRY FOCADO                                                    │
    │     ┌──────────────────────────────────────────────────────────────┐│
    │     │ Faltando: ["INC-005-III"]                                    ││
    │     │ → Retry com prompt focado apenas em incisos                  ││
    │     └──────────────────────────────────────────────────────────────┘│
    │                          │                                          │
    │                          ▼                                          │
    │  7. MONTA ArticleChunk                                              │
    │     ┌──────────────────────────────────────────────────────────────┐│
    │     │ ArticleChunk(                                                ││
    │     │   article_id="ART-005",                                      ││
    │     │   citations=["ART-005", "PAR-005-1", "INC-005-I", ...],      ││
    │     │   status=VALID,                                              ││
    │     │   coverage_paragrafos=1.0,                                   ││
    │     │   coverage_incisos=1.0                                       ││
    │     │ )                                                            ││
    │     └──────────────────────────────────────────────────────────────┘│
    └──────────────────────────────────────────────────────────────────────┘

Classes Principais:
==================

    | Classe                   | Descrição                                    |
    |--------------------------|----------------------------------------------|
    | ValidationStatus         | Enum: VALID, SUSPECT, INVALID                |
    | ArticleChunk             | Chunk materializado de um artigo             |
    | ArticleExtractionResult  | Resultado da extração de todos os artigos    |
    | OrchestratorConfig       | Configurações (retry, threshold, etc.)       |
    | ArticleOrchestrator      | Classe principal do orquestrador             |

ValidationStatus:
----------------

    | Status   | Significado                                           |
    |----------|-------------------------------------------------------|
    | VALID    | 100% de cobertura, sem duplicatas, IDs válidos        |
    | SUSPECT  | Cobertura < threshold (80%), mas sem erros críticos   |
    | INVALID  | IDs inválidos, duplicatas, ou erros de parent         |

OrchestratorConfig:
------------------

    | Parâmetro                   | Default | Descrição                      |
    |-----------------------------|---------|--------------------------------|
    | temperature                 | 0.0     | Temperatura do LLM             |
    | max_tokens                  | 512     | Tokens máximos (IDs são curtos)|
    | strict_validation           | True    | Falhar em IDs inválidos        |
    | coverage_threshold          | 0.8     | Mínimo para não ser SUSPECT    |
    | validate_parent_consistency | True    | Validar parent_id de incisos   |
    | enable_retry                | True    | Habilitar retry focado         |
    | max_retries                 | 2       | Max retries (1 PAR + 1 INC)    |
    | auto_fix_article_ids        | True    | Corrigir ART-1 → ART-001       |
    | auto_fix_child_ids          | False   | Não adivinhar PAR/INC/ALI      |

ArticleChunk (Resultado):
------------------------

    @dataclass
    class ArticleChunk:
        article_id: str           # "ART-005"
        article_number: str       # "5º"
        text: str                 # Texto completo reconstruído
        citations: list[str]      # ["ART-005", "PAR-005-1", "INC-005-I"]
        inciso_ids: list[str]     # IDs de incisos extraídos
        paragrafo_ids: list[str]  # IDs de parágrafos extraídos
        status: ValidationStatus  # VALID, SUSPECT, INVALID
        validation_notes: list    # Notas de validação
        coverage_paragrafos: float # 0.0 - 1.0
        coverage_incisos: float    # 0.0 - 1.0

Exemplo de Uso:
==============

    ```python
    from parsing import ArticleOrchestrator, OrchestratorConfig
    from parsing import SpanParser
    from llm import VLLMClient

    # 1. Parseia documento
    parser = SpanParser()
    parsed_doc = parser.parse(markdown_text)

    # 2. Configura orquestrador
    config = OrchestratorConfig(
        enable_retry=True,
        max_retries=2,
        coverage_threshold=0.8
    )

    # 3. Extrai hierarquia de todos os artigos
    llm = VLLMClient(...)
    orchestrator = ArticleOrchestrator(llm, config)
    result = orchestrator.extract_all_articles(parsed_doc)

    # 4. Verifica resultados
    print(f"Taxa de sucesso: {result.success_rate:.0%}")
    print(f"Válidos: {result.valid_articles}")
    print(f"Suspeitos: {result.suspect_articles}")
    print(f"Inválidos: {result.invalid_articles}")

    # 5. Processa chunks
    for chunk in result.chunks:
        if chunk.status == ValidationStatus.VALID:
            print(f"{chunk.article_id}: {len(chunk.citations)} citações")
        else:
            print(f"{chunk.article_id}: {chunk.status} - {chunk.validation_notes}")
    ```

Função de Conveniência:
======================

    ```python
    from parsing import extract_articles_with_hierarchy

    result = extract_articles_with_hierarchy(
        parsed_doc,
        llm_client,
        config=OrchestratorConfig(enable_retry=True)
    )
    ```

Prompts Utilizados:
==================

    | Prompt                   | Uso                                          |
    |--------------------------|----------------------------------------------|
    | ARTICLE_SYSTEM_PROMPT    | Instruções gerais para o LLM                 |
    | ARTICLE_USER_PROMPT      | Template para extração inicial               |
    | ARTICLE_RETRY_PROMPT     | Template para retry focado                   |

    O system prompt inclui "/no_think" para desabilitar thinking mode do Qwen 3.

Módulos Relacionados:
====================

    - parsing/span_parser.py: Gera ParsedDocument via regex
    - parsing/span_models.py: SpanType, Span, ParsedDocument
    - parsing/span_extraction_models.py: ArticleSpans schema
    - chunking/chunk_materializer.py: Converte ArticleChunk em chunks indexáveis
    - llm/vllm_client.py: Cliente LLM com chat_with_schema()

@author: Equipe VectorGov
@version: 1.0.0
@since: 23/12/2024
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum

from .span_models import ParsedDocument, Span, SpanType
from .span_extraction_models import ArticleSpans

logger = logging.getLogger(__name__)


class ValidationStatus(str, Enum):
    """Status de validação do artigo."""
    VALID = "valid"
    SUSPECT = "suspect"  # Cobertura incompleta
    INVALID = "invalid"  # IDs inválidos ou duplicatas


@dataclass
class ArticleChunk:
    """Chunk materializado de um artigo."""

    article_id: str
    article_number: str

    # Texto reconstruído
    text: str

    # Citations (lista de span_ids usados)
    citations: list[str] = field(default_factory=list)

    # Hierarquia extraída
    inciso_ids: list[str] = field(default_factory=list)
    paragrafo_ids: list[str] = field(default_factory=list)

    # Validação
    status: ValidationStatus = ValidationStatus.VALID
    validation_notes: list[str] = field(default_factory=list)

    # Métricas de cobertura por tipo
    parser_paragrafos_count: int = 0
    parser_incisos_count: int = 0
    llm_paragrafos_count: int = 0
    llm_incisos_count: int = 0

    # Retry info
    retry_count: int = 0

    @property
    def coverage_paragrafos(self) -> float:
        """Cobertura de parágrafos."""
        if self.parser_paragrafos_count == 0:
            return 1.0
        return self.llm_paragrafos_count / self.parser_paragrafos_count

    @property
    def coverage_incisos(self) -> float:
        """Cobertura de incisos."""
        if self.parser_incisos_count == 0:
            return 1.0
        return self.llm_incisos_count / self.parser_incisos_count

    @property
    def coverage_ratio(self) -> float:
        """Razão de cobertura geral (min das duas)."""
        return min(self.coverage_paragrafos, self.coverage_incisos)


@dataclass
class ArticleExtractionResult:
    """Resultado da extração por artigo."""

    # Chunks materializados
    chunks: list[ArticleChunk] = field(default_factory=list)

    # Estatísticas
    total_articles: int = 0
    valid_articles: int = 0
    suspect_articles: int = 0
    invalid_articles: int = 0

    # Detalhes de erros
    errors: list[dict] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Taxa de sucesso (válidos / total)."""
        if self.total_articles == 0:
            return 0.0
        return self.valid_articles / self.total_articles


@dataclass
class OrchestratorConfig:
    """Configuração do orquestrador."""

    # LLM
    temperature: float = 0.0
    max_tokens: int = 512  # Suficiente para JSON de IDs
    model_context_limit: int = 8192  # Limite de contexto do modelo (Qwen3-8B)
    chars_per_token: float = 4.0  # Estimativa de caracteres por token
    context_safety_margin: int = 256  # Margem de segurança para evitar overflow

    # Validação
    strict_validation: bool = True  # Falhar em IDs inválidos
    coverage_threshold: float = 0.8  # Mínimo de cobertura para não ser SUSPECT
    validate_parent_consistency: bool = True  # Validar sufixo ↔ parent

    # Retry focado por janela
    enable_retry: bool = True  # Retry quando cobertura < 100%
    max_retries: int = 2  # Máximo de retries (1 para PAR, 1 para INC)

    # Auto-fix (apenas para ART/CAP, não para PAR/INC/ALI)
    auto_fix_article_ids: bool = True
    auto_fix_child_ids: bool = False  # Não adivinhar PAR/INC/ALI


# =============================================================================
# PROMPTS
# =============================================================================

ARTICLE_SYSTEM_PROMPT = """Você é um especialista em documentos legais brasileiros. /no_think

Você receberá o texto de UM ÚNICO ARTIGO com marcações de span no formato:
[SPAN_ID] texto do span

Sua tarefa é COPIAR os IDs dos parágrafos e incisos que aparecem NO TEXTO.

Tipos de span (apenas copie se aparecer no texto):
- PAR-{art}-{n}: Parágrafo (ex: PAR-005-1, PAR-005-2, PAR-005-UNICO)
- INC-{art}-{romano}: Inciso (ex: INC-005-I, INC-005-II, INC-005-I_2)

REGRAS CRÍTICAS:
1. COPIE APENAS IDs que aparecem entre colchetes [ID] no texto
2. NUNCA invente ou "adivinhe" IDs - se não está no texto, não existe
3. Se o artigo não tem parágrafos, retorne paragrafo_ids: []
4. Se o artigo não tem incisos, retorne inciso_ids: []
5. Mantenha a ordem exata de aparição no texto

CHECKLIST ANTES DE RESPONDER:
- Verifique: cada ID que você listou aparece literalmente no texto?
- Se não há [PAR-...] no texto, a lista de parágrafos deve estar VAZIA
- Se não há [INC-...] no texto, a lista de incisos deve estar VAZIA
"""

ARTICLE_USER_PROMPT = """Extraia os IDs de parágrafos e incisos do artigo abaixo.

ARTIGO:
{annotated_text}

---

INSTRUÇÕES:
1. Procure por IDs no formato [PAR-XXX-N] e [INC-XXX-N] no texto acima
2. COPIE apenas os IDs que você encontrar - não invente nenhum
3. Se não encontrar nenhum parágrafo, retorne paragrafo_ids: []
4. Se não encontrar nenhum inciso, retorne inciso_ids: []

Retorne JSON com:
- article_id: "{article_id}"
- paragrafo_ids: [lista dos PAR-* encontrados, ou [] se nenhum]
- inciso_ids: [lista dos INC-* encontrados, ou [] se nenhum]"""

# Prompt para retry focado
ARTICLE_RETRY_PROMPT = """O artigo abaixo tem spans que podem ter sido omitidos na extração anterior.

ARTIGO (revise com atenção):
{annotated_text}

IDs JÁ ENCONTRADOS:
- Parágrafos: {found_paragrafos}
- Incisos: {found_incisos}

PROCURE por IDs adicionais que podem ter sido omitidos.
Retorne a lista COMPLETA (incluindo os já encontrados + novos):

- article_id: "{article_id}"
- paragrafo_ids: [lista completa]
- inciso_ids: [lista completa]"""


class ArticleOrchestrator:
    """
    Orquestrador de extração por artigo.

    Itera sobre cada artigo do documento e extrai a hierarquia completa
    usando LLM com guided JSON.
    """

    def __init__(
        self,
        llm_client: Any,
        config: Optional[OrchestratorConfig] = None
    ):
        """
        Inicializa o orquestrador.

        Args:
            llm_client: Cliente LLM com chat_with_schema() ou chat()
            config: Configuração do orquestrador
        """
        self.llm = llm_client
        self.config = config or OrchestratorConfig()

    def extract_all_articles(
        self,
        parsed_doc: ParsedDocument
    ) -> ArticleExtractionResult:
        """
        Extrai hierarquia completa de todos os artigos.

        NOTA: Este método agora usa extract_all_articles_with_split() para
        dividir automaticamente artigos grandes (>10k chars) em partes menores,
        evitando erros HTTP 422 do GPU Server durante o enriquecimento.

        Args:
            parsed_doc: Documento já parseado pelo SpanParser

        Returns:
            ArticleExtractionResult com chunks e validação
        """
        # Delega para versão com split de artigos grandes
        return self.extract_all_articles_with_split(parsed_doc)

    def _extract_article(
        self,
        article: Span,
        parsed_doc: ParsedDocument,
        retry_count: int = 0
    ) -> ArticleChunk:
        """Extrai hierarquia de um único artigo."""

        # 1. Obtém filhos esperados do parser
        parser_children = self._get_all_descendants(article.span_id, parsed_doc)
        parser_paragrafos = [s for s in parser_children if s.span_type == SpanType.PARAGRAFO]
        parser_incisos = [s for s in parser_children if s.span_type == SpanType.INCISO]
        parser_alineas = [s for s in parser_children if s.span_type == SpanType.ALINEA]

        # 2. CURTO-CIRCUITO: se não há filhos, não chama LLM
        if not parser_paragrafos and not parser_incisos and not parser_alineas:
            logger.debug(f"Curto-circuito: {article.span_id} não tem filhos")
            return ArticleChunk(
                article_id=article.span_id,
                article_number=article.identifier or "",
                text=article.text,
                citations=[article.span_id],
                inciso_ids=[],
                paragrafo_ids=[],
                status=ValidationStatus.VALID,
                validation_notes=["Artigo sem filhos (curto-circuito)"],
                parser_paragrafos_count=0,
                parser_incisos_count=0,
                llm_paragrafos_count=0,
                llm_incisos_count=0,
                retry_count=0,
            )

        # 3. Gera documento anotado filtrado para este artigo
        annotated_text = self._generate_article_annotated(article, parsed_doc)

        # 4. Extrai IDs permitidos para enum dinâmico
        allowed_paragrafos = [s.span_id for s in parser_paragrafos]
        allowed_incisos = [s.span_id for s in parser_incisos]

        # 5. Chama LLM com IDs permitidos (para schema dinâmico)
        llm_response = self._call_llm(
            article.span_id, annotated_text, allowed_paragrafos, allowed_incisos
        )

        # 6. Parseia resposta
        article_spans = self._parse_response(llm_response, article.span_id)

        # 7. Valida IDs retornados
        valid_incisos, invalid_incisos = self._validate_ids(
            article_spans.inciso_ids, "INC", parsed_doc
        )
        valid_paragrafos, invalid_paragrafos = self._validate_ids(
            article_spans.paragrafo_ids, "PAR", parsed_doc
        )

        # 8. Detecta duplicatas
        all_ids = valid_incisos + valid_paragrafos
        duplicates = self._find_duplicates(all_ids)

        # 9. Valida consistência parent-suffix (INC-005-I_2 => parent=PAR-005-2)
        parent_errors = []
        if self.config.validate_parent_consistency:
            parent_errors = self._validate_parent_consistency(
                valid_incisos, parsed_doc
            )

        # 10. Valida cobertura POR TIPO
        validation_notes = []
        status = ValidationStatus.VALID

        # Cobertura de parágrafos
        cov_par = len(valid_paragrafos) / len(parser_paragrafos) if parser_paragrafos else 1.0
        cov_inc = len(valid_incisos) / len(parser_incisos) if parser_incisos else 1.0

        if cov_par < self.config.coverage_threshold and parser_paragrafos:
            validation_notes.append(
                f"Cobertura parágrafos: {len(valid_paragrafos)}/{len(parser_paragrafos)} ({cov_par:.0%})"
            )
        if cov_inc < self.config.coverage_threshold and parser_incisos:
            validation_notes.append(
                f"Cobertura incisos: {len(valid_incisos)}/{len(parser_incisos)} ({cov_inc:.0%})"
            )

        # 11. Determina status
        if invalid_incisos or invalid_paragrafos:
            validation_notes.append(f"IDs inválidos: {invalid_incisos + invalid_paragrafos}")
            status = ValidationStatus.INVALID if self.config.strict_validation else ValidationStatus.SUSPECT

        if duplicates:
            validation_notes.append(f"Duplicatas: {duplicates}")
            status = ValidationStatus.INVALID

        if parent_errors:
            validation_notes.append(f"Inconsistência parent: {parent_errors}")
            status = ValidationStatus.INVALID

        # Se cobertura baixa mas sem erros críticos -> SUSPECT
        if validation_notes and status == ValidationStatus.VALID:
            status = ValidationStatus.SUSPECT

        # 12. RETRY FOCADO POR JANELA: retry específico para tipo faltante
        if self.config.enable_retry and retry_count < self.config.max_retries:
            # Identifica o que está faltando
            missing_pars = [p for p in allowed_paragrafos if p not in valid_paragrafos]
            missing_incs = [i for i in allowed_incisos if i not in valid_incisos]

            # Retry para parágrafos se necessário
            if missing_pars and cov_par < 1.0:
                logger.info(f"Retry PAR para {article.span_id}: {len(missing_pars)} faltando")
                par_response = self._call_llm_retry_focused(
                    article.span_id, annotated_text,
                    "paragrafos", missing_pars, valid_paragrafos
                )
                par_spans = self._parse_response(par_response, article.span_id)
                new_pars, _ = self._validate_ids(par_spans.paragrafo_ids, "PAR", parsed_doc)
                for p in new_pars:
                    if p not in valid_paragrafos:
                        valid_paragrafos.append(p)
                retry_count += 1

            # Retry para incisos se necessário
            if missing_incs and cov_inc < 1.0 and retry_count < self.config.max_retries:
                logger.info(f"Retry INC para {article.span_id}: {len(missing_incs)} faltando")
                inc_response = self._call_llm_retry_focused(
                    article.span_id, annotated_text,
                    "incisos", missing_incs, valid_incisos
                )
                inc_spans = self._parse_response(inc_response, article.span_id)
                new_incs, _ = self._validate_ids(inc_spans.inciso_ids, "INC", parsed_doc)
                for i in new_incs:
                    if i not in valid_incisos:
                        valid_incisos.append(i)
                retry_count += 1

            # Recalcula cobertura
            cov_par = len(valid_paragrafos) / len(parser_paragrafos) if parser_paragrafos else 1.0
            cov_inc = len(valid_incisos) / len(parser_incisos) if parser_incisos else 1.0

            # Atualiza status se melhorou
            if cov_par >= self.config.coverage_threshold and cov_inc >= self.config.coverage_threshold:
                if status == ValidationStatus.SUSPECT:
                    status = ValidationStatus.VALID
                    validation_notes = [n for n in validation_notes if "Cobertura" not in n]

        # 11. Monta citations (artigo + todos os filhos válidos)
        citations = [article.span_id] + valid_paragrafos + valid_incisos

        # Adiciona alíneas dos incisos
        for inc_id in valid_incisos:
            for child in parsed_doc.get_children(inc_id):
                if child.span_type == SpanType.ALINEA:
                    citations.append(child.span_id)

        # 12. Reconstrói texto
        text = self._reconstruct_text(article.span_id, parsed_doc)

        return ArticleChunk(
            article_id=article.span_id,
            article_number=article.identifier or "",
            text=text,
            citations=citations,
            inciso_ids=valid_incisos,
            paragrafo_ids=valid_paragrafos,
            status=status,
            validation_notes=validation_notes,
            parser_paragrafos_count=len(parser_paragrafos),
            parser_incisos_count=len(parser_incisos),
            llm_paragrafos_count=len(valid_paragrafos),
            llm_incisos_count=len(valid_incisos),
            retry_count=retry_count,
        )

    def _generate_article_annotated(
        self,
        article: Span,
        parsed_doc: ParsedDocument
    ) -> str:
        """Gera documento anotado contendo apenas o artigo e seus descendentes."""
        lines = []

        # Adiciona o artigo
        lines.append(f"[{article.span_id}] {article.text}")

        # Adiciona todos os descendentes
        descendants = self._get_all_descendants(article.span_id, parsed_doc)
        for span in descendants:
            lines.append(f"[{span.span_id}] {span.text}")

        return "\n".join(lines)

    def _get_all_descendants(
        self,
        parent_id: str,
        parsed_doc: ParsedDocument
    ) -> list[Span]:
        """Obtém todos os descendentes de um span (recursivo)."""
        descendants = []

        for child in parsed_doc.get_children(parent_id):
            descendants.append(child)
            # Recursão para filhos dos filhos
            descendants.extend(self._get_all_descendants(child.span_id, parsed_doc))

        return descendants

    def _estimate_tokens(self, text: str) -> int:
        """
        Estima número de tokens em um texto.

        Usa proporção chars/token configurável (default: 4 chars = 1 token).
        Esta é uma estimativa conservadora para português.
        """
        return int(len(text) / self.config.chars_per_token)

    def _calculate_dynamic_max_tokens(self, input_text: str) -> int:
        """
        Calcula max_tokens disponível para resposta baseado no input.

        Usa estimativa CONSERVADORA de tokens para evitar erros 400.

        Args:
            input_text: Texto completo que será enviado ao modelo

        Returns:
            max_tokens que pode ser solicitado sem estourar contexto
        """
        # Estima tokens do input com fator conservador
        # Tokenizer Qwen pode gerar mais tokens que o esperado para português
        chars_per_token = 2.5  # conservador (era 4.0)
        input_tokens = int(len(input_text) / chars_per_token)

        # Calcula espaço disponível com margem extra
        safety_margin = 512  # margem de segurança aumentada
        available = (
            self.config.model_context_limit
            - input_tokens
            - safety_margin
        )

        # Limita entre min e max permitido
        min_tokens = 128  # Mínimo para resposta JSON válida
        max_allowed = 256  # Limita max_tokens para safety (IDs são curtos)
        max_tokens = min(max_allowed, max(min_tokens, available))

        if available < min_tokens:
            logger.warning(
                f"Input muito grande: {input_tokens} tokens estimados, "
                f"disponível: {available} tokens. Usando mínimo: {min_tokens}"
            )

        logger.info(
            f"Tokens dinâmicos: input_chars={len(input_text)}, "
            f"input_tokens_est={input_tokens}, disponível={available}, "
            f"usando max_tokens={max_tokens}"
        )

        return max_tokens

    def _input_exceeds_context(self, input_text: str) -> bool:
        """
        Verifica se o input excede o limite de contexto do modelo.

        Retorna True se o input for muito grande para caber no contexto
        mesmo com max_tokens mínimo.
        """
        input_tokens = self._estimate_tokens(input_text)
        min_output_tokens = 256  # Mínimo necessário para resposta JSON
        max_input_allowed = self.config.model_context_limit - min_output_tokens - self.config.context_safety_margin

        if input_tokens > max_input_allowed:
            logger.warning(
                f"Input excede contexto: {input_tokens} tokens > {max_input_allowed} permitidos"
            )
            return True
        return False

    def _truncate_annotated_text(self, annotated_text: str, max_chars: int = 28000) -> str:
        """
        Trunca texto anotado preservando estrutura.

        Mantém os primeiros N caracteres, cortando em linha completa.
        Adiciona nota de truncamento.
        """
        if len(annotated_text) <= max_chars:
            return annotated_text

        # Corta na última linha completa antes do limite
        truncated = annotated_text[:max_chars]
        last_newline = truncated.rfind('\n')
        if last_newline > 0:
            truncated = truncated[:last_newline]

        # Adiciona nota
        truncated += "\n\n[... texto truncado por limite de contexto ...]"

        logger.info(
            f"Texto anotado truncado: {len(annotated_text)} -> {len(truncated)} chars"
        )

        return truncated

    def _call_llm(
        self,
        article_id: str,
        annotated_text: str,
        allowed_paragrafos: list[str],
        allowed_incisos: list[str]
    ) -> str:
        """Chama LLM para extrair hierarquia do artigo com schema dinâmico."""
        # Verifica e trunca se input for muito grande
        # O prompt total inclui: system prompt + headers + allowed IDs + annotated_text
        # Estimativa MUITO conservadora: 2.5 chars/token (tokenizer Qwen pode ser mais agressivo)
        # contexto 8192, output 256 (mínimo para JSON), margem 512, overhead prompts 3000 chars
        prompt_overhead_chars = 3500  # system + headers + instruções + allowed_ids
        chars_per_token_conservative = 2.5  # muito conservador para safety
        max_input_chars = int(
            (self.config.model_context_limit - 256 - 512)  # reserva 256 output, 512 margem
            * chars_per_token_conservative
            - prompt_overhead_chars
        )
        # Garante mínimo de 15000 chars para artigos pequenos
        max_input_chars = max(max_input_chars, 15000)

        if len(annotated_text) > max_input_chars:
            logger.warning(
                f"Artigo {article_id}: texto muito grande ({len(annotated_text)} chars), "
                f"truncando para {max_input_chars} chars"
            )
            annotated_text = self._truncate_annotated_text(annotated_text, max_input_chars)

        # Prompt com contadores para evitar alucinação
        user_prompt = f"""Extraia os IDs de parágrafos e incisos do artigo abaixo.

ESTATÍSTICAS DO PARSER:
- Parágrafos detectados: {len(allowed_paragrafos)} → IDs permitidos: {allowed_paragrafos}
- Incisos detectados: {len(allowed_incisos)} → IDs permitidos: {allowed_incisos}

ARTIGO:
{annotated_text}

---

INSTRUÇÕES:
1. Procure por IDs no formato [PAR-XXX-N] e [INC-XXX-N] no texto acima
2. COPIE apenas os IDs que aparecem na lista de IDs permitidos
3. Se a lista permitida está vazia, retorne array vazio para aquela categoria
4. NÃO invente IDs que não estão na lista permitida

Retorne JSON com:
- article_id: "{article_id}"
- paragrafo_ids: [IDs dos parágrafos encontrados, ou [] se nenhum permitido]
- inciso_ids: [IDs dos incisos encontrados, ou [] se nenhum permitido]"""

        messages = [
            {"role": "system", "content": ARTICLE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Gera schema dinâmico com enum
        dynamic_schema = self._build_dynamic_schema(
            article_id, allowed_paragrafos, allowed_incisos
        )

        # Calcula max_tokens dinâmico para evitar estouro de contexto
        full_input = ARTICLE_SYSTEM_PROMPT + user_prompt
        dynamic_max_tokens = self._calculate_dynamic_max_tokens(full_input)

        # Tenta guided JSON se disponível
        if hasattr(self.llm, 'chat_with_schema'):
            response = self.llm.chat_with_schema(
                messages=messages,
                schema=dynamic_schema,
                temperature=self.config.temperature,
                max_tokens=dynamic_max_tokens,
            )
        else:
            response = self.llm.chat(
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=dynamic_max_tokens,
            )

        return response

    def _build_dynamic_schema(
        self,
        article_id: str,
        allowed_paragrafos: list[str],
        allowed_incisos: list[str]
    ) -> dict:
        """Constrói JSON schema dinâmico com enum de IDs permitidos."""
        schema = {
            "type": "object",
            "properties": {
                "article_id": {
                    "type": "string",
                    "const": article_id  # Força valor exato
                },
                "paragrafo_ids": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                },
                "inciso_ids": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                }
            },
            "required": ["article_id", "paragrafo_ids", "inciso_ids"],
            "additionalProperties": False
        }

        # Adiciona enum apenas se há IDs permitidos
        # Se lista vazia, permite qualquer string (será validado depois)
        if allowed_paragrafos:
            schema["properties"]["paragrafo_ids"]["items"]["enum"] = allowed_paragrafos
        if allowed_incisos:
            schema["properties"]["inciso_ids"]["items"]["enum"] = allowed_incisos

        return schema

    def _parse_response(self, response: Any, article_id: str) -> ArticleSpans:
        """Parseia resposta do LLM."""
        try:
            # Resposta já é dict (de chat_with_schema)
            if isinstance(response, dict):
                data = response
            elif isinstance(response, str):
                # Remove tags <think> do Qwen3
                response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

                # Encontra JSON
                start = response.find('{')
                end = response.rfind('}') + 1

                if start >= 0 and end > start:
                    response = response[start:end]

                data = json.loads(response)
            else:
                # Tipo inesperado
                logger.warning(f"Tipo inesperado de resposta: {type(response)}")
                data = {}

            # Garante que article_id está correto
            if isinstance(data, dict):
                data['article_id'] = article_id
            else:
                data = {'article_id': article_id, 'inciso_ids': [], 'paragrafo_ids': []}

            return ArticleSpans(**data)

        except Exception as e:
            logger.warning(f"Erro ao parsear resposta para {article_id}: {e}")
            # Retorna vazio
            return ArticleSpans(
                article_id=article_id,
                inciso_ids=[],
                paragrafo_ids=[]
            )

    def _validate_ids(
        self,
        ids: list[str],
        expected_prefix: str,
        parsed_doc: ParsedDocument
    ) -> tuple[list[str], list[str]]:
        """
        Valida lista de IDs.

        Returns:
            (ids_válidos, ids_inválidos)
        """
        valid = []
        invalid = []

        for span_id in ids:
            if parsed_doc.get_span(span_id):
                valid.append(span_id)
            else:
                # Tenta auto-fix apenas se habilitado
                fixed = self._try_fix_id(span_id, expected_prefix, parsed_doc)
                if fixed:
                    valid.append(fixed)
                else:
                    invalid.append(span_id)

        return valid, invalid

    def _try_fix_id(
        self,
        span_id: str,
        expected_prefix: str,
        parsed_doc: ParsedDocument
    ) -> Optional[str]:
        """Tenta corrigir ID comum (apenas para ART/CAP se habilitado)."""
        # Só corrige ART/CAP automaticamente
        if expected_prefix not in ("ART", "CAP"):
            if not self.config.auto_fix_child_ids:
                return None

        if not self.config.auto_fix_article_ids:
            return None

        # Tenta padding de zeros: ART-1 -> ART-001
        match = re.match(rf'^{expected_prefix}-(\d+)$', span_id)
        if match:
            num = match.group(1)
            fixed = f"{expected_prefix}-{num.zfill(3)}"
            if parsed_doc.get_span(fixed):
                return fixed

        return None

    def _find_duplicates(self, ids: list[str]) -> list[str]:
        """Encontra IDs duplicados na lista."""
        seen = set()
        duplicates = []

        for span_id in ids:
            if span_id in seen:
                duplicates.append(span_id)
            seen.add(span_id)

        return duplicates

    def _validate_parent_consistency(
        self,
        inciso_ids: list[str],
        parsed_doc: ParsedDocument
    ) -> list[str]:
        """
        Valida consistência de parent_id dos incisos.

        NOTA: O sufixo _N em INC-XXX-I_N é apenas um DISAMBIGUADOR para quando
        há múltiplos incisos com o mesmo número romano no artigo (ex: inciso I
        aparece tanto no caput quanto em §1 ou §2). O sufixo NÃO indica qual
        parágrafo é o parent - isso é determinado pela posição no texto.

        Esta validação apenas verifica se:
        1. O parent existe no documento
        2. O parent pertence ao mesmo artigo

        Returns:
            Lista de erros encontrados
        """
        errors = []

        for inc_id in inciso_ids:
            span = parsed_doc.get_span(inc_id)
            if not span:
                continue

            # Extrai número do artigo do inciso (ex: "005" de INC-005-I ou INC-005-I_2)
            match = re.match(r'^INC-(\d+)-', inc_id)
            if not match:
                continue

            art_num = match.group(1)

            # Parent deve existir
            if not span.parent_id:
                errors.append(f"{inc_id}: parent_id vazio")
                continue

            parent = parsed_doc.get_span(span.parent_id)
            if not parent:
                errors.append(f"{inc_id}: parent '{span.parent_id}' não encontrado")
                continue

            # Parent deve ser do mesmo artigo (ART-XXX ou PAR-XXX-N)
            if span.parent_id.startswith("ART-"):
                parent_art = span.parent_id.replace("ART-", "")
                if parent_art != art_num:
                    errors.append(
                        f"{inc_id}: artigo mismatch (inciso de art {art_num}, "
                        f"parent de art {parent_art})"
                    )
            elif span.parent_id.startswith("PAR-"):
                parent_match = re.match(r'^PAR-(\d+)-', span.parent_id)
                if parent_match and parent_match.group(1) != art_num:
                    errors.append(
                        f"{inc_id}: artigo mismatch (inciso de art {art_num}, "
                        f"parent de art {parent_match.group(1)})"
                    )

        return errors

    def _call_llm_retry_focused(
        self,
        article_id: str,
        annotated_text: str,
        target_type: str,  # "paragrafos" ou "incisos"
        missing_ids: list[str],
        found_ids: list[str]
    ) -> str:
        """Retry focado em um tipo específico (parágrafos OU incisos)."""
        # Aplica mesma lógica de truncação do _call_llm
        prompt_overhead_chars = 4000  # retry tem mais overhead (listas de IDs)
        chars_per_token_conservative = 2.5
        max_input_chars = int(
            (self.config.model_context_limit - 256 - 512)
            * chars_per_token_conservative
            - prompt_overhead_chars
        )
        max_input_chars = max(max_input_chars, 12000)

        if len(annotated_text) > max_input_chars:
            logger.warning(
                f"Retry {article_id}: texto muito grande ({len(annotated_text)} chars), "
                f"truncando para {max_input_chars} chars"
            )
            annotated_text = self._truncate_annotated_text(annotated_text, max_input_chars)

        user_prompt = f"""BUSCA FOCADA: Procure especificamente por {target_type.upper()} no texto abaixo.

IDs FALTANDO (preciso encontrar esses):
{missing_ids}

IDs JÁ ENCONTRADOS:
{found_ids}

ARTIGO:
{annotated_text}

INSTRUÇÕES:
1. Procure APENAS pelos IDs listados em "FALTANDO"
2. Verifique se eles aparecem no texto entre colchetes [ID]
3. Retorne a lista COMPLETA (encontrados + novos)

Retorne JSON com article_id="{article_id}" e as listas completas."""

        messages = [
            {"role": "system", "content": ARTICLE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Schema focado apenas no tipo solicitado
        if target_type == "paragrafos":
            allowed_par = found_ids + missing_ids
            allowed_inc = []
        else:
            allowed_par = []
            allowed_inc = found_ids + missing_ids

        dynamic_schema = self._build_dynamic_schema(
            article_id, allowed_par, allowed_inc
        )

        # Calcula max_tokens dinâmico para evitar estouro de contexto
        full_input = ARTICLE_SYSTEM_PROMPT + user_prompt
        dynamic_max_tokens = self._calculate_dynamic_max_tokens(full_input)

        if hasattr(self.llm, 'chat_with_schema'):
            response = self.llm.chat_with_schema(
                messages=messages,
                schema=dynamic_schema,
                temperature=self.config.temperature,
                max_tokens=dynamic_max_tokens,
            )
        else:
            response = self.llm.chat(
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=dynamic_max_tokens,
            )

        return response

    def _call_llm_retry(
        self,
        article_id: str,
        annotated_text: str,
        found_paragrafos: list[str],
        found_incisos: list[str],
        allowed_paragrafos: list[str],
        allowed_incisos: list[str]
    ) -> str:
        """Chama LLM com prompt de retry focado e schema dinâmico."""
        # Aplica mesma lógica de truncação
        prompt_overhead_chars = 4500  # retry tem mais overhead (listas de IDs)
        chars_per_token_conservative = 2.5
        max_input_chars = int(
            (self.config.model_context_limit - 256 - 512)
            * chars_per_token_conservative
            - prompt_overhead_chars
        )
        max_input_chars = max(max_input_chars, 12000)

        if len(annotated_text) > max_input_chars:
            logger.warning(
                f"Retry genérico {article_id}: texto muito grande ({len(annotated_text)} chars), "
                f"truncando para {max_input_chars} chars"
            )
            annotated_text = self._truncate_annotated_text(annotated_text, max_input_chars)

        user_prompt = f"""O artigo abaixo tem spans que podem ter sido omitidos na extração anterior.

IDs PERMITIDOS (do parser):
- Parágrafos: {allowed_paragrafos}
- Incisos: {allowed_incisos}

IDs JÁ ENCONTRADOS:
- Parágrafos: {found_paragrafos}
- Incisos: {found_incisos}

ARTIGO (revise com atenção):
{annotated_text}

Procure por IDs adicionais que podem ter sido omitidos.
IMPORTANTE: Só retorne IDs que estão na lista de PERMITIDOS.

Retorne a lista COMPLETA:
- article_id: "{article_id}"
- paragrafo_ids: [lista completa dos parágrafos encontrados]
- inciso_ids: [lista completa dos incisos encontrados]"""

        messages = [
            {"role": "system", "content": ARTICLE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Usa schema dinâmico com enum
        dynamic_schema = self._build_dynamic_schema(
            article_id, allowed_paragrafos, allowed_incisos
        )

        # Calcula max_tokens dinâmico para evitar estouro de contexto
        full_input = ARTICLE_SYSTEM_PROMPT + user_prompt
        dynamic_max_tokens = self._calculate_dynamic_max_tokens(full_input)

        if hasattr(self.llm, 'chat_with_schema'):
            response = self.llm.chat_with_schema(
                messages=messages,
                schema=dynamic_schema,
                temperature=self.config.temperature,
                max_tokens=dynamic_max_tokens,
            )
        else:
            response = self.llm.chat(
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=dynamic_max_tokens,
            )

        return response

    def _reconstruct_text(
        self,
        article_id: str,
        parsed_doc: ParsedDocument
    ) -> str:
        """Reconstrói texto do artigo com todos os filhos."""
        article = parsed_doc.get_span(article_id)
        if not article:
            return ""

        lines = [article.text]

        # Adiciona filhos recursivamente com indentação
        self._add_children_text(article_id, parsed_doc, lines, indent=1)

        return "\n".join(lines)

    def _add_children_text(
        self,
        parent_id: str,
        parsed_doc: ParsedDocument,
        lines: list[str],
        indent: int
    ):
        """Adiciona texto dos filhos recursivamente."""
        prefix = "  " * indent

        for child in parsed_doc.get_children(parent_id):
            lines.append(f"{prefix}{child.text}")
            self._add_children_text(child.span_id, parsed_doc, lines, indent + 1)

    def _split_large_article(
        self,
        article: Span,
        parsed_doc: ParsedDocument,
        max_chars: int = 5000
    ) -> list[tuple[str, str, list[str]]]:
        """
        Divide artigo grande em partes menores preservando estrutura.

        Args:
            article: Span do artigo a dividir
            parsed_doc: Documento completo parseado
            max_chars: Tamanho máximo de cada parte (default: 5000)

        Returns:
            Lista de tuplas (part_id, text, child_ids) para cada parte
            Exemplo: [
                ("ART-006-PART-1", "Art. 6º Para...", ["INC-006-I", "INC-006-II"]),
                ("ART-006-PART-2", "[continuação]...", ["INC-006-III", "INC-006-IV"])
            ]

        Estratégia de Split:
            1. Extrai todos os filhos (parágrafos, incisos, alíneas)
            2. Divide filhos em grupos que não excedam max_chars
            3. Cada grupo se torna uma "parte" do artigo
            4. Preserva IDs originais dos filhos em cada parte
        """
        # 1. Obtém todos os filhos do artigo
        children = self._get_all_descendants(article.span_id, parsed_doc)

        # Se o artigo inteiro cabe em max_chars, não divide
        if len(article.text) <= max_chars:
            return [(article.span_id, article.text, [article.span_id])]

        # 2. Ordena filhos por posição no texto
        children_sorted = sorted(children, key=lambda s: article.text.find(s.text))

        # 3. Divide em partes
        parts = []
        current_part_text = ""
        current_part_children = []
        part_number = 1

        # Cabeçalho do artigo (primeira linha, ex: "Art. 6º Para os fins desta Lei:")
        article_header = article.text.split('\n')[0] if '\n' in article.text else article.text[:200]

        for child in children_sorted:
            # Se adicionar este filho exceder o limite, inicia nova parte
            child_text = child.text
            if current_part_text and len(current_part_text) + len(child_text) > max_chars:
                # Salva parte atual
                part_id = f"{article.span_id}-PART-{part_number}"
                parts.append((
                    part_id,
                    f"[PARTE {part_number}]\n{article_header}\n\n{current_part_text}",
                    current_part_children
                ))

                # Inicia nova parte
                part_number += 1
                current_part_text = child_text + "\n"
                current_part_children = [child.span_id]
            else:
                # Adiciona à parte atual
                current_part_text += child_text + "\n"
                current_part_children.append(child.span_id)

        # Salva última parte
        if current_part_text:
            part_id = f"{article.span_id}-PART-{part_number}"
            parts.append((
                part_id,
                f"[PARTE {part_number}]\n{article_header}\n\n{current_part_text}",
                current_part_children
            ))

        logger.info(
            f"Artigo {article.span_id} dividido em {len(parts)} partes "
            f"(original: {len(article.text)} chars)"
        )

        return parts

    def extract_all_articles_with_split(
        self,
        parsed_doc: ParsedDocument,
        large_article_threshold: int = 10000
    ) -> ArticleExtractionResult:
        """
        Versão modificada de extract_all_articles que divide artigos grandes.

        Args:
            parsed_doc: Documento parseado
            large_article_threshold: Artigos >threshold chars são divididos (default: 10k)

        Returns:
            ArticleExtractionResult com chunks (incluindo partes de artigos grandes)
        """
        result = ArticleExtractionResult(total_articles=len(parsed_doc.articles))

        for article in parsed_doc.articles:
            try:
                # Detecta se artigo é grande
                if len(article.text) > large_article_threshold:
                    logger.warning(
                        f"Artigo grande detectado: {article.span_id} "
                        f"({len(article.text)} chars) - dividindo em partes"
                    )

                    # Divide artigo em partes
                    parts = self._split_large_article(article, parsed_doc)

                    # Processa cada parte como um chunk independente
                    for part_id, part_text, child_ids in parts:
                        # Cria ArticleChunk simplificado para esta parte
                        chunk = ArticleChunk(
                            article_id=part_id,
                            article_number=article.identifier or "",
                            text=part_text,
                            citations=child_ids,  # IDs dos filhos nesta parte
                            inciso_ids=[],  # Não extrai hierarquia para partes
                            paragrafo_ids=[],
                            status=ValidationStatus.VALID,
                            validation_notes=[f"Parte de artigo grande (split automático)"],
                            parser_paragrafos_count=0,
                            parser_incisos_count=0,
                            llm_paragrafos_count=0,
                            llm_incisos_count=0,
                            retry_count=0,
                        )
                        result.chunks.append(chunk)
                        result.valid_articles += 1
                else:
                    # Processa normalmente (artigo pequeno)
                    chunk = self._extract_article(article, parsed_doc)
                    result.chunks.append(chunk)

                    # Atualiza estatísticas
                    if chunk.status == ValidationStatus.VALID:
                        result.valid_articles += 1
                    elif chunk.status == ValidationStatus.SUSPECT:
                        result.suspect_articles += 1
                    else:
                        result.invalid_articles += 1

            except Exception as e:
                logger.error(f"Erro ao extrair {article.span_id}: {e}")
                result.errors.append({
                    "article_id": article.span_id,
                    "error": str(e)
                })
                result.invalid_articles += 1

        logger.info(
            f"Extração concluída: {result.valid_articles}/{result.total_articles} válidos, "
            f"{result.suspect_articles} suspeitos, {result.invalid_articles} inválidos"
        )

        return result


# =============================================================================
# FUNÇÕES DE CONVENIÊNCIA
# =============================================================================

def extract_articles_with_hierarchy(
    parsed_doc: ParsedDocument,
    llm_client: Any,
    config: Optional[OrchestratorConfig] = None
) -> ArticleExtractionResult:
    """
    Função de conveniência para extrair hierarquia de todos os artigos.

    Args:
        parsed_doc: Documento parseado
        llm_client: Cliente LLM
        config: Configuração opcional

    Returns:
        ArticleExtractionResult
    """
    orchestrator = ArticleOrchestrator(llm_client, config)
    return orchestrator.extract_all_articles(parsed_doc)
