"""
Validador de artigos pos-Docling.

Valida:
1. Sequencia de artigos (gaps)
2. Duplicatas
3. Splits (partes de artigos grandes)
4. Manifesto de chunks para validacao pos-Milvus
"""

import re
import logging
from typing import Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SplitArticle:
    """Informacoes de um artigo que foi splitado."""
    article_number: str
    parts_count: int
    parts: list[str] = field(default_factory=list)


@dataclass
class ArticleValidationResult:
    """Resultado da validacao de artigos."""

    # Configuracao
    validate_enabled: bool = False
    expected_first_article: Optional[int] = None
    expected_last_article: Optional[int] = None

    # Artigos
    expected_articles: list[str] = field(default_factory=list)
    found_articles: list[str] = field(default_factory=list)
    missing_articles: list[str] = field(default_factory=list)
    duplicate_articles: list[str] = field(default_factory=list)

    # Splits
    split_articles: list[SplitArticle] = field(default_factory=list)

    # Chunks
    total_chunks_generated: int = 0
    chunks_manifest: list[str] = field(default_factory=list)

    # Metricas
    total_found: int = 0
    first_article: Optional[int] = None
    last_article: Optional[int] = None
    has_gaps: bool = False
    has_duplicates: bool = False
    coverage_percent: float = 0.0

    # Status
    status: str = "passed"  # passed, warning, failed

    def to_dict(self) -> dict:
        """Converte para dicionario para JSON."""
        return {
            "validate_enabled": self.validate_enabled,
            "expected_first_article": self.expected_first_article,
            "expected_last_article": self.expected_last_article,
            "expected_articles": self.expected_articles,
            "found_articles": self.found_articles,
            "found_list": self.found_articles,  # Alias para compatibilidade
            "missing_articles": self.missing_articles,
            "article_gaps": self.missing_articles,  # Alias para compatibilidade
            "duplicate_articles": self.duplicate_articles,
            "split_articles": [
                {
                    "article_number": s.article_number,
                    "parts_count": s.parts_count,
                    "parts": s.parts
                }
                for s in self.split_articles
            ],
            "total_chunks_generated": self.total_chunks_generated,
            "chunks_manifest": self.chunks_manifest,
            "total_found": self.total_found,
            "first_article": self.first_article,
            "last_article": self.last_article,
            "has_gaps": self.has_gaps,
            "has_duplicates": self.has_duplicates,
            "coverage_percent": self.coverage_percent,
            "status": self.status,
        }


class ArticleValidator:
    """Validador de artigos extraidos."""

    # Regex para extrair numero do artigo de span_ids
    # Matches: ART-006, ART-006-P1, ART-006-P2
    ARTICLE_PATTERN = re.compile(r'^ART-(\d+)(?:-P(\d+))?$')

    def __init__(
        self,
        validate_enabled: bool = False,
        expected_first: Optional[int] = None,
        expected_last: Optional[int] = None,
    ):
        self.validate_enabled = validate_enabled
        self.expected_first = expected_first
        self.expected_last = expected_last

    def validate(self, chunks: list[Any]) -> ArticleValidationResult:
        """
        Valida os chunks extraidos.

        Args:
            chunks: Lista de MaterializedChunk ou objetos com span_id

        Returns:
            ArticleValidationResult com todos os dados de validacao
        """
        result = ArticleValidationResult(
            validate_enabled=self.validate_enabled,
            expected_first_article=self.expected_first,
            expected_last_article=self.expected_last,
        )

        # Coletar span_ids de todos os chunks
        span_ids = []
        for chunk in chunks:
            # Suporta objetos com atributo ou dicts
            if hasattr(chunk, 'span_id'):
                span_id = chunk.span_id
            elif isinstance(chunk, dict):
                span_id = chunk.get('span_id', '')
            else:
                span_id = ''

            if span_id:
                span_ids.append(span_id)

        # Registrar manifesto de chunks (para validacao pos-Milvus)
        result.chunks_manifest = span_ids
        result.total_chunks_generated = len(span_ids)

        # Extrair artigos dos span_ids
        articles_found: dict[str, list[str]] = {}  # article_number -> list of span_ids
        splits_found: dict[str, list[str]] = {}    # article_number -> list of parts (ART-006-P1, etc)

        for span_id in span_ids:
            match = self.ARTICLE_PATTERN.match(span_id)
            if match:
                article_num = match.group(1)  # "006"
                part_num = match.group(2)     # "1" ou None

                # Normalizar numero do artigo (remover zeros a esquerda)
                article_num_normalized = str(int(article_num))

                if article_num_normalized not in articles_found:
                    articles_found[article_num_normalized] = []
                articles_found[article_num_normalized].append(span_id)

                # Registrar splits
                if part_num:
                    if article_num_normalized not in splits_found:
                        splits_found[article_num_normalized] = []
                    splits_found[article_num_normalized].append(span_id)

        # Lista de artigos unicos encontrados (ordenada numericamente)
        found_list = sorted(articles_found.keys(), key=lambda x: int(x))
        result.found_articles = found_list
        result.total_found = len(found_list)

        # Primeiro e ultimo artigo encontrado
        if found_list:
            result.first_article = int(found_list[0])
            result.last_article = int(found_list[-1])

        # Detectar duplicatas (mesmo artigo com multiplos span_ids NAO splitados)
        for article_num, span_ids_list in articles_found.items():
            # Se tem mais de um span_id E nao sao splits (nao tem -P)
            non_split_spans = [s for s in span_ids_list if '-P' not in s]
            if len(non_split_spans) > 1:
                result.duplicate_articles.append(article_num)

        result.has_duplicates = len(result.duplicate_articles) > 0

        # Registrar artigos splitados
        for article_num, parts in splits_found.items():
            result.split_articles.append(SplitArticle(
                article_number=article_num,
                parts_count=len(parts),
                parts=sorted(parts)
            ))

        # Gerar lista de artigos esperados
        # IMPORTANTE: Só inferir o range se a validação estiver EXPLICITAMENTE habilitada
        # Caso contrário, reporta apenas estatísticas sem calcular "faltando"
        if self.validate_enabled:
            if self.expected_first and self.expected_last:
                # Usuário especificou o range esperado
                result.expected_articles = [
                    str(i) for i in range(self.expected_first, self.expected_last + 1)
                ]
            elif result.first_article and result.last_article:
                # Inferir do range encontrado (apenas com validação habilitada)
                result.expected_articles = [
                    str(i) for i in range(result.first_article, result.last_article + 1)
                ]
        # Se validate_enabled=False, expected_articles fica vazio e não calcula gaps

        # Detectar gaps (artigos faltando na sequencia)
        # Só calcula gaps se há artigos esperados definidos
        if result.expected_articles:
            expected_set = set(result.expected_articles)
            found_set = set(result.found_articles)
            missing = expected_set - found_set
            result.missing_articles = sorted(missing, key=lambda x: int(x))
            result.has_gaps = len(result.missing_articles) > 0
        else:
            # Sem validação explícita, não reporta gaps
            result.missing_articles = []
            result.has_gaps = False

        # Calcular cobertura
        if result.expected_articles:
            result.coverage_percent = round(
                (len(result.found_articles) / len(result.expected_articles)) * 100, 2
            )
        else:
            result.coverage_percent = 100.0

        # Determinar status
        if result.has_gaps or result.has_duplicates:
            if result.coverage_percent >= 95:
                result.status = "warning"
            else:
                result.status = "failed"
        else:
            result.status = "passed"

        # Log do resultado
        logger.info(
            f"Validacao de artigos: {result.total_found} encontrados, "
            f"{len(result.missing_articles)} faltando, "
            f"{len(result.split_articles)} splitados, "
            f"status={result.status}"
        )

        return result
