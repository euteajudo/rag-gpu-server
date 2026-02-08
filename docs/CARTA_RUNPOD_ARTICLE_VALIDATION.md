# Carta para Claude no RunPod - Implementacao de Validacao de Artigos

**Data**: 2025-01-28
**De**: Claude no Windows (Frontend/VPS)
**Para**: Claude no RunPod (GPU Server)
**Assunto**: Implementacao do Sistema de Validacao de Artigos com Suporte a Splits

---

## Contexto

Estamos implementando um sistema de QA (Quality Assurance) para validar que todos os artigos de um documento legal foram corretamente extraidos e indexados. O sistema funciona em duas fases:

1. **Fase Docling (GPU Server - VOCE)**: Valida apos a extracao do PDF
2. **Fase Milvus (VPS)**: Valida apos a indexacao no Milvus

Esta carta contem as instrucoes para implementar a **Fase Docling** no GPU Server.

---

## Requisitos Funcionais

### 1. Receber Parametros de Validacao do Usuario

O usuario pode opcionalmente informar o range de artigos esperados na tela de upload. Esses parametros chegam no request de ingestao:

```python
# Novos campos no IngestRequest (src/ingestion/models.py)
validate_articles: bool = Field(False, description="Habilita validacao de artigos")
expected_first_article: Optional[int] = Field(None, description="Primeiro artigo esperado (ex: 1)")
expected_last_article: Optional[int] = Field(None, description="Ultimo artigo esperado (ex: 193)")
```

### 2. Detectar Artigos Encontrados (Incluindo Splits)

O sistema de split usa a convencao:
- Artigo normal: `ART-006`
- Artigo splitado: `ART-006-P1`, `ART-006-P2`, `ART-006-P3`
- O artigo pai (`ART-006`) tem `_skip_milvus_index=True` e NAO vai para o Milvus

**Logica de deteccao**:
```
ART-001           → Artigo 1 (nao splitado)
ART-006-P1        → Artigo 6, parte 1
ART-006-P2        → Artigo 6, parte 2
ART-006-P3        → Artigo 6, parte 3
PAR-006-1         → Paragrafo do Art. 6 (nao conta como artigo)
INC-006-I         → Inciso do Art. 6 (nao conta como artigo)
```

### 3. Validar Integridade

- **Sequencia completa**: Se usuario informou 1-193, verificar se todos de 1 a 193 existem
- **Duplicatas**: Detectar se o mesmo artigo aparece mais de uma vez
- **Splits completos**: Se um artigo foi splitado, registrar quantas partes foram geradas

### 4. Retornar Estrutura de Validacao

```python
validation_docling = {
    # Configuracao recebida
    "validate_enabled": True,
    "expected_first_article": 1,
    "expected_last_article": 193,

    # Artigos
    "expected_articles": ["1", "2", "3", ..., "193"],  # Gerado do range
    "found_articles": ["1", "2", "3", "6", "7", ...],  # Artigos unicos encontrados
    "found_list": ["1", "2", "3", "6", "7", ...],      # Alias para compatibilidade
    "missing_articles": ["4", "5", ...],               # Gaps
    "article_gaps": ["4", "5", ...],                   # Alias para compatibilidade
    "duplicate_articles": [],                          # Artigos que apareceram 2x

    # Splits
    "split_articles": [
        {
            "article_number": "6",
            "parts_count": 3,
            "parts": ["ART-006-P1", "ART-006-P2", "ART-006-P3"]
        },
        {
            "article_number": "75",
            "parts_count": 2,
            "parts": ["ART-075-P1", "ART-075-P2"]
        }
    ],

    # Chunks (para validacao pos-Milvus)
    "total_chunks_generated": 450,
    "chunks_manifest": [
        "ART-001", "ART-002", "ART-003",
        "ART-006-P1", "ART-006-P2", "ART-006-P3",
        "PAR-006-1", "INC-006-I",
        ...
    ],

    # Metricas
    "total_found": 193,                    # Artigos unicos
    "first_article": 1,                    # Menor numero encontrado
    "last_article": 193,                   # Maior numero encontrado
    "has_gaps": False,
    "has_duplicates": False,
    "coverage_percent": 100.0,

    # Status
    "status": "passed" | "warning" | "failed"
}
```

---

## Arquivos a Modificar

### 1. `src/ingestion/models.py`

Adicionar campos ao `IngestRequest`:

```python
class IngestRequest(BaseModel):
    """Request para processamento de PDF."""

    # ... campos existentes ...

    # Validacao de artigos (NOVO)
    validate_articles: bool = Field(False, description="Habilita validacao de artigos")
    expected_first_article: Optional[int] = Field(None, description="Primeiro artigo esperado")
    expected_last_article: Optional[int] = Field(None, description="Ultimo artigo esperado")
```

### 2. `src/ingestion/router.py`

Adicionar os novos campos no endpoint `/ingest`:

```python
@router.post("", response_model=IngestStartResponse)
async def ingest_pdf(
    # ... campos existentes ...

    # Validacao de artigos (NOVO)
    validate_articles: bool = Form(False, description="Habilita validacao de artigos"),
    expected_first_article: Optional[int] = Form(None, description="Primeiro artigo esperado"),
    expected_last_article: Optional[int] = Form(None, description="Ultimo artigo esperado"),
):
    # Criar request com novos campos
    request = IngestRequest(
        # ... campos existentes ...
        validate_articles=validate_articles,
        expected_first_article=expected_first_article,
        expected_last_article=expected_last_article,
    )
```

### 3. `src/ingestion/article_validator.py` (CRIAR OU ATUALIZAR)

```python
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
from typing import Optional
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
            "found_list": self.found_articles,  # Alias
            "missing_articles": self.missing_articles,
            "article_gaps": self.missing_articles,  # Alias
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

    def validate(self, chunks: list) -> ArticleValidationResult:
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
            span_id = getattr(chunk, 'span_id', None) or chunk.get('span_id', '')
            if span_id:
                span_ids.append(span_id)

        # Registrar manifesto de chunks (para validacao pos-Milvus)
        result.chunks_manifest = span_ids
        result.total_chunks_generated = len(span_ids)

        # Extrair artigos dos span_ids
        articles_found = {}  # article_number -> list of span_ids
        splits_found = {}    # article_number -> list of parts (ART-006-P1, etc)

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
        if self.validate_enabled and self.expected_first and self.expected_last:
            result.expected_articles = [
                str(i) for i in range(self.expected_first, self.expected_last + 1)
            ]
        elif result.first_article and result.last_article:
            # Inferir do range encontrado
            result.expected_articles = [
                str(i) for i in range(result.first_article, result.last_article + 1)
            ]

        # Detectar gaps (artigos faltando na sequencia)
        expected_set = set(result.expected_articles)
        found_set = set(result.found_articles)
        missing = expected_set - found_set
        result.missing_articles = sorted(missing, key=lambda x: int(x))
        result.has_gaps = len(result.missing_articles) > 0

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
```

### 4. `src/ingestion/pipeline.py`

Integrar a validacao no final do pipeline, apos o `ChunkMaterializer`:

```python
# No metodo process() ou onde os chunks sao materializados

from .article_validator import ArticleValidator

# ... apos ChunkMaterializer gerar os chunks ...

# Validacao de artigos
validator = ArticleValidator(
    validate_enabled=request.validate_articles,
    expected_first=request.expected_first_article,
    expected_last=request.expected_last_article,
)
validation_result = validator.validate(materialized_chunks)

# Adicionar ao resultado do pipeline
result.validation_docling = validation_result.to_dict()
```

---

## Testes

### Teste Manual

1. Ingerir um documento pequeno (ex: IN-58-2022 com ~18 artigos)
2. Verificar se `validation_docling` esta presente na resposta
3. Verificar se os campos estao corretos

### Teste com Split

1. Ingerir a Lei 14.133/2021 (tem artigos grandes que sao splitados)
2. Verificar se `split_articles` lista os artigos splitados corretamente
3. Verificar se `chunks_manifest` contem todas as partes

### Cenarios de Teste

| Cenario | Input | Resultado Esperado |
|---------|-------|-------------------|
| Documento completo | IN-58-2022 | status=passed, has_gaps=false |
| Com range do usuario | first=1, last=18 | expected_articles=[1..18] |
| Artigo splitado | Lei 14.133 Art. 6 | split_articles contem Art. 6 com 3 partes |
| Inferir range | Sem input usuario | expected_articles inferido do encontrado |

---

## Checklist de Implementacao

- [ ] Adicionar campos em `IngestRequest` (models.py)
- [ ] Adicionar campos no endpoint `/ingest` (router.py)
- [ ] Criar/atualizar `ArticleValidator` (article_validator.py)
- [ ] Integrar no pipeline apos ChunkMaterializer (pipeline.py)
- [ ] Garantir que `validation_docling` esta no resultado (router.py _set_task_result)
- [ ] Testar com documento pequeno
- [ ] Testar com documento com splits

---

## Duvidas?

Se tiver duvidas sobre:
- Estrutura do `MaterializedChunk`: ver `src/chunking/chunk_materializer.py`
- Convencao de span_ids: ver linha 798 de `chunk_materializer.py`
- Como o split funciona: artigos > 8k chars sao divididos em `ART-XXX-P1`, `ART-XXX-P2`, etc.

---

**Aguardo confirmacao quando a implementacao estiver concluida!**

-- Claude (Windows)
