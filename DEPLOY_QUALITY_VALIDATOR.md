# Deploy do Quality Validator - Pipeline Robusto

## Resumo das Mudanças

Este deploy adiciona um sistema de validação de qualidade de extração que:

1. **Detecta PDFs com texto corrompido** (como LCP 123 com fontes não-Unicode)
2. **Faz fallback automático para OCR** quando a qualidade é baixa
3. **Valida a extração** antes de continuar o pipeline

## Arquivos Modificados

| Arquivo | Ação | Descrição |
|---------|------|-----------|
| `src/ingestion/quality_validator.py` | **NOVO** | Módulo de validação de qualidade |
| `src/ingestion/pipeline.py` | **ATUALIZADO** | Pipeline com validação + OCR fallback |
| `src/ingestion/__init__.py` | **ATUALIZADO** | Exports das novas classes |

## Instruções de Deploy no RunPod

### Opção 1: Via Web Terminal do RunPod

1. Acesse o painel do RunPod: https://runpod.io/
2. Encontre o pod `vectorgov-gpu`
3. Clique em "Connect" → "Web Terminal"
4. Execute os comandos abaixo

### Opção 2: Via SSH (se disponível)

Se você tiver acesso SSH ao RunPod, use os comandos abaixo.

---

## Passo 1: Backup dos Arquivos Originais

```bash
cd /workspace/rag-gpu-server/src/ingestion
cp pipeline.py pipeline.py.bak
cp __init__.py __init__.py.bak
```

## Passo 2: Criar quality_validator.py

```bash
cat > /workspace/rag-gpu-server/src/ingestion/quality_validator.py << 'ENDOFFILE'
"""
Validador de Qualidade de Extração de PDF.

Detecta problemas comuns em PDFs:
- Fontes com encoding corrompido (caracteres "lixo")
- PDFs escaneados sem OCR
- PDFs com camada de texto ilegível
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Relatório de qualidade da extração."""
    is_valid: bool
    score: float  # 0.0 a 1.0
    readable_ratio: float  # Proporção de caracteres legíveis
    article_count: int  # Número de artigos detectados
    word_count: int  # Número de palavras válidas
    issues: list  # Lista de problemas encontrados
    recommendation: str  # "continue", "retry_ocr", "manual_review"


class QualityValidator:
    """
    Valida a qualidade do texto extraído de PDFs.

    Detecta problemas como:
    - Fontes com encoding não-Unicode (caracteres aleatórios)
    - Texto muito curto para o tamanho do PDF
    - Ausência de estrutura legal (Art., §, etc.)
    """

    # Palavras comuns em português que devem aparecer em documentos legais
    PORTUGUESE_WORDS = {
        "de", "da", "do", "das", "dos", "em", "no", "na", "nos", "nas",
        "para", "por", "com", "sem", "sobre", "entre", "até", "desde",
        "que", "qual", "quais", "como", "quando", "onde", "porque",
        "lei", "decreto", "artigo", "parágrafo", "inciso", "alínea",
        "será", "serão", "deve", "devem", "pode", "podem", "dispõe",
        "federal", "estadual", "municipal", "público", "privado",
        "contrato", "licitação", "dispensa", "inexigibilidade",
        "prazo", "valor", "preço", "pagamento", "execução",
    }

    # Padrões que indicam texto legal válido
    LEGAL_PATTERNS = [
        r'\bArt\.?\s*\d+',  # Art. 1, Art 2
        r'§\s*\d+',  # § 1º
        r'\b[IVX]+\s*[-–]',  # I -, II -
        r'\balínea\b',
        r'\binciso\b',
        r'\bparágrafo\b',
        r'\bcapítulo\b',
        r'\bseção\b',
    ]

    # Caracteres que indicam texto corrompido (sequências sem sentido)
    GARBAGE_PATTERNS = [
        r'[A-Z]{10,}',  # Sequências longas de maiúsculas sem espaço
        r'[\x00-\x08\x0b\x0c\x0e-\x1f]',  # Caracteres de controle
        r'[^\x00-\x7F\u00C0-\u024F\u1E00-\u1EFF]{5,}',  # Sequências de caracteres estranhos
    ]

    def __init__(
        self,
        min_readable_ratio: float = 0.7,
        min_article_count: int = 1,
        min_word_ratio: float = 0.3,
    ):
        self.min_readable_ratio = min_readable_ratio
        self.min_article_count = min_article_count
        self.min_word_ratio = min_word_ratio

    def validate(self, text: str, expected_pages: int = 0) -> QualityReport:
        """Valida a qualidade do texto extraído."""
        issues = []

        if not text or len(text.strip()) < 100:
            return QualityReport(
                is_valid=False,
                score=0.0,
                readable_ratio=0.0,
                article_count=0,
                word_count=0,
                issues=["Texto extraído muito curto ou vazio"],
                recommendation="retry_ocr",
            )

        # 1. Calcula proporção de caracteres legíveis
        readable_ratio = self._calculate_readable_ratio(text)
        if readable_ratio < self.min_readable_ratio:
            issues.append(f"Baixa proporção de caracteres legíveis: {readable_ratio:.1%}")

        # 2. Conta artigos detectados
        article_count = self._count_articles(text)
        if article_count < self.min_article_count:
            issues.append(f"Poucos artigos encontrados: {article_count}")

        # 3. Verifica palavras portuguesas válidas
        word_count, word_ratio = self._count_valid_words(text)
        if word_ratio < self.min_word_ratio:
            issues.append(f"Baixa proporção de palavras válidas: {word_ratio:.1%}")

        # 4. Detecta padrões de texto corrompido
        garbage_detected = self._detect_garbage(text)
        if garbage_detected:
            issues.append("Detectado texto corrompido (encoding de fonte)")

        # 5. Verifica padrões legais
        legal_patterns_found = self._count_legal_patterns(text)
        if legal_patterns_found == 0:
            issues.append("Nenhum padrão de documento legal encontrado")

        # Calcula score final (0-1)
        score = self._calculate_score(
            readable_ratio=readable_ratio,
            article_count=article_count,
            word_ratio=word_ratio,
            garbage_detected=garbage_detected,
            legal_patterns_found=legal_patterns_found,
        )

        # Determina se é válido e recomendação
        is_valid = score >= 0.6 and not garbage_detected

        if is_valid:
            recommendation = "continue"
        elif garbage_detected or readable_ratio < 0.5:
            recommendation = "retry_ocr"
        else:
            recommendation = "manual_review"

        return QualityReport(
            is_valid=is_valid,
            score=score,
            readable_ratio=readable_ratio,
            article_count=article_count,
            word_count=word_count,
            issues=issues,
            recommendation=recommendation,
        )

    def _calculate_readable_ratio(self, text: str) -> float:
        """Calcula proporção de caracteres legíveis no texto."""
        if not text:
            return 0.0

        readable_count = 0
        total_count = 0

        for char in text:
            if char.isspace():
                continue
            total_count += 1

            category = unicodedata.category(char)
            if category.startswith(('L', 'N', 'P', 'S')):
                if category not in ('Co', 'Cn', 'Cc'):
                    readable_count += 1

        return readable_count / total_count if total_count > 0 else 0.0

    def _count_articles(self, text: str) -> int:
        """Conta número de artigos no texto."""
        pattern = r'\bArt\.?\s*\d+'
        matches = re.findall(pattern, text, re.IGNORECASE)
        return len(set(matches))

    def _count_valid_words(self, text: str) -> Tuple[int, float]:
        """Conta palavras portuguesas válidas."""
        words = re.findall(r'\b[a-záàâãéèêíïóôõöúçñ]+\b', text.lower())

        if not words:
            return 0, 0.0

        valid_count = sum(1 for w in words if w in self.PORTUGUESE_WORDS)
        ratio = valid_count / len(words)

        return valid_count, ratio

    def _detect_garbage(self, text: str) -> bool:
        """Detecta se o texto contém padrões de texto corrompido."""
        for pattern in self.GARBAGE_PATTERNS:
            matches = re.findall(pattern, text)
            if matches:
                total_garbage = sum(len(m) for m in matches)
                if total_garbage > len(text) * 0.1:
                    logger.warning(f"Garbage detected: {matches[:3]}...")
                    return True

        non_standard = re.findall(r'[^\x00-\x7F\u00C0-\u00FF]{3,}', text)
        if len(non_standard) > 10:
            logger.warning(f"Non-standard sequences: {non_standard[:3]}...")
            return True

        return False

    def _count_legal_patterns(self, text: str) -> int:
        """Conta padrões de documento legal encontrados."""
        count = 0
        for pattern in self.LEGAL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                count += 1
        return count

    def _calculate_score(
        self,
        readable_ratio: float,
        article_count: int,
        word_ratio: float,
        garbage_detected: bool,
        legal_patterns_found: int,
    ) -> float:
        """Calcula score final de qualidade (0-1)."""
        score = 0.0

        # Readable ratio (30%)
        score += readable_ratio * 0.3

        # Artigos encontrados (25%)
        article_score = min(article_count / 10, 1.0)
        score += article_score * 0.25

        # Palavras válidas (25%)
        score += word_ratio * 0.25

        # Padrões legais (20%)
        pattern_score = min(legal_patterns_found / 5, 1.0)
        score += pattern_score * 0.2

        # Penalidade por garbage
        if garbage_detected:
            score *= 0.3

        return min(max(score, 0.0), 1.0)


def quick_quality_check(text: str) -> Tuple[bool, str]:
    """Verificação rápida de qualidade para uso inline."""
    validator = QualityValidator()
    report = validator.validate(text)

    if report.is_valid:
        return True, "OK"
    else:
        return False, "; ".join(report.issues)
ENDOFFILE
```

## Passo 3: Atualizar pipeline.py

O arquivo `pipeline.py` é muito grande para incluir aqui. Copie-o do repositório local:

```bash
# Se você tiver acesso ao repositório git no RunPod:
cd /workspace/rag-gpu-server
git pull origin main

# Ou copie manualmente via SCP/SFTP
```

## Passo 4: Atualizar __init__.py

```bash
cat > /workspace/rag-gpu-server/src/ingestion/__init__.py << 'ENDOFFILE'
"""
Módulo de Ingestão de PDFs com GPU.

Pipeline completo de processamento:
1. Docling (PDF → Markdown) - GPU accelerated
2. Validação de Qualidade (detecta texto corrompido)
3. OCR Fallback (se qualidade baixa)
4. SpanParser (Markdown → Spans)
5. ArticleOrchestrator (LLM extraction)
6. ChunkMaterializer (parent-child chunks)
7. Enriquecimento (context, thesis, questions)
8. Embeddings (BGE-M3)

Retorna chunks prontos para indexação no Milvus.
"""

from .models import (
    IngestRequest,
    IngestResponse,
    ProcessedChunk,
    IngestStatus,
    IngestError,
)
from .pipeline import IngestionPipeline, ExtractionMethod, PipelineResult
from .quality_validator import QualityValidator, QualityReport
from .router import router as ingestion_router

__all__ = [
    # Models
    "IngestRequest",
    "IngestResponse",
    "ProcessedChunk",
    "IngestStatus",
    "IngestError",
    # Pipeline
    "IngestionPipeline",
    "PipelineResult",
    "ExtractionMethod",
    # Quality Validation
    "QualityValidator",
    "QualityReport",
    # Router
    "ingestion_router",
]
ENDOFFILE
```

## Passo 5: Reiniciar o Servidor

```bash
# Via systemd (se configurado)
sudo systemctl restart rag-gpu

# OU via script de init
/workspace/init-after-restart.sh

# OU manualmente
cd /workspace/rag-gpu-server
source .venv/bin/activate
pkill -f "uvicorn src.main"
nohup python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 &
```

## Passo 6: Verificar se Funcionou

```bash
# Verificar health
curl http://localhost:8000/health

# Verificar logs
tail -f /workspace/rag-gpu-server/logs/app.log
```

---

## Comportamento Esperado Após o Deploy

### Para PDFs com texto nativo (IN-58, Lei 14133):
- Extração rápida (~4s para documentos pequenos)
- Quality score > 0.6
- Sem OCR fallback

### Para PDFs com texto corrompido (LCP 123):
- Detecção automática de texto "lixo"
- Fallback para OCR (força reconhecimento de caracteres na imagem)
- Tempo maior (~2-5 minutos dependendo do tamanho)
- Quality score pode ainda ser baixo se OCR não conseguir ler

### Log de Exemplo (PDF problemático):

```
Fase 1.1: Docling iniciando (texto nativo)...
Fase 1.2: Qualidade do texto nativo: score=0.15, readable=45.0%, articles=0
Fase 1.3: Qualidade baixa (0.15), issues: ['Detectado texto corrompido']. Tentando OCR...
Fase 1.4: Qualidade após OCR: score=0.72, readable=89.0%, articles=12 (tempo: 180s)
Fase 1: Docling concluida em 185s (OCR fallback)
```
