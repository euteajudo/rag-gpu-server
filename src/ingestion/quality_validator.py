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
    issues: list[str]  # Lista de problemas encontrados
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
        """
        Args:
            min_readable_ratio: Proporção mínima de caracteres legíveis (0-1)
            min_article_count: Número mínimo de artigos esperados
            min_word_ratio: Proporção mínima de palavras portuguesas válidas
        """
        self.min_readable_ratio = min_readable_ratio
        self.min_article_count = min_article_count
        self.min_word_ratio = min_word_ratio

    def validate(self, text: str, expected_pages: int = 0) -> QualityReport:
        """
        Valida a qualidade do texto extraído.

        Args:
            text: Texto extraído do PDF
            expected_pages: Número de páginas do PDF (para calcular proporção)

        Returns:
            QualityReport com resultado da validação
        """
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

            # Caractere é legível se:
            # - É letra (qualquer script)
            # - É número
            # - É pontuação comum
            # - É caractere especial legal (§, º, ª, etc.)
            category = unicodedata.category(char)
            if category.startswith(('L', 'N', 'P', 'S')):
                # Verifica se não é caractere de controle ou privado
                if category not in ('Co', 'Cn', 'Cc'):
                    readable_count += 1

        return readable_count / total_count if total_count > 0 else 0.0

    def _count_articles(self, text: str) -> int:
        """Conta número de artigos no texto."""
        pattern = r'\bArt\.?\s*\d+'
        matches = re.findall(pattern, text, re.IGNORECASE)
        return len(set(matches))  # Únicos

    def _count_valid_words(self, text: str) -> Tuple[int, float]:
        """Conta palavras portuguesas válidas."""
        # Extrai palavras (letras apenas)
        words = re.findall(r'\b[a-záàâãéèêíïóôõöúçñ]+\b', text.lower())

        if not words:
            return 0, 0.0

        valid_count = sum(1 for w in words if w in self.PORTUGUESE_WORDS)
        ratio = valid_count / len(words)

        return valid_count, ratio

    def _detect_garbage(self, text: str) -> bool:
        """Detecta se o texto contém padrões de texto corrompido."""
        # Verifica sequências longas de maiúsculas sem sentido
        # Exemplo do LCP 123: "FGHIKLMFNOPQRSTUOSOVHFGVLWXYZMOW"

        for pattern in self.GARBAGE_PATTERNS:
            matches = re.findall(pattern, text)
            if matches:
                # Se encontrou muitas ocorrências, é provável que seja lixo
                total_garbage = sum(len(m) for m in matches)
                if total_garbage > len(text) * 0.1:  # Mais de 10% é lixo
                    logger.warning(f"Garbage detected: {matches[:3]}...")
                    return True

        # Verifica se há muitos caracteres não-ASCII consecutivos que não são acentos
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
        # Pesos para cada métrica
        score = 0.0

        # Readable ratio (30%)
        score += readable_ratio * 0.3

        # Artigos encontrados (25%)
        article_score = min(article_count / 10, 1.0)  # Cap em 10 artigos
        score += article_score * 0.25

        # Palavras válidas (25%)
        score += word_ratio * 0.25

        # Padrões legais (20%)
        pattern_score = min(legal_patterns_found / 5, 1.0)  # Cap em 5 padrões
        score += pattern_score * 0.2

        # Penalidade por garbage
        if garbage_detected:
            score *= 0.3  # Penaliza 70%

        return min(max(score, 0.0), 1.0)


def quick_quality_check(text: str) -> Tuple[bool, str]:
    """
    Verificação rápida de qualidade para uso inline.

    Returns:
        Tuple (is_ok, reason)
    """
    validator = QualityValidator()
    report = validator.validate(text)

    if report.is_valid:
        return True, "OK"
    else:
        return False, "; ".join(report.issues)
