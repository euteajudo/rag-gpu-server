"""
Sanitizador de Markdown gerado pelo Docling.

Remove anomalias conhecidas que podem vazar para o usuário final:
- `<!-- image -->` - Placeholders de imagens
- Outros marcadores de elementos não-textuais
"""

import re
import logging
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SanitizationReport:
    """Relatório de sanitização do markdown."""
    original_length: int
    sanitized_length: int
    anomalies_removed: int
    anomalies_found: List[Tuple[str, int]]  # (tipo, contagem)
    changes_made: List[str]  # Descrição das mudanças


class MarkdownSanitizer:
    """
    Sanitiza markdown gerado pelo Docling.

    Remove placeholders e anomalias que não devem chegar ao Milvus.
    """

    # Padrões de anomalias a remover
    ANOMALY_PATTERNS = [
        # HTML comments de imagem (Docling)
        (r'<!--\s*image\s*-->', 'html_image_comment'),

        # Markdown image placeholders
        (r'!\[.*?\]\(.*?\)', 'markdown_image'),

        # Variações de [image]
        (r'\[image\]', 'image_bracket'),
        (r'\[IMAGE\]', 'image_bracket_upper'),

        # Placeholders genéricos
        (r'--image--', 'double_dash_image'),
        (r'\[figure\]', 'figure_bracket'),
        (r'\[FIGURE\]', 'figure_bracket_upper'),

        # Tags HTML soltas
        (r'<image\s*/?\s*>', 'html_image_tag'),
        (r'<figure\s*/?\s*>', 'html_figure_tag'),
        (r'</figure>', 'html_figure_close'),

        # Linhas em branco múltiplas (cleanup)
        (r'\n{4,}', 'multiple_blank_lines'),
    ]

    def __init__(self, aggressive: bool = False):
        """
        Args:
            aggressive: Se True, remove mais padrões (pode afetar conteúdo válido)
        """
        self.aggressive = aggressive
        self._compile_patterns()

    def _compile_patterns(self):
        """Pré-compila os padrões regex."""
        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE | re.MULTILINE), name)
            for pattern, name in self.ANOMALY_PATTERNS
        ]

    def sanitize(self, markdown: str) -> Tuple[str, SanitizationReport]:
        """
        Sanitiza o markdown removendo anomalias.

        Args:
            markdown: Texto markdown a sanitizar

        Returns:
            Tuple (markdown_sanitizado, relatório)
        """
        if not markdown:
            return markdown, SanitizationReport(
                original_length=0,
                sanitized_length=0,
                anomalies_removed=0,
                anomalies_found=[],
                changes_made=[],
            )

        original_length = len(markdown)
        sanitized = markdown
        anomalies_found = []
        changes_made = []
        total_removed = 0

        for pattern, name in self._compiled_patterns:
            matches = pattern.findall(sanitized)
            count = len(matches)

            if count > 0:
                anomalies_found.append((name, count))
                total_removed += count

                # Substitui por espaço ou string vazia dependendo do tipo
                if name == 'multiple_blank_lines':
                    # Mantém apenas duas linhas em branco
                    sanitized = pattern.sub('\n\n\n', sanitized)
                    changes_made.append(f"Normalizado {count} sequências de linhas em branco")
                else:
                    # Remove completamente
                    sanitized = pattern.sub('', sanitized)
                    changes_made.append(f"Removido {count}x '{name}'")

                logger.info(f"Sanitização: removido {count}x '{name}'")

        # Limpeza final: remove espaços duplos e linhas com apenas espaços
        sanitized = re.sub(r' {2,}', ' ', sanitized)
        sanitized = re.sub(r'\n +\n', '\n\n', sanitized)

        # Remove linhas que ficaram vazias após remoção de anomalias
        sanitized = re.sub(r'\n\s*\n\s*\n', '\n\n', sanitized)

        return sanitized, SanitizationReport(
            original_length=original_length,
            sanitized_length=len(sanitized),
            anomalies_removed=total_removed,
            anomalies_found=anomalies_found,
            changes_made=changes_made,
        )

    def detect_anomalies(self, markdown: str) -> List[Tuple[str, int, List[str]]]:
        """
        Detecta anomalias sem remover (para relatório).

        Returns:
            Lista de (nome_anomalia, contagem, exemplos)
        """
        results = []

        for pattern, name in self._compiled_patterns:
            matches = pattern.findall(markdown)
            if matches:
                # Pega até 3 exemplos únicos
                examples = list(set(matches[:3]))
                results.append((name, len(matches), examples))

        return results


def sanitize_markdown(markdown: str) -> str:
    """
    Função utilitária para sanitização rápida.

    Uso:
        from ingestion.markdown_sanitizer import sanitize_markdown
        clean_md = sanitize_markdown(raw_markdown)
    """
    sanitizer = MarkdownSanitizer()
    sanitized, _ = sanitizer.sanitize(markdown)
    return sanitized


def detect_markdown_anomalies(markdown: str) -> List[Tuple[str, int]]:
    """
    Função utilitária para detecção de anomalias.

    Returns:
        Lista de (tipo, contagem)
    """
    sanitizer = MarkdownSanitizer()
    anomalies = sanitizer.detect_anomalies(markdown)
    return [(name, count) for name, count, _ in anomalies]
