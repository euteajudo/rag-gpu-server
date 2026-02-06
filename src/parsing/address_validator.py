"""
AddressValidator - Valida consistência entre span_id e texto do chunk.

Este módulo implementa a validação ADDRESS_MISMATCH para detectar chunks
cujo span_id (endereço) não corresponde ao conteúdo real do texto.

Bug corrigido:
    O SpanParser pode criar spans com ID incorreto quando citações internas
    (ex: "conforme § 1º deste artigo") são detectadas como novos dispositivos.

    Exemplo do bug:
        span_id: PAR-040-1 (indica § 1º do Art. 40)
        text: "§ 4º A fase preparatória..." (texto do § 4º, não § 1º!)

Uso:
    from parsing.address_validator import AddressValidator

    validator = AddressValidator()
    result = validator.validate_span(span)

    if result.is_mismatch:
        logger.error(f"ADDRESS_MISMATCH: {result.message}")

@since: 2026-02-06
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Resultado da validação de endereço."""
    is_valid: bool
    is_mismatch: bool
    message: str
    span_id: str
    expected_prefix: str
    actual_prefix: str


class AddressValidator:
    """
    Valida se o span_id corresponde ao conteúdo do texto.

    Regras de validação:
    - PAR-040-1 → texto deve começar com "§ 1" (variantes: §1º, § 1°, etc)
    - PAR-040-UNICO → texto deve conter "parágrafo único" ou "§ único"
    - ART-044 → texto deve começar com "Art. 44" (ou "Artigo 44")
    - INC-036-V → texto deve começar com "V -" (ou "V–", "V—")
    - ALI-036-V-a → texto deve começar com "a)" (ou "a -")
    """

    # Padrões para extrair o identificador esperado do texto
    PATTERNS = {
        "PAR": [
            # § 1º, §1°, § 1, etc.
            r'^[§]\s*(\d+)[ºo°]?',
            # Parágrafo único
            r'^[Pp]ar[áa]grafo\s+[úu]nico',
        ],
        "ART": [
            # Art. 44, Art 44, Artigo 44
            r'^Art(?:igo)?\.?\s*(\d+)',
        ],
        "INC": [
            # I -, II –, III —, IV-
            r'^([IVXLC]+)\s*[-–—]',
        ],
        "ALI": [
            # a), b), c)
            r'^([a-z])\)',
        ],
    }

    def __init__(self):
        """Inicializa o validador com padrões compilados."""
        self._compiled_patterns = {}
        for span_type, patterns in self.PATTERNS.items():
            self._compiled_patterns[span_type] = [
                re.compile(p, re.IGNORECASE | re.MULTILINE)
                for p in patterns
            ]

    def validate_span(self, span) -> ValidationResult:
        """
        Valida se o span_id corresponde ao texto.

        Args:
            span: Objeto Span com span_id e text

        Returns:
            ValidationResult com status da validação
        """
        span_id = getattr(span, 'span_id', '') or ''
        text = getattr(span, 'text', '') or ''

        # Extrai tipo do span (PAR, ART, INC, ALI)
        span_type = span_id.split('-')[0] if '-' in span_id else ''

        if span_type not in self._compiled_patterns:
            # Tipo desconhecido ou não validável (HDR, CAP, etc.)
            return ValidationResult(
                is_valid=True,
                is_mismatch=False,
                message="Tipo de span não requer validação",
                span_id=span_id,
                expected_prefix="",
                actual_prefix="",
            )

        # Extrai identificador do span_id
        expected_id = self._extract_expected_id(span_id, span_type)

        # Extrai identificador do texto
        actual_id = self._extract_actual_id(text, span_type)

        # Compara
        is_match = self._ids_match(expected_id, actual_id, span_type)

        if is_match:
            return ValidationResult(
                is_valid=True,
                is_mismatch=False,
                message="OK",
                span_id=span_id,
                expected_prefix=expected_id,
                actual_prefix=actual_id,
            )
        else:
            return ValidationResult(
                is_valid=False,
                is_mismatch=True,
                message=f"ADDRESS_MISMATCH: span_id={span_id} indica '{expected_id}' "
                       f"mas texto começa com '{actual_id}'",
                span_id=span_id,
                expected_prefix=expected_id,
                actual_prefix=actual_id,
            )

    def _extract_expected_id(self, span_id: str, span_type: str) -> str:
        """Extrai o identificador esperado do span_id."""
        parts = span_id.split('-')

        if span_type == "PAR":
            # PAR-040-1 → "1", PAR-040-UNICO → "UNICO"
            return parts[-1] if len(parts) >= 3 else ""

        elif span_type == "ART":
            # ART-044 → "44", ART-337-E → "337-E"
            if len(parts) >= 2:
                # Remove zeros à esquerda
                art_num = parts[1].lstrip('0') or '0'
                # Se tem sufixo letra (ART-337-E)
                if len(parts) >= 3 and len(parts[2]) == 1 and parts[2].isalpha():
                    return f"{art_num}-{parts[2]}"
                return art_num
            return ""

        elif span_type == "INC":
            # INC-040-I → "I", INC-040-II_2 → "II"
            if len(parts) >= 3:
                inciso = parts[2]
                # Remove sufixo de desambiguação (_2, _3, etc.)
                if '_' in inciso:
                    inciso = inciso.split('_')[0]
                return inciso
            return ""

        elif span_type == "ALI":
            # ALI-040-I-a → "a"
            return parts[-1] if len(parts) >= 4 else ""

        return ""

    def _extract_actual_id(self, text: str, span_type: str) -> str:
        """Extrai o identificador real do início do texto."""
        text = text.strip()

        for pattern in self._compiled_patterns.get(span_type, []):
            match = pattern.match(text)
            if match:
                if match.groups():
                    return match.group(1)
                # Para "Parágrafo único" que não tem grupo
                if span_type == "PAR" and "único" in text.lower()[:30]:
                    return "UNICO"

        # Fallback: primeiros 20 chars para diagnóstico
        return text[:20].replace('\n', ' ') if text else "(vazio)"

    def _ids_match(self, expected: str, actual: str, span_type: str) -> bool:
        """Verifica se os identificadores correspondem."""
        if not expected or not actual:
            return False

        expected = expected.upper()
        actual = actual.upper()

        if span_type == "PAR":
            # Para parágrafos, compara números ou "UNICO"
            if expected == "UNICO":
                return "UNICO" in actual or "ÚNICO" in actual.replace("U", "Ú")
            # Compara números (§ 1 == 1)
            return expected == actual

        elif span_type == "ART":
            # Remove zeros à esquerda para comparação
            return expected.lstrip('0') == actual.lstrip('0')

        elif span_type == "INC":
            # Romanos são case-insensitive
            return expected == actual

        elif span_type == "ALI":
            # Letras são case-insensitive
            return expected == actual

        return expected == actual

    def validate_all(self, spans: list) -> dict:
        """
        Valida todos os spans e retorna estatísticas.

        Args:
            spans: Lista de objetos Span

        Returns:
            dict com estatísticas de validação
        """
        results = {
            "total": len(spans),
            "valid": 0,
            "mismatches": 0,
            "skipped": 0,
            "mismatch_details": [],
        }

        for span in spans:
            result = self.validate_span(span)

            if result.is_mismatch:
                results["mismatches"] += 1
                results["mismatch_details"].append({
                    "span_id": result.span_id,
                    "expected": result.expected_prefix,
                    "actual": result.actual_prefix,
                    "message": result.message,
                })
                logger.warning(result.message)
            elif result.is_valid:
                results["valid"] += 1
            else:
                results["skipped"] += 1

        return results
