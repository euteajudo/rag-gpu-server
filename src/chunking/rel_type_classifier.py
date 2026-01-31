# PR5 REL_TYPE CLASSIFIER
"""
Classificador de tipos de relacionamento para citações normativas.

Este módulo classifica citações extraídas em tipos semânticos específicos:
- REVOGA_EXPRESSAMENTE: "fica revogado o art. X"
- ALTERA_EXPRESSAMENTE: "passa a vigorar com a seguinte redação"
- REGULAMENTA: "Este Decreto regulamenta..."
- DEPENDE_DE: "nos termos do art. X"
- EXCEPCIONA: "salvo o disposto no art. X"
- REFERENCIA: menção sem efeito jurídico
- CITA: citação genérica (default)

NOTA: REVOGA_TACITAMENTE NÃO é emitido por este classificador.
      Conflitos implícitos requerem análise manual/LLM especializada.

@author: Equipe VectorGov
@since: 30/01/2025
@pr: PR5 - Classificação de rel_type
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RelTypePattern:
    """
    Padrão para classificação de tipo de relacionamento.

    Attributes:
        pattern: Regex compilado para matching
        rel_type: Tipo de relacionamento a atribuir
        confidence: Confiança base do padrão (0.0 a 1.0)
        description: Descrição do padrão para debug
    """
    pattern: re.Pattern
    rel_type: str
    confidence: float
    description: str = ""


# =============================================================================
# PADRÕES DE CLASSIFICAÇÃO (ordem define precedência)
# =============================================================================
# IMPORTANTE: Padrões mais específicos devem vir primeiro.
# O primeiro match ganha.

REL_TYPE_PATTERNS: list[RelTypePattern] = [
    # -------------------------------------------------------------------------
    # REVOGA_EXPRESSAMENTE (HIGH priority)
    # Textos que explicitamente revogam dispositivos
    # -------------------------------------------------------------------------
    RelTypePattern(
        pattern=re.compile(
            r"(?:fica[mn]?\s+)?revogad[oa]s?(?:\s+expressamente)?",
            re.IGNORECASE
        ),
        rel_type="REVOGA_EXPRESSAMENTE",
        confidence=0.95,
        description="Revogação expressa: 'fica revogado', 'ficam revogados'"
    ),
    RelTypePattern(
        pattern=re.compile(
            r"revoga(?:m|-se)?\s+(?:o\s+)?(?:art|§|inciso|al[ií]nea)",
            re.IGNORECASE
        ),
        rel_type="REVOGA_EXPRESSAMENTE",
        confidence=0.90,
        description="Revogação direta: 'revoga o art.', 'revogam-se'"
    ),
    RelTypePattern(
        pattern=re.compile(
            r"revoga[çc][aã]o\s+(?:expressa|total|integral)",
            re.IGNORECASE
        ),
        rel_type="REVOGA_EXPRESSAMENTE",
        confidence=0.92,
        description="Revogação nominal: 'revogação expressa'"
    ),

    # -------------------------------------------------------------------------
    # ALTERA_EXPRESSAMENTE (HIGH priority)
    # Textos que explicitamente alteram dispositivos
    # -------------------------------------------------------------------------
    RelTypePattern(
        pattern=re.compile(
            r"passa[mn]?\s+a\s+vigorar\s+com\s+(?:a\s+seguinte\s+)?reda[çc][aã]o",
            re.IGNORECASE
        ),
        rel_type="ALTERA_EXPRESSAMENTE",
        confidence=0.95,
        description="Alteração com nova redação: 'passa a vigorar com a seguinte redação'"
    ),
    RelTypePattern(
        pattern=re.compile(
            r"altera(?:m|-se)?\s+(?:o\s+)?(?:art|§|inciso|al[ií]nea|caput)",
            re.IGNORECASE
        ),
        rel_type="ALTERA_EXPRESSAMENTE",
        confidence=0.90,
        description="Alteração direta: 'altera o art.', 'alteram-se'"
    ),
    RelTypePattern(
        pattern=re.compile(
            r"nova\s+reda[çc][aã]o\s+(?:d[oa]\s+)?(?:art|§|inciso|al[ií]nea)",
            re.IGNORECASE
        ),
        rel_type="ALTERA_EXPRESSAMENTE",
        confidence=0.88,
        description="Nova redação: 'nova redação do art.'"
    ),
    RelTypePattern(
        pattern=re.compile(
            r"(?:d[aá]\s+)?nova\s+reda[çc][aã]o\s+a[os]?",
            re.IGNORECASE
        ),
        rel_type="ALTERA_EXPRESSAMENTE",
        confidence=0.85,
        description="Dá nova redação: 'dá nova redação ao'"
    ),
    RelTypePattern(
        pattern=re.compile(
            r"altera[çc][aã]o\s+(?:expressa|do\s+art)",
            re.IGNORECASE
        ),
        rel_type="ALTERA_EXPRESSAMENTE",
        confidence=0.85,
        description="Alteração nominal: 'alteração do art.'"
    ),

    # -------------------------------------------------------------------------
    # REGULAMENTA (MEDIUM priority)
    # Decreto/norma que regulamenta lei
    # -------------------------------------------------------------------------
    RelTypePattern(
        pattern=re.compile(
            r"regulamenta(?:r)?\s+(?:a\s+|o\s+)?(?:lei|art)",
            re.IGNORECASE
        ),
        rel_type="REGULAMENTA",
        confidence=0.90,
        description="Regulamentação: 'regulamenta a Lei', 'regulamentar o art.'"
    ),
    RelTypePattern(
        pattern=re.compile(
            r"(?:este|esta)\s+(?:decreto|instru[çc][aã]o|portaria|resolu[çc][aã]o)\s+regulamenta",
            re.IGNORECASE
        ),
        rel_type="REGULAMENTA",
        confidence=0.95,
        description="Regulamentação no preâmbulo: 'Este Decreto regulamenta'"
    ),
    RelTypePattern(
        pattern=re.compile(
            r"para\s+fins\s+de\s+regulamenta[çc][aã]o",
            re.IGNORECASE
        ),
        rel_type="REGULAMENTA",
        confidence=0.85,
        description="Fins de regulamentação"
    ),

    # -------------------------------------------------------------------------
    # EXCEPCIONA (MEDIUM priority)
    # Exceções a regras gerais - TEM PRECEDÊNCIA sobre DEPENDE_DE
    # -------------------------------------------------------------------------
    RelTypePattern(
        pattern=re.compile(
            r"salvo\s+(?:o\s+)?(?:disposto|previsto|estabelecido)",
            re.IGNORECASE
        ),
        rel_type="EXCEPCIONA",
        confidence=0.92,
        description="Exceção: 'salvo o disposto', 'salvo previsto'"
    ),
    RelTypePattern(
        pattern=re.compile(
            r"exceto\s+(?:o\s+)?(?:disposto|previsto|estabelecido|quando|nos?\s+casos?)",
            re.IGNORECASE
        ),
        rel_type="EXCEPCIONA",
        confidence=0.90,
        description="Exceção: 'exceto o disposto', 'exceto quando'"
    ),
    RelTypePattern(
        pattern=re.compile(
            r"ressalvad[oa]s?\s+(?:o\s+)?(?:disposto|previsto|art)",
            re.IGNORECASE
        ),
        rel_type="EXCEPCIONA",
        confidence=0.88,
        description="Ressalva: 'ressalvado o disposto'"
    ),
    RelTypePattern(
        pattern=re.compile(
            r"com\s+exce[çc][aã]o\s+d[eo]",
            re.IGNORECASE
        ),
        rel_type="EXCEPCIONA",
        confidence=0.85,
        description="Com exceção de"
    ),
    RelTypePattern(
        pattern=re.compile(
            r"excepcionalmente",
            re.IGNORECASE
        ),
        rel_type="EXCEPCIONA",
        confidence=0.75,
        description="Excepcionalmente"
    ),

    # -------------------------------------------------------------------------
    # DEPENDE_DE (MEDIUM priority)
    # Dependências e referências condicionais
    # -------------------------------------------------------------------------
    RelTypePattern(
        pattern=re.compile(
            r"nos\s+termos\s+(?:d[oa]\s+)?(?:art|§|inciso|lei|decreto)",
            re.IGNORECASE
        ),
        rel_type="DEPENDE_DE",
        confidence=0.85,
        description="Nos termos: 'nos termos do art.', 'nos termos da Lei'"
    ),
    RelTypePattern(
        pattern=re.compile(
            r"na\s+forma\s+(?:d[oa]\s+)?(?:art|§|inciso|lei|decreto|regulamento)",
            re.IGNORECASE
        ),
        rel_type="DEPENDE_DE",
        confidence=0.85,
        description="Na forma: 'na forma do art.', 'na forma da Lei'"
    ),
    RelTypePattern(
        pattern=re.compile(
            r"observad[oa]s?\s+(?:o\s+)?(?:disposto|previsto|art|§)",
            re.IGNORECASE
        ),
        rel_type="DEPENDE_DE",
        confidence=0.82,
        description="Observado: 'observado o disposto'"
    ),
    RelTypePattern(
        pattern=re.compile(
            r"atendid[oa]s?\s+(?:o\s+)?(?:disposto|previsto|requisitos?)",
            re.IGNORECASE
        ),
        rel_type="DEPENDE_DE",
        confidence=0.80,
        description="Atendido: 'atendido o disposto'"
    ),
    RelTypePattern(
        pattern=re.compile(
            r"em\s+conson[aâ]ncia\s+com",
            re.IGNORECASE
        ),
        rel_type="DEPENDE_DE",
        confidence=0.78,
        description="Em consonância com"
    ),
    RelTypePattern(
        pattern=re.compile(
            r"de\s+acordo\s+com\s+(?:o\s+)?(?:art|§|disposto|previsto)",
            re.IGNORECASE
        ),
        rel_type="DEPENDE_DE",
        confidence=0.80,
        description="De acordo com: 'de acordo com o art.'"
    ),

    # -------------------------------------------------------------------------
    # REFERENCIA (LOW priority)
    # Menções informativas sem efeito jurídico direto
    # -------------------------------------------------------------------------
    RelTypePattern(
        pattern=re.compile(
            r"(?:conforme|v(?:ide|er))\s+(?:o\s+)?(?:art|§|inciso|disposto)",
            re.IGNORECASE
        ),
        rel_type="REFERENCIA",
        confidence=0.75,
        description="Referência: 'conforme o art.', 'vide art.', 'ver art.'"
    ),
    RelTypePattern(
        pattern=re.compile(
            r"(?:mencionad[oa]|referid[oa]|citad[oa])\s+(?:no|na|n[oa]s?)\s+(?:art|§|inciso)",
            re.IGNORECASE
        ),
        rel_type="REFERENCIA",
        confidence=0.70,
        description="Menção: 'mencionado no art.', 'referido no §'"
    ),
    RelTypePattern(
        pattern=re.compile(
            r"a\s+que\s+(?:se\s+)?refere\s+(?:o\s+)?(?:art|§|inciso)",
            re.IGNORECASE
        ),
        rel_type="REFERENCIA",
        confidence=0.72,
        description="A que se refere: 'a que se refere o art.'"
    ),
]


def classify_rel_type(
    text: str,
    start: int,
    end: int,
    context_window: int = 120,
) -> tuple[str, float]:
    """
    Classifica o tipo de relacionamento de uma citação.

    Analisa o contexto ao redor da citação para determinar o tipo
    de relacionamento jurídico (CITA, ALTERA, REVOGA, etc.).

    Args:
        text: Texto completo do documento/chunk
        start: Posição inicial da citação no texto
        end: Posição final da citação no texto
        context_window: Tamanho da janela de contexto (caracteres antes/depois)

    Returns:
        tuple: (rel_type, rel_type_confidence)
            - rel_type: Tipo de relacionamento (default: "CITA")
            - rel_type_confidence: Confiança na classificação (0.0 a 1.0)

    Example:
        >>> text = "Fica revogado o art. 5º da Lei 14.133/2021."
        >>> rel_type, confidence = classify_rel_type(text, 15, 42)
        >>> print(rel_type)
        'REVOGA_EXPRESSAMENTE'
        >>> print(confidence)
        0.95
    """
    if not text:
        return "CITA", 0.0

    # Extrai janela de contexto
    ctx_start = max(0, start - context_window)
    ctx_end = min(len(text), end + context_window)
    context = text[ctx_start:ctx_end].lower()

    # Procura match nos padrões (primeiro match ganha)
    for pattern_def in REL_TYPE_PATTERNS:
        match = pattern_def.pattern.search(context)
        if match:
            logger.debug(
                f"Classificado como {pattern_def.rel_type} "
                f"(conf={pattern_def.confidence:.2f}): {pattern_def.description}"
            )
            return pattern_def.rel_type, pattern_def.confidence

    # Default: CITA genérico com baixa confiança de classificação
    # (a citação existe, mas não sabemos o tipo específico)
    return "CITA", 0.5


def classify_rel_type_from_match(
    text: str,
    match: "re.Match",
    context_window: int = 120,
) -> tuple[str, float]:
    """
    Classifica rel_type a partir de um objeto Match de regex.

    Wrapper conveniente para usar com resultados de re.finditer().

    Args:
        text: Texto completo
        match: Objeto Match com a citação encontrada
        context_window: Tamanho da janela de contexto

    Returns:
        tuple: (rel_type, rel_type_confidence)
    """
    return classify_rel_type(
        text=text,
        start=match.start(),
        end=match.end(),
        context_window=context_window,
    )


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def get_rel_type_description(rel_type: str) -> str:
    """
    Retorna descrição legível do tipo de relacionamento.

    Args:
        rel_type: Tipo de relacionamento

    Returns:
        Descrição em português
    """
    descriptions = {
        "CITA": "Citação genérica",
        "REFERENCIA": "Menção sem efeito jurídico",
        "ALTERA_EXPRESSAMENTE": "Alteração expressa de dispositivo",
        "REVOGA_EXPRESSAMENTE": "Revogação expressa de dispositivo",
        "REGULAMENTA": "Regulamentação de norma superior",
        "DEPENDE_DE": "Dependência condicional",
        "EXCEPCIONA": "Exceção a regra geral",
        # REVOGA_TACITAMENTE não é emitido pelo classificador
        "REVOGA_TACITAMENTE": "Revogação tácita (requer análise manual)",
    }
    return descriptions.get(rel_type, f"Tipo desconhecido: {rel_type}")


def get_all_patterns() -> list[dict]:
    """
    Retorna todos os padrões de classificação para documentação/debug.

    Returns:
        Lista de dicts com info dos padrões
    """
    return [
        {
            "rel_type": p.rel_type,
            "confidence": p.confidence,
            "description": p.description,
            "pattern": p.pattern.pattern,
        }
        for p in REL_TYPE_PATTERNS
    ]
