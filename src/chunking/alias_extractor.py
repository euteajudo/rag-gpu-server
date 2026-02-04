"""
Alias Extractor - Extrai siglas e acrônimos de textos legais.

Este módulo identifica padrões de definição de siglas em textos jurídicos
para melhorar o recall na busca esparsa.

Padrões detectados:
- "Termo Completo - SIGLA" (ex: "Estudo Técnico Preliminar - ETP")
- "SIGLA (Termo Completo)" (ex: "ETP (Estudo Técnico Preliminar)")
- "Termo Completo (SIGLA)" (ex: "Estudo Técnico Preliminar (ETP)")
- "SIGLA – Termo Completo" (ex: "ETP – Estudo Técnico Preliminar")

Uso:
    from chunking.alias_extractor import extract_aliases

    text = "O Estudo Técnico Preliminar - ETP é documento constitutivo..."
    aliases = extract_aliases(text)
    # aliases = ["ETP", "Estudo Técnico Preliminar"]
"""

import re
from typing import List, Set


# Siglas a ignorar (termos genéricos que não agregam valor)
IGNORED_ALIASES: Set[str] = {
    "ART", "ARTS", "LEI", "LEIS", "IN", "INS",
    "DEC", "DECRETO", "DECRETOS",
    "PAR", "INC", "CAP", "SEC",
    "CF", "CC", "CP", "CPC", "CPP",  # Códigos
    "STF", "STJ", "TST", "TSE",  # Tribunais (muito genéricos)
    "RG", "CPF", "CNPJ", "CEP",  # Documentos
    "II", "III", "IV", "VI", "VII", "VIII", "IX", "XI", "XII",  # Numerais romanos
    "RS", "SP", "RJ", "MG", "BA", "PR", "SC",  # Estados
    "DF", "GO", "MT", "MS", "TO", "PA", "AM",  # Estados
    "AC", "AL", "AP", "CE", "ES", "MA", "PB",  # Estados
    "PE", "PI", "RN", "RO", "RR", "SE",  # Estados
    "HTTP", "HTTPS", "WWW", "URL", "PDF",  # Tecnologia
}

# Padrão para sigla válida: 2-10 letras maiúsculas, pode ter números no final
SIGLA_PATTERN = r"[A-Z]{2,10}(?:-?[A-Z0-9]{1,4})?"


def extract_aliases(text: str) -> List[str]:
    """
    Extrai siglas e seus termos expandidos de um texto legal.

    Args:
        text: Texto do chunk legal

    Returns:
        Lista de aliases únicos (siglas e termos expandidos)

    Examples:
        >>> extract_aliases("O Estudo Técnico Preliminar - ETP é documento...")
        ['ETP', 'Estudo Técnico Preliminar']

        >>> extract_aliases("A IN (Instrução Normativa) estabelece...")
        ['IN', 'Instrução Normativa']
    """
    if not text or not text.strip():
        return []

    aliases: Set[str] = set()

    # Padrão 1: "Termo Completo - SIGLA" ou "Termo Completo – SIGLA"
    # Ex: "Estudo Técnico Preliminar - ETP"
    pattern1 = rf"([A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Úa-zà-ú]+){{1,6}})\s*[-–—]\s*({SIGLA_PATTERN})\b"
    for match in re.finditer(pattern1, text):
        termo = match.group(1).strip()
        sigla = match.group(2).strip()
        if _is_valid_alias(sigla) and _is_valid_termo(termo):
            aliases.add(sigla)
            aliases.add(termo)

    # Padrão 2: "SIGLA (Termo Completo)" ou "SIGLA – Termo Completo"
    # Ex: "ETP (Estudo Técnico Preliminar)"
    pattern2 = rf"\b({SIGLA_PATTERN})\s*\(\s*([A-ZÀ-Úa-zà-ú][a-zà-ú]+(?:\s+[A-ZÀ-Úa-zà-ú]+){{1,6}})\s*\)"
    for match in re.finditer(pattern2, text):
        sigla = match.group(1).strip()
        termo = match.group(2).strip()
        if _is_valid_alias(sigla) and _is_valid_termo(termo):
            aliases.add(sigla)
            aliases.add(termo)

    # Padrão 3: "Termo Completo (SIGLA)"
    # Ex: "Estudo Técnico Preliminar (ETP)"
    pattern3 = rf"([A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Úa-zà-ú]+){{1,6}})\s*\(\s*({SIGLA_PATTERN})\s*\)"
    for match in re.finditer(pattern3, text):
        termo = match.group(1).strip()
        sigla = match.group(2).strip()
        if _is_valid_alias(sigla) and _is_valid_termo(termo):
            aliases.add(sigla)
            aliases.add(termo)

    # Padrão 4: "SIGLA - Termo Completo" (sigla antes do traço)
    # Ex: "ETP - Estudo Técnico Preliminar"
    pattern4 = rf"\b({SIGLA_PATTERN})\s*[-–—]\s*([A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Úa-zà-ú]+){{1,6}})"
    for match in re.finditer(pattern4, text):
        sigla = match.group(1).strip()
        termo = match.group(2).strip()
        if _is_valid_alias(sigla) and _is_valid_termo(termo):
            aliases.add(sigla)
            aliases.add(termo)

    # Converte para lista e ordena (siglas primeiro, depois termos por tamanho)
    result = sorted(aliases, key=lambda x: (len(x) > 10, x))

    return result


def _is_valid_alias(sigla: str) -> bool:
    """
    Verifica se uma sigla é válida para inclusão nos aliases.

    Args:
        sigla: Candidato a sigla

    Returns:
        True se a sigla for válida
    """
    if not sigla:
        return False

    # Remove da lista de ignorados
    if sigla.upper() in IGNORED_ALIASES:
        return False

    # Deve ter pelo menos 2 caracteres
    if len(sigla) < 2:
        return False

    # Deve ter no máximo 10 caracteres
    if len(sigla) > 10:
        return False

    # Deve ser predominantemente maiúscula
    upper_count = sum(1 for c in sigla if c.isupper())
    if upper_count < len(sigla) * 0.5:
        return False

    return True


def _is_valid_termo(termo: str) -> bool:
    """
    Verifica se um termo expandido é válido.

    Args:
        termo: Candidato a termo expandido

    Returns:
        True se o termo for válido
    """
    if not termo:
        return False

    # Deve ter pelo menos 2 palavras
    words = termo.split()
    if len(words) < 2:
        return False

    # Deve ter no máximo 60 caracteres
    if len(termo) > 60:
        return False

    # Não deve ser apenas números ou pontuação
    if not any(c.isalpha() for c in termo):
        return False

    return True


# ==============================================================================
# TESTES INLINE (para validação rápida)
# ==============================================================================

if __name__ == "__main__":
    # Testes básicos
    test_cases = [
        (
            "O Estudo Técnico Preliminar - ETP é documento constitutivo da primeira etapa.",
            ["ETP", "Estudo Técnico Preliminar"]
        ),
        (
            "A IN (Instrução Normativa) estabelece as regras.",
            []  # IN está na lista de ignorados
        ),
        (
            "O Sistema ETP Digital permite a elaboração de ETPs.",
            []  # Sem padrão de definição
        ),
        (
            "O Termo de Referência (TR) deve conter as especificações.",
            ["TR", "Termo de Referência"]
        ),
        (
            "A Análise de Riscos - AR e o Mapa de Riscos (MR) são obrigatórios.",
            ["AR", "Análise de Riscos", "MR", "Mapa de Riscos"]
        ),
        (
            "SEGES – Secretaria de Gestão estabelece diretrizes.",
            ["SEGES", "Secretaria de Gestão"]
        ),
    ]

    print("=" * 60)
    print("TESTES DE EXTRAÇÃO DE ALIASES")
    print("=" * 60)

    all_passed = True
    for text, expected in test_cases:
        result = extract_aliases(text)
        # Ordena ambos para comparação
        result_set = set(result)
        expected_set = set(expected)

        if result_set == expected_set:
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
            all_passed = False

        print(f"\nTexto: {text[:60]}...")
        print(f"Esperado: {expected}")
        print(f"Obtido:   {result}")
        print(f"Status:   {status}")

    print("\n" + "=" * 60)
    if all_passed:
        print("TODOS OS TESTES PASSARAM!")
    else:
        print("ALGUNS TESTES FALHARAM!")
    print("=" * 60)
