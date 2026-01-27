"""
Normalizacao de identificadores de documentos legais.

Este modulo fornece funcoes para normalizar document_ids para um formato
canonico, garantindo consistencia entre:
- CitationExtractor (extraindo referencias do texto)
- Pipeline de Ingestao (criando document_id dos chunks)
- Neo4j (edges :CITA)
- Milvus (document_id nos chunks)

Problema original:
- CitationExtractor gerava: leis:LEI-14133-2021#ART-018 (sem ponto)
- Milvus tinha: LEI-14.133-2021 (com ponto)
- Resultado: citacoes nao resolviam

Solucao:
- Funcao unica de normalizacao usada em todos os pontos
- Formato canonico: LEI-14.133-2021 (com ponto de milhar para numeros >= 1000)
"""

import re
from typing import Optional


def normalize_document_id(raw_id: str) -> str:
    """
    Normaliza document_id para formato canonico.

    Regras:
    1. Uppercase: "lei" -> "LEI"
    2. Separador hifen: "LEI 14133/2021" -> "LEI-14133-2021"
    3. Ponto de milhar para numeros >= 1000: "14133" -> "14.133"
    4. Remove "no", espacos extras, caracteres especiais

    Args:
        raw_id: Document ID em qualquer formato

    Returns:
        Document ID normalizado

    Exemplos:
        >>> normalize_document_id("LEI 14133/2021")
        'LEI-14.133-2021'
        >>> normalize_document_id("lei-14.133-2021")
        'LEI-14.133-2021'
        >>> normalize_document_id("LEI-14133-2021")
        'LEI-14.133-2021'
        >>> normalize_document_id("Lei no 14.133")
        'LEI-14.133'
        >>> normalize_document_id("IN-58-2022")
        'IN-58-2022'
        >>> normalize_document_id("DECRETO-10947-2022")
        'DECRETO-10.947-2022'
    """
    if not raw_id:
        return ""

    # 1. Uppercase e remove espacos extras
    normalized = raw_id.upper().strip()

    # 2. Remove "No", "N.", "No.", etc.
    normalized = re.sub(r'\bN[oOº°]?\.?\s*', '', normalized, flags=re.IGNORECASE)

    # 3. Substitui separadores por hifen (espaco, barra, underline)
    normalized = re.sub(r'[\s/_]+', '-', normalized)

    # 4. Remove hifens duplicados
    normalized = re.sub(r'-+', '-', normalized)

    # 5. Adiciona ponto de milhar APENAS no numero do documento (nao no ano)
    # Formato esperado: TIPO-NUMERO-ANO
    # Exemplo: LEI-14133-2021 -> LEI-14.133-2021
    parts = normalized.split('-')
    if len(parts) >= 2:
        new_parts = []
        for i, part in enumerate(parts):
            # Se e o ultimo elemento e tem 4 digitos, e provavelmente o ano - nao mexe
            is_year = (i == len(parts) - 1) and part.isdigit() and len(part) == 4
            # Se ja tem ponto, nao mexe
            has_dot = '.' in part

            if part.isdigit() and not is_year and not has_dot:
                num = int(part)
                if num >= 1000:
                    # Formata com ponto de milhar brasileiro
                    part = f"{num:,}".replace(",", ".")
            new_parts.append(part)
        normalized = '-'.join(new_parts)

    # 6. Remove hifen no inicio/fim
    normalized = normalized.strip('-')

    return normalized


def normalize_node_id(raw_node_id: str) -> str:
    """
    Normaliza um node_id completo (com prefixo e fragmento).

    Formato: "leis:DOCUMENT_ID#SPAN_ID"

    Args:
        raw_node_id: Node ID em qualquer formato

    Returns:
        Node ID normalizado

    Exemplos:
        >>> normalize_node_id("leis:LEI-14133-2021#ART-018")
        'leis:LEI-14.133-2021#ART-018'
        >>> normalize_node_id("leis:lei-14.133-2021#art-018")
        'leis:LEI-14.133-2021#ART-018'
    """
    if not raw_node_id:
        return ""

    # Divide em prefixo, document_id e span_id
    parts = raw_node_id.split(":", 1)
    if len(parts) != 2:
        return raw_node_id

    prefix = parts[0].lower()  # leis, acordaos, etc.
    rest = parts[1]

    # Divide document_id e span_id
    doc_parts = rest.split("#", 1)
    document_id = normalize_document_id(doc_parts[0])

    if len(doc_parts) == 2:
        span_id = doc_parts[1].upper()  # ART-018, PAR-005-1, etc.
        return f"{prefix}:{document_id}#{span_id}"
    else:
        return f"{prefix}:{document_id}"


def extract_document_id_from_text(text: str) -> Optional[str]:
    """
    Extrai e normaliza document_id de texto livre.

    Util para extrair referencias como "Lei 14.133/2021" ou "IN 65/2021".

    Args:
        text: Texto contendo referencia a documento

    Returns:
        Document ID normalizado ou None

    Exemplos:
        >>> extract_document_id_from_text("conforme a Lei 14.133/2021")
        'LEI-14.133-2021'
        >>> extract_document_id_from_text("IN no 65 de 2021")
        'IN-65-2021'
    """
    if not text:
        return None

    # Padroes comuns de referencia a documentos
    patterns = [
        # Lei 14.133/2021, Lei no 14133/2021
        r'\b(LEI)\s*[nNoOº°\.]*\s*(\d+[\.\d]*)[/-]?(\d{4})?\b',
        # IN 65/2021, IN-65-2021
        r'\b(IN)\s*[nNoOº°\.]*\s*(\d+)[/-]?(\d{4})?\b',
        # Decreto 10.947/2022
        r'\b(DECRETO)\s*[nNoOº°\.]*\s*(\d+[\.\d]*)[/-]?(\d{4})?\b',
        # Portaria 123/2021
        r'\b(PORTARIA)\s*[nNoOº°\.]*\s*(\d+)[/-]?(\d{4})?\b',
    ]

    text_upper = text.upper()

    for pattern in patterns:
        match = re.search(pattern, text_upper, re.IGNORECASE)
        if match:
            tipo = match.group(1)
            numero = match.group(2).replace(".", "")  # Remove ponto existente
            ano = match.group(3) if match.group(3) else ""

            if ano:
                raw_id = f"{tipo}-{numero}-{ano}"
            else:
                raw_id = f"{tipo}-{numero}"

            return normalize_document_id(raw_id)

    return None


# =============================================================================
# TESTES
# =============================================================================

if __name__ == "__main__":
    print("=== TESTES DE NORMALIZACAO ===\n")

    # Teste 1: Normalizacao basica
    print("Teste 1: Normalizacao basica")
    tests = [
        ("LEI 14133/2021", "LEI-14.133-2021"),
        ("lei-14.133-2021", "LEI-14.133-2021"),
        ("LEI-14133-2021", "LEI-14.133-2021"),
        ("Lei no 14.133", "LEI-14.133"),
        ("LEI No 14133/2021", "LEI-14.133-2021"),
    ]
    for raw, expected in tests:
        result = normalize_document_id(raw)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] '{raw}' -> '{result}' (esperado: '{expected}')")

    # Teste 2: Numeros menores que 1000 (sem ponto)
    print("\nTeste 2: Numeros < 1000 (sem ponto)")
    tests = [
        ("IN-58-2022", "IN-58-2022"),
        ("IN-65-2021", "IN-65-2021"),
        ("PORTARIA-100-2021", "PORTARIA-100-2021"),
    ]
    for raw, expected in tests:
        result = normalize_document_id(raw)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] '{raw}' -> '{result}' (esperado: '{expected}')")

    # Teste 3: Numeros maiores que 1000 (com ponto)
    print("\nTeste 3: Numeros >= 1000 (com ponto)")
    tests = [
        ("DECRETO-10947-2022", "DECRETO-10.947-2022"),
        ("LEI-8666-1993", "LEI-8.666-1993"),
        ("LEI-14133-2021", "LEI-14.133-2021"),
    ]
    for raw, expected in tests:
        result = normalize_document_id(raw)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] '{raw}' -> '{result}' (esperado: '{expected}')")

    # Teste 4: Preservar ponto existente
    print("\nTeste 4: Preservar ponto existente")
    tests = [
        ("LEI-14.133-2021", "LEI-14.133-2021"),
        ("DECRETO-10.947-2022", "DECRETO-10.947-2022"),
    ]
    for raw, expected in tests:
        result = normalize_document_id(raw)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] '{raw}' -> '{result}' (esperado: '{expected}')")

    # Teste 5: normalize_node_id
    print("\nTeste 5: normalize_node_id")
    tests = [
        ("leis:LEI-14133-2021#ART-018", "leis:LEI-14.133-2021#ART-018"),
        ("leis:lei-14.133-2021#art-018", "leis:LEI-14.133-2021#ART-018"),
        ("leis:IN-58-2022#PAR-005-1", "leis:IN-58-2022#PAR-005-1"),
    ]
    for raw, expected in tests:
        result = normalize_node_id(raw)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] '{raw}' -> '{result}' (esperado: '{expected}')")

    print("\n=== TESTES CONCLUIDOS ===")
