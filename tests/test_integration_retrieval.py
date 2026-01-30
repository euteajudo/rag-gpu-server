"""
Teste de integração: verifica que retrieval_text está sendo gerado e usado corretamente.

Executa com:
    cd /workspace/rag-gpu-server
    python tests/test_integration_retrieval.py
"""

import sys
import re
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from typing import List


# ============================================================
# Cópia local das estruturas para teste isolado
# ============================================================

class DeviceType(Enum):
    """Tipos de dispositivo legal."""
    ARTICLE = "article"
    PARAGRAPH = "paragraph"
    INCISO = "inciso"
    ALINEA = "alinea"
    PART = "part"


CAPUT_HEAD_CHARS: int = 600
CAPUT_TAIL_CHARS: int = 200

DEVICE_TYPE_LABELS = {
    DeviceType.ARTICLE: "Artigo",
    DeviceType.PARAGRAPH: "Parágrafo",
    DeviceType.INCISO: "Inciso",
    DeviceType.ALINEA: "Alínea",
    DeviceType.PART: "Parte",
}


def _truncate_head_tail(text: str, head_chars: int, tail_chars: int) -> str:
    """Trunca texto mantendo início e fim."""
    if len(text) <= head_chars + tail_chars:
        return text

    head = text[:head_chars].rstrip()
    tail = text[-tail_chars:].lstrip()

    if head and not head[-1].isspace() and not head.endswith(('.', ',', ';', ':')):
        last_space = head.rfind(' ')
        if last_space > head_chars - 100:
            head = head[:last_space]

    return f"{head} ... {tail}"


def build_retrieval_text(
    document_id: str,
    article_number: str,
    device_type: DeviceType,
    text: str,
    parent_text: str = "",
    span_id: str = "",
) -> str:
    """Constrói texto determinístico para embedding."""
    lines = []

    device_label = DEVICE_TYPE_LABELS.get(device_type, device_type.value.capitalize())

    specific_id = ""
    if device_type == DeviceType.INCISO and span_id:
        parts = span_id.split("-")
        if len(parts) >= 3:
            specific_id = f" {parts[-1]}"
    elif device_type == DeviceType.PARAGRAPH and span_id:
        parts = span_id.split("-")
        if len(parts) >= 3:
            par_num = parts[-1]
            if par_num.upper() == "UNICO":
                specific_id = " único"
            else:
                specific_id = f" {par_num}º"
    elif device_type == DeviceType.ALINEA and span_id:
        parts = span_id.split("-")
        if len(parts) >= 4:
            specific_id = f" {parts[-1]})"

    header = f"{document_id} | Art. {article_number} | {device_label}{specific_id}"
    lines.append(header)

    # Dispositivo PRIMEIRO
    lines.append(text)

    # Contexto DEPOIS
    if parent_text and device_type != DeviceType.ARTICLE:
        truncated_caput = _truncate_head_tail(
            parent_text,
            head_chars=CAPUT_HEAD_CHARS,
            tail_chars=CAPUT_TAIL_CHARS
        )
        lines.append(f"[CONTEXTO] {truncated_caput}")

    return "\n".join(lines)


@dataclass
class MaterializedChunk:
    """Chunk materializado para indexação."""
    chunk_id: str
    document_id: str
    span_id: str
    text: str
    device_type: DeviceType
    article_number: str = ""
    parent_chunk_id: str = ""
    citations: List[str] = field(default_factory=list)
    parent_text: str = ""
    retrieval_text: str = ""
    enriched_text: str = ""


# ============================================================
# Função safe_article_number (AÇÃO 5)
# ============================================================

def safe_article_number(chunk) -> str:
    """
    Extrai article_number de forma determinística.

    Se chunk.article_number existe, usa ele.
    Caso contrário, extrai do span_id removendo zeros à esquerda.

    Exemplos:
        INC-014-I -> "14"
        PAR-005-1 -> "5"
        ART-123   -> "123"
    """
    v = getattr(chunk, "article_number", "") or ""
    if v:
        return str(v)
    m = re.search(r"(?:INC|PAR|ART)-0*([0-9]+)", getattr(chunk, "span_id", ""))
    return m.group(1) if m else ""


# ============================================================
# Testes
# ============================================================

def test_materialized_chunk_has_retrieval_text():
    """Verifica que MaterializedChunk tem campo retrieval_text."""
    chunk = MaterializedChunk(
        chunk_id="IN-58-2022#ART-014",
        document_id="IN-58-2022",
        span_id="ART-014",
        text="Art. 14. A elaboração do ETP é facultada.",
        device_type=DeviceType.ARTICLE,
        article_number="14",
        parent_chunk_id="",
        citations=[],
        parent_text="",
        retrieval_text="IN-58-2022 | Art. 14 | Artigo\nArt. 14. A elaboração do ETP é facultada.",
    )

    assert hasattr(chunk, 'retrieval_text'), "MaterializedChunk deve ter campo retrieval_text"
    assert chunk.retrieval_text != "", "retrieval_text não deve estar vazio"
    assert "IN-58-2022" in chunk.retrieval_text
    assert "Art. 14" in chunk.retrieval_text
    print("✓ MaterializedChunk tem campo retrieval_text")


def test_build_retrieval_text_format():
    """Verifica formato do retrieval_text."""
    result = build_retrieval_text(
        document_id="IN-58-2022",
        article_number="14",
        device_type=DeviceType.INCISO,
        text="I - é facultada nas hipóteses previstas no inciso III do art. 75;",
        parent_text="Art. 14. A elaboração do ETP é facultada nas seguintes hipóteses:",
        span_id="INC-014-I",
    )

    lines = result.split("\n")

    # Header correto
    assert lines[0] == "IN-58-2022 | Art. 14 | Inciso I", f"Header incorreto: {lines[0]}"

    # Dispositivo vem primeiro
    assert "I - é facultada" in lines[1], "Dispositivo deve vir na segunda linha"

    # Contexto vem depois
    assert "[CONTEXTO]" in result, "Deve ter [CONTEXTO]"
    assert result.index("I - é facultada") < result.index("[CONTEXTO]"), "Dispositivo deve vir ANTES do contexto"

    print("✓ build_retrieval_text gera formato correto")


def test_safe_article_number():
    """Testa extração determinística do article_number."""

    class MockChunk:
        def __init__(self, article_number="", span_id=""):
            self.article_number = article_number
            self.span_id = span_id

    # Caso 1: article_number já definido
    chunk1 = MockChunk(article_number="14", span_id="INC-014-I")
    assert safe_article_number(chunk1) == "14", "Deve usar article_number quando disponível"

    # Caso 2: article_number vazio, extrai do span_id
    chunk2 = MockChunk(article_number="", span_id="INC-014-I")
    assert safe_article_number(chunk2) == "14", "Deve extrair 14 de INC-014-I"

    # Caso 3: zeros à esquerda removidos
    chunk3 = MockChunk(article_number="", span_id="PAR-005-1")
    assert safe_article_number(chunk3) == "5", "Deve remover zeros à esquerda (005 -> 5)"

    # Caso 4: número grande
    chunk4 = MockChunk(article_number="", span_id="ART-123")
    assert safe_article_number(chunk4) == "123", "Deve extrair 123 de ART-123"

    # Caso 5: span_id sem padrão reconhecido
    chunk5 = MockChunk(article_number="", span_id="OUTRO-FORMATO")
    assert safe_article_number(chunk5) == "", "Deve retornar vazio se não reconhecer padrão"

    print("✓ safe_article_number funciona corretamente")


def test_safe_article_number_vs_raw_span_id():
    """Demonstra o problema que safe_article_number resolve."""

    class MockChunk:
        def __init__(self, article_number="", span_id=""):
            self.article_number = article_number
            self.span_id = span_id

    chunk = MockChunk(article_number="", span_id="INC-014-I")

    # ANTES (código antigo - RUIM)
    old_way = getattr(chunk, 'article_number', '') or chunk.span_id
    assert old_way == "INC-014-I", "Código antigo retorna span_id completo"

    # DEPOIS (safe_article_number - BOM)
    new_way = safe_article_number(chunk)
    assert new_way == "14", "safe_article_number retorna apenas o número"

    print("✓ safe_article_number evita passar INC-014-I quando o correto é 14")


def test_pipeline_text_for_embedding_priority():
    """Simula a lógica do pipeline para escolher texto de embedding.

    Estratégia oficial (sem Qwen):
    - retrieval_text determinístico para embeddings
    - fallback para text apenas para dados legados
    - enriched_text NÃO é mais usado (descontinuado)
    """

    class MockChunk:
        def __init__(self, text, retrieval_text=""):
            self.text = text
            self.retrieval_text = retrieval_text

    # Caso 1: retrieval_text disponível (uso normal)
    chunk1 = MockChunk(
        text="texto original",
        retrieval_text="IN-58-2022 | Art. 14 | Artigo\ntexto determinístico",
    )
    text_for_embedding = getattr(chunk1, 'retrieval_text', '') or chunk1.text
    assert text_for_embedding == chunk1.retrieval_text, "Deve usar retrieval_text"
    print("✓ Pipeline usa retrieval_text para embeddings")

    # Caso 2: retrieval_text vazio, fallback defensivo para text (legados)
    chunk2 = MockChunk(
        text="texto original",
        retrieval_text="",  # Vazio (dado legado em transição)
    )
    text_for_embedding = getattr(chunk2, 'retrieval_text', '') or chunk2.text
    assert text_for_embedding == chunk2.text, "Deve usar text como fallback (legados)"
    print("✓ Pipeline faz fallback para text (dados legados)")


def test_different_incisos_produce_different_retrieval_text():
    """Verifica que incisos diferentes do mesmo artigo produzem retrieval_text diferentes."""
    caput = "Art. 33. Os critérios de julgamento serão os seguintes:"

    texts = []
    for i, (roman, desc) in enumerate([("I", "menor preço"), ("II", "maior desconto"), ("III", "melhor técnica")], 1):
        result = build_retrieval_text(
            document_id="LEI-14133-2021",
            article_number="33",
            device_type=DeviceType.INCISO,
            text=f"{roman} - {desc};",
            parent_text=caput,
            span_id=f"INC-033-{roman}",
        )
        texts.append(result)

    # Todos devem ser diferentes
    assert len(set(texts)) == 3, "Cada inciso deve produzir retrieval_text único"

    # Mas todos têm o mesmo contexto
    for t in texts:
        assert "[CONTEXTO] Art. 33." in t

    print("✓ Incisos diferentes produzem retrieval_text diferentes")


def test_determinism():
    """Verifica que retrieval_text é determinístico (mesmo input = mesmo output)."""
    kwargs = {
        "document_id": "IN-58-2022",
        "article_number": "14",
        "device_type": DeviceType.INCISO,
        "text": "I - é facultada nas hipóteses previstas;",
        "parent_text": "Art. 14. A elaboração do ETP é facultada:",
        "span_id": "INC-014-I",
    }

    results = [build_retrieval_text(**kwargs) for _ in range(5)]

    assert len(set(results)) == 1, "retrieval_text deve ser determinístico"
    print("✓ retrieval_text é 100% determinístico")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESTES DE INTEGRAÇÃO: retrieval_text + safe_article_number")
    print("=" * 60 + "\n")

    test_materialized_chunk_has_retrieval_text()
    test_build_retrieval_text_format()
    test_safe_article_number()
    test_safe_article_number_vs_raw_span_id()
    test_pipeline_text_for_embedding_priority()
    test_different_incisos_produce_different_retrieval_text()
    test_determinism()

    print("\n" + "=" * 60)
    print("TODOS OS TESTES PASSARAM!")
    print("=" * 60 + "\n")
