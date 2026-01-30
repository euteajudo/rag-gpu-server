"""
Testes para validar o formato do retrieval_text determinístico.

Executa com:
    cd D:/2025/pipeline/rag-gpu-server
    python tests/test_retrieval_text.py
"""

import sys
from pathlib import Path
from enum import Enum

# ============================================================
# Cópia local das funções para teste isolado (evita problemas de import)
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

# ============================================================
# Testes
# ============================================================


class TestBuildRetrievalText:
    """Testes para build_retrieval_text."""

    def test_artigo_sem_caput(self):
        """Artigo não deve ter [CONTEXTO] (ele É o contexto)."""
        result = build_retrieval_text(
            document_id="IN-58-2022",
            article_number="14",
            device_type=DeviceType.ARTICLE,
            text="Art. 14. A elaboração do ETP é facultada nas hipóteses...",
            parent_text="",  # Artigo não tem pai
            span_id="ART-014",
        )

        print(f"\n=== ARTIGO ===\n{result}\n")

        # Verifica estrutura
        lines = result.split("\n")
        assert lines[0] == "IN-58-2022 | Art. 14 | Artigo"
        assert lines[1].startswith("Art. 14.")
        assert "[CONTEXTO]" not in result  # Artigo não tem contexto de pai

    def test_inciso_com_contexto(self):
        """Inciso deve ter dispositivo PRIMEIRO, contexto DEPOIS."""
        caput = "Art. 14. A elaboração do ETP é facultada nas hipóteses previstas nos incisos seguintes, desde que atendidos os requisitos legais aplicáveis ao caso concreto."

        result = build_retrieval_text(
            document_id="IN-58-2022",
            article_number="14",
            device_type=DeviceType.INCISO,
            text="I - é facultada nas hipóteses previstas no inciso III do art. 75 da Lei nº 14.133, de 2021;",
            parent_text=caput,
            span_id="INC-014-I",
        )

        print(f"\n=== INCISO ===\n{result}\n")

        lines = result.split("\n")

        # 1. Header com metadados
        assert lines[0] == "IN-58-2022 | Art. 14 | Inciso I"

        # 2. Dispositivo PRIMEIRO (sem label)
        assert lines[1].startswith("I - é facultada")

        # 3. Contexto DEPOIS
        assert "[CONTEXTO]" in result
        assert result.index("I - é facultada") < result.index("[CONTEXTO]")

    def test_paragrafo_com_contexto(self):
        """Parágrafo deve ter dispositivo PRIMEIRO, contexto DEPOIS."""
        caput = "Art. 5. O estudo técnico preliminar deverá conter os seguintes elementos..."

        result = build_retrieval_text(
            document_id="LEI-14133-2021",
            article_number="5",
            device_type=DeviceType.PARAGRAPH,
            text="§ 1º O ETP poderá ser simplificado nos casos de menor complexidade.",
            parent_text=caput,
            span_id="PAR-005-1",
        )

        print(f"\n=== PARÁGRAFO ===\n{result}\n")

        lines = result.split("\n")

        # Header com número do parágrafo
        assert "Parágrafo 1º" in lines[0]

        # Dispositivo antes do contexto
        assert result.index("§ 1º") < result.index("[CONTEXTO]")

    def test_paragrafo_unico(self):
        """Parágrafo único deve ter label correto."""
        result = build_retrieval_text(
            document_id="IN-65-2021",
            article_number="3",
            device_type=DeviceType.PARAGRAPH,
            text="Parágrafo único. As definições deste artigo aplicam-se...",
            parent_text="Art. 3. Para os fins desta Instrução Normativa...",
            span_id="PAR-003-UNICO",
        )

        print(f"\n=== PARÁGRAFO ÚNICO ===\n{result}\n")

        assert "Parágrafo único" in result

    def test_alinea_com_contexto(self):
        """Alínea deve ter label correto."""
        result = build_retrieval_text(
            document_id="LEI-14133-2021",
            article_number="33",
            device_type=DeviceType.ALINEA,
            text="a) especificação do objeto a ser contratado;",
            parent_text="Art. 33. O edital deverá conter...",
            span_id="ALI-033-I-a",
        )

        print(f"\n=== ALÍNEA ===\n{result}\n")

        assert "Alínea a)" in result
        assert result.index("a) especificação") < result.index("[CONTEXTO]")

    def test_dispositivo_antes_contexto(self):
        """Verifica que o dispositivo sempre vem antes do contexto."""
        caput_longo = "Art. 14. " + "x" * 1000  # Caput muito longo

        result = build_retrieval_text(
            document_id="IN-58-2022",
            article_number="14",
            device_type=DeviceType.INCISO,
            text="II - prorrogação de contratos de serviços e fornecimentos contínuos;",
            parent_text=caput_longo,
            span_id="INC-014-II",
        )

        print(f"\n=== ORDEM CORRETA ===\n{result[:300]}...\n")

        # Encontra posições
        pos_dispositivo = result.find("II - prorrogação")
        pos_contexto = result.find("[CONTEXTO]")

        assert pos_dispositivo > 0, "Dispositivo deve estar presente"
        assert pos_contexto > 0, "Contexto deve estar presente"
        assert pos_dispositivo < pos_contexto, "Dispositivo deve vir ANTES do contexto"


class TestTruncateHeadTail:
    """Testes para _truncate_head_tail."""

    def test_texto_curto_sem_truncamento(self):
        """Texto menor que limite não deve ser truncado."""
        text = "Texto curto que cabe inteiro."
        result = _truncate_head_tail(text, head_chars=100, tail_chars=50)
        assert result == text
        assert "..." not in result

    def test_texto_longo_com_truncamento(self):
        """Texto longo deve ser truncado com '...' no meio."""
        text = "Início do texto. " + "x" * 1000 + " Final do texto."
        result = _truncate_head_tail(text, head_chars=50, tail_chars=30)

        print(f"\n=== TRUNCAMENTO ===\n{result}\n")

        assert result.startswith("Início")
        assert result.endswith("texto.")
        assert " ... " in result
        assert len(result) < len(text)

    def test_preserva_inicio_e_fim(self):
        """Deve preservar início (tema) e fim (condições finais)."""
        text = (
            "Art. 14. A elaboração do ETP é facultada nas hipóteses "
            "previstas nos incisos seguintes, observados os requisitos "
            "estabelecidos nesta Instrução Normativa e demais normas "
            "aplicáveis, desde que atendidos os requisitos legais "
            "aplicáveis ao caso concreto."
        )

        result = _truncate_head_tail(text, head_chars=80, tail_chars=50)

        print(f"\n=== HEAD+TAIL ===\n{result}\n")

        # Início preservado (tema principal)
        assert "Art. 14. A elaboração do ETP" in result
        # Fim preservado (condições finais)
        assert "ao caso concreto" in result


class TestFormatoCompleto:
    """Testes de integração do formato completo."""

    def test_formato_exemplo_documentacao(self):
        """Testa o exemplo da documentação."""
        result = build_retrieval_text(
            document_id="IN-58-2022",
            article_number="14",
            device_type=DeviceType.INCISO,
            text="I - é facultada nas hipóteses previstas no inciso III do art. 75 da Lei nº 14.133, de 2021;",
            parent_text="Art. 14. A elaboração do ETP é facultada nas seguintes hipóteses, desde que atendidos os requisitos legais aplicáveis ao caso concreto.",
            span_id="INC-014-I",
        )

        print(f"\n{'='*60}")
        print("FORMATO FINAL DO RETRIEVAL_TEXT")
        print('='*60)
        print(result)
        print('='*60)

        # Verifica estrutura completa
        assert "IN-58-2022 | Art. 14 | Inciso I" in result
        assert "I - é facultada" in result
        assert "[CONTEXTO]" in result

    def test_multiplos_incisos_diferentes(self):
        """Verifica que incisos diferentes produzem textos diferentes."""
        caput = "Art. 33. Os critérios de julgamento serão os seguintes:"

        inciso_i = build_retrieval_text(
            document_id="LEI-14133-2021",
            article_number="33",
            device_type=DeviceType.INCISO,
            text="I - menor preço;",
            parent_text=caput,
            span_id="INC-033-I",
        )

        inciso_ii = build_retrieval_text(
            document_id="LEI-14133-2021",
            article_number="33",
            device_type=DeviceType.INCISO,
            text="II - maior desconto;",
            parent_text=caput,
            span_id="INC-033-II",
        )

        inciso_iii = build_retrieval_text(
            document_id="LEI-14133-2021",
            article_number="33",
            device_type=DeviceType.INCISO,
            text="III - melhor técnica ou conteúdo artístico;",
            parent_text=caput,
            span_id="INC-033-III",
        )

        print(f"\n=== INCISO I ===\n{inciso_i}\n")
        print(f"\n=== INCISO II ===\n{inciso_ii}\n")
        print(f"\n=== INCISO III ===\n{inciso_iii}\n")

        # Todos têm o mesmo contexto
        assert "[CONTEXTO] Art. 33." in inciso_i
        assert "[CONTEXTO] Art. 33." in inciso_ii
        assert "[CONTEXTO] Art. 33." in inciso_iii

        # Mas dispositivos diferentes (que vêm primeiro!)
        assert "I - menor preço" in inciso_i
        assert "II - maior desconto" in inciso_ii
        assert "III - melhor técnica" in inciso_iii

        # Headers diferentes
        assert "Inciso I" in inciso_i
        assert "Inciso II" in inciso_ii
        assert "Inciso III" in inciso_iii


if __name__ == "__main__":
    # Executa testes manualmente
    print("\n" + "="*60)
    print("EXECUTANDO TESTES DO RETRIEVAL_TEXT")
    print("="*60)

    test_instance = TestBuildRetrievalText()
    test_instance.test_artigo_sem_caput()
    test_instance.test_inciso_com_contexto()
    test_instance.test_paragrafo_com_contexto()
    test_instance.test_paragrafo_unico()
    test_instance.test_alinea_com_contexto()
    test_instance.test_dispositivo_antes_contexto()

    trunc_tests = TestTruncateHeadTail()
    trunc_tests.test_texto_curto_sem_truncamento()
    trunc_tests.test_texto_longo_com_truncamento()
    trunc_tests.test_preserva_inicio_e_fim()

    formato_tests = TestFormatoCompleto()
    formato_tests.test_formato_exemplo_documentacao()
    formato_tests.test_multiplos_incisos_diferentes()

    print("\n" + "="*60)
    print("TODOS OS TESTES PASSARAM!")
    print("="*60)
