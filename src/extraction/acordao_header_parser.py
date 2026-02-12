"""
AcordaoHeaderParser — Extração de metadados do cabeçalho de acórdãos do TCU.

Extrai: numero, ano, colegiado, processo, natureza, relator,
        unidade_tecnica, data_sessao, sumario, resultado.
"""

import re
import logging

logger = logging.getLogger(__name__)


# === Regexes ===

RE_ACORDAO_NUM = re.compile(
    r'AC[OÓ]RD[AÃ]O\s+(?:N[°ºo.]?\s*)?(\d+)/(\d{4})',
    re.IGNORECASE,
)

RE_COLEGIADO = re.compile(
    r'(Plen[aá]rio|1[ªa]\s*C[aâ]mara|2[ªa]\s*C[aâ]mara|'
    r'Primeira\s+C[aâ]mara|Segunda\s+C[aâ]mara)',
    re.IGNORECASE,
)

RE_PROCESSO = re.compile(r'TC\s+(\d{3}\.\d{3}/\d{4}-\d)')

RE_NATUREZA = re.compile(r'Natureza:\s*(.+?)(?:\n|$)')

RE_RELATOR = re.compile(r'Relator:\s*(?:Ministro\s+)?(.+?)(?:\n|$)')

RE_DATA_SESSAO = re.compile(r'Data\s+da\s+Sess[aã]o:\s*(\d{1,2}/\d{1,2}/\d{4})')

RE_UNIDADE_TECNICA = re.compile(
    r'Unidade\s+T[eé]cnica:\s*(.+?)(?:\n|$)',
    re.IGNORECASE,
)

RE_SUMARIO = re.compile(
    r'SUM[AÁ]RIO:\s*(.+?)(?=\n\s*(?:RELAT[OÓ]RIO|VOTO|AC[OÓ]RD[AÃ]O)\s)',
    re.IGNORECASE | re.DOTALL,
)

RE_RESULTADO = re.compile(
    r'considerar\s+(?:a\s+\w+\s+)?'
    r'(parcialmente\s+procedente|procedente|improcedente)',
    re.IGNORECASE,
)


def _normalize_colegiado(raw: str) -> str:
    """Normaliza nome do colegiado para formato canônico."""
    low = raw.lower().strip()
    if 'plen' in low:
        return "Plenario"
    if '1' in low or 'primeira' in low:
        return "1a_Camara"
    if '2' in low or 'segunda' in low:
        return "2a_Camara"
    return raw.strip()


class AcordaoHeaderParser:
    """Extrai metadados estruturados do cabeçalho do acórdão."""

    def parse_header(self, text: str) -> dict:
        """
        Extrai metadados do cabeçalho do acórdão.

        Args:
            text: Texto integral do acórdão (canonical_text).

        Returns:
            Dict com: numero, ano, colegiado, processo, natureza,
                      relator, unidade_tecnica, data_sessao, sumario, resultado.
        """
        result = {
            "numero": "",
            "ano": "",
            "colegiado": "",
            "processo": "",
            "natureza": "",
            "relator": "",
            "unidade_tecnica": "",
            "data_sessao": "",
            "sumario": "",
            "resultado": "",
        }

        # Numero e ano
        m = RE_ACORDAO_NUM.search(text)
        if m:
            result["numero"] = m.group(1)
            result["ano"] = m.group(2)
        else:
            logger.warning("AcordaoHeaderParser: não encontrou número/ano do acórdão")

        # Colegiado
        m = RE_COLEGIADO.search(text)
        if m:
            result["colegiado"] = _normalize_colegiado(m.group(1))

        # Processo
        m = RE_PROCESSO.search(text)
        if m:
            result["processo"] = f"TC {m.group(1)}"

        # Natureza
        m = RE_NATUREZA.search(text)
        if m:
            result["natureza"] = m.group(1).strip()

        # Relator
        m = RE_RELATOR.search(text)
        if m:
            result["relator"] = m.group(1).strip()

        # Unidade técnica
        m = RE_UNIDADE_TECNICA.search(text)
        if m:
            result["unidade_tecnica"] = m.group(1).strip()

        # Data da sessão
        m = RE_DATA_SESSAO.search(text)
        if m:
            result["data_sessao"] = m.group(1)

        # Sumário
        m = RE_SUMARIO.search(text)
        if m:
            result["sumario"] = " ".join(m.group(1).split())

        # Resultado
        m = RE_RESULTADO.search(text)
        if m:
            result["resultado"] = m.group(1).strip().capitalize()

        logger.info(
            f"AcordaoHeaderParser: {result['numero']}/{result['ano']} "
            f"- {result['colegiado']} - {result['relator']}"
        )
        return result
