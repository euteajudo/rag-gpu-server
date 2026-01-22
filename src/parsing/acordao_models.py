"""
Modelos de dados para Acordaos do TCU.

Estrutura de um Acordao:
========================

    CABEÇALHO
    - Processo: TC 002.019/2024-8
    - Código eletrônico: AC-2724-47/25-P
    - Grupo/Classe: GRUPO II – CLASSE VII – Plenário
    - Relator: Ministro Benjamin Zymler
    - Interessado: ...
    - Unidade Técnica: SecexAdministração

    SUMÁRIO
    - Texto resumido do acordao

    RELATÓRIO
    - Parágrafos numerados (1, 2, 3...)
    - Subseções opcionais (D, E, F...)

    VOTO
    - Parágrafos numerados (1, 2, 3...)

    ACÓRDÃO
    - Bloco principal
    - Deliberações numeradas (9.1, 9.2, 9.3...)

Formato de IDs:
==============

| Tipo        | Formato              | Exemplo                       |
|-------------|----------------------|-------------------------------|
| acordao_id  | AC-{num}-{ano}-{col} | AC-2724-2025-P                |
| span_id     | {tipo}-{seq}         | SUMARIO, REL-001, ACORDAO-9-1 |
| node_id     | acordaos:{ac}#{span} | acordaos:AC-2724-2025-P#REL-001|
| chunk_id    | {ac}#{span}          | AC-2724-2025-P#REL-001        |

@author: VectorGov
@version: 1.0.0
@since: 22/01/2025
"""

import re
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any


class AcordaoSpanType(Enum):
    """Tipos de spans em um acordao."""
    HEADER = "header"
    SUMARIO = "sumario"
    RELATORIO = "relatorio"
    VOTO = "voto"
    ACORDAO = "acordao"
    DELIBERACAO = "deliberacao"


@dataclass
class AcordaoSpan:
    """
    Um span (trecho) do acordao com ID unico.

    Attributes:
        span_id: ID do span (ex: "REL-001", "ACORDAO-9-1")
        span_type: Tipo do span (sumario, relatorio, voto, etc.)
        text: Texto do span
        identifier: Identificador numerico (1, 2, 9.1, etc.)
        parent_id: ID do span pai (apenas para deliberacoes)
        start_pos: Posicao inicial no texto fonte
        end_pos: Posicao final no texto fonte
        metadata: Metadados adicionais
    """
    span_id: str
    span_type: AcordaoSpanType
    text: str
    identifier: str = ""
    parent_id: Optional[str] = None
    start_pos: int = 0
    end_pos: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AcordaoMetadata:
    """Metadados extraidos do cabecalho do acordao."""
    numero: int = 0
    ano: int = 0
    colegiado: str = ""  # P, 1C, 2C
    processo: str = ""  # TC 002.019/2024-8
    codigo_eletronico: str = ""  # AC-2724-47/25-P
    relator: str = ""
    data_sessao: str = ""  # DD/MM/YYYY
    unidade_tecnica: str = ""
    interessado: str = ""
    assunto: str = ""
    grupo: str = ""  # GRUPO II
    classe: str = ""  # CLASSE VII

    @property
    def acordao_id(self) -> str:
        """Gera o acordao_id canonico."""
        if self.numero and self.ano and self.colegiado:
            return f"AC-{self.numero}-{self.ano}-{self.colegiado}"
        return ""


@dataclass
class ParsedAcordao:
    """
    Acordao parseado com todos os spans indexados.

    Attributes:
        source_text: Texto fonte (markdown do Docling)
        metadata: Metadados do acordao
        spans: Lista de todos os spans
        _span_index: Indice para lookup rapido por span_id
    """
    source_text: str = ""
    metadata: AcordaoMetadata = field(default_factory=AcordaoMetadata)
    spans: List[AcordaoSpan] = field(default_factory=list)
    _span_index: Dict[str, AcordaoSpan] = field(default_factory=dict)

    def add_span(self, span: AcordaoSpan):
        """Adiciona span ao documento."""
        self.spans.append(span)
        self._span_index[span.span_id] = span

    def get_span(self, span_id: str) -> Optional[AcordaoSpan]:
        """Busca span por ID."""
        return self._span_index.get(span_id)

    def get_children(self, parent_id: str) -> List[AcordaoSpan]:
        """Retorna filhos de um span."""
        return [s for s in self.spans if s.parent_id == parent_id]

    @property
    def acordao_id(self) -> str:
        """Retorna o acordao_id canonico."""
        return self.metadata.acordao_id

    @property
    def sumario(self) -> Optional[AcordaoSpan]:
        """Retorna o span do sumario."""
        return self._span_index.get("SUMARIO")

    @property
    def relatorios(self) -> List[AcordaoSpan]:
        """Retorna spans do relatorio."""
        return [s for s in self.spans if s.span_type == AcordaoSpanType.RELATORIO]

    @property
    def votos(self) -> List[AcordaoSpan]:
        """Retorna spans do voto."""
        return [s for s in self.spans if s.span_type == AcordaoSpanType.VOTO]

    @property
    def acordao(self) -> Optional[AcordaoSpan]:
        """Retorna o span principal do acordao."""
        return self._span_index.get("ACORDAO")

    @property
    def deliberacoes(self) -> List[AcordaoSpan]:
        """Retorna deliberacoes (9.1, 9.2, etc.)."""
        return [s for s in self.spans if s.span_type == AcordaoSpanType.DELIBERACAO]

    def to_annotated_markdown(self) -> str:
        """
        Gera markdown anotado com span_ids.

        Formato:
            [SUMARIO] Sumario do acordao...
            [REL-001] 1. Trata-se de...
            [REL-002] 2. A analise...
            [VOTO-001] 1. Concordo com...
            [ACORDAO] ACORDAM os Ministros...
            [ACORDAO-9-1] 9.1. dar ciencia...
        """
        lines = []
        for span in self.spans:
            # Adiciona marcador de span
            first_line = span.text.split('\n')[0][:100]
            lines.append(f"[{span.span_id}] {first_line}...")
        return '\n'.join(lines)

    def generate_chunk_id(self, span: AcordaoSpan) -> str:
        """Gera chunk_id para um span."""
        return f"{self.acordao_id}#{span.span_id}"

    def generate_node_id(self, span: AcordaoSpan) -> str:
        """Gera node_id canonico para um span."""
        return f"acordaos:{self.acordao_id}#{span.span_id}"


def normalize_acordao_id(codigo_eletronico: str) -> str:
    """
    Normaliza codigo eletronico para acordao_id canonico.

    Entrada: "AC-2724-47/25-P"
    Saida: "AC-2724-2025-P"

    Args:
        codigo_eletronico: Codigo eletronico do PDF

    Returns:
        acordao_id normalizado
    """
    # Pattern: AC-{numero}-{sessao}/{ano}-{colegiado}
    # Exemplo: AC-2724-47/25-P
    match = re.match(r'AC[–-](\d+)[–-]\d+/(\d+)[–-]([A-Z0-9]+)', codigo_eletronico)
    if match:
        numero = match.group(1)
        ano_curto = match.group(2)
        colegiado = match.group(3)

        # Converte ano curto para completo
        ano = int(ano_curto)
        if ano < 50:
            ano = 2000 + ano
        else:
            ano = 1900 + ano

        return f"AC-{numero}-{ano}-{colegiado}"

    return ""


def parse_colegiado(colegiado: str) -> str:
    """
    Normaliza colegiado para formato canonico.

    P -> P (Plenario)
    1C ou 1a Camara -> 1C
    2C ou 2a Camara -> 2C
    """
    colegiado = colegiado.upper().strip()
    if colegiado in ('P', 'PLENARIO', 'PLENÁRIO'):
        return 'P'
    if '1' in colegiado or 'PRIMEIRA' in colegiado:
        return '1C'
    if '2' in colegiado or 'SEGUNDA' in colegiado:
        return '2C'
    return colegiado
