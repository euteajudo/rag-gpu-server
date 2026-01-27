"""
Extrator de Citações Normativas para documentos legais brasileiros.

Este módulo extrai referências a normas/documentos de texto legal:
- Leis, Decretos, Instruções Normativas, Portarias, Resoluções, Acórdãos
- Referências internas (art. 9º, inciso III, alínea a)

Formato de saída (JSON string para Milvus VarChar):
[
    {
        "raw": "art. 9º da IN 58/2022",
        "type": "IN",
        "doc_id": "IN-58-2022",
        "span_ref": "ART-009",
        "target_node_id": "leis:IN-58-2022#ART-009"
    },
    {
        "raw": "Lei nº 14.133/2021",
        "type": "LEI",
        "doc_id": "LEI-14133-2021",
        "span_ref": null,
        "target_node_id": "leis:LEI-14133-2021"
    }
]

@author: Equipe VectorGov
@since: 22/01/2025
"""

import re
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum

logger = logging.getLogger(__name__)


# Catálogo de documentos conhecidos (pode ser expandido ou carregado de arquivo)
KNOWN_DOCUMENTS = {
    # Leis principais
    "lei de licitações": "LEI-14133-2021",
    "lei de licitações e contratos": "LEI-14133-2021",
    "nova lei de licitações": "LEI-14133-2021",
    "lei 14133": "LEI-14133-2021",
    "lei 8666": "LEI-8666-1993",
    "lei das estatais": "LEI-13303-2016",
    "lei 13303": "LEI-13303-2016",
    "lei de responsabilidade fiscal": "LC-101-2000",
    "lrf": "LC-101-2000",
    "estatuto das micro e pequenas empresas": "LC-123-2006",
    "lei complementar 123": "LC-123-2006",
    # Decretos
    "decreto do pregão eletrônico": "DECRETO-10024-2019",
    "decreto 10024": "DECRETO-10024-2019",
    "decreto 10947": "DECRETO-10947-2022",
    # INs
    "in 65": "IN-65-2021",
    "in 58": "IN-58-2022",
    "in 5": "IN-5-2017",
    # Constituição
    "constituição federal": "CF-1988",
    "cf": "CF-1988",
    "carta magna": "CF-1988",
}

# Tabela canônica de normas: (tipo, número) -> ano correto
# Usado para validar/corrigir anos extraídos incorretamente
CANONICAL_NORMS = {
    # Leis Federais
    ("LEI", "8666"): 1993,       # Lei de Licitações (antiga)
    ("LEI", "10520"): 2002,      # Lei do Pregão
    ("LEI", "12462"): 2011,      # RDC
    ("LEI", "13303"): 2016,      # Lei das Estatais
    ("LEI", "14133"): 2021,      # Nova Lei de Licitações
    ("LEI", "8429"): 1992,       # Lei de Improbidade
    ("LEI", "9784"): 1999,       # Processo Administrativo Federal
    ("LEI", "12527"): 2011,      # Lei de Acesso à Informação
    ("LEI", "13709"): 2018,      # LGPD
    ("LEI", "4320"): 1964,       # Normas Gerais de Direito Financeiro
    ("LEI", "8112"): 1990,       # Estatuto do Servidor Público Federal
    ("LEI", "10406"): 2002,      # Código Civil
    ("LEI", "5172"): 1966,       # CTN - Código Tributário Nacional
    ("LEI", "6404"): 1976,       # Lei das S.A.
    ("LEI", "9472"): 1997,       # Lei Geral de Telecomunicações
    ("LEI", "9478"): 1997,       # Lei do Petróleo
    ("LEI", "11079"): 2004,      # Lei das PPPs
    ("LEI", "11107"): 2005,      # Lei dos Consórcios Públicos
    ("LEI", "8987"): 1995,       # Lei de Concessões
    ("LEI", "13019"): 2014,      # Marco Regulatório das OSCs

    # Leis Complementares
    ("LC", "101"): 2000,         # LRF
    ("LC", "123"): 2006,         # Estatuto ME/EPP
    ("LC", "116"): 2003,         # ISS
    ("LC", "87"): 1996,          # Lei Kandir (ICMS)

    # Decretos
    ("DECRETO", "10024"): 2019,  # Pregão Eletrônico
    ("DECRETO", "10947"): 2022,  # Regulamenta Lei 14.133
    ("DECRETO", "7892"): 2013,   # SRP
    ("DECRETO", "9507"): 2018,   # Terceirização
    ("DECRETO", "8538"): 2015,   # ME/EPP
    ("DECRETO", "6170"): 2007,   # Convênios
    ("DECRETO", "93872"): 1986,  # Unificação de recursos de caixa

    # Instruções Normativas SEGES/ME
    ("IN", "5"): 2017,           # Terceirização
    ("IN", "40"): 2020,          # Elaboração de ETP
    ("IN", "58"): 2022,          # ETP e TR
    ("IN", "65"): 2021,          # Pesquisa de Preços
    ("IN", "73"): 2020,          # Contratação de TIC
    ("IN", "81"): 2022,          # Contratações de TIC
    ("IN", "98"): 2022,          # Dispensa eletrônica

    # Portarias
    ("PORTARIA", "938"): 2022,   # Catálogo de Soluções de TIC
    ("PORTARIA", "8678"): 2021,  # Modelo de Contratação de TIC
}

# Anos mínimos e máximos válidos por tipo de norma
NORM_YEAR_BOUNDS = {
    "LEI": (1824, 2030),           # Desde o Império
    "LC": (1967, 2030),            # LC só existe desde 1967
    "DECRETO": (1889, 2030),       # Desde a República
    "DL": (1937, 1988),            # Decretos-Lei só até 1988
    "IN": (1990, 2030),            # INs modernas
    "PORTARIA": (1950, 2030),
    "RESOLUCAO": (1950, 2030),
    "ACORDAO": (1990, 2030),       # Acórdãos modernos
    "MP": (1988, 2030),            # MPs só desde 1988
    "EC": (1992, 2030),            # ECs à CF/88
}

# Padrões que indicam contexto normativo (acionam LLM fallback)
NORMATIVE_CONTEXT_PATTERNS = [
    r"\bconforme\b",
    r"\bnos termos\b",
    r"\bde acordo com\b",
    r"\bvedado\b",
    r"\bdispensa\b",
    r"\binexigibilidade\b",
    r"\bprevisto\b",
    r"\bestablecido\b",
    r"\bregulament",
    r"\bdispõe\b",
]


class NormativeType(str, Enum):
    """Tipos de normas reconhecidas."""
    LEI = "LEI"
    LEI_COMPLEMENTAR = "LC"
    DECRETO = "DECRETO"
    DECRETO_LEI = "DL"
    INSTRUCAO_NORMATIVA = "IN"
    PORTARIA = "PORTARIA"
    RESOLUCAO = "RESOLUCAO"
    ACORDAO = "ACORDAO"
    MEDIDA_PROVISORIA = "MP"
    EMENDA_CONSTITUCIONAL = "EC"
    CONSTITUICAO = "CF"
    INTERNO = "INTERNO"  # Referência interna (mesmo documento)


@dataclass
class NormativeReference:
    """Uma referência normativa extraída do texto."""

    raw: str                          # Texto original encontrado
    type: str                         # Tipo da norma (LEI, IN, DECRETO...)
    doc_id: Optional[str] = None      # ID normalizado (LEI-14133-2021)
    span_ref: Optional[str] = None    # Referência ao dispositivo (ART-009)
    target_node_id: Optional[str] = None  # node_id alvo se existir
    method: str = "regex"             # "regex" ou "llm"
    confidence: float = 1.0           # 0.0 a 1.0 - confiança na extração
    is_ambiguous: bool = False        # True se a referência é ambígua/incompleta

    def to_dict(self) -> dict:
        """Converte para dicionário serializável."""
        return {
            "raw": self.raw,
            "type": self.type,
            "doc_id": self.doc_id,
            "span_ref": self.span_ref,
            "target_node_id": self.target_node_id,
            "method": self.method,
            "confidence": self.confidence,
        }


class CitationExtractor:
    """
    Extrai citações normativas de texto legal.

    Uso:
        extractor = CitationExtractor(current_document_id="IN-58-2022")
        citations = extractor.extract("conforme art. 9º da Lei 14.133/2021")
        json_str = extractor.to_json(citations)
    """

    # Padrões para identificar tipos de norma
    NORM_PATTERNS = {
        NormativeType.LEI_COMPLEMENTAR: [
            r"Lei\s+Complementar\s+(?:n[ºo°]?\s*)?(\d+[\d\.]*)",
            r"LC\s+(?:n[ºo°]?\s*)?(\d+[\d\.]*)",
        ],
        NormativeType.LEI: [
            r"Lei\s+(?:Federal\s+)?(?:n[ºo°]?\s*)?(\d+[\d\.]*)",
            r"Lei\s+(\d+[\d\.]*)(?:/(\d{2,4}))?",
        ],
        NormativeType.DECRETO_LEI: [
            r"Decreto[-\s]Lei\s+(?:n[ºo°]?\s*)?(\d+[\d\.]*)",
            r"DL\s+(?:n[ºo°]?\s*)?(\d+[\d\.]*)",
        ],
        NormativeType.DECRETO: [
            r"Decreto\s+(?:Federal\s+)?(?:n[ºo°]?\s*)?(\d+[\d\.]*)",
        ],
        NormativeType.INSTRUCAO_NORMATIVA: [
            r"Instru[çc][aã]o\s+Normativa\s+(?:[\w\-/]+\s+)?(?:n[ºo°]?\s*)?(\d+)",
            r"IN\s+(?:[\w\-/]+\s+)?(?:n[ºo°]?\s*)?(\d+)",
        ],
        NormativeType.PORTARIA: [
            r"Portaria\s+(?:[\w\-/]+\s+)?(?:n[ºo°]?\s*)?(\d+)",
        ],
        NormativeType.RESOLUCAO: [
            r"Resolu[çc][aã]o\s+(?:[\w\-/]+\s+)?(?:n[ºo°]?\s*)?(\d+)",
        ],
        NormativeType.ACORDAO: [
            r"Ac[oó]rd[aã]o\s+(?:n[ºo°]?\s*)?(\d+)",
            r"Ac[oó]rd[aã]o\s+(\d+)(?:/(\d{4}))?(?:\s*[-–]\s*(\w+))?",
        ],
        NormativeType.MEDIDA_PROVISORIA: [
            r"Medida\s+Provis[oó]ria\s+(?:n[ºo°]?\s*)?(\d+[\d\.]*)",
            r"MP\s+(?:n[ºo°]?\s*)?(\d+[\d\.]*)",
        ],
        NormativeType.EMENDA_CONSTITUCIONAL: [
            r"Emenda\s+Constitucional\s+(?:n[ºo°]?\s*)?(\d+)",
            r"EC\s+(?:n[ºo°]?\s*)?(\d+)",
        ],
        NormativeType.CONSTITUICAO: [
            r"Constitui[çc][aã]o\s+(?:Federal|da\s+Rep[úu]blica)?",
            r"\bCF(?:/\d{2,4})?\b",
        ],
    }

    # Padrões para dispositivos (artigo, parágrafo, inciso, alínea)
    DEVICE_PATTERNS = {
        "artigo": r"(?:art\.?|artigo)\s*(\d+)[ºo°]?",
        "paragrafo": r"(?:§|par[aá]grafo)\s*(\d+|[úu]nico)[ºo°]?",
        "inciso": r"inciso\s+([IVXLCDM]+)",
        "alinea": r"al[ií]nea\s+['\"]?([a-z])['\"]?",
    }

    # Padrão para capturar ano
    YEAR_PATTERN = r"[/\s](\d{2,4})"

    def __init__(
        self,
        current_document_id: Optional[str] = None,
        known_documents: Optional[dict[str, str]] = None,
        llm_resolver: Optional[Callable[[str, list[str]], Optional[str]]] = None,
        enable_llm_fallback: bool = False,
    ):
        """
        Args:
            current_document_id: ID do documento atual (para referências internas)
            known_documents: Dicionário de nomes conhecidos -> doc_id
            llm_resolver: Função callback para resolver via LLM
                          Assinatura: (text: str, candidates: list[str]) -> Optional[str]
            enable_llm_fallback: Se True, usa LLM para resolver referências ambíguas
        """
        self.current_document_id = current_document_id
        self.known_documents = {**KNOWN_DOCUMENTS, **(known_documents or {})}
        self.llm_resolver = llm_resolver
        self.enable_llm_fallback = enable_llm_fallback
        self._compile_patterns()

        # Métricas de telemetria
        self.stats = {
            "regex_extractions": 0,
            "llm_fallback_calls": 0,
            "llm_resolved": 0,
            "ambiguous_refs": 0,
        }

    def _compile_patterns(self):
        """Compila padrões regex para melhor performance."""
        self._compiled_norms = {}
        for norm_type, patterns in self.NORM_PATTERNS.items():
            self._compiled_norms[norm_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

        self._compiled_devices = {
            k: re.compile(v, re.IGNORECASE)
            for k, v in self.DEVICE_PATTERNS.items()
        }

    def extract(self, text: str) -> list[NormativeReference]:
        """
        Extrai todas as referências normativas do texto.

        Args:
            text: Texto legal para análise

        Returns:
            Lista de NormativeReference encontradas
        """
        if not text:
            return []

        references = []
        seen_raw = set()  # Evita duplicatas por texto
        seen_doc_ids = set()  # Evita duplicatas por doc_id + span_ref

        # 1. Extrai referências a normas externas
        for norm_type, patterns in self._compiled_norms.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    raw = match.group(0)
                    if raw.lower() in seen_raw:
                        continue
                    seen_raw.add(raw.lower())

                    ref = self._parse_normative_match(match, norm_type, text)
                    if ref:
                        # Evita duplicatas pelo doc_id + span_ref
                        ref_key = f"{ref.doc_id}#{ref.span_ref or ''}"
                        if ref_key in seen_doc_ids:
                            continue
                        seen_doc_ids.add(ref_key)
                        references.append(ref)

        # 2. Extrai referências internas (artigos/parágrafos/incisos)
        internal_refs = self._extract_internal_references(text, seen_raw)
        references.extend(internal_refs)

        return references

    def _parse_normative_match(
        self,
        match: re.Match,
        norm_type: NormativeType,
        full_text: str
    ) -> Optional[NormativeReference]:
        """Parseia um match de norma para NormativeReference."""
        raw = match.group(0)

        # Extrai número da norma
        number = match.group(1) if match.lastindex and match.lastindex >= 1 else None
        if number:
            number = number.replace(".", "")

        # Tenta extrair ano do contexto
        year = None
        # Busca ano logo após o match
        remaining_text = full_text[match.end():match.end() + 20]
        year_match = re.search(self.YEAR_PATTERN, remaining_text)
        if year_match:
            year = year_match.group(1)
            if len(year) == 2:
                year = f"20{year}" if int(year) < 50 else f"19{year}"
            raw = full_text[match.start():match.end() + year_match.end()]

        # Também verifica se ano veio no próprio match
        if match.lastindex and match.lastindex >= 2:
            captured_year = match.group(2)
            if captured_year and captured_year.isdigit():
                year = captured_year
                if len(year) == 2:
                    year = f"20{year}" if int(year) < 50 else f"19{year}"

        # Monta doc_id normalizado
        doc_id = self._build_doc_id(norm_type, number, year)

        # Tenta extrair dispositivo referenciado
        # Primeiro busca ANTES do match (ex: "art. 9º da Lei 14.133")
        span_ref = self._extract_device_reference_before(full_text, match.start())
        # Se não encontrou antes, busca APÓS
        if not span_ref:
            span_ref = self._extract_device_reference(full_text, match.end())

        # Monta target_node_id
        target_node_id = None
        if doc_id:
            if span_ref:
                target_node_id = f"leis:{doc_id}#{span_ref}"
            else:
                target_node_id = f"leis:{doc_id}"

        # Calcula confiança e detecta ambiguidade
        confidence, is_ambiguous = self._calculate_confidence(
            number=number,
            year=year,
            doc_id=doc_id,
            norm_type=norm_type
        )

        return NormativeReference(
            raw=raw.strip(),
            type=norm_type.value,
            doc_id=doc_id,
            span_ref=span_ref,
            target_node_id=target_node_id,
            method="regex",
            confidence=confidence,
            is_ambiguous=is_ambiguous,
        )

    def _calculate_confidence(
        self,
        number: Optional[str],
        year: Optional[str],
        doc_id: Optional[str],
        norm_type: NormativeType
    ) -> tuple[float, bool]:
        """
        Calcula confiança da extração e detecta ambiguidade.

        Returns:
            tuple: (confidence: float, is_ambiguous: bool)
        """
        confidence = 1.0
        is_ambiguous = False

        # Sem número = muito baixa confiança
        if not number:
            confidence = 0.3
            is_ambiguous = True

        # Sem ano = média confiança (pode ser ambíguo)
        elif not year:
            confidence = 0.6
            is_ambiguous = True

        # doc_id não resolvido
        elif not doc_id:
            confidence = 0.5
            is_ambiguous = True

        # Constituição é especial (não precisa de número/ano)
        elif norm_type == NormativeType.CONSTITUICAO:
            confidence = 0.95
            is_ambiguous = False

        # Tudo OK
        else:
            confidence = 0.95

        return confidence, is_ambiguous

    def _extract_internal_references(
        self,
        text: str,
        seen_raw: set
    ) -> list[NormativeReference]:
        """Extrai referências internas (art. 9º, inciso III)."""
        references = []

        # Padrão para artigo com contexto opcional de documento
        art_pattern = re.compile(
            r"(?:art\.?|artigo)\s*(\d+)[ºo°]?"
            r"(?:\s*,?\s*(?:§|par[aá]grafo)\s*(\d+|[úu]nico)[ºo°]?)?"
            r"(?:\s*,?\s*inciso\s+([IVXLCDM]+))?"
            r"(?:\s*,?\s*al[ií]nea\s+['\"]?([a-z])['\"]?)?",
            re.IGNORECASE
        )

        for match in art_pattern.finditer(text):
            raw = match.group(0)
            if raw.lower() in seen_raw or len(raw) < 4:
                continue
            seen_raw.add(raw.lower())

            art_num = match.group(1)
            par_num = match.group(2)
            inc_num = match.group(3)
            ali_num = match.group(4)

            # Verifica se é referência ao próprio documento ou a outro
            # Busca contexto após o match
            after_match = text[match.end():match.end() + 100].lower()
            is_external = any(
                kw in after_match[:50]
                for kw in ["da lei", "do decreto", "da in", "da portaria", "desta"]
            )

            if is_external and "desta" not in after_match[:30]:
                # É referência externa, já será capturada pelos padrões de norma
                continue

            # Monta span_ref
            span_ref = self._build_span_ref(art_num, par_num, inc_num, ali_num)

            # Se temos documento atual, monta target_node_id
            target_node_id = None
            doc_id = None
            if self.current_document_id:
                doc_id = self.current_document_id
                target_node_id = f"leis:{doc_id}#{span_ref}"

            # Referências internas têm alta confiança se temos document_id
            confidence = 0.9 if self.current_document_id else 0.5
            is_ambiguous = not self.current_document_id

            references.append(NormativeReference(
                raw=raw.strip(),
                type=NormativeType.INTERNO.value,
                doc_id=doc_id,
                span_ref=span_ref,
                target_node_id=target_node_id,
                method="regex",
                confidence=confidence,
                is_ambiguous=is_ambiguous,
            ))

        return references

    def _build_doc_id(
        self,
        norm_type: NormativeType,
        number: Optional[str],
        year: Optional[str]
    ) -> Optional[str]:
        """
        Constrói doc_id normalizado com validação de ano.

        Usa CANONICAL_NORMS para:
        1. Fornecer ano quando ausente
        2. Corrigir ano quando claramente errado
        """
        if not number:
            return None

        type_prefix = norm_type.value
        number_clean = number.replace(".", "").lstrip("0") or number

        # Busca ano canônico na tabela
        canonical_key = (type_prefix, number_clean)
        canonical_year = CANONICAL_NORMS.get(canonical_key)

        # Valida ano extraído
        validated_year = self._validate_year(
            type_prefix=type_prefix,
            number=number_clean,
            extracted_year=year,
            canonical_year=canonical_year
        )

        parts = [type_prefix, number_clean]
        if validated_year:
            parts.append(str(validated_year))

        return "-".join(parts)

    def _validate_year(
        self,
        type_prefix: str,
        number: str,
        extracted_year: Optional[str],
        canonical_year: Optional[int]
    ) -> Optional[int]:
        """
        Valida e corrige o ano extraído.

        Regras:
        1. Se temos ano canônico, usa ele (mais confiável)
        2. Se ano extraído está fora dos limites válidos, descarta
        3. Se ano extraído é claramente errado (ex: Lei 8.666/2021), corrige
        """
        # Se temos ano canônico, é sempre preferido
        if canonical_year:
            if extracted_year:
                try:
                    ext_year = int(extracted_year)
                    # Log se há discrepância significativa
                    if abs(ext_year - canonical_year) > 2:
                        logger.debug(
                            f"Ano corrigido: {type_prefix}-{number}/{extracted_year} "
                            f"-> {canonical_year}"
                        )
                except ValueError:
                    pass
            return canonical_year

        # Se não temos canônico mas temos extraído, valida
        if extracted_year:
            try:
                year_int = int(extracted_year)

                # Verifica limites por tipo de norma
                bounds = NORM_YEAR_BOUNDS.get(type_prefix, (1900, 2030))
                min_year, max_year = bounds

                if year_int < min_year or year_int > max_year:
                    logger.warning(
                        f"Ano fora dos limites: {type_prefix}-{number}/{extracted_year} "
                        f"(válido: {min_year}-{max_year})"
                    )
                    return None

                return year_int

            except ValueError:
                return None

        return None

    def _build_span_ref(
        self,
        art_num: str,
        par_num: Optional[str] = None,
        inc_num: Optional[str] = None,
        ali_num: Optional[str] = None
    ) -> str:
        """Constrói span_ref no formato padrão."""
        art_padded = art_num.zfill(3)

        if ali_num:
            return f"ALI-{art_padded}-{inc_num}-{ali_num}"
        elif inc_num:
            return f"INC-{art_padded}-{inc_num}"
        elif par_num:
            par_str = "UNICO" if par_num.lower() in ("único", "unico") else par_num
            return f"PAR-{art_padded}-{par_str}"
        else:
            return f"ART-{art_padded}"

    def _extract_device_reference_before(
        self,
        text: str,
        end_pos: int
    ) -> Optional[str]:
        """Extrai referência de dispositivo ANTES da menção a norma."""
        # Busca até 100 caracteres antes da menção da norma
        start = max(0, end_pos - 100)
        search_text = text[start:end_pos]

        # Padrão para "art. X, inciso Y, alínea Z da/do" - busca artigo próximo do final
        # Ex: "art. 9º da Lei", "art. 75, inciso II, alínea 'a', da Lei"
        art_pattern = re.compile(
            r"(?:art\.?|artigo)\s*(\d+)[ºo°]?"
            r"(?:\s*,?\s*(?:§|par[aá]grafo)\s*(\d+|[úu]nico)[ºo°]?)?"
            r"(?:\s*,?\s*inciso\s+([IVXLCDM]+))?"
            r"(?:\s*,?\s*al[ií]nea\s+['\"]?([a-z])['\"]?)?"
            r"\s*(?:,\s*)?(?:d[aoe]s?|n[aoe]s?)\s*$",
            re.IGNORECASE
        )

        match = art_pattern.search(search_text)
        if not match:
            return None

        art_num = match.group(1)
        par_num = match.group(2)
        inc_num = match.group(3)
        ali_num = match.group(4)

        return self._build_span_ref(art_num, par_num, inc_num, ali_num)

    def _extract_device_reference(
        self,
        text: str,
        start_pos: int
    ) -> Optional[str]:
        """Extrai referência de dispositivo após menção a norma."""
        # Busca no texto após a menção da norma
        search_text = text[start_pos:start_pos + 100]

        # Procura por "art. X", "artigo X"
        art_match = self._compiled_devices["artigo"].search(search_text)
        if not art_match:
            return None

        art_num = art_match.group(1)

        # Continua buscando parágrafo/inciso/alínea
        remaining = search_text[art_match.end():]

        par_num = None
        inc_num = None
        ali_num = None

        par_match = self._compiled_devices["paragrafo"].search(remaining[:50])
        if par_match:
            par_num = par_match.group(1)
            remaining = remaining[par_match.end():]

        inc_match = self._compiled_devices["inciso"].search(remaining[:50])
        if inc_match:
            inc_num = inc_match.group(1)
            remaining = remaining[inc_match.end():]

        ali_match = self._compiled_devices["alinea"].search(remaining[:30])
        if ali_match:
            ali_num = ali_match.group(1)

        return self._build_span_ref(art_num, par_num, inc_num, ali_num)

    def to_json(self, references: list[NormativeReference]) -> str:
        """Converte lista de referências para JSON string."""
        return json.dumps(
            [ref.to_dict() for ref in references],
            ensure_ascii=False,
            indent=None  # Compacto para VarChar
        )

    def extract_and_serialize(self, text: str) -> str:
        """Extrai referências e retorna JSON string."""
        refs = self.extract(text)
        return self.to_json(refs)



def normalize_citations(
    citations: list[str | dict] | None,
    chunk_node_id: str,
    parent_chunk_id: str | None = None,
    document_type: str | None = None,
    device_type: str | None = None,
) -> list[str]:
    """
    Normaliza citações removendo self-loops, parent-loops e duplicatas.

    Args:
        citations: Lista de citações (strings ou dicts com target_node_id)
        chunk_node_id: Node ID do chunk atual (ex: "leis:LEI-14.133-2021#ART-006-P1")
        parent_chunk_id: ID do chunk pai sem prefixo (ex: "LEI-14.133-2021#ART-006")
        document_type: Tipo do documento (LEI, DECRETO, IN, ACORDAO, etc.)
        device_type: Tipo do dispositivo (article, paragraph, inciso, alinea)

    Returns:
        Lista de target_node_ids normalizados (sem self-loops, sem parent-loops, sem duplicatas)

    Regras aplicadas:
    1. Remove valores vazios e None
    2. Extrai target_node_id de dicts
    3. Remove self-loops (citation == chunk_node_id)
    4. Remove parent-loops (citation == parent_node_id)
    5. Remove duplicatas preservando ordem
    """
    if not citations:
        return []

    # Mapeamento de document_type para prefixo
    PREFIX_MAP = {
        "LEI": "leis",
        "DECRETO": "leis",
        "IN": "leis",
        "LC": "leis",
        "ACORDAO": "acordaos",
        "SUMULA": "sumulas",
    }

    # Calcula parent_node_id se parent_chunk_id foi fornecido
    parent_node_id = None
    if parent_chunk_id:
        # Determina o prefixo
        prefix = None
        if document_type:
            prefix = PREFIX_MAP.get(document_type.upper(), "leis")
        else:
            # Tenta inferir do chunk_node_id
            if chunk_node_id and ":" in chunk_node_id:
                prefix = chunk_node_id.split(":")[0]

        if prefix:
            parent_node_id = f"{prefix}:{parent_chunk_id}"

    seen = set()
    normalized = []

    for citation in citations:
        # Extrai target_node_id
        if isinstance(citation, dict):
            target = citation.get("target_node_id")
        else:
            target = citation

        # Pula valores vazios
        if not target or (isinstance(target, str) and not target.strip()):
            continue

        target = target.strip()

        # Pula self-loop
        if target == chunk_node_id:
            continue

        # Pula parent-loop
        if parent_node_id and target == parent_node_id:
            continue

        # Pula duplicatas
        if target in seen:
            continue

        seen.add(target)
        normalized.append(target)

    return normalized

def extract_citations_from_chunk(
    text: str,
    document_id: Optional[str] = None,
    known_documents: Optional[dict[str, str]] = None,
    chunk_node_id: Optional[str] = None,
    parent_chunk_id: Optional[str] = None,
    document_type: Optional[str] = None,
) -> list[str]:
    """
    Função utilitária para extrair citações de um chunk.

    Args:
        text: Texto do chunk
        document_id: ID do documento atual
        known_documents: Dicionário de nomes conhecidos -> doc_id
        chunk_node_id: Node ID do chunk (para remover self-loops)
        parent_chunk_id: ID do chunk pai (para remover parent-loops)
        document_type: Tipo do documento (LEI, DECRETO, etc.)

    Returns:
        Lista de target_node_ids encontrados (sem self-loops e parent-loops)
    """
    extractor = CitationExtractor(
        current_document_id=document_id,
        known_documents=known_documents,
    )

    refs = extractor.extract(text)

    # Retorna lista de target_node_ids (sem None)
    citations = [ref.target_node_id for ref in refs if ref.target_node_id]
    if chunk_node_id:
        citations = normalize_citations(
            citations=citations,
            chunk_node_id=chunk_node_id,
            parent_chunk_id=parent_chunk_id,
            document_type=document_type,
        )
    return citations
