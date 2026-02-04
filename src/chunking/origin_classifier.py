"""
OriginClassifier - Classificador de Origem Material de Chunks.

Este modulo resolve o problema de "ilhas de material externo" em documentos legais.
Exemplo: A Lei 14.133/2021 contem artigos do Codigo Penal (Art. 337-E a 337-P)
que foram INSERIDOS pela lei, mas nao SAO da lei.

Problema:
    O PDF da Lei 14.133 contem o Art. 337-E do Codigo Penal. Se indexarmos
    como se fosse da Lei 14.133, o LLM pode responder "Segundo a Lei 14.133,
    Art. 337-E..." quando deveria ser "Segundo o Codigo Penal, Art. 337-E,
    inserido pela Lei 14.133..."

Solucao:
    Manter document_id como esta (cadeia de custodia do PDF), mas adicionar
    metadados de origem material para tratamento diferenciado no retrieval.

Posicao no pipeline:
    PDF -> Docling -> SpanParser -> ChunkMaterializer -> [OriginClassifier] -> Embeddings -> Milvus

Campos adicionados:
    - origin_type: "self" (material da lei) ou "external" (material de outra lei)
    - origin_reference: identificador da lei externa (ex: "DL-2848-1940")
    - origin_reference_name: nome legivel (ex: "Codigo Penal")
    - is_external_material: bool para filtros rapidos
    - origin_confidence: "high", "medium", "low"
    - origin_reason: regra que disparou a classificacao

Uso no retrieval:
    - Filtrar por origin_type="self" para respostas estritas
    - Incluir external com aviso para contexto completo
    - Boost implicito para "self" no reranking

@author: Claude (RunPod)
@date: 2026-02-04
@version: 1.0.0
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


@dataclass
class OriginRule:
    """
    Regra de deteccao de material externo.

    Cada regra define um padrao regex que, quando encontrado no texto de um chunk,
    indica que o chunk contem material de uma lei externa especifica.

    Attributes:
        name: Identificador unico da regra (ex: "codigo_penal_art337")
        pattern: Regex compilado para match no texto
        origin_reference: ID da lei externa (ex: "DL-2848-1940")
        origin_reference_name: Nome legivel (ex: "Codigo Penal")
        confidence: Nivel de confianca ("high", "medium", "low")
        priority: Prioridade (menor = mais prioritario, aplicado primeiro)

    Example:
        >>> rule = OriginRule(
        ...     name="codigo_penal_art337",
        ...     pattern=re.compile(r'Art\\.\\s*337-[A-Z]', re.IGNORECASE),
        ...     origin_reference="DL-2848-1940",
        ...     origin_reference_name="Codigo Penal",
        ...     confidence="high",
        ...     priority=1,
        ... )
    """
    name: str
    pattern: re.Pattern
    origin_reference: str
    origin_reference_name: str
    confidence: str = "high"
    priority: int = 10


class OriginClassifier:
    """
    Classifica chunks por origem material.

    Detecta quando um chunk contem material de outra lei citada/modificada
    pelo documento principal, permitindo tratamento diferenciado no retrieval.

    O classificador aplica uma lista de regras em ordem de prioridade.
    A primeira regra que faz match determina a classificacao do chunk.

    Attributes:
        rules: Lista de OriginRule ordenadas por prioridade

    Example:
        >>> classifier = OriginClassifier()
        >>> chunk = {"text": "Art. 337-E. Admitir, possibilitar..."}
        >>> result = classifier.classify(chunk)
        >>> print(result["origin_type"])  # "external"
        >>> print(result["origin_reference"])  # "DL-2848-1940"
    """

    # Regras de deteccao ordenadas por prioridade
    #
    # PRINCIPIO ULTRA-CONSERVADOR:
    # Apenas marca como "external" quando o chunk CONTEM o texto real de outra lei,
    # NAO quando apenas MENCIONA outra lei.
    #
    # Exemplo:
    #   - "Art. 337-E. Admitir, possibilitar..." -> EXTERNAL (texto DO Codigo Penal)
    #   - "Art. 178. O Codigo Penal passa a vigorar..." -> SELF (texto DA Lei 14.133 SOBRE o CP)
    #
    DEFAULT_RULES: List[OriginRule] = [
        # =====================================================================
        # CODIGO PENAL (DL 2.848/1940) - UNICA REGRA HIGH CONFIDENCE
        # Apenas Art. 337-* que COMECAM o texto sao material externo real
        # =====================================================================
        OriginRule(
            name="codigo_penal_art337",
            # Deve COMECAR com Art. 337-* (nao apenas conter)
            pattern=re.compile(r'^[\s\-\*]*Art\.?\s*337-[A-Z]', re.IGNORECASE | re.MULTILINE),
            origin_reference="DL-2848-1940",
            origin_reference_name="Codigo Penal",
            confidence="high",
            priority=1,
        ),
        # Mencoes ao Decreto-Lei 2.848 sao LOW confidence (apenas referencia)
        OriginRule(
            name="codigo_penal_decreto_lei",
            pattern=re.compile(r'Decreto-Lei\s+n?[°º]?\s*2\.?848', re.IGNORECASE),
            origin_reference="DL-2848-1940",
            origin_reference_name="Codigo Penal",
            confidence="low",  # Rebaixado para LOW
            priority=2,
        ),
        OriginRule(
            name="codigo_penal_mention",
            pattern=re.compile(r'\bC[oó]digo\s+Penal\b', re.IGNORECASE),
            origin_reference="DL-2848-1940",
            origin_reference_name="Codigo Penal",
            confidence="low",  # Rebaixado para LOW
            priority=3,
        ),

        # =====================================================================
        # CODIGO DE PROCESSO CIVIL (Lei 13.105/2015) - LOW confidence
        # Mencoes ao CPC sao referencias, nao material externo
        # =====================================================================
        OriginRule(
            name="cpc_lei_13105",
            pattern=re.compile(r'Lei\s+n?[°º]?\s*13\.?105', re.IGNORECASE),
            origin_reference="LEI-13105-2015",
            origin_reference_name="Codigo de Processo Civil",
            confidence="low",  # Rebaixado para LOW
            priority=5,
        ),
        OriginRule(
            name="cpc_mention",
            pattern=re.compile(r'\bC[oó]digo\s+de\s+Processo\s+Civil\b', re.IGNORECASE),
            origin_reference="LEI-13105-2015",
            origin_reference_name="Codigo de Processo Civil",
            confidence="low",  # Rebaixado para LOW
            priority=6,
        ),

        # =====================================================================
        # LEI DE INTRODUCAO AS NORMAS (DL 4.657/1942 - LINDB) - LOW confidence
        # =====================================================================
        OriginRule(
            name="lindb",
            pattern=re.compile(r'Decreto-Lei\s+n?[°º]?\s*4\.?657|LINDB', re.IGNORECASE),
            origin_reference="DL-4657-1942",
            origin_reference_name="LINDB",
            confidence="low",  # Rebaixado para LOW
            priority=7,
        ),

        # =====================================================================
        # REGRAS DE BAIXA CONFIANCA (mencoes a outras leis)
        #
        # IMPORTANTE: Estas regras detectam MENCOES a outras leis, nao
        # necessariamente material EXTERNO. Por exemplo:
        #   - "A Lei 8.666 fica revogada" -> mencao, mas o texto E da Lei 14.133
        #   - Art. 337-E do Codigo Penal -> material externo REAL
        #
        # Use origin_confidence para distinguir:
        #   - "high": Material externo real (ex: Art. 337-*)
        #   - "low": Apenas mencao (ex: "conforme a Lei 8.666")
        # =====================================================================

        # LEI 8.987/1995 (Concessoes de Servicos Publicos)
        OriginRule(
            name="lei_8987",
            pattern=re.compile(r'Lei\s+n?[°º]?\s*8\.?987', re.IGNORECASE),
            origin_reference="LEI-8987-1995",
            origin_reference_name="Lei de Concessoes",
            confidence="low",  # Mencao, nao material externo
            priority=20,
        ),

        # LEI 8.666/1993 (Licitacoes - revogada pela 14.133)
        OriginRule(
            name="lei_8666",
            pattern=re.compile(r'Lei\s+n?[°º]?\s*8\.?666', re.IGNORECASE),
            origin_reference="LEI-8666-1993",
            origin_reference_name="Lei de Licitacoes (revogada)",
            confidence="low",  # Mencao, nao material externo
            priority=20,
        ),

        # LEI 10.520/2002 (Pregao - revogada pela 14.133)
        OriginRule(
            name="lei_10520",
            pattern=re.compile(r'Lei\s+n?[°º]?\s*10\.?520', re.IGNORECASE),
            origin_reference="LEI-10520-2002",
            origin_reference_name="Lei do Pregao (revogada)",
            confidence="low",
            priority=20,
        ),

        # LEI 12.462/2011 (RDC - Regime Diferenciado de Contratacoes)
        OriginRule(
            name="lei_12462",
            pattern=re.compile(r'Lei\s+n?[°º]?\s*12\.?462', re.IGNORECASE),
            origin_reference="LEI-12462-2011",
            origin_reference_name="Lei do RDC",
            confidence="low",
            priority=20,
        ),

        # LEI 11.079/2004 (PPPs - Parcerias Publico-Privadas)
        OriginRule(
            name="lei_11079",
            pattern=re.compile(r'Lei\s+n?[°º]?\s*11\.?079', re.IGNORECASE),
            origin_reference="LEI-11079-2004",
            origin_reference_name="Lei das PPPs",
            confidence="low",
            priority=20,
        ),

        # LEI 12.846/2013 (Anticorrupcao)
        OriginRule(
            name="lei_12846",
            pattern=re.compile(r'Lei\s+n?[°º]?\s*12\.?846', re.IGNORECASE),
            origin_reference="LEI-12846-2013",
            origin_reference_name="Lei Anticorrupcao",
            confidence="low",
            priority=20,
        ),

        # LEI 13.303/2016 (Estatais)
        OriginRule(
            name="lei_13303",
            pattern=re.compile(r'Lei\s+n?[°º]?\s*13\.?303', re.IGNORECASE),
            origin_reference="LEI-13303-2016",
            origin_reference_name="Lei das Estatais",
            confidence="low",
            priority=20,
        ),

        # LEI 4.320/1964 (Normas de Direito Financeiro)
        OriginRule(
            name="lei_4320",
            pattern=re.compile(r'Lei\s+n?[°º]?\s*4\.?320', re.IGNORECASE),
            origin_reference="LEI-4320-1964",
            origin_reference_name="Lei de Direito Financeiro",
            confidence="low",
            priority=20,
        ),

        # LEI 8.212/1991 (Seguridade Social)
        OriginRule(
            name="lei_8212",
            pattern=re.compile(r'Lei\s+n?[°º]?\s*8\.?212', re.IGNORECASE),
            origin_reference="LEI-8212-1991",
            origin_reference_name="Lei da Seguridade Social",
            confidence="low",
            priority=20,
        ),
    ]

    def __init__(self, rules: Optional[List[OriginRule]] = None):
        """
        Inicializa o classificador.

        Args:
            rules: Lista de regras customizadas. Se None, usa DEFAULT_RULES.
        """
        self.rules = sorted(
            rules or self.DEFAULT_RULES,
            key=lambda r: r.priority
        )
        logger.info(f"OriginClassifier inicializado com {len(self.rules)} regras")

    def classify(self, chunk: dict) -> dict:
        """
        Adiciona metadados de origem ao chunk.

        Aplica as regras em ordem de prioridade. A primeira regra que faz match
        determina a classificacao. Se nenhuma regra faz match, o chunk e
        classificado como "self" (material da propria lei).

        Args:
            chunk: dict com pelo menos 'text' (e opcionalmente 'span_id', etc.)

        Returns:
            chunk atualizado com campos origin_*:
                - origin_type: "self" | "external"
                - origin_reference: str | None
                - origin_reference_name: str | None
                - is_external_material: bool
                - origin_confidence: "high" | "medium" | "low"
                - origin_reason: str | None

        Example:
            >>> classifier = OriginClassifier()
            >>> chunk = {"text": "Art. 337-E. Admitir..."}
            >>> result = classifier.classify(chunk)
            >>> result["origin_type"]
            'external'
        """
        text = chunk.get("text", "")

        # Default: material proprio (self)
        chunk["origin_type"] = "self"
        chunk["origin_reference"] = None
        chunk["origin_reference_name"] = None
        chunk["is_external_material"] = False
        chunk["origin_confidence"] = "high"
        chunk["origin_reason"] = None

        if not text:
            return chunk

        # Aplica regras em ordem de prioridade
        for rule in self.rules:
            if rule.pattern.search(text):
                # LOW confidence = apenas mencao, NAO marca como external
                # Isso evita falsos positivos como "A Lei 8.666 fica revogada"
                # que e texto DA Lei 14.133 falando SOBRE a Lei 8.666
                if rule.confidence == "low":
                    # Registra a deteccao mas mantem como "self"
                    chunk["origin_type"] = "self"
                    chunk["origin_reference"] = rule.origin_reference
                    chunk["origin_reference_name"] = rule.origin_reference_name
                    chunk["is_external_material"] = False
                    chunk["origin_confidence"] = "low"
                    chunk["origin_reason"] = f"mention:{rule.name}"  # Prefixo diferente

                    logger.debug(
                        f"Chunk com MENCAO detectada (mantido como self): "
                        f"span_id={chunk.get('span_id', 'N/A')}, "
                        f"rule={rule.name}"
                    )
                else:
                    # HIGH/MEDIUM confidence = material externo real
                    chunk["origin_type"] = "external"
                    chunk["origin_reference"] = rule.origin_reference
                    chunk["origin_reference_name"] = rule.origin_reference_name
                    chunk["is_external_material"] = True
                    chunk["origin_confidence"] = rule.confidence
                    chunk["origin_reason"] = f"rule:{rule.name}"

                    logger.debug(
                        f"Chunk classificado como external: "
                        f"span_id={chunk.get('span_id', 'N/A')}, "
                        f"rule={rule.name}, ref={rule.origin_reference}"
                    )
                break  # Primeira regra que match (por prioridade)

        return chunk

    def classify_batch(
        self,
        chunks: List[dict]
    ) -> Tuple[List[dict], Dict[str, any]]:
        """
        Classifica batch de chunks.

        Args:
            chunks: Lista de chunks para classificar

        Returns:
            Tuple contendo:
                - chunks_classificados: Lista de chunks com campos origin_*
                - stats: Dicionario com estatisticas:
                    - total: numero total de chunks
                    - self: quantidade de chunks "self"
                    - external: quantidade de chunks "external"
                    - external_refs: dict de origin_reference -> count
                    - rules_triggered: dict de rule_name -> count

        Example:
            >>> classifier = OriginClassifier()
            >>> chunks = [{"text": "Art. 1..."}, {"text": "Art. 337-E..."}]
            >>> results, stats = classifier.classify_batch(chunks)
            >>> stats["self"]
            1
            >>> stats["external"]
            1
        """
        stats = {
            "total": len(chunks),
            "self": 0,
            "external": 0,
            "mentions": 0,  # Chunks self que tem mencao a outra lei
            "external_refs": {},  # ref -> count (apenas external)
            "mention_refs": {},   # ref -> count (apenas mencoes)
            "rules_triggered": {},  # rule_name -> count
        }

        for chunk in chunks:
            self.classify(chunk)

            origin_type = chunk["origin_type"]
            stats[origin_type] += 1

            reason = chunk.get("origin_reason")
            ref = chunk.get("origin_reference")

            if origin_type == "external":
                if ref:
                    stats["external_refs"][ref] = stats["external_refs"].get(ref, 0) + 1
                if reason:
                    stats["rules_triggered"][reason] = stats["rules_triggered"].get(reason, 0) + 1
            elif reason and reason.startswith("mention:"):
                # E "self" mas tem mencao detectada
                stats["mentions"] += 1
                if ref:
                    stats["mention_refs"][ref] = stats["mention_refs"].get(ref, 0) + 1
                if reason:
                    stats["rules_triggered"][reason] = stats["rules_triggered"].get(reason, 0) + 1

        logger.info(
            f"OriginClassifier: {stats['self']} self, {stats['external']} external "
            f"(refs: {list(stats['external_refs'].keys())})"
        )

        return chunks, stats

    def get_rules_summary(self) -> List[dict]:
        """
        Retorna resumo das regras configuradas.

        Returns:
            Lista de dicts com informacoes de cada regra.
        """
        return [
            {
                "name": rule.name,
                "origin_reference": rule.origin_reference,
                "origin_reference_name": rule.origin_reference_name,
                "confidence": rule.confidence,
                "priority": rule.priority,
            }
            for rule in self.rules
        ]

    def classify_materialized_chunk(self, chunk) -> None:
        """
        Classifica um MaterializedChunk (dataclass) in-place.

        Diferente de classify() que trabalha com dicts, este metodo
        trabalha com objetos dataclass que tem atributos.

        Args:
            chunk: MaterializedChunk ou objeto similar com atributos:
                   - text (str): texto do chunk
                   - origin_type, origin_reference, etc. (atributos para setar)
        """
        text = getattr(chunk, 'text', '')

        # Default: material proprio (self)
        chunk.origin_type = "self"
        chunk.origin_reference = None
        chunk.origin_reference_name = None
        chunk.is_external_material = False
        chunk.origin_confidence = "high"
        chunk.origin_reason = None

        if not text:
            return

        # Aplica regras em ordem de prioridade
        for rule in self.rules:
            if rule.pattern.search(text):
                # LOW confidence = apenas mencao, NAO marca como external
                if rule.confidence == "low":
                    chunk.origin_type = "self"
                    chunk.origin_reference = rule.origin_reference
                    chunk.origin_reference_name = rule.origin_reference_name
                    chunk.is_external_material = False
                    chunk.origin_confidence = "low"
                    chunk.origin_reason = f"mention:{rule.name}"

                    logger.debug(
                        f"MaterializedChunk com MENCAO (mantido self): "
                        f"chunk_id={getattr(chunk, 'chunk_id', 'N/A')}, "
                        f"rule={rule.name}"
                    )
                else:
                    # HIGH/MEDIUM confidence = material externo real
                    chunk.origin_type = "external"
                    chunk.origin_reference = rule.origin_reference
                    chunk.origin_reference_name = rule.origin_reference_name
                    chunk.is_external_material = True
                    chunk.origin_confidence = rule.confidence
                    chunk.origin_reason = f"rule:{rule.name}"

                    logger.debug(
                        f"MaterializedChunk classificado como external: "
                        f"chunk_id={getattr(chunk, 'chunk_id', 'N/A')}, "
                        f"rule={rule.name}"
                    )
                break

    def classify_materialized_batch(self, chunks: List) -> Dict[str, any]:
        """
        Classifica batch de MaterializedChunk in-place.

        Args:
            chunks: Lista de MaterializedChunk

        Returns:
            stats: Dicionario com estatisticas
        """
        stats = {
            "total": len(chunks),
            "self": 0,
            "external": 0,
            "mentions": 0,  # Chunks self com mencao a outra lei
            "external_refs": {},
            "mention_refs": {},
            "rules_triggered": {},
        }

        for chunk in chunks:
            self.classify_materialized_chunk(chunk)

            origin_type = getattr(chunk, 'origin_type', 'self')
            stats[origin_type] += 1

            ref = getattr(chunk, 'origin_reference', None)
            reason = getattr(chunk, 'origin_reason', None)

            if origin_type == "external":
                if ref:
                    stats["external_refs"][ref] = stats["external_refs"].get(ref, 0) + 1
                if reason:
                    stats["rules_triggered"][reason] = stats["rules_triggered"].get(reason, 0) + 1
            elif reason and reason.startswith("mention:"):
                stats["mentions"] += 1
                if ref:
                    stats["mention_refs"][ref] = stats["mention_refs"].get(ref, 0) + 1
                if reason:
                    stats["rules_triggered"][reason] = stats["rules_triggered"].get(reason, 0) + 1

        logger.info(
            f"OriginClassifier: {stats['self']} self ({stats['mentions']} com mencoes), "
            f"{stats['external']} external (refs: {list(stats['external_refs'].keys())})"
        )

        return stats


# =============================================================================
# Funcoes utilitarias
# =============================================================================

def classify_chunk_origins(chunks: List[dict]) -> Tuple[List[dict], Dict]:
    """
    Funcao utilitaria para classificar chunks.

    Cria uma instancia de OriginClassifier com regras default e classifica
    todos os chunks do batch.

    Args:
        chunks: Lista de chunks para classificar

    Returns:
        Tuple (chunks_classificados, estatisticas)

    Example:
        >>> from chunking.origin_classifier import classify_chunk_origins
        >>> chunks, stats = classify_chunk_origins(my_chunks)
        >>> print(f"Self: {stats['self']}, External: {stats['external']}")
    """
    classifier = OriginClassifier()
    return classifier.classify_batch(chunks)


def is_external_material(chunk: dict) -> bool:
    """
    Verifica se um chunk contem material externo.

    Args:
        chunk: Chunk ja classificado (com campos origin_*)

    Returns:
        True se origin_type == "external", False caso contrario
    """
    return chunk.get("origin_type") == "external" or chunk.get("is_external_material", False)


def get_origin_warning(chunk: dict) -> Optional[str]:
    """
    Gera mensagem de aviso para chunks de material externo.

    Util para incluir no prompt do LLM ou na resposta ao usuario.

    Args:
        chunk: Chunk classificado

    Returns:
        Mensagem de aviso ou None se for material proprio

    Example:
        >>> warning = get_origin_warning(chunk)
        >>> if warning:
        ...     response += f"\\n\\nNota: {warning}"
    """
    if not is_external_material(chunk):
        return None

    ref_name = chunk.get("origin_reference_name", "outra lei")
    ref_id = chunk.get("origin_reference", "")

    return (
        f"Este trecho esta no documento consultado, mas refere-se ao "
        f"{ref_name} ({ref_id}), nao a lei principal do documento."
    )
