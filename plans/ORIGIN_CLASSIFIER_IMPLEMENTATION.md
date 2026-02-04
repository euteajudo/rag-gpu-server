# Plano de Implementação: OriginClassifier

**Data**: 2026-02-04
**Autor**: Claude (RunPod)
**Status**: PLANEJAMENTO
**Prioridade**: Alta

---

## 1. Contexto do Problema

### 1.1 Descrição

O PDF da Lei 14.133/2021 contém "ilhas" de material externo - artigos de outras leis que são **citados** ou **modificados** pela Lei 14.133. Exemplo:

- **Art. 178 da Lei 14.133**: "O Código Penal passa a vigorar acrescido do seguinte artigo:"
- **Art. 337-E do Código Penal**: "Admitir, possibilitar ou dar causa à contratação direta..."

O Art. 337-E **está no PDF** da Lei 14.133, mas **não é** da Lei 14.133 - é do Código Penal.

### 1.2 Risco Semântico

Se indexarmos o Art. 337-E como se fosse ontologia da Lei 14.133:
- O LLM pode responder "Segundo a Lei 14.133, Art. 337-E..." (ERRADO)
- Deveria ser "Segundo o Código Penal, Art. 337-E, inserido pela Lei 14.133..."

### 1.3 Solução Aprovada

Manter `document_id` e `node_id` como estão (cadeia de custódia), mas adicionar metadados de **origem material** em cada chunk.

---

## 2. Arquitetura da Solução

### 2.1 Posição no Pipeline

```
PDF → Docling → SpanParser → ChunkMaterializer → [OriginClassifier] → Embeddings → Milvus
                                                        ↑
                                                   NOVO MÓDULO
```

### 2.2 Fluxo de Dados

```
ChunkMaterializer.materialize()
        │
        ▼
   List[dict] chunks (sem origin_*)
        │
        ▼
OriginClassifier.classify_batch()
        │
        ▼
   List[dict] chunks (com origin_*)
        │
        ▼
   Embeddings + Milvus Insert
```

### 2.3 Campos de Metadados

| Campo | Tipo | Valores | Descrição |
|-------|------|---------|-----------|
| `origin_type` | VARCHAR(16) | "self" \| "external" | Origem material do chunk |
| `origin_reference` | VARCHAR(128) | "DL-2848-1940" \| null | Identificador da lei externa |
| `is_external_material` | BOOL | true \| false | Flag para filtros rápidos |
| `origin_confidence` | VARCHAR(8) | "high" \| "medium" \| "low" | Confiança da classificação |
| `origin_reason` | VARCHAR(256) | "rule:codigo_penal_art337" | Regra que disparou |

---

## 3. Lista de Tarefas (To-Do)

### 3.1 RunPod (GPU Server)

#### Fase 1: Criar Módulo OriginClassifier
- [ ] **T1.1** Criar arquivo `src/chunking/origin_classifier.py`
- [ ] **T1.2** Implementar classe `OriginRule` (dataclass)
- [ ] **T1.3** Implementar classe `OriginClassifier`
  - [ ] Método `classify(chunk: dict) -> dict`
  - [ ] Método `classify_batch(chunks: List[dict]) -> Tuple[List[dict], dict]`
- [ ] **T1.4** Definir regras iniciais:
  - [ ] Código Penal (Art. 337-*, DL 2.848)
  - [ ] CPC (Lei 13.105)
  - [ ] Lei 8.987 (concessões)
  - [ ] Lei 8.666 (licitações antiga)
  - [ ] Outras leis frequentemente citadas

#### Fase 2: Integrar no Pipeline
- [ ] **T2.1** Modificar `IngestionPipeline._process_document()` para chamar OriginClassifier
- [ ] **T2.2** Adicionar estatísticas de origem no retorno de `/ingest/result`
- [ ] **T2.3** Logar classificações no log de ingestão

#### Fase 3: Atualizar Schema Milvus
- [ ] **T3.1** Adicionar campos no `MilvusChunkPayload` (ou modelo equivalente)
- [ ] **T3.2** Atualizar script de criação de collection (se existir)
- [ ] **T3.3** Documentar campos novos

#### Fase 4: Testes
- [ ] **T4.1** Criar `tests/test_origin_classifier.py`
  - [ ] Teste com Art. 337-E (deve ser external)
  - [ ] Teste com Art. 1 da Lei 14.133 (deve ser self)
  - [ ] Teste com menção a "Código Penal" no texto
  - [ ] Teste com Lei 8.666 citada
  - [ ] Teste de batch classification
- [ ] **T4.2** Teste de integração com Lei 14.133 completa
- [ ] **T4.3** Validar que chunks external têm metadados corretos

#### Fase 5: Documentação
- [ ] **T5.1** Docstrings completas no módulo
- [ ] **T5.2** Atualizar README ou docs com novo campo
- [ ] **T5.3** Comentário no código explicando a decisão arquitetural

### 3.2 VPS (API Server) - Ver seção 5

---

## 4. Implementação Detalhada

### 4.1 Arquivo: `src/chunking/origin_classifier.py`

```python
"""
OriginClassifier - Classificador de Origem Material de Chunks.

Este módulo resolve o problema de "ilhas de material externo" em documentos legais.
Exemplo: A Lei 14.133/2021 contém artigos do Código Penal (Art. 337-E a 337-P)
que foram INSERIDOS pela lei, mas não SÃO da lei.

Posição no pipeline: após ChunkMaterializer, antes de embeddings.

Campos adicionados:
- origin_type: "self" (material da lei) ou "external" (material de outra lei)
- origin_reference: identificador da lei externa (ex: "DL-2848-1940")
- is_external_material: bool para filtros rápidos
- origin_confidence: "high", "medium", "low"
- origin_reason: regra que disparou a classificação

Uso no retrieval:
- Filtrar por origin_type="self" para respostas estritas
- Incluir external com aviso para contexto completo

@author: Claude (RunPod)
@date: 2026-02-04
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


@dataclass
class OriginRule:
    """
    Regra de detecção de material externo.

    Attributes:
        name: Identificador único da regra
        pattern: Regex compilado para match no texto
        origin_reference: ID da lei externa (ex: "DL-2848-1940")
        origin_reference_name: Nome legível (ex: "Código Penal")
        confidence: Nível de confiança ("high", "medium", "low")
        priority: Prioridade (menor = mais prioritário)
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

    Detecta quando um chunk contém material de outra lei citada/modificada
    pelo documento principal, permitindo tratamento diferenciado no retrieval.
    """

    # Regras de detecção ordenadas por prioridade
    DEFAULT_RULES: List[OriginRule] = [
        # === CÓDIGO PENAL (alta prioridade) ===
        OriginRule(
            name="codigo_penal_art337",
            pattern=re.compile(r'Art\.?\s*337-[A-Z]', re.IGNORECASE),
            origin_reference="DL-2848-1940",
            origin_reference_name="Código Penal",
            confidence="high",
            priority=1,
        ),
        OriginRule(
            name="codigo_penal_mention_dl",
            pattern=re.compile(r'Decreto-Lei\s+n?[°º]?\s*2\.?848', re.IGNORECASE),
            origin_reference="DL-2848-1940",
            origin_reference_name="Código Penal",
            confidence="high",
            priority=2,
        ),
        OriginRule(
            name="codigo_penal_mention_name",
            pattern=re.compile(r'\bCódigo\s+Penal\b', re.IGNORECASE),
            origin_reference="DL-2848-1940",
            origin_reference_name="Código Penal",
            confidence="medium",
            priority=3,
        ),

        # === CPC (Código de Processo Civil) ===
        OriginRule(
            name="cpc_lei",
            pattern=re.compile(r'Lei\s+n?[°º]?\s*13\.?105', re.IGNORECASE),
            origin_reference="LEI-13105-2015",
            origin_reference_name="Código de Processo Civil",
            confidence="high",
            priority=5,
        ),
        OriginRule(
            name="cpc_mention",
            pattern=re.compile(r'\bCódigo\s+de\s+Processo\s+Civil\b', re.IGNORECASE),
            origin_reference="LEI-13105-2015",
            origin_reference_name="Código de Processo Civil",
            confidence="medium",
            priority=6,
        ),

        # === LEI 8.987 (Concessões) ===
        OriginRule(
            name="lei_8987",
            pattern=re.compile(r'Lei\s+n?[°º]?\s*8\.?987', re.IGNORECASE),
            origin_reference="LEI-8987-1995",
            origin_reference_name="Lei de Concessões",
            confidence="medium",
            priority=10,
        ),

        # === LEI 8.666 (Licitações antiga) ===
        OriginRule(
            name="lei_8666",
            pattern=re.compile(r'Lei\s+n?[°º]?\s*8\.?666', re.IGNORECASE),
            origin_reference="LEI-8666-1993",
            origin_reference_name="Lei de Licitações (revogada)",
            confidence="medium",
            priority=10,
        ),

        # === LEI 10.520 (Pregão) ===
        OriginRule(
            name="lei_10520",
            pattern=re.compile(r'Lei\s+n?[°º]?\s*10\.?520', re.IGNORECASE),
            origin_reference="LEI-10520-2002",
            origin_reference_name="Lei do Pregão",
            confidence="medium",
            priority=10,
        ),

        # === LEI 12.462 (RDC) ===
        OriginRule(
            name="lei_12462",
            pattern=re.compile(r'Lei\s+n?[°º]?\s*12\.?462', re.IGNORECASE),
            origin_reference="LEI-12462-2011",
            origin_reference_name="Lei do RDC",
            confidence="medium",
            priority=10,
        ),

        # === LEI 11.079 (PPPs) ===
        OriginRule(
            name="lei_11079",
            pattern=re.compile(r'Lei\s+n?[°º]?\s*11\.?079', re.IGNORECASE),
            origin_reference="LEI-11079-2004",
            origin_reference_name="Lei das PPPs",
            confidence="medium",
            priority=10,
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

        Args:
            chunk: dict com pelo menos 'text' (e opcionalmente 'span_id', etc.)

        Returns:
            chunk atualizado com campos origin_*
        """
        text = chunk.get("text", "")

        # Default: material próprio (self)
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
            Tuple (chunks_classificados, estatísticas)
        """
        stats = {
            "total": len(chunks),
            "self": 0,
            "external": 0,
            "external_refs": {},  # ref -> count
            "rules_triggered": {},  # rule_name -> count
        }

        for chunk in chunks:
            self.classify(chunk)

            origin_type = chunk["origin_type"]
            stats[origin_type] += 1

            if origin_type == "external":
                ref = chunk["origin_reference"]
                rule = chunk["origin_reason"]

                stats["external_refs"][ref] = stats["external_refs"].get(ref, 0) + 1
                stats["rules_triggered"][rule] = stats["rules_triggered"].get(rule, 0) + 1

        logger.info(
            f"OriginClassifier: {stats['self']} self, {stats['external']} external "
            f"(refs: {list(stats['external_refs'].keys())})"
        )

        return chunks, stats


# Função utilitária para uso rápido
def classify_chunk_origins(chunks: List[dict]) -> Tuple[List[dict], Dict]:
    """
    Função utilitária para classificar chunks.

    Uso:
        from chunking.origin_classifier import classify_chunk_origins
        chunks, stats = classify_chunk_origins(chunks)
    """
    classifier = OriginClassifier()
    return classifier.classify_batch(chunks)
```

### 4.2 Integração no Pipeline

Modificar `src/ingestion/pipeline.py`:

```python
# Após materialização, antes de embeddings

from src.chunking.origin_classifier import OriginClassifier

# Na função _process_document() ou similar:

# ... código existente de materialização ...
chunks = materializer.materialize(parsed_doc, ...)

# NOVO: Classificar origens
origin_classifier = OriginClassifier()
chunks, origin_stats = origin_classifier.classify_batch(chunks)

# Adicionar stats ao resultado
result["origin_stats"] = origin_stats

# ... código existente de embeddings ...
```

### 4.3 Retorno do Endpoint `/ingest/result`

```json
{
  "success": true,
  "document_id": "LEI-14133-2021",
  "total_chunks": 206,
  "origin_stats": {
    "total": 206,
    "self": 187,
    "external": 19,
    "external_refs": {
      "DL-2848-1940": 12,
      "LEI-8987-1995": 4,
      "LEI-13105-2015": 3
    },
    "rules_triggered": {
      "rule:codigo_penal_art337": 12,
      "rule:lei_8987": 4,
      "rule:cpc_lei": 3
    }
  }
}
```

---

## 5. Instruções para Claude da VPS

### 5.1 Texto para Enviar

```
═══════════════════════════════════════════════════════════════════════════════
INSTRUÇÕES PARA IMPLEMENTAÇÃO: OriginClassifier (VPS)
═══════════════════════════════════════════════════════════════════════════════

Contexto:
---------
Implementamos no RunPod (GPU Server) o módulo OriginClassifier que adiciona
metadados de "origem material" em cada chunk. Isso resolve o problema de
"ilhas de material externo" - ex: Art. 337-E do Código Penal que está no
PDF da Lei 14.133 mas não É da Lei 14.133.

Novos campos no payload de chunks:
----------------------------------
| Campo                  | Tipo         | Valores                    |
|------------------------|--------------|----------------------------|
| origin_type            | VARCHAR(16)  | "self" | "external"         |
| origin_reference       | VARCHAR(128) | "DL-2848-1940" | null       |
| origin_reference_name  | VARCHAR(128) | "Código Penal" | null       |
| is_external_material   | BOOL         | true | false                |
| origin_confidence      | VARCHAR(8)   | "high"|"medium"|"low"       |
| origin_reason          | VARCHAR(256) | "rule:codigo_penal_art337" |

Tarefas para VPS:
-----------------

1. MILVUS SCHEMA
   - Adicionar os 6 campos acima na collection `leis_v4` (ou criar v5)
   - Criar índice escalar em `is_external_material` se for filtrar frequentemente
   - Campos VARCHAR são suficientes, não precisa de índice full-text

2. RETRIEVAL/SEARCH
   - Opção A (recomendada): Adicionar parâmetro `include_external: bool = True`
     - Se False: filter += "is_external_material == false"
     - Se True: retorna tudo, mas marca os externos no resultado

   - Opção B: Boost implícito
     - origin_type="self" tem score multiplicado por 1.0
     - origin_type="external" tem score multiplicado por 0.8

3. RESPOSTA DO RAG
   - Quando um chunk external for usado na resposta, incluir nota:
     "Este trecho está no documento da Lei 14.133, mas refere-se ao
      {origin_reference_name} ({origin_reference})."

   - Ou no header da citação:
     "[Fonte: LEI-14133-2021, material externo do Código Penal]"

4. API DE INGESTÃO (receber do RunPod)
   - Verificar se o payload de chunks do RunPod agora inclui os campos origin_*
   - Mapear para os campos do Milvus
   - Logar estatísticas de origem no log de ingestão

5. ENDPOINT DE STATS (opcional)
   - Adicionar em /stats ou /collection/info:
     - Contagem de chunks por origin_type
     - Lista de origin_references únicos

Exemplo de chunk que o RunPod vai enviar:
-----------------------------------------
{
  "chunk_id": "LEI-14133-2021#ART-178#INC-337E",
  "text": "Art. 337-E. Admitir, possibilitar ou dar causa...",
  "document_id": "LEI-14133-2021",
  "span_id": "ART-178",
  "origin_type": "external",
  "origin_reference": "DL-2848-1940",
  "origin_reference_name": "Código Penal",
  "is_external_material": true,
  "origin_confidence": "high",
  "origin_reason": "rule:codigo_penal_art337",
  "dense_vector": [0.123, ...],
  ...
}

Prioridade:
-----------
1. Schema Milvus (bloqueante para testes)
2. Receber/mapear campos do RunPod
3. Filtro no retrieval
4. Nota no RAG (pode ser depois)

Dúvidas? Avisa que a gente sincroniza.
═══════════════════════════════════════════════════════════════════════════════
```

---

## 6. Testes Planejados

### 6.1 Testes Unitários (`tests/test_origin_classifier.py`)

```python
import pytest
from src.chunking.origin_classifier import OriginClassifier, OriginRule

class TestOriginClassifier:

    @pytest.fixture
    def classifier(self):
        return OriginClassifier()

    def test_self_origin_regular_article(self, classifier):
        """Artigo normal da lei deve ser 'self'."""
        chunk = {"text": "Art. 1º Esta Lei estabelece normas gerais de licitação."}
        result = classifier.classify(chunk)
        assert result["origin_type"] == "self"
        assert result["is_external_material"] == False

    def test_external_codigo_penal_art337(self, classifier):
        """Art. 337-E deve ser detectado como Código Penal."""
        chunk = {"text": "Art. 337-E. Admitir, possibilitar ou dar causa à contratação direta..."}
        result = classifier.classify(chunk)
        assert result["origin_type"] == "external"
        assert result["origin_reference"] == "DL-2848-1940"
        assert result["is_external_material"] == True
        assert "codigo_penal" in result["origin_reason"]

    def test_external_cpc_mention(self, classifier):
        """Menção ao CPC deve ser detectada."""
        chunk = {"text": "Conforme o Código de Processo Civil, Art. 1.048..."}
        result = classifier.classify(chunk)
        assert result["origin_type"] == "external"
        assert result["origin_reference"] == "LEI-13105-2015"

    def test_external_lei_8666(self, classifier):
        """Menção à Lei 8.666 deve ser detectada."""
        chunk = {"text": "A Lei nº 8.666, de 21 de junho de 1993, fica revogada."}
        result = classifier.classify(chunk)
        assert result["origin_type"] == "external"
        assert result["origin_reference"] == "LEI-8666-1993"

    def test_batch_classification(self, classifier):
        """Teste de classificação em batch."""
        chunks = [
            {"text": "Art. 1º Esta Lei estabelece normas."},
            {"text": "Art. 337-E. Admitir, possibilitar..."},
            {"text": "Art. 2º Aplicam-se as disposições."},
        ]
        results, stats = classifier.classify_batch(chunks)

        assert stats["total"] == 3
        assert stats["self"] == 2
        assert stats["external"] == 1
        assert "DL-2848-1940" in stats["external_refs"]

    def test_priority_order(self, classifier):
        """Regra de maior prioridade deve vencer."""
        # Art. 337-E menciona tanto padrão específico quanto "Código Penal"
        chunk = {"text": "Art. 337-E do Código Penal..."}
        result = classifier.classify(chunk)
        # Deve usar a regra mais específica (art337) não a genérica (mention)
        assert "art337" in result["origin_reason"]

    def test_empty_text(self, classifier):
        """Texto vazio deve retornar 'self'."""
        chunk = {"text": ""}
        result = classifier.classify(chunk)
        assert result["origin_type"] == "self"
```

### 6.2 Teste de Integração

```python
def test_lei_14133_full_classification():
    """Teste com Lei 14.133 completa."""
    # Carrega e processa a lei
    # Verifica que ~187 são self e ~19 são external
    # Verifica que todos os Art. 337-* são external
```

---

## 7. Riscos e Mitigações

| Risco | Probabilidade | Impacto | Mitigação |
|-------|---------------|---------|-----------|
| Falso positivo (self → external) | Baixa | Médio | Regras conservadoras, confidence level |
| Falso negativo (external → self) | Média | Médio | Expandir regras conforme descoberto |
| Schema Milvus incompatível | Baixa | Alto | Campos opcionais, default null |
| Performance em batch grande | Baixa | Baixo | Regex é O(n), ~1ms/chunk |

---

## 8. Cronograma Estimado

| Fase | Descrição | Tempo |
|------|-----------|-------|
| 1 | Criar módulo OriginClassifier | 1h |
| 2 | Integrar no pipeline | 30min |
| 3 | Testes unitários | 1h |
| 4 | Teste integração Lei 14.133 | 30min |
| 5 | Documentação | 30min |
| **Total RunPod** | | **3.5h** |

VPS (paralelo):
| Fase | Descrição | Tempo |
|------|-----------|-------|
| 1 | Atualizar schema Milvus | 1h |
| 2 | Receber/mapear campos | 30min |
| 3 | Filtro no retrieval | 1h |
| 4 | Nota no RAG | 1h |
| **Total VPS** | | **3.5h** |

---

## 9. Checklist de Conclusão

### RunPod
- [ ] Código implementado e funcionando
- [ ] Testes passando
- [ ] Lei 14.133 classificada corretamente
- [ ] Documentação atualizada
- [ ] Commit com mensagem descritiva

### VPS
- [ ] Schema Milvus atualizado
- [ ] Campos mapeados corretamente
- [ ] Filtro de retrieval funcionando
- [ ] Nota de origem no RAG

### Integração
- [ ] Teste end-to-end: ingestão → retrieval → resposta
- [ ] Verificar que external chunks têm nota correta
- [ ] Performance aceitável

---

## 10. Referências

- Discussão original: Conversa Claude RunPod + ChatGPT + Claude VPS (2026-02-04)
- Problema identificado: Artigos duplicados/citações na Lei 14.133
- Commits relacionados:
  - `3e1c9fb` - fix(PR13): offsets absolutos
  - `f4529a7` - fix(docling): desabilitar fast_mode
