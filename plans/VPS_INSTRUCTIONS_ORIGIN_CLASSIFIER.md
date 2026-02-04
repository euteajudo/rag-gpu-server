# Instruções para Claude da VPS: OriginClassifier

**Copie o texto abaixo e envie para o Claude da VPS**

---

## Texto para Enviar

```
═══════════════════════════════════════════════════════════════════════════════
INSTRUÇÕES PARA IMPLEMENTAÇÃO: OriginClassifier (VPS)
Data: 2026-02-04
De: Claude RunPod
Para: Claude VPS
═══════════════════════════════════════════════════════════════════════════════

CONTEXTO
--------
Implementamos no RunPod (GPU Server) o módulo OriginClassifier que adiciona
metadados de "origem material" em cada chunk durante a ingestão.

PROBLEMA QUE RESOLVE:
O PDF da Lei 14.133/2021 contém "ilhas" de material externo - artigos de
outras leis que são citados ou modificados. Exemplo:

  - Art. 178 da Lei 14.133: "O Código Penal passa a vigorar com..."
  - Art. 337-E: "Admitir, possibilitar ou dar causa à contratação..."

O Art. 337-E ESTÁ no PDF da Lei 14.133, mas NÃO É da Lei 14.133 - é do
Código Penal. Se o LLM citar "Segundo a Lei 14.133, Art. 337-E..." está
semanticamente ERRADO.

SOLUÇÃO:
Manter document_id como está (cadeia de custódia), mas adicionar metadados
de origem para permitir tratamento diferenciado no retrieval e na resposta.

═══════════════════════════════════════════════════════════════════════════════
NOVOS CAMPOS NO PAYLOAD DE CHUNKS (RunPod → VPS)
═══════════════════════════════════════════════════════════════════════════════

| Campo                  | Tipo         | Valores Possíveis            |
|------------------------|--------------|------------------------------|
| origin_type            | VARCHAR(16)  | "self" | "external"           |
| origin_reference       | VARCHAR(128) | "DL-2848-1940" | null         |
| origin_reference_name  | VARCHAR(128) | "Código Penal" | null         |
| is_external_material   | BOOL         | true | false                  |
| origin_confidence      | VARCHAR(8)   | "high" | "medium" | "low"     |
| origin_reason          | VARCHAR(256) | "rule:codigo_penal_art337"   |

Valores de origin_type:
- "self": Material da própria lei (ex: Art. 1 da Lei 14.133)
- "external": Material de outra lei citada (ex: Art. 337-E do Código Penal)

═══════════════════════════════════════════════════════════════════════════════
TAREFAS PARA VPS
═══════════════════════════════════════════════════════════════════════════════

1. MILVUS SCHEMA
   ─────────────
   Adicionar os 6 campos na collection (leis_v4 ou criar v5):

   - origin_type: VARCHAR(16), default "self"
   - origin_reference: VARCHAR(128), nullable
   - origin_reference_name: VARCHAR(128), nullable
   - is_external_material: BOOL, default false
   - origin_confidence: VARCHAR(8), default "high"
   - origin_reason: VARCHAR(256), nullable

   Se for filtrar frequentemente por is_external_material, criar índice escalar.

2. RECEBER/MAPEAR CAMPOS DO RUNPOD
   ────────────────────────────────
   O payload de chunks do RunPod agora inclui os campos origin_*.
   Mapear para os campos correspondentes do Milvus no insert.

3. RETRIEVAL/SEARCH
   ─────────────────
   Opção A (recomendada): Adicionar parâmetro no endpoint de search

     GET /search?query=...&include_external=false

     Se include_external=false:
       expr += " and is_external_material == false"

     Se include_external=true (default):
       Retorna tudo, mas marca os externos no resultado

   Opção B: Boost implícito no reranking
     - origin_type="self" → score * 1.0
     - origin_type="external" → score * 0.8

4. RESPOSTA DO RAG
   ────────────────
   Quando um chunk com origin_type="external" for usado na resposta,
   incluir nota contextual. Opções:

   Opção A - No texto da resposta:
     "Este trecho está no documento da Lei 14.133, mas refere-se ao
      Código Penal (DL-2848-1940), inserido pela Lei 14.133."

   Opção B - No header da citação:
     "[Fonte: LEI-14133-2021 | Material externo: Código Penal]"

   Opção C - Metadata separado:
     {
       "citation": {...},
       "origin_warning": "Material do Código Penal citado na Lei 14.133"
     }

5. ENDPOINT DE STATS (opcional, baixa prioridade)
   ───────────────────────────────────────────────
   Em /stats ou /collection/info, adicionar:
   - Contagem de chunks por origin_type (self vs external)
   - Lista de origin_references únicos na collection

═══════════════════════════════════════════════════════════════════════════════
EXEMPLO DE CHUNK QUE O RUNPOD VAI ENVIAR
═══════════════════════════════════════════════════════════════════════════════

{
  "chunk_id": "LEI-14133-2021#ART-178",
  "node_id": "leis:LEI-14133-2021#ART-178",
  "text": "Art. 178. O Decreto-Lei nº 2.848, de 7 de dezembro de 1940 (Código Penal), passa a vigorar acrescido...",
  "document_id": "LEI-14133-2021",
  "span_id": "ART-178",
  "article_number": "178",

  // CAMPOS NOVOS:
  "origin_type": "self",
  "origin_reference": null,
  "origin_reference_name": null,
  "is_external_material": false,
  "origin_confidence": "high",
  "origin_reason": null,

  "dense_vector": [0.0123, -0.0456, ...],
  "sparse_vector": {...}
}

// Chunk de material EXTERNO:
{
  "chunk_id": "LEI-14133-2021#CIT-CP-337E",
  "node_id": "leis:LEI-14133-2021#CIT-CP-337E",
  "text": "Art. 337-E. Admitir, possibilitar ou dar causa à contratação direta fora das hipóteses previstas em lei...",
  "document_id": "LEI-14133-2021",  // Está no PDF da 14.133
  "span_id": "CIT-CP-337E",

  // CAMPOS NOVOS - EXTERNAL:
  "origin_type": "external",
  "origin_reference": "DL-2848-1940",
  "origin_reference_name": "Código Penal",
  "is_external_material": true,
  "origin_confidence": "high",
  "origin_reason": "rule:codigo_penal_art337",

  "dense_vector": [0.0789, -0.0123, ...],
  "sparse_vector": {...}
}

═══════════════════════════════════════════════════════════════════════════════
ESTATÍSTICAS ESPERADAS (Lei 14.133)
═══════════════════════════════════════════════════════════════════════════════

Após ingestão da Lei 14.133/2021:
- Total chunks: ~206
- origin_type="self": ~187 (artigos da Lei 14.133)
- origin_type="external": ~19 (principalmente Código Penal)

external_refs esperados:
- DL-2848-1940 (Código Penal): ~12 chunks (Art. 337-E a 337-P)
- LEI-8987-1995 (Concessões): ~4 chunks
- LEI-13105-2015 (CPC): ~3 chunks

═══════════════════════════════════════════════════════════════════════════════
PRIORIDADE DE IMPLEMENTAÇÃO
═══════════════════════════════════════════════════════════════════════════════

1. [BLOQUEANTE] Schema Milvus - Adicionar campos
2. [BLOQUEANTE] Receber/mapear campos do RunPod
3. [ALTA] Filtro include_external no retrieval
4. [MÉDIA] Nota de origem na resposta do RAG
5. [BAIXA] Endpoint de stats

═══════════════════════════════════════════════════════════════════════════════
DÚVIDAS?
═══════════════════════════════════════════════════════════════════════════════

Se tiver dúvidas sobre:
- Formato dos campos
- Lógica de classificação
- Casos de borda

Avisa que a gente sincroniza. O código do OriginClassifier está em:
  /workspace/rag-gpu-server/src/chunking/origin_classifier.py
  /workspace/rag-gpu-server/plans/ORIGIN_CLASSIFIER_IMPLEMENTATION.md

═══════════════════════════════════════════════════════════════════════════════
```

---

## Resumo Rápido (se precisar de versão curta)

```
RunPod vai adicionar 6 campos novos nos chunks:
- origin_type: "self" | "external"
- origin_reference: "DL-2848-1940" | null
- origin_reference_name: "Código Penal" | null
- is_external_material: bool
- origin_confidence: "high" | "medium" | "low"
- origin_reason: string | null

Tarefas VPS:
1. Adicionar campos no schema Milvus
2. Mapear no insert
3. Filtro include_external no search
4. Nota de origem no RAG quando external
```
