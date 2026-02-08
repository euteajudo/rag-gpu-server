# PATCH: Remoção de Parent-Loops em normalize_citations()

## Problema
Chunks filhos (ex: ART-006-P1) estavam citando seus pais (ART-006), criando parent-loops falsos no grafo Neo4j.

### Casos Reais Encontrados na Lei 14.133:
- **ART-006-P1**: citations = `["leis:LEI-14.133-2021#ART-006"]` → deveria ser `[]`
- **ART-075-P1**: citations = `["leis:LEI-12850#ART-003", "leis:LEI-14.133-2021#ART-075"]` → deveria manter apenas `["leis:LEI-12850#ART-003"]`

## Solução: Double Shield

### 1ª Blindagem (Pipeline → Milvus)
Arquivo: `/workspace/rag-gpu-server/src/chunking/citation_extractor.py`

A função `normalize_citations()` precisa aceitar novos parâmetros para detectar e remover parent-loops.

### 2ª Blindagem (Sync → Neo4j)
Já implementada na VPS em `sync_service.py`.

---

## INSTRUÇÕES PARA APLICAR O PATCH

### Passo 1: Backup do arquivo original
```bash
cp /workspace/rag-gpu-server/src/chunking/citation_extractor.py \
   /workspace/rag-gpu-server/src/chunking/citation_extractor.py.backup
```

### Passo 2: Localizar a função normalize_citations()

A função está no final do arquivo `citation_extractor.py`. Procure por:
```python
def normalize_citations(
    citations: list[str | dict] | None,
    chunk_node_id: str,
) -> list[str]:
```

### Passo 3: Substituir pela versão atualizada

Substitua a função `normalize_citations()` inteira pelo código abaixo:

```python
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
```

### Passo 4: Atualizar chamada no pipeline.py

Arquivo: `/workspace/rag-gpu-server/src/ingestion/pipeline.py`

Procure pela chamada de `normalize_citations()` (deve estar perto da linha 825-830) e atualize para passar os novos parâmetros:

**ANTES:**
```python
chunk.citations = normalize_citations(
    citations=chunk_citations,
    chunk_node_id=chunk.node_id,
)
```

**DEPOIS:**
```python
chunk.citations = normalize_citations(
    citations=chunk_citations,
    chunk_node_id=chunk.node_id,
    parent_chunk_id=chunk.parent_chunk_id,
    document_type=request.tipo_documento,
)
```

### Passo 5: Reiniciar o servidor
```bash
pkill -f uvicorn
cd /workspace/rag-gpu-server
nohup /workspace/venv/bin/python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 > /workspace/gpu-server.log 2>&1 &
```

---

## COMO TESTAR A CORREÇÃO

### Teste 1: Teste unitário rápido
```python
import sys
sys.path.insert(0, '/workspace/rag-gpu-server/src')

from chunking.citation_extractor import normalize_citations

# Teste ART-006-P1 (deve retornar [])
result = normalize_citations(
    citations=["leis:LEI-14.133-2021#ART-006"],
    chunk_node_id="leis:LEI-14.133-2021#ART-006-P1",
    parent_chunk_id="LEI-14.133-2021#ART-006",
    document_type="LEI"
)
assert result == [], f"FALHOU ART-006-P1: esperado [], obteve {result}"
print("✓ ART-006-P1: OK (parent-loop removido)")

# Teste ART-075-P1 (deve manter apenas LEI-12850)
result = normalize_citations(
    citations=["leis:LEI-12850#ART-003", "leis:LEI-14.133-2021#ART-075"],
    chunk_node_id="leis:LEI-14.133-2021#ART-075-P1",
    parent_chunk_id="LEI-14.133-2021#ART-075",
    document_type="LEI"
)
assert result == ["leis:LEI-12850#ART-003"], f"FALHOU ART-075-P1: esperado ['leis:LEI-12850#ART-003'], obteve {result}"
print("✓ ART-075-P1: OK (manteve externa, removeu parent-loop)")

# Teste self-loop
result = normalize_citations(
    citations=["leis:DECRETO-10.947-2022#ART-003"],
    chunk_node_id="leis:DECRETO-10.947-2022#ART-003",
)
assert result == [], f"FALHOU self-loop: esperado [], obteve {result}"
print("✓ Self-loop: OK")

# Teste sem parent_chunk_id (comportamento legado)
result = normalize_citations(
    citations=["leis:LEI-14.133-2021#ART-006"],
    chunk_node_id="leis:LEI-14.133-2021#ART-006-P1",
)
assert result == ["leis:LEI-14.133-2021#ART-006"], f"FALHOU legado: esperado manter citação"
print("✓ Legado (sem parent_chunk_id): OK")

print("\n✅ TODOS OS TESTES PASSARAM!")
```

### Teste 2: Re-ingerir um documento pequeno
Após aplicar o patch, re-ingira a IN-65-2021 ou IN-58-2022 e verifique se os chunks com parent_chunk_id têm citations corretas.

---

## ROLLBACK (se necessário)
```bash
cp /workspace/rag-gpu-server/src/chunking/citation_extractor.py.backup \
   /workspace/rag-gpu-server/src/chunking/citation_extractor.py
```

---

## CONTEXTO TÉCNICO

### Por que parent-loops são problemáticos?
1. Criam edges falsos no Neo4j (ART-006-P1 → ART-006)
2. Poluem os resultados de citation expansion
3. Distorcem métricas de conectividade do grafo

### Formato dos IDs:
- `chunk_node_id`: `"leis:LEI-14.133-2021#ART-006-P1"` (com prefixo)
- `parent_chunk_id`: `"LEI-14.133-2021#ART-006"` (SEM prefixo)
- A função calcula: `parent_node_id = prefix + ":" + parent_chunk_id`

### Mapeamento de prefixos:
- LEI, DECRETO, IN, LC → `"leis"`
- ACORDAO → `"acordaos"`
- SUMULA → `"sumulas"`
