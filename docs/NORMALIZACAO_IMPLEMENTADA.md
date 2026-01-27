# Normalizacao de Document ID - IMPLEMENTADO

**Data:** 2026-01-27
**Autor:** Claude (sessao RunPod GPU Server)
**Status:** Implementado e testado

---

## Implementacao Realizada

### 1. Novo Modulo: `src/utils/normalization.py`

Criado modulo com funcoes de normalizacao:

```python
from src.utils.normalization import normalize_document_id, normalize_node_id

# Exemplos:
normalize_document_id("LEI-14133-2021")  # -> "LEI-14.133-2021"
normalize_document_id("LEI 14133/2021")  # -> "LEI-14.133-2021"
normalize_document_id("IN-58-2022")      # -> "IN-58-2022" (< 1000, sem ponto)

normalize_node_id("leis:LEI-14133-2021#ART-018")  # -> "leis:LEI-14.133-2021#ART-018"
```

### 2. Integracao no CitationExtractor

Modificado `src/chunking/citation_extractor.py`:

```python
# Adicionado import
from src.utils.normalization import normalize_document_id

# Modificado _build_doc_id() para aplicar normalizacao
raw_doc_id = "-".join(parts)
return normalize_document_id(raw_doc_id)
```

### 3. Testes Realizados

```
Teste 1: art. 18 da Lei 14.133/2021
  doc_id: LEI-14.133-2021
  target_node_id: leis:LEI-14.133-2021#ART-018

Teste 2: art. 25 da Lei 14133
  doc_id: LEI-14.133-2021
  target_node_id: leis:LEI-14.133-2021#ART-025

Teste 3: IN 58/2022
  doc_id: IN-58-2022
  target_node_id: leis:IN-58-2022

=== TODOS OS TESTES PASSARAM ===
```

---

## Regras de Normalizacao

| Entrada | Saida | Regra |
|---------|-------|-------|
| `LEI 14133/2021` | `LEI-14.133-2021` | Ponto de milhar (>= 1000) |
| `lei-14.133-2021` | `LEI-14.133-2021` | Uppercase |
| `LEI-14133-2021` | `LEI-14.133-2021` | Ponto de milhar |
| `IN-58-2022` | `IN-58-2022` | Sem mudanca (< 1000) |
| `DECRETO-10947-2022` | `DECRETO-10.947-2022` | Ponto de milhar |
| `LEI-8666-1993` | `LEI-8.666-1993` | Ponto de milhar |

**IMPORTANTE:** Anos (4 digitos no final) NAO recebem ponto:
- `2021` fica `2021` (nao `2.021`)
- `1993` fica `1993` (nao `1.993`)

---

## Proximos Passos (VPS)

### Fase 4: Migracao de Dados Existentes (VPS)

Os dados existentes no Milvus e Neo4j precisam ser atualizados:

1. **Milvus**: Os chunks ja existentes com `LEI-14.133-2021` estao corretos
2. **Edges Neo4j**: Precisam ser recriados com target_node_id normalizado

Script sugerido para re-sync:
```python
# Na VPS, apos nova ingestao
# Os edges serao criados com target_node_id normalizado
```

### Fase 5: Validacao (VPS)

Apos nova ingestao da IN-58-2022:
- [ ] Verificar que citacoes para LEI-14.133-2021 resolvem
- [ ] Verificar que edges :CITA apontam para node_ids existentes
- [ ] Testar GraphRetriever end-to-end

---

## Arquivos Modificados

| Arquivo | Mudanca |
|---------|---------|
| `src/utils/__init__.py` | Criado (exporta normalize_document_id) |
| `src/utils/normalization.py` | Criado (funcoes de normalizacao) |
| `src/chunking/citation_extractor.py` | Import + usa normalize_document_id |

---

## Servidor

```
Status: Reiniciado com mudancas aplicadas
Endpoint: http://localhost:8000
Health: /health
```

Pronto para nova ingestao!

---
RESPOSTA ENVIADA EM: 2026-01-27 16:45 UTC
