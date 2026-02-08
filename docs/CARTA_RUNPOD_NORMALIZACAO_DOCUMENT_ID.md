# Carta: Integração do Normalizador de document_id no Pipeline de Ingestão

**De:** Claude (VPS/Local)
**Para:** Claude (RunPod GPU Server)
**Data:** 2025-01-28
**Prioridade:** Alta

---

## Contexto do Problema

Existe uma inconsistência de formato entre document_ids no Milvus e citations no Neo4j:

| Local | Formato Atual | Formato Canônico |
|-------|---------------|------------------|
| Milvus | `LEI-14133-2021` (sem ponto) | `LEI-14.133-2021` (com ponto) |
| Neo4j citations | `LEI-14.133-2021` (com ponto) | `LEI-14.133-2021` (com ponto) ✅ |

**Resultado:** As citações extraídas pelo `CitationExtractor` não resolvem porque o target `leis:LEI-14.133-2021#ART-036` não encontra match no Milvus que tem `LEI-14133-2021`.

---

## O Normalizador Já Existe

O arquivo `src/utils/normalization.py` já implementa a normalização correta:

```python
from utils.normalization import normalize_document_id

normalize_document_id("LEI-14133-2021")   # -> "LEI-14.133-2021"
normalize_document_id("LEI 14133/2021")   # -> "LEI-14.133-2021"
normalize_document_id("IN-58-2022")       # -> "IN-58-2022" (sem ponto, <1000)
normalize_document_id("DECRETO-10947-2022") # -> "DECRETO-10.947-2022"
```

**Regra:** Números >= 1000 recebem ponto de milhar brasileiro.

---

## Tarefa: Integrar Normalizador no Router de Ingestão

### Arquivo a Modificar

`/workspace/rag-gpu-server/src/ingestion/router.py`

### Mudança Necessária

**Linha ~176** (onde `document_id` é recebido do Form):

```python
# ANTES
document_id: str = Form(..., description="ID unico do documento (ex: LEI-14133-2021)"),
```

**Linha ~219** (onde `IngestRequest` é criado):

```python
# ANTES
IngestRequest(
    pdf_content=pdf_content,
    document_id=document_id,  # <-- SEM normalização
    ...
)

# DEPOIS
from utils.normalization import normalize_document_id

IngestRequest(
    pdf_content=pdf_content,
    document_id=normalize_document_id(document_id),  # <-- COM normalização
    ...
)
```

### Import a Adicionar

No topo do arquivo `router.py`:

```python
from utils.normalization import normalize_document_id
```

---

## Verificação

Após a mudança, teste com:

```bash
# Ingerir um documento de teste
curl -X POST http://localhost:8000/ingest \
  -F "file=@test.pdf" \
  -F "document_id=LEI-14133-2021" \
  -F "tipo_documento=LEI" \
  -F "numero=14133" \
  -F "ano=2021"

# Verificar no log se o document_id foi normalizado para LEI-14.133-2021
```

---

## Por Que Isso é Importante

1. **Citations funcionarão:** O Neo4j cria edges com targets normalizados. Se o Milvus também tiver format normalizado, as expansões via grafo encontrarão os chunks.

2. **Consistência:** Todos os document_ids seguirão o mesmo padrão em todo o sistema.

3. **Formato brasileiro:** Números como 14.133 são mais legíveis para usuários brasileiros.

---

## Não Esquecer

- **NÃO** modificar dados existentes no Milvus (isso será feito por migração separada na VPS)
- **APENAS** adicionar a normalização para novas ingestões
- Fazer commit das mudanças no repositório

---

## Checklist

- [ ] Adicionar import de `normalize_document_id` no `router.py`
- [ ] Aplicar `normalize_document_id(document_id)` ao criar `IngestRequest`
- [ ] Testar ingestão e verificar que document_id está normalizado
- [ ] Commit: `fix: Normaliza document_id na ingestão para formato canônico`

---

Obrigado!
