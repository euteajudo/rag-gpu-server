# TAREFA: Executar Re-Ingestao da Lei 14.133/2021

## Contexto
Os patches foram aplicados com sucesso:
1. **Patch 1**: Fallback em `vllm_client.py` para erros de JSON (retorna dict vazio)
2. **Patch 2**: Filtro em `pipeline.py` para chunks com `_skip_milvus_index=True`

O Milvus e Neo4j foram purgados (0 chunks da LEI-14.133-2021).

## O Que Voce Precisa Fazer

Execute a ingestao diretamente no RunPod usando o venv do servidor.

### Opcao 1: Criar e executar script Python

Crie o arquivo `/tmp/run_lei_ingest.py` com este conteudo:

```python
import sys
import os
import asyncio

sys.path.insert(0, "/workspace/rag-gpu-server")
os.chdir("/workspace/rag-gpu-server")
os.environ["VLLM_BASE_URL"] = "http://localhost:8002/v1"

from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.models import IngestRequest

async def run():
    print("=== INICIANDO INGESTAO LEI 14.133/2021 ===")

    request = IngestRequest(
        document_id="LEI-14.133-2021",
        tipo_documento="LEI",
        numero="14.133",
        ano=2021,
        orgao_emissor="Presidencia da Republica",
        max_articles=250
    )

    pdf_path = "/workspace/pdf/Lei 14.133_2021.pdf"
    print(f"Lendo PDF: {pdf_path}")

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    print(f"PDF: {len(pdf_bytes)} bytes")

    pipeline = IngestionPipeline()
    result = await pipeline.process(pdf_bytes, request)

    print(f"
=== RESULTADO ===")
    print(f"Status: {result.status}")
    print(f"Total chunks: {len(result.chunks) if result.chunks else 0}")

    if result.error:
        print(f"ERRO: {result.error}")

    if result.chunks:
        from collections import Counter
        types = Counter(c.device_type for c in result.chunks if hasattr(c, 'device_type'))
        print(f"
Chunks por tipo:")
        for t, count in types.most_common():
            print(f"  {t}: {count}")

        art_ids = set(c.span_id for c in result.chunks if hasattr(c, 'span_id') and c.span_id)
        print(f"
=== VALIDACAO ARTIGOS CRITICOS ===")
        for art in ["ART-006", "ART-056", "ART-057"]:
            status = "PRESENTE" if art in art_ids else "AUSENTE"
            print(f"  {art}: {status}")

        art075_status = "PRESENTE (ERRO!)" if "ART-075" in art_ids else "AUSENTE (correto)"
        print(f"  ART-075 (parent): {art075_status}")

        skipped = [c for c in result.chunks if getattr(c, '_skip_milvus_index', False)]
        print(f"
Chunks com _skip_milvus_index=True: {len(skipped)}")

    return result

asyncio.run(run())
```

Depois execute:
```bash
/workspace/venv/bin/python /tmp/run_lei_ingest.py
```

## Criterios de Sucesso

1. **ART-006, ART-056, ART-057**: Devem estar PRESENTES
2. **ART-075 (parent)**: Deve estar AUSENTE (so as partes P1, P2, P3)
3. **Total de artigos**: Proximo de 194
4. **Erros de JSON**: Nao devem mais ocorrer

## Responda Com

1. Output completo da execucao
2. Quantos chunks foram gerados
3. Status dos artigos criticos (ART-006, 056, 057, 075)
4. Se houve algum erro

---
MENSAGEM ENVIADA EM: 2026-01-27 01:00 UTC
EXECUTE DIRETAMENTE NO TERMINAL, NAO VIA API HTTP

---

## RESPOSTA DO CLAUDE RUNPOD - 2026-01-27 01:15 UTC

### RESULTADO DA INGESTAO

```
=== RESULTADO ===
Status: IngestStatus.COMPLETED
Total chunks: 1263
```

**A ingestao foi executada com sucesso!**

### OBSERVACOES

1. **Erros de JSON capturados (patch funcionando):**
   ```
   JSON invalido com json_schema, tentando extrair: Unterminated string...
   Nao foi possivel recuperar JSON, retornando estrutura vazia
   ```
   O patch capturou 2 erros de JSON e continuou o processamento.

2. **Milvus nao esta no RunPod:**
   O pipeline processa os chunks e retorna no objeto `result.chunks`.
   Ele NAO insere no Milvus - isso e feito pela VPS.

3. **Para inserir no Milvus:**
   Voce precisa chamar o endpoint `/ingest` via HTTP da VPS.
   O GPU Server retornara os chunks no response JSON.

### PROXIMOS PASSOS

Execute a ingestao via API HTTP da VPS:

```bash
curl -X POST http://GPU_SERVER:8000/ingest \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/Lei 14.133_2021.pdf" \
  -F "document_id=LEI-14.133-2021" \
  -F "tipo_documento=LEI" \
  -F "numero=14.133" \
  -F "ano=2021"
```

O response tera os chunks com embeddings prontos para o Milvus.

---

## ATUALIZACAO - 2026-01-27 01:30 UTC

### VALIDACAO COMPLETA

Executei nova ingestao e salvei os resultados:

```
Status: COMPLETED
Total chunks: 1263

=== ART-006 (artigo de definicoes) ===
ART-006 parent: NAO indexado (correto!)
ART-006 partes: 6 chunks
  - ART-006-P1: 4400 chars
  - ART-006-P2: 3984 chars
  - ART-006-P3: 4112 chars
  - ART-006-P4: 4074 chars
  - ART-006-P5: 3917 chars
  - ART-006-P6: 2689 chars
  Total: ~23,176 chars (COMPLETO!)

=== ART-075 ===
ART-075 parent: NAO indexado (correto!)
ART-075 partes: P1, P2, P3 (OK)

=== PROBLEMA: ART-056 e ART-057 ===
Status: AUSENTES
A lista de span_ids pula de ART-055 para ART-058.

Verifiquei na web: esses artigos EXISTEM na Lei 14.133/2021.
- Art. 56: Modos de disputa (aberto/fechado)
- Art. 57: Combinacao dos modos

CAUSA PROVAVEL: O Docling ou o SpanParser nao esta
detectando esses artigos no PDF. Pode ser problema de:
1. Formatacao do PDF nessa regiao
2. Regex do SpanParser nao matching
3. Artigos curtos sendo ignorados

SUGESTAO: Verificar o markdown do Docling para ver se
os artigos 56 e 57 aparecem no texto extraido.
```

### RESUMO

| Artigo | Status | Observacao |
|--------|--------|------------|
| ART-006 | OK | Splitado em 6 partes |
| ART-056 | FALHA | Nao extraido do PDF |
| ART-057 | FALHA | Nao extraido do PDF |
| ART-075 | OK | Splitado em 3 partes |

---

## PATCH ADICIONAL - 2026-01-27 01:45 UTC

### CAUSA RAIZ DO ART-056 e ART-057

O Docling interpreta algumas partes do PDF como listas numeradas:
```
Art. 55. Os prazos...          ← OK
11. Art. 56. O modo de...      ← Prefixo "11." adicionado
21. Art. 57. O edital...       ← Prefixo "21." adicionado
```

### PATCH APLICADO

**Arquivo:** `src/parsing/span_parser.py`

1. **Regex atualizado** para aceitar prefixos numéricos:
   ```python
   # Antes:
   r'^[-*]?\s*Art\.?\s*(\d+)...'

   # Depois:
   r'^(?:\d+\.\s*)?[-*]?\s*Art\.?\s*(\d+)...'
   ```

2. **Limpeza do conteúdo** para remover prefixos das linhas:
   ```python
   clean_line = re.sub(r'^\d+\.\s*', '', line)
   ```

3. Atualizado PATTERN_ARTIGO, PATTERN_PARAGRAFO, PATTERN_INCISO, PATTERN_ALINEA

### RESULTADO DO TESTE

```
Parsing markdown...
Total spans: 1390
Total articles: 206  ← Antes: 194

ART-055: ENCONTRADO
ART-056: ENCONTRADO  ← CORRIGIDO!
ART-057: ENCONTRADO  ← CORRIGIDO!
ART-058: ENCONTRADO

Texto limpo: "Art. 56º O modo de disputa..."  ← Sem "11."
```

### STATUS

- Servidor reiniciado e saudável
- Pronto para re-ingestão via API

---

## RELATORIO FINAL DE VALIDACAO - 2026-01-27 02:00 UTC

### RESULTADO GERAL

```
Status: COMPLETED ✓
Total chunks: 1299
Artigos unicos: 195

Chunks por tipo:
  inciso: 638
  paragraph: 448
  article: 204
  part: 9
```

### VALIDACAO DE ARTIGOS CRITICOS

#### ART-006 (artigo de definicoes - ~23k chars)
```
✓ Parent ART-006 NAO indexado (correto!)
✓ Splitado em 6 partes:
  - ART-006-P1: 4400 chars
  - ART-006-P2: 3984 chars
  - ART-006-P3: 4112 chars
  - ART-006-P4: 4074 chars
  - ART-006-P5: 3917 chars
  - ART-006-P6: 2689 chars
  Total: ~23,176 chars (COMPLETO!)
```

#### ART-056 e ART-057 (antes faltando)
```
ART-056: ✓ PRESENTE
  Texto: "Art. 56º O modo de disputa poderá ser..."

ART-057: ✓ PRESENTE
  Texto: "Art. 57º O edital de licitação poderá..."

✓ Texto limpo, sem prefixos numericos do Docling
```

#### ART-075 (outro artigo grande)
```
✓ Parent ART-075 NAO indexado (correto!)
✓ Splitado em 3 partes: P1, P2, P3
```

### ERROS DE JSON CAPTURADOS

```
JSON invalido com json_schema, tentando extrair: Unterminated string...
Nao foi possivel recuperar JSON, retornando estrutura vazia
```
O patch de fallback capturou 2 erros de JSON e o processo continuou normalmente.

### RESUMO DOS PATCHES APLICADOS

| # | Arquivo | Correcao |
|---|---------|----------|
| 1 | `vllm_client.py` | Fallback para JSONDecodeError |
| 2 | `pipeline.py` | Filtro `_skip_milvus_index` |
| 3 | `span_parser.py` | Aceita prefixos numericos do Docling |

### STATUS FINAL

```
✓ Todos os artigos criticos validados
✓ Split de artigos grandes funcionando
✓ Texto limpo (sem prefixos)
✓ Parents nao indexados
✓ Servidor saudavel em http://localhost:8000
✓ Pronto para ingestao via API HTTP
```

---

## PATCH PARENT-LOOPS APLICADO - 2026-01-27 03:20 UTC

### Arquivos Modificados

1. **`src/chunking/citation_extractor.py`**
   - `normalize_citations()` agora aceita `parent_chunk_id` e `document_type`
   - Remove citações onde chunk filho cita seu parent

2. **`src/ingestion/pipeline.py`**
   - Atualizada chamada de `extract_citations_from_chunk()` para passar novos parâmetros

### Testes Executados

```
=== TESTE DE PARENT-LOOPS ===

OK ART-006-P1: esperado [], obteve []
   (citação ao parent ART-006 removida)

OK ART-075-P1: esperado ['leis:LEI-12850#ART-003'], obteve ['leis:LEI-12850#ART-003']
   (manteve citação externa, removeu parent-loop)

OK Self-loop: esperado [], obteve []
OK Legado (sem parent_chunk_id): esperado manter citação

=== TODOS OS TESTES PASSARAM ===
```

### Status

```
✓ Patch aplicado
✓ Testes passaram
✓ Servidor reiniciado e saudável
✓ Pronto para nova ingestão
```

---
RELATORIO ENVIADO EM: 2026-01-27 03:20 UTC

---

## PATCH parent_chunk_id PARA INCISOS - 2026-01-27 05:35 UTC

### Problema Identificado

Incisos tinham `parent_chunk_id` incorreto, sempre apontando para o artigo
mesmo quando pertenciam a um parágrafo.

**Exemplo incorreto:**
```
INC-018-V → parent_chunk_id = LEI-14.133-2021#ART-018  ❌
```

**Esperado:**
```
INC-018-V → parent_chunk_id = LEI-14.133-2021#PAR-018-1  ✓
```

### Causa

O `ChunkMaterializer` ignorava o `parent_id` do Span (definido pelo SpanParser)
e sempre usava o artigo como pai dos incisos.

### Correção Aplicada

**Arquivo:** `src/chunking/chunk_materializer.py`

**Lógica adicionada** (em 2 locais - `materialize_article` e `_materialize_large_article`):

```python
# Determina parent_chunk_id correto: parágrafo (se existir) ou artigo
if inc_span.parent_id and inc_span.parent_id.startswith("PAR-"):
    correct_parent_chunk_id = f"{self.document_id}#{inc_span.parent_id}"
else:
    correct_parent_chunk_id = parent_chunk_id  # Fallback para artigo
```

### Testes Executados

```
============================================================
TESTE DE CORREÇÃO parent_chunk_id
============================================================

=== TESTE: Inciso sob Parágrafo ===
Inciso span_id: INC-018-V
Inciso parent_chunk_id: LEI-14.133-2021#PAR-018-1
Esperado: LEI-14.133-2021#PAR-018-1
✓ PASSOU: Inciso aponta para parágrafo correto!

=== TESTE: Inciso diretamente sob Artigo ===
Inciso span_id: INC-010-I
Inciso parent_chunk_id: LEI-14.133-2021#ART-010
✓ PASSOU: Inciso aponta para artigo corretamente!

=== TESTE: Inciso sem parent_id (fallback) ===
Inciso span_id: INC-020-I
Inciso parent_chunk_id: LEI-14.133-2021#ART-020
✓ PASSOU: Fallback para artigo funcionou!

Total: 3/3 testes passaram
✓ TODOS OS TESTES PASSARAM!
```

### Status

```
✓ Patch aplicado
✓ 3/3 testes passaram
✓ Servidor reiniciado e saudável
✓ Pronto para re-ingestão
```

---
ATUALIZADO EM: 2026-01-27 05:35 UTC
