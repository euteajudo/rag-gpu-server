# Carta para Claude RunPod: Problema ADDRESS_MISMATCH

**Data:** 2025-02-05
**De:** Claude VPS (extracao/rag-api)
**Para:** Claude RunPod (rag-gpu-server)
**Assunto:** Bug crítico no SpanParser causando chunks com endereço incorreto

---

## 1. Problema Identificado

Durante validação de evidências, descobrimos que alguns chunks estão sendo indexados com **node_id incorreto** - o endereço (span_id) não corresponde ao texto real do chunk.

### Exemplo Concreto do Bug

**Chunk indexado:**
```
node_id: leis:LEI-14133-2021#PAR-040-1
span_id: PAR-040-1
text: "§ 4º A fase preparatória do processo licitatório..."
```

**Problema:** O span_id diz `PAR-040-1` (§ 1º do Art. 40), mas o texto começa com `§ 4º` (deveria ser `PAR-040-4`).

### Impacto

- Usuário pergunta sobre § 1º do Art. 40
- Sistema retorna chunk com texto do § 4º
- Citação aponta para lugar errado
- **Perda de confiança no sistema**

---

## 2. Causa Raiz Identificada

O problema está no **SpanParser** (`rag-gpu-server/src/parsing/span_parser.py`).

### O que acontece

O regex de detecção de parágrafos (`PATTERN_PARAGRAFO`) não diferencia entre:

1. **Novo parágrafo real** - `§ 2º` no início de uma linha, marcando novo dispositivo legal
2. **Citação interna** - `§ 1º deste artigo` no meio do texto, referenciando outro dispositivo

### Exemplo de texto problemático

```markdown
Art. 40. O planejamento de compras...

§ 1º O estudo técnico preliminar...

§ 2º Para os fins do disposto no § 1º deste artigo, considera-se...
                                   ^^^^^^^^^^^^^^^^
                                   CITAÇÃO, não novo parágrafo!

§ 3º As contratações de que trata o § 2º...

§ 4º A fase preparatória do processo licitatório...
```

O SpanParser detecta `§ 1º deste artigo` (dentro do texto do § 2º) como se fosse um novo parágrafo, quebrando a estrutura.

### Padrões de citação interna que causam falsos positivos

- `conforme § 1º deste artigo`
- `nos termos do § 2º`
- `previsto no § 3º acima`
- `de que trata o § 1º do art. 40`
- `segundo o § único`

---

## 3. O Que Já Fizemos (VPS)

Implementamos um **guard-rail** para detectar e bloquear esses chunks incorretos **antes de servir ao usuário**.

### Componentes criados

| Arquivo | Descrição |
|---------|-----------|
| `src/evidence/address_validator.py` | Valida se node_id corresponde ao texto |
| `src/evidence/integrity_validator.py` | Integrado como step 6 de validação |
| `src/api/db/models/ingest_alarm.py` | Novo tipo `ADDRESS_MISMATCH` |
| `tests/test_address_validator.py` | Suite de testes (25+ casos) |

### Lógica de validação

```python
# Regras implementadas:
PAR-040-1  → texto deve começar com "§ 1" (ou variantes: §1º, § 1°, etc)
ART-044    → texto deve começar com "Art. 44" (ou "Artigo 44", "Art 44")
INC-036-V  → texto deve começar com "V -" (ou "V–", "V—")
ALI-036-V-a → texto deve começar com "a)" (ou "a -")
```

### Commits realizados

```
extracao @ 782ca8a - feat: Add AddressValidator for chunk address consistency validation
vector_govi_2 @ eeedaea - chore: Atualiza submodule extracao com AddressValidator
```

### Limitação

Isso é um **guard-rail**, não uma correção. Os chunks incorretos ainda são indexados no Milvus - apenas não são servidos. Precisamos corrigir na origem.

---

## 4. Sugestão de Correção no SpanParser

### Opção A: Regex mais restritivo (simples, risco médio)

Exigir que `§` esteja **no início da linha** (após `\n` ou início do texto):

```python
# ANTES (problemático):
PATTERN_PARAGRAFO = r'§\s*(\d+)[ºo°]?'

# DEPOIS (mais restritivo):
PATTERN_PARAGRAFO = r'(?:^|\n)\s*§\s*(\d+)[ºo°]?'
```

**Risco:** Pode não funcionar se o Markdown não tiver quebras de linha consistentes.

### Opção B: Detecção de contexto léxico (robusta, recomendada)

Antes de criar um novo span de parágrafo, verificar se o `§` está em contexto de citação:

```python
# Padrões que indicam CITAÇÃO (ignorar, não é novo dispositivo)
CITATION_CONTEXT_PATTERNS = [
    r'(?:conforme|nos termos d[oa]|segundo [oa]|previsto n[oa]|de que trata [oa])\s*§',
    r'§\s*\d+[ºo°]?\s*(?:deste|desta|do|da|acima|anterior|seguinte|do caput)',
    r'(?:o|a|os|as)\s+§\s*\d+[ºo°]?(?:\s+(?:do|da|deste|desta))',
    r'art(?:igo)?\.?\s*\d+[,\s]+§',  # "art. 40, § 1º" = citação composta
    r'§\s*\d+[ºo°]?\s+do\s+art',     # "§ 1º do art. 40" = citação
]

def is_citation_context(text: str, match_start: int) -> bool:
    """Verifica se o § encontrado está em contexto de citação."""
    # Pega 50 chars antes e 30 depois do match
    context_before = text[max(0, match_start - 50):match_start]
    context_after = text[match_start:match_start + 30]
    context = context_before + context_after

    for pattern in CITATION_CONTEXT_PATTERNS:
        if re.search(pattern, context, re.IGNORECASE):
            return True
    return False
```

### Opção C: Validação pós-parse (complementar)

Adicionar o AddressValidator **durante a ingestão** no pipeline, rejeitando chunks com mismatch antes de indexar:

```python
# No pipeline.py, após ChunkMaterializer:
from evidence.address_validator import validate_chunk_address

for chunk in materialized_chunks:
    result = validate_chunk_address(
        node_id=chunk.node_id,
        chunk_text=chunk.text,
        document_id=chunk.document_id,
    )
    if result.address_mismatch:
        logger.error(f"[ADDRESS_MISMATCH] Rejeitando chunk: {chunk.node_id}")
        # Não indexar, ou marcar para revisão
        continue
```

---

## 5. Recomendação Final

Implementar **Opção B + Opção C** (defesa em profundidade):

1. **Opção B** (SpanParser) - Previne o problema na origem
2. **Opção C** (Pipeline) - Rejeita qualquer erro que escape

### Ordem de implementação sugerida

1. Primeiro implementar Opção C (mais rápida, usa código que já existe na VPS)
2. Depois implementar Opção B (requer análise mais cuidadosa dos padrões)

---

## 6. Arquivos Relevantes para Referência

### No RunPod (rag-gpu-server)

```
src/parsing/span_parser.py          # SpanParser - onde está o bug
src/parsing/span_models.py          # Modelos Span, SpanType
src/ingestion/pipeline.py           # Pipeline principal
src/chunking/chunk_materializer.py  # Onde os chunks são criados
```

### Na VPS (extracao) - código do AddressValidator para copiar

```
src/evidence/address_validator.py   # Validador completo
tests/test_address_validator.py     # Testes para referência
```

---

## 7. Testes para Validar a Correção

Após implementar, rodar com documento de teste que contenha citações internas:

```python
# Texto de teste com citação interna
test_markdown = """
Art. 40. O planejamento de compras...

§ 1º O estudo técnico preliminar a que se refere o inciso I do caput...

§ 2º Para os fins do disposto no § 1º deste artigo, considera-se...

§ 3º As contratações de que trata o § 2º serão...

§ 4º A fase preparatória do processo licitatório...
"""

# Esperado após parse:
# - PAR-040-1 → texto começa com "§ 1º O estudo"
# - PAR-040-2 → texto começa com "§ 2º Para os fins" (contém "§ 1º" mas como citação)
# - PAR-040-3 → texto começa com "§ 3º As contratações"
# - PAR-040-4 → texto começa com "§ 4º A fase"

# NÃO deve criar PAR-040-1 extra por causa do "§ 1º deste artigo" dentro do § 2º
```

---

## 8. Contato

Se precisar do código do AddressValidator ou tiver dúvidas sobre a lógica de validação, o código está disponível em:

- Repo: `euteajudo/rag-api`
- Branch: `main`
- Commit: `782ca8a`
- Arquivo: `src/evidence/address_validator.py`

---

**Prioridade:** ALTA - Afeta confiabilidade das citações
**Esforço estimado:** 2-4 horas para Opção B + C
