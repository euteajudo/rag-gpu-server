# Normalização de Document ID - Correção Crítica

**Data:** 2025-01-27
**Autor:** Claude (sessão de testes VPS)
**Prioridade:** Alta

---

## 1. Problema Identificado

Durante testes de validação das citações da IN-58-2022, descobrimos uma **inconsistência crítica de normalização** entre o CitationExtractor e os document_ids armazenados no Milvus.

### Sintoma

```
Citação extraída:  leis:LEI-14133-2021#ART-018
Document ID no Milvus: LEI-14.133-2021

Resultado: Citation NÃO RESOLVE (node_id não encontrado)
```

### Dados do Teste

| Categoria | Quantidade | Status |
|-----------|------------|--------|
| Citações únicas IN-58-2022 | 18 | |
| Citações internas (mesma IN) | 3 | ✅ Válidas |
| Citações externas (outros docs) | 15 | ❌ Não resolvidas |

**Citações que falharam por normalização:**
- `leis:LEI-14133-2021#ART-011` → Milvus tem `LEI-14.133-2021`
- `leis:LEI-14133-2021#ART-018` → Milvus tem `LEI-14.133-2021`
- `leis:LEI-14133-2021#ART-025` → Milvus tem `LEI-14.133-2021`
- `leis:LEI-14133-2021#ART-036` → Milvus tem `LEI-14.133-2021`
- (e mais 4 artigos da Lei 14.133)

---

## 2. Causa Raiz

O CitationExtractor extrai referências normativas do texto e gera node_ids, mas **não normaliza** o número do documento com ponto de milhar.

**Fluxo atual (bugado):**
```
Texto: "conforme art. 18 da Lei nº 14.133/2021"
                           │
                           ▼
CitationExtractor.extract()
                           │
                           ▼
target_node_id = "leis:LEI-14133-2021#ART-018"  ← SEM PONTO
```

**Enquanto a ingestão:**
```
Usuário informa: "Lei 14.133/2021"
                           │
                           ▼
Pipeline de ingestão
                           │
                           ▼
document_id = "LEI-14.133-2021"  ← COM PONTO
```

---

## 3. Consequências

### 3.1 Graph Retrieval Quebrado
- Edges `:CITA` no Neo4j apontam para node_ids inexistentes
- Expansão por citação retorna 0 resultados
- GraphRetriever não consegue seguir referências normativas

### 3.2 Citation Expansion Ineficaz
- Campo `citations` dos chunks contém stubs inválidos
- Verificação de validade de citações falha
- Sistema não consegue resolver referências cruzadas

### 3.3 Qualidade do RAG Comprometida
- Contexto expandido não inclui artigos citados
- Respostas perdem informação de dispositivos referenciados
- Usuário não recebe contexto completo

---

## 4. Estratégia Adotada: Normalização Centralizada

### 4.1 Decisão
Criar uma **função única de normalização** (`normalize_document_id()`) que será usada em todos os pontos do sistema.

### 4.2 Justificativa
- Não podemos garantir que o usuário sempre escreva no formato esperado
- Normalização em um único ponto garante consistência
- Manutenção centralizada facilita correções futuras

### 4.3 Regras de Normalização

```python
def normalize_document_id(raw_id: str) -> str:
    """
    Normaliza document_id para formato canônico.

    Regras:
    1. Uppercase: "lei" → "LEI"
    2. Separador hífen: "LEI 14133/2021" → "LEI-14133-2021"
    3. Ponto de milhar para números >= 1000: "14133" → "14.133"
    4. Remove "nº", espaços extras, caracteres especiais

    Exemplos:
        "LEI 14133/2021"    → "LEI-14.133-2021"
        "lei-14.133-2021"   → "LEI-14.133-2021"
        "LEI-14133-2021"    → "LEI-14.133-2021"
        "Lei nº 14.133"     → "LEI-14.133"
        "IN-58-2022"        → "IN-58-2022" (sem mudança, < 1000)
        "DECRETO-10947-2022" → "DECRETO-10.947-2022"
    """
    pass  # Implementar
```

### 4.4 Pontos de Integração

| Local | Arquivo | Uso |
|-------|---------|-----|
| **CitationExtractor** | `src/chunking/citation_extractor.py` | Ao extrair referências do texto |
| **Pipeline de Ingestão** | `src/ingestion/pipeline.py` | Ao criar document_id do chunk |
| **SDK Python** | `vectorgov-sdk/src/vectorgov/client.py` | Ao validar entrada do usuário |
| **Frontend** | Formulário de upload | Máscara visual + validação |

### 4.5 Implementação Sugerida

```python
import re

def normalize_document_id(raw_id: str) -> str:
    if not raw_id:
        return ""

    # 1. Uppercase e remove espaços extras
    normalized = raw_id.upper().strip()

    # 2. Remove "Nº", "N.", etc.
    normalized = re.sub(r'\bN[ºO°]?\.?\s*', '', normalized)

    # 3. Substitui separadores por hífen
    normalized = re.sub(r'[\s/]+', '-', normalized)

    # 4. Remove hífens duplicados
    normalized = re.sub(r'-+', '-', normalized)

    # 5. Adiciona ponto de milhar em números >= 1000
    def add_thousands_separator(match):
        num = match.group(0)
        if len(num) >= 4 and num.isdigit():
            # Formata com ponto de milhar
            return f"{int(num):,}".replace(",", ".")
        return num

    normalized = re.sub(r'\d+', add_thousands_separator, normalized)

    # 6. Remove hífen no início/fim
    normalized = normalized.strip('-')

    return normalized
```

---

## 5. Plano de Ação

### Fase 1: Implementar função de normalização
- [ ] Criar `normalize_document_id()` em `src/utils/normalization.py`
- [ ] Adicionar testes unitários

### Fase 2: Integrar no CitationExtractor
- [ ] Importar função em `citation_extractor.py`
- [ ] Aplicar normalização ao gerar `target_node_id`
- [ ] Testar com IN-58-2022

### Fase 3: Integrar no Pipeline de Ingestão
- [ ] Aplicar normalização ao criar `document_id` dos chunks
- [ ] Garantir que `node_id` canônico use mesmo formato

### Fase 4: Migração de Dados Existentes
- [ ] Script para normalizar document_ids existentes no Milvus
- [ ] Script para atualizar citations existentes
- [ ] Re-sync Neo4j após correção

### Fase 5: Validação
- [ ] Re-testar IN-58-2022 citations
- [ ] Verificar resolução de citações para LEI-14.133-2021
- [ ] Testar GraphRetriever end-to-end

---

## 6. Testes de Validação Esperados

Após implementação, os seguintes testes devem passar:

```python
# Teste 1: Normalização básica
assert normalize_document_id("LEI 14133/2021") == "LEI-14.133-2021"
assert normalize_document_id("lei-14.133-2021") == "LEI-14.133-2021"
assert normalize_document_id("LEI-14133-2021") == "LEI-14.133-2021"

# Teste 2: Números menores que 1000 (sem ponto)
assert normalize_document_id("IN-58-2022") == "IN-58-2022"
assert normalize_document_id("IN-65-2021") == "IN-65-2021"

# Teste 3: Números maiores que 1000 (com ponto)
assert normalize_document_id("DECRETO-10947-2022") == "DECRETO-10.947-2022"
assert normalize_document_id("LEI-8666-1993") == "LEI-8.666-1993"

# Teste 4: Citations resolvem corretamente
# Dado: Milvus tem document_id = "LEI-14.133-2021"
# Quando: CitationExtractor extrai "art. 18 da Lei 14133"
# Então: target_node_id = "leis:LEI-14.133-2021#ART-018" (com ponto)
```

---

## 7. Referências

- **Sessão de origem:** Testes de validação IN-58-2022 (2025-01-27)
- **Collection testada:** leis_v4
- **Documentos afetados:** Todos que citam leis com número >= 1000

---

**Status:** Aguardando implementação no GPU Server
