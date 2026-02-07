# origin_classifier.py — Especificação v2

> **Contexto**: Classificador de proveniência de chunks no pipeline VectorGov
> **Posição no pipeline**: Pós-Reconciliator, pré-canonical_validation
> **Input**: chunks reconciliados (texto limpo PyMuPDF + hierarquia VLM) + canonical_text
> **Output**: campos origin_* preenchidos em cada chunk
> **Custo GPU**: zero (texto-only, regex + heurística + máquina de estados)
> **Abordagem**: Score híbrido para detectar transições + máquina de estados para propagar zona

---

## 1. O Problema

### 1.1 O que é incorporação normativa

A técnica legislativa brasileira permite que uma lei **altere o texto de outras leis**
diretamente no seu próprio corpo. Quando o Congresso quer criar um novo crime no
Código Penal, por exemplo, ele não edita o Código Penal diretamente — ele publica uma
nova lei que diz "o Código Penal passa a vigorar acrescido do seguinte artigo:" e transcreve
o conteúdo do novo artigo dentro da lei nova.

Isso significa que o PDF de uma lei pode conter, fisicamente dentro dele, **texto que
pertence a outra norma**. O texto está lá, mas a autoridade normativa é de outro diploma.

### 1.2 Tipos de incorporação

| Tipo | Descrição | Frequência | Exemplo |
|---|---|---|---|
| **Alteração por inserção** | Lei hospedeira insere artigos/capítulos inteiros em outra norma | Comum | Lei 14.133 Art. 178 insere Cap. II-B no Código Penal |
| **Alteração por nova redação** | Lei hospedeira reescreve dispositivos existentes de outra norma | Muito comum | Lei 14.133 Art. 179 altera incisos da Lei 8.987 |
| **Alteração de caput** | Lei hospedeira reescreve apenas o caput de artigo de outra norma | Comum | Lei 14.133 Art. 180 altera art. 10 da Lei 11.079 |
| **Anexos e apêndices** | Material técnico, tabelas, quadros comparativos anexados | Ocasional | Tabelas de valores, modelos de editais |
| **Transcrição por remissão** | Trechos copiados de outras normas para contexto | Raro | Consolidações que citam texto original |

### 1.3 Por que é tão comum

Na tradição jurídica brasileira, a alteração legislativa por transcrição é o mecanismo padrão.
A Lei 14.133/2021 (Nova Lei de Licitações), por exemplo, além de seus 194 artigos próprios:
- Inseriu 12 artigos novos no Código Penal (Arts. 337-E a 337-P)
- Alterou dispositivos de 4 outras leis (CPC, Lei 8.987, Lei 11.079, Lei 13.303)
- Revogou 3 leis inteiras (Lei 8.666, Lei 10.520, arts. da Lei 12.462)

Praticamente toda lei de médio/grande porte faz isso. Não é exceção — é regra.

### 1.4 Exemplo concreto

A Lei 14.133/2021, páginas 69-72:

```
Art. 178. O Título XI da Parte Especial do Decreto-Lei nº 2.848, de 7 de dezembro de 1940
(Código Penal), passa a vigorar acrescido do seguinte Capítulo II-B:

    "CAPÍTULO II-B
    DOS CRIMES EM LICITAÇÕES E CONTRATOS ADMINISTRATIVOS

    Contratação direta ilegal
    Art. 337-E. Admitir, possibilitar ou dar causa à contratação direta fora das
    hipóteses previstas em lei:
    Pena - reclusão, de 4 (quatro) a 8 (oito) anos, e multa.

    [... Arts. 337-F a 337-O ...]

    Art. 337-P. A pena de multa cominada aos crimes previstos neste Capítulo seguirá a
    metodologia de cálculo prevista neste Código..." 
```

O Art. 178 é dispositivo da Lei 14.133. Mas os Arts. 337-E a 337-P são dispositivos do
**Código Penal** — foram inseridos lá pela Lei 14.133. A autoridade normativa é do DL 2.848/1940.

---

## 2. As Causas do Problema no Pipeline

### 2.1 Por que o RAG erra sem classificação

Um pipeline RAG padrão trata todo texto dentro de um PDF como conteúdo daquele documento.
Quando processa o PDF da Lei 14.133, extrai todos os blocos de texto, gera embeddings, e
indexa no Milvus com `document_id = "LEI-14133-2021"`.

Resultado: o Art. 337-E ("Contratação direta ilegal") é indexado como se fosse parte da
Lei 14.133. Quando um usuário pergunta "quais são os crimes em licitações?", o sistema
retorna o chunk e diz: **"Segundo a Lei 14.133, Art. 337-E..."** — tecnicamente incorreto.
O Art. 337-E é dispositivo do Código Penal.

### 2.2 Por que o pipeline anterior (Docling + SpanParser) falhava

O pipeline V1 usava regex para detectar esses padrões. O problema:

1. **Texto sujo do Docling**: o Docling frequentemente quebrava aspas, reticências e marcadores
   como "(NR)" — os mesmos sinais que o regex precisava para funcionar
2. **Offsets inconsistentes**: com offsets errados, a janela de contexto ao redor do chunk
   capturava texto desalinhado, degradando a detecção
3. **Fragmentação de blocos**: o Docling às vezes quebrava "passa a vigorar acrescido do seguinte"
   em dois blocos separados, e o regex não via a frase completa

### 2.3 Por que o pipeline novo (PyMuPDF + VLM) resolve

O PyMuPDF entrega **texto determinístico e limpo** com **offsets corretos**. Isso significa:

- Aspas, reticências e "(NR)" chegam intactos → regex funciona corretamente
- Offsets confiáveis → janela de contexto captura o texto certo
- Blocos não fragmentados arbitrariamente → frases gatilho ficam completas

O classificador é o mesmo (regex + heurística), mas opera sobre **dados muito mais limpos**.
A melhoria vem do input, não do algoritmo.

---

## 3. Efeitos Downstream (sem classificação)

### 3.1 ADDRESS_MISMATCH nos Evidence Links

O Evidence Drawer monta links na forma `LEI-14133-2021#ART-337-E`. Mas o Art. 337-E
não existe na Lei 14.133 — ele existe no Código Penal. O link aponta para um endereço
que semanticamente não é verdadeiro. Isso é uma instância do ADDRESS_MISMATCH que
motivou toda a migração VLM.

### 3.2 PR13 com snippet atribuído à norma errada

O PR13 valida que o snippet (texto extraído do canonical) bate com o chunk indexado.
Se o chunk está marcado como `document_id = "LEI-14133-2021"` mas o texto é do Código Penal,
o PR13 pode validar a integridade técnica mas a **atribuição semântica** está errada.
O módulo de segurança não consegue distinguir "texto integro mas atribuído errado" de
"texto correto e bem atribuído".

### 3.3 Grafo Neo4j com relações falsas

O Neo4j armazena hierarquia: `LEI-14133-2021#ART-178 → LEI-14133-2021#ART-337-E`.
Essa relação pai-filho é **fisicamente** correta (no PDF, o 337-E está dentro do bloco do 178),
mas **logicamente** falsa (o 337-E é dispositivo autônomo do Código Penal, não filho do Art. 178).

Com a classificação, o sistema pode:
- Manter a relação física (útil para navegação no documento)
- Adicionar uma relação lógica: `DL-2848-1940#ART-337-E` → norma real

### 3.4 Resposta ao usuário final

Sem classificação:
> "De acordo com o Art. 337-E da **Lei 14.133/2021**, a contratação direta ilegal é punida com..."

Com classificação:
> "De acordo com o Art. 337-E do **Código Penal** (inserido pela Lei 14.133/2021), a contratação direta ilegal é punida com..."

A diferença é sutil mas juridicamente relevante. Um operador de licitações precisa saber
que o tipo penal está no Código Penal — é lá que ele vai buscar jurisprudência, doutrina,
e precedentes, não na Lei 14.133.

### 3.5 Impacto na busca semântica

Chunks externos indexados como `self` poluem os resultados de busca quando o usuário
filtra por documento. Uma query com `tipo_documento = "LEI"` e `document_id = "LEI-14133-2021"`
retorna chunks do Código Penal — porque estavam no PDF da 14.133. O campo
`is_external_material = True` permite filtrar esses chunks ou reduzir seu peso no ranking.

---

## 4. Campos Populados

```python
class OriginFields:
    origin_type: str            # "self" | "external"
    origin_reference: str       # ID canônico da norma-alvo (ex: "DL-2848-1940") ou ""
    origin_reference_name: str  # Nome legível (ex: "Código Penal") ou ""
    is_external_material: bool  # origin_type == "external"
    origin_confidence: str      # "high" | "medium" | "low"
    origin_reason: str          # Features que contribuíram para a decisão
```

**Princípio de independência**: `origin_type` e `origin_reference` são resoluções separadas.
Um chunk pode ser `external` sem que a norma-alvo tenha sido identificada:

```python
# Sabe que é externo, não sabe de onde
origin_type = "external"
origin_reference = ""
origin_reference_name = ""
origin_confidence = "medium"
origin_reason = "trigger_phrase + quote_block (norma-alvo não identificada)"
```

---

## 5. Arquitetura: Score + Máquina de Estados

### 5.1 Dois problemas distintos

| Problema | Mecanismo | Pergunta |
|---|---|---|
| **Detecção de transição** | Score (enter_score / exit_score) | "Estou entrando ou saindo de uma zona externa?" |
| **Propagação dentro da zona** | Máquina de estados | "Este chunk genérico herda o estado da zona?" |
| **Identificação da norma-alvo** | Extração regex independente | "De qual norma é este material externo?" |

### 5.2 Por que score sozinho não basta

O Art. 337-F ("Frustração do caráter competitivo de licitação") é texto genérico do Código Penal.
Sem marcadores próprios, o score daria ~0.0 e seria classificado como `self`.
Mas ele está **dentro** da zona aberta pelo Art. 178. A máquina de estados propaga o `external`.

### 5.3 Por que máquina de estados sozinha não basta

Se o Reconciliator cortou o chunk de forma que "passa a vigorar" ficou no chunk anterior
e o conteúdo externo começa no próximo, a máquina não vê o gatilho.
O score com janela de contexto (ctx_before) resolve: enxerga 800 chars antes do chunk.

---

## 6. Janela de Contexto

Cada chunk é analisado com contexto do canonical_text ao redor:

```python
CTX_WINDOW = 800  # caracteres

def get_context(canonical_text: str, start: int, end: int) -> tuple[str, str]:
    ctx_before = canonical_text[max(0, start - CTX_WINDOW):start]
    ctx_after = canonical_text[end:min(len(canonical_text), end + CTX_WINDOW)]
    return ctx_before, ctx_after
```

A janela captura sinais que podem estar fora do chunk mas semanticamente relacionados:
- "passa a vigorar..." no chunk anterior
- "(NR)" no chunk seguinte
- "da Lei nº..." no parágrafo acima

---

## 7. Sistema de Score

### 7.1 Dois scores separados

Cada chunk recebe **dois** scores independentes:

- **enter_score**: evidência de que este chunk INICIA uma zona externa
- **exit_score**: evidência de que este chunk ENCERRA uma zona externa

### 7.2 Features de entrada (enter_score)

| ID | Feature | Peso | Onde buscar | Exemplo |
|---|---|---|---|---|
| E1 | Frase gatilho (`passa a vigorar`, `acrescido do seguinte`, etc.) | +0.40 | ctx_before + chunk | "passa a vigorar acrescido do seguinte Capítulo II-B:" |
| E2 | Aspas abrindo próximo ao início do chunk | +0.20 | ctx_before[-200:] + chunk[:200] | `"Art. 337-E.` |
| E3 | "Art." com numeração fora da sequência da lei hospedeira | +0.50 | chunk | Art. 337-E dentro de doc que está no Art. 178 |
| E4 | "CAPÍTULO" / "Seção" / "TÍTULO" dentro de aspas | +0.40 | chunk | "CAPÍTULO II-B" |
| E5 | Referência explícita a norma-alvo (`da Lei nº`, `do Decreto-Lei nº`) | +0.30 | ctx_before + chunk | "do Decreto-Lei nº 2.848..." |
| E6 | Nome legível entre parênteses | +0.20 | ctx_before + chunk | "(Código Penal)" |
| E7 | Marcador de anexo/apêndice (header isolado) | +0.50 | chunk | "ANEXO I" |

**Frases gatilho** (E1) — lista expandida:

```python
TRIGGER_PHRASES = [
    "passa a vigorar acrescido do seguinte",
    "passa a vigorar com a seguinte redação",
    "passam a vigorar com a seguinte redação",
    "fica acrescido do seguinte",
    "fica acrescida do seguinte",
    "dá-se a seguinte redação",
    "a seguinte redação ao",
    "passa a vigorar acrescido de",
    "com a redação dada por",
    "na redação da",
]
```

### 7.3 Features de saída (exit_score)

| ID | Feature | Peso | Onde buscar | Exemplo |
|---|---|---|---|---|
| S1 | `" (NR)` ou `..." (NR)` | +0.70 | chunk + ctx_after[:200] | `............." (NR)` |
| S2 | Aspas fechando seguidas de retomada da numeração sequencial | +0.50 | chunk + ctx_after | `"` → Art. 179 |
| S3 | Retomada de numeração sequencial da lei hospedeira (sem aspas) | +0.30 | ctx_after[:400] | próximo chunk é Art. 179 |
| S4 | Nova frase gatilho de entrada (outra zona) | +0.40 | ctx_after | "O art. X da Lei Y passa a vigorar..." |

### 7.4 Score de permanência (dentro da zona)

Quando a máquina está em estado EXTERNAL, chunks genéricos (sem features de entrada ou saída)
herdam o estado. Mas o score de permanência serve como diagnóstico:

| Feature | Peso | Significado |
|---|---|---|
| Numeração compatível com a zona (337-F, 337-G...) | +0.30 | Sequência da norma-alvo |
| Padrão "Pena - reclusão/detenção" | +0.20 | Dispositivo penal típico |
| Ausência total de sinais | +0.00 | Herda por inércia (state machine) |

---

## 8. Thresholds com Histerese

```python
T_ENTER = 0.60    # Alto: difícil abrir zona (evita falso positivo de external)
T_EXIT  = 0.40    # Baixo: fácil fechar zona (evita ficar preso em zona aberta)
```

**Por que histerese?**

Falso positivo de `external` é mais perigoso que falso negativo:
- FP: marca texto normativo da lei como "externo" → sistema ignora dispositivo válido
- FN: marca texto externo como "self" → sistema atribui autoridade errada, mas o texto ainda é encontrável

Então: exigimos mais evidência para ENTRAR do que para SAIR.

**Exemplos de decisão:**

| Situação | enter_score | exit_score | Decisão |
|---|---|---|---|
| "passa a vigorar" (+0.4) + aspas (+0.2) + ref norma (+0.3) | 0.90 | — | ABRE zona (≥ T_ENTER) |
| "passa a vigorar" (+0.4) sozinho | 0.40 | — | NÃO abre (< T_ENTER) |
| `" (NR)` (+0.7) | — | 0.70 | FECHA zona (≥ T_EXIT) |
| Retomada numeração (+0.3) sozinha | — | 0.30 | NÃO fecha (< T_EXIT), mas acumula |

---

## 9. TTL (Time-To-Live) como Guard Rail

```python
TTL_CHUNKS = 50   # Máximo de chunks consecutivos em zona EXTERNAL
```

**Problema que resolve**: se o classificador abriu uma zona e o marcador de saída está corrompido,
ausente, ou foi cortado de forma estranha, a máquina ficaria em EXTERNAL para sempre
— marcando o resto inteiro do documento como material externo.

**Comportamento**:

```
Se chunks_na_zona >= TTL_CHUNKS e nenhum exit_score >= T_EXIT nos últimos N chunks:
    → Fecha zona automaticamente
    → origin_confidence = "low" para os últimos chunks da zona
    → origin_reason inclui "ttl_forced_close"
    → Loga warning para revisão humana
```

**Calibração**: 50 chunks é conservador. A maior zona externa real na Lei 14.133 é o Capítulo II-B
do Código Penal (Arts. 337-E a 337-P = ~12 artigos = ~15-20 chunks). TTL de 50 dá folga enorme
mas impede catástrofe.

---

## 10. Máquina de Estados

### 10.1 Estados

```python
class ClassifierState:
    mode: str = "SELF"                   # "SELF" | "EXTERNAL"
    zone_target_id: str = ""             # origin_reference da zona ativa
    zone_target_name: str = ""           # origin_reference_name da zona ativa
    zone_entry_chunk_id: str = ""        # chunk onde a zona abriu
    zone_host_article: str = ""          # artigo da lei hospedeira que abriu a zona
    zone_chunk_count: int = 0            # contador para TTL
    zone_enter_score: float = 0.0        # score que abriu a zona (para diagnóstico)
    zone_reasons: list[str] = []         # features acumuladas da zona
```

### 10.2 Transições

```
                        ┌────────────────────┐
                        │   SELF (default)   │
                        └─────────┬──────────┘
                                  │
                     enter_score >= T_ENTER (0.60)
                                  │
                                  ▼
                        ┌────────────────────┐
                        │     EXTERNAL       │◄──── chunks genéricos herdam
                        │  (zona ativa)      │      por propagação
                        └─────────┬──────────┘
                                  │
                     exit_score >= T_EXIT (0.40)
                     OU zone_chunk_count >= TTL
                                  │
                                  ▼
                        ┌────────────────────┐
                        │   SELF (retorno)   │
                        └────────────────────┘
```

### 10.3 Algoritmo principal

```python
def classify_document(chunks: list[Chunk], canonical_text: str, host_doc_id: str) -> list[Chunk]:
    """
    Processa chunks em ordem canônica (canonical_start crescente).
    Retorna chunks com campos origin_* preenchidos.
    """
    state = ClassifierState()

    # Ordenar por posição canônica
    chunks_sorted = sorted(chunks, key=lambda c: c.canonical_start)

    for chunk in chunks_sorted:
        ctx_before, ctx_after = get_context(canonical_text, chunk.canonical_start, chunk.canonical_end)

        # --- Calcular scores ---
        enter_score, enter_reasons = compute_enter_score(chunk.text, ctx_before, ctx_after, host_doc_id)
        exit_score, exit_reasons = compute_exit_score(chunk.text, ctx_before, ctx_after, host_doc_id)

        # --- Resolver referência (independente do estado) ---
        ref_id, ref_name = resolve_reference(chunk.text, ctx_before, ctx_after)

        # --- Transições ---
        if state.mode == "SELF":
            if enter_score >= T_ENTER:
                # ABRE zona
                state.mode = "EXTERNAL"
                state.zone_target_id = ref_id
                state.zone_target_name = ref_name
                state.zone_entry_chunk_id = chunk.chunk_id
                state.zone_chunk_count = 0
                state.zone_enter_score = enter_score
                state.zone_reasons = enter_reasons

        elif state.mode == "EXTERNAL":
            state.zone_chunk_count += 1

            # Atualizar referência se encontrou agora (resolução tardia)
            if ref_id and not state.zone_target_id:
                state.zone_target_id = ref_id
                state.zone_target_name = ref_name

            if exit_score >= T_EXIT:
                # FECHA zona — este chunk é o último da zona ou o primeiro fora
                assign_origin(chunk, state, exit_reasons)
                state = ClassifierState()  # reset
                continue

            if state.zone_chunk_count >= TTL_CHUNKS:
                # FORCE CLOSE — guard rail
                assign_origin(chunk, state, ["ttl_forced_close"])
                state = ClassifierState()
                continue

        # --- Atribuir campos ---
        assign_origin(chunk, state, enter_reasons if state.mode == "EXTERNAL" else [])

    return chunks_sorted


def assign_origin(chunk: Chunk, state: ClassifierState, reasons: list[str]):
    """Preenche campos origin_* no chunk baseado no estado atual."""
    if state.mode == "EXTERNAL":
        chunk.origin_type = "external"
        chunk.is_external_material = True
        chunk.origin_reference = state.zone_target_id
        chunk.origin_reference_name = state.zone_target_name
        chunk.origin_confidence = compute_confidence(state)
        chunk.origin_reason = format_reasons(state.zone_reasons + reasons)
    else:
        chunk.origin_type = "self"
        chunk.is_external_material = False
        chunk.origin_reference = ""
        chunk.origin_reference_name = ""
        chunk.origin_confidence = "high"
        chunk.origin_reason = ""
```

---

## 11. Resolução de Referência (independente)

Módulo separado que tenta identificar a norma-alvo. Pode falhar — isso não impede
a classificação como `external`.

### 11.1 Regexes de extração

```python
REFERENCE_PATTERNS = [
    # "da Lei nº 13.105, de 16 de março de 2015 (Código de Processo Civil)"
    r'da Lei nº ([\d.]+),?\s*de\s+(.+?)\s*\((.+?)\)',

    # "da Lei nº 8.987, de 13 de fevereiro de 1995,"
    r'da Lei nº ([\d.]+),?\s*de\s+(.+?)[\.,;]',

    # "do Decreto-Lei nº 2.848, de 7 de dezembro de 1940 (Código Penal)"
    r'do Decreto-Lei nº ([\d.]+),?\s*de\s+(.+?)\s*\((.+?)\)',

    # "do Decreto nº 10.024"
    r'do Decreto nº ([\d.]+)',

    # "da Lei Complementar nº 123"
    r'da Lei Complementar nº ([\d.]+)',

    # "da Medida Provisória nº 1.047"
    r'da Medida Provisória nº ([\d.]+)',
]
```

### 11.2 Mapa de nomes conhecidos

```python
KNOWN_REFERENCES = {
    "2.848":  {"tipo": "DL",  "id": "DL-2848-1940",   "nome": "Código Penal"},
    "13.105": {"tipo": "LEI", "id": "LEI-13105-2015",  "nome": "Código de Processo Civil"},
    "8.987":  {"tipo": "LEI", "id": "LEI-8987-1995",   "nome": "Lei de Concessões"},
    "11.079": {"tipo": "LEI", "id": "LEI-11079-2004",  "nome": "Lei de PPPs"},
    "8.666":  {"tipo": "LEI", "id": "LEI-8666-1993",   "nome": "Lei de Licitações (revogada)"},
    "10.520": {"tipo": "LEI", "id": "LEI-10520-2002",  "nome": "Lei do Pregão (revogada)"},
    "12.462": {"tipo": "LEI", "id": "LEI-12462-2011",  "nome": "RDC"},
    "13.303": {"tipo": "LEI", "id": "LEI-13303-2016",  "nome": "Lei das Estatais"},
    "12.232": {"tipo": "LEI", "id": "LEI-12232-2010",  "nome": "Lei de Publicidade Institucional"},
    "11.107": {"tipo": "LEI", "id": "LEI-11107-2005",  "nome": "Lei dos Consórcios Públicos"},
    "10.406": {"tipo": "LEI", "id": "LEI-10406-2002",  "nome": "Código Civil"},
    "5.172":  {"tipo": "LEI", "id": "LEI-5172-1966",   "nome": "Código Tributário Nacional"},
    "9.784":  {"tipo": "LEI", "id": "LEI-9784-1999",   "nome": "Lei do Processo Administrativo"},
}
```

### 11.3 Resolução tardia

A referência pode não estar no chunk que abre a zona. Pode estar no ctx_before
(no artigo hospedeiro acima). A resolução é tentada em cada chunk da zona:

```
Chunk 1 (Art. 178 caput): "...Decreto-Lei nº 2.848... Código Penal..."
    → ref_id = "DL-2848-1940", ref_name = "Código Penal"
    → zona abre COM referência

Chunk 2 (Art. 337-E): texto genérico sem referência
    → ref_id extraída do ctx_before (Art. 178 está na janela)
    → OU herdada do state.zone_target_id (já resolvida)
```

Se a zona fecha sem ter encontrado referência:
```python
origin_type = "external"
origin_reference = ""            # não identificada
origin_reference_name = ""
origin_confidence = "medium"     # sabe que é externo mas não sabe de onde
origin_reason = "trigger_phrase + quote_block (norma-alvo não identificada)"
```

---

## 12. Cálculo de Confidence

```python
def compute_confidence(state: ClassifierState) -> str:
    """
    Confidence baseada na riqueza de evidências da zona.
    Independente do score de entrada (que já passou do threshold).
    """
    evidence_score = 0.0

    # Tem referência identificada?
    if state.zone_target_id:
        evidence_score += 0.4
    if state.zone_target_name:
        evidence_score += 0.2

    # Score de entrada era forte?
    if state.zone_enter_score >= 0.80:
        evidence_score += 0.3
    elif state.zone_enter_score >= 0.60:
        evidence_score += 0.1

    # Tem múltiplas features?
    if len(state.zone_reasons) >= 3:
        evidence_score += 0.1

    # Decisão
    if evidence_score >= 0.7:
        return "high"      # ref + nome + score forte
    elif evidence_score >= 0.4:
        return "medium"    # ref sem nome, ou score forte sem ref
    else:
        return "low"       # evidência fraca, entrou por threshold mínimo
```

**Caso especial — TTL forced close:**
```python
if "ttl_forced_close" in reasons:
    return "low"  # sempre low quando TTL forçou o fechamento
```

---

## 13. Posição no Pipeline

```
╔══ ZONA DE MUDANÇA ══════════════════╗
║  PyMuPDF → VLM → Reconciliator     ║
╚═══════════════╪═════════════════════╝
                │
                ▼
       OriginClassifier               ← AQUI
       (score + state machine)
       Input: chunks + canonical_text
       Output: chunks com origin_*
                │
                ▼
       canonical_validation
                │
                ▼
       pr13_validator → alarm_service
                │
                ▼
       BGE-M3 → Milvus + Neo4j
```

Roda no **RunPod CPU**. Recebe `canonical_text` do PyMuPDF e chunks do Reconciliator.

---

## 14. Casos Especiais

### 14.1 Artigo hospedeiro como "ponte"

O Art. 178 da Lei 14.133 é dispositivo da própria lei, mas abre a zona externa.

- **Art. 178 (caput)**: `origin_type = "self"` — é o comando de alteração
- A zona EXTERNAL abre no **próximo chunk** (o que contém o conteúdo transcrito)
- Isso acontece naturalmente: o enter_score é calculado no chunk seguinte,
  com ctx_before capturando "passa a vigorar" do Art. 178

### 14.2 Zonas curtas (1-2 chunks)

Art. 180 altera apenas o caput do art. 10 da Lei 11.079. Zona de 1 chunk.
O score de entrada abre, o `" (NR)` no mesmo chunk ou ctx_after fecha.
Sem problemas — a máquina entra e sai no mesmo ciclo.

### 14.3 Zonas consecutivas sem gap

Art. 177 abre zona (CPC), fecha com "(NR)". Logo em seguida Art. 178 abre outra zona (CP).
A máquina volta para SELF entre as duas — Art. 178 caput é SELF, depois abre EXTERNAL.

### 14.4 Zonas aninhadas

Não implementar. Na prática legislativa brasileira, não existe "alteração dentro de alteração".
Se detectado (enter_score >= T_ENTER enquanto mode == EXTERNAL), logar como anomalia
e manter a zona atual.

### 14.5 Artigos revogatórios

Art. 193 revoga leis inteiras. `origin_type = "self"` — a revogação é ato da lei hospedeira.
Não há conteúdo transcrito.

### 14.6 "(VETADO)"

"§ 2º (VETADO)." → `origin_type = "self"`. Texto vetado não consta no documento.

### 14.7 Reticências (omissões)

"Art. 1.048. ........................................" — são omissões do texto da norma-alvo.
Dentro de zona EXTERNAL, tratadas como parte da zona.

---

## 15. Princípio de Generalidade

**O origin_classifier NÃO é específico para a Lei 14.133/2021.**

A Lei 14.133 é usada como caso de teste neste documento porque é a norma principal do
domínio do VectorGov e contém exemplos claros de todos os padrões. Mas o classificador
deve funcionar para **qualquer lei brasileira** que contenha incorporação normativa.

### 15.1 Padrões universais da técnica legislativa

Os marcadores documentados nas seções 7.2 e 7.3 são **padrões da técnica legislativa brasileira**,
não idiossincrasias da Lei 14.133. Qualquer lei que altere outra usa as mesmas construções:

- "passa a vigorar" → presente em toda lei que faz alteração
- "(NR)" → marcador padronizado de nova redação
- "da Lei nº X, de DD de MM de AAAA" → formato canônico de referência legislativa
- Aspas delimitando texto transcrito → convenção universal no Diário Oficial

Exemplos de outras leis que usam os mesmos padrões:
- Lei 13.146/2015 (Estatuto da Pessoa com Deficiência) → altera Código Civil, CPC, CLT
- Lei 13.709/2018 (LGPD) → altera Lei do Marco Civil da Internet
- Lei 12.846/2013 (Lei Anticorrupção) → altera Lei de Licitações (8.666)
- Qualquer Medida Provisória que altera legislação existente

### 15.2 O que NÃO deve ser hardcoded

| Elemento | Abordagem | Por quê |
|---|---|---|
| Frases gatilho (E1) | Lista expandível de strings | Padrão da técnica legislativa, não da lei |
| Regexes de referência (C1-C6) | Padrões genéricos com captura | "da Lei nº X" funciona para qualquer lei |
| KNOWN_REFERENCES | Mapa expandível, **não obrigatório** | Classificação funciona sem ele (confidence menor) |
| Numeração "fora de sequência" (E3) | Comparar com sequência do documento atual | Genérico por definição |
| Marcadores de saída (S1-S4) | Padrões do Diário Oficial | Universais |

### 15.3 O que PODE ser expandido por domínio

O `KNOWN_REFERENCES` é o único componente que se beneficia de conhecimento de domínio.
Para o VectorGov (licitações), as referências mais frequentes já estão mapeadas. Mas se
o sistema processar leis de outros domínios (tributário, trabalhista, ambiental), basta
expandir o mapa:

```python
# Expansão futura — domínio trabalhista
KNOWN_REFERENCES["5.452"] = {"tipo": "DL", "id": "DL-5452-1943", "nome": "CLT"}
KNOWN_REFERENCES["8.213"] = {"tipo": "LEI", "id": "LEI-8213-1991", "nome": "Lei de Benefícios da Previdência"}

# Expansão futura — domínio tributário
KNOWN_REFERENCES["5.172"] = {"tipo": "LEI", "id": "LEI-5172-1966", "nome": "Código Tributário Nacional"}
```

Se a referência não está no mapa, o classificador ainda funciona:
- `origin_type = "external"` ✓ (detectou a zona)
- `origin_reference = "LEI-5452-1943"` ✓ (extraiu do regex)
- `origin_reference_name = ""` ← (não sabe o nome legível)
- `origin_confidence = "medium"` ← (rebaixado por falta de nome)

---

## 16. Testes

### 16.1 Caso de referência: Lei 14.133/2021

| Trecho | origin_type | origin_reference | confidence | Por quê |
|---|---|---|---|---|
| Art. 1 a 176 | self | — | high | Dispositivos próprios |
| Art. 177 (caput) | self | — | high | Comando de alteração |
| "Art. 1.048..." (CPC) | external | LEI-13105-2015 | high | Zona CPC |
| Art. 178 (caput) | self | — | high | Comando de alteração |
| "CAPÍTULO II-B" | external | DL-2848-1940 | high | Zona CP (abertura) |
| Art. 337-E | external | DL-2848-1940 | high | Zona CP (propagação) |
| Art. 337-F a 337-O | external | DL-2848-1940 | high | Zona CP (propagação — sem marcadores próprios) |
| Art. 337-P | external | DL-2848-1940 | high | Zona CP (último) |
| Art. 179 (caput) | self | — | high | Comando de alteração |
| "Art. 2º ..." (8.987) | external | LEI-8987-1995 | high | Zona 8.987 |
| Art. 180 (caput) | self | — | high | Comando de alteração |
| "Art. 10. ..." (11.079) | external | LEI-11079-2004 | high | Zona 11.079 |
| Art. 181 a 194 | self | — | high | Disposições transitórias |

### 16.2 Invariantes (devem valer para QUALQUER documento)

- Zero chunks `external/high` com `origin_reference = ""` (se é high, tem referência)
- Zero chunks `self` dentro de zona delimitada por marcadores E+S
- Zero TTL forced closes na Lei 14.133 (todas as zonas fecham normalmente)
- Todos os chunks 337-E a 337-P marcados como `external` com mesma `origin_reference`

### 16.3 Métricas esperadas para Lei 14.133 (exemplo, não spec)

> Estes valores são **referência para calibração**, não requisitos rígidos.
> Cada documento terá distribuição diferente dependendo do seu conteúdo.

- ~85-90% chunks `self`, ~10-15% `external` (para a 14.133 especificamente)
- 4 zonas externas (CPC, CP, 8.987, 11.079)
- 0 anomalias (zonas aninhadas, TTL forced close)

### 16.4 Teste de generalidade

O classificador deve ser testado com **pelo menos 3 documentos** além da Lei 14.133:

| Documento | Expectativa | O que testa |
|---|---|---|
| Lei 14.133/2021 | 4 zonas externas, ~15% external | Caso principal, múltiplas zonas |
| Lei simples sem alterações (ex: lei municipal curta) | 0 zonas, 100% self | Falso positivo zero |
| Lei com uma única alteração (ex: MP que altera 1 artigo) | 1 zona curta | Zona mínima (1-2 chunks) |
| Decreto regulamentador com anexos | Zonas de anexo | Marcadores A1-A5 |

Se o classificador retorna 0 anomalias e 0 falsos positivos nesses 4 cenários,
está pronto para produção.

---

## 17. Configuração

```python
# Thresholds de transição (histerese)
T_ENTER = 0.60      # Alto: difícil abrir zona
T_EXIT  = 0.40      # Baixo: fácil fechar zona

# Guard rail
TTL_CHUNKS = 50     # Máximo de chunks em zona sem exit

# Janela de contexto
CTX_WINDOW = 800    # Caracteres antes/depois do chunk

# Feature weights — entrada
W_TRIGGER_PHRASE = 0.40
W_QUOTE_OPEN = 0.20
W_OUT_OF_SEQUENCE = 0.50
W_CHAPTER_IN_QUOTES = 0.40
W_TARGET_REF = 0.30
W_TARGET_NAME = 0.20
W_ANNEX_HEADER = 0.50

# Feature weights — saída
W_NR_MARKER = 0.70
W_QUOTE_CLOSE_RESUME = 0.50
W_RESUME_SEQUENCE = 0.30
W_NEW_TRIGGER = 0.40
```

Todos os pesos e thresholds são constantes no topo do módulo, fáceis de ajustar
após testes com documentos reais.

---

## 18. Atribuição de Tarefas

> Este documento é enviado para ambos os Claudes (RunPod e VPS).
> As seções 1-17 são contexto compartilhado.
> Esta seção define **quem faz o quê**.

---

### 18.1 RUNPOD — Implementação do Classificador (80% do trabalho)

**Responsabilidade**: criar o módulo `origin_classifier.py` e integrá-lo no pipeline de inspeção/ingestão.

**Artefatos a criar**:

| Arquivo | Descrição |
|---|---|
| `src/classification/origin_classifier.py` | Módulo principal (seções 5-12 deste doc) |
| `src/classification/__init__.py` | Exports |
| `tests/test_origin_classifier.py` | Testes unitários (seção 15 deste doc) |

**O que implementar**:

1. **`ClassifierState`** — dataclass com estado da máquina (seção 10.1)
2. **`get_context()`** — janela de 800 chars antes/depois (seção 6)
3. **`compute_enter_score()`** — score de entrada com features E1-E7 (seção 7.2)
4. **`compute_exit_score()`** — score de saída com features S1-S4 (seção 7.3)
5. **`resolve_reference()`** — extração regex da norma-alvo, independente (seção 11)
6. **`compute_confidence()`** — high/medium/low baseado em evidências (seção 12)
7. **`classify_document()`** — loop principal com score + máquina de estados (seção 10.3)
8. **`assign_origin()`** — preenche os 6 campos origin_* no chunk (seção 10.3)

**Integração no pipeline**:

```python
# Em pipeline.py, DEPOIS do Reconciliator, ANTES da canonical_validation:

from ..classification.origin_classifier import classify_document

# Dentro do fluxo de ingestão/inspeção:
chunks = reconciliator.reconcile(pymupdf_blocks, vlm_elements)
chunks = classify_document(chunks, canonical_text, host_doc_id)  # ← NOVO
chunks = canonical_validation.validate(chunks)
```

**Integração no Inspector** (nova aba ou sub-aba de Reconciliação):

Os resultados do classificador devem ser visíveis no Inspector para revisão humana.
Sugestão: adicionar ao `ReconciliationArtifact` ou criar artefato separado com:
- Contagem de zonas detectadas (ex: "4 zonas externas")
- Lista de zonas com: chunk_id de entrada/saída, norma-alvo, confidence
- Chunks com `origin_confidence = "low"` destacados para atenção

**Inputs disponíveis** (já existem no pipeline):
- `canonical_text`: string do PyMuPDF (já produzido na fase 1)
- `chunks`: lista de ProcessedChunk do Reconciliator (já produzidos na fase 3)
- `host_doc_id`: string do document_id (ex: "LEI-14133-2021", já disponível nos metadados)

**Padrões a seguir**:
- Lazy loading (mesmo padrão do `_docling_converter` e `_span_parser`)
- `storage.save_artifact()` para salvar resultado no Redis (Inspector)
- Imports relativos (`from ..classification import classify_document`)
- Progress callback: reportar progresso durante classificação

**Dados que o RunPod envia para a VPS** (nos chunks):

```python
# Cada chunk já deve chegar na VPS com esses campos preenchidos:
chunk.origin_type            # "self" | "external"
chunk.origin_reference       # "DL-2848-1940" | ""
chunk.origin_reference_name  # "Código Penal" | ""
chunk.is_external_material   # True | False
chunk.origin_confidence      # "high" | "medium" | "low"
chunk.origin_reason          # "trigger_phrase + target_id:DL-2848-1940 + ..."
```

A VPS não precisa saber como esses campos foram calculados — só recebe e persiste.

---

### 18.2 VPS — Integração e Uso em Query-Time (20% do trabalho)

**Responsabilidade**: receber os campos origin_* dos chunks, persistir no Milvus/Neo4j, e usar em query-time.

**O que fazer**:

#### A) Milvus — Nenhuma alteração de schema

Os 6 campos origin_* **já existem** no schema `leis_v4`:

```
origin_type          VarChar(16)    ← já existe
origin_reference     VarChar(128)   ← já existe
origin_reference_name VarChar(128)  ← já existe
is_external_material Bool           ← já existe
origin_confidence    VarChar(8)     ← já existe
origin_reason        VarChar(256)   ← já existe
```

**Verificar**: que o código de insert no Milvus está passando esses campos dos chunks
recebidos do RunPod. Se hoje está hardcoded como `""` ou `"self"`, precisa ler do chunk.

#### B) Neo4j — Relação ORIGINATES_FROM (opcional, futuro)

Quando `origin_type = "external"` e `origin_reference != ""`:

```cypher
// Relação lógica: este chunk pertence a outra norma
MERGE (chunk:Chunk {node_id: $node_id})
MERGE (target:Document {document_id: $origin_reference})
CREATE (chunk)-[:ORIGINATES_FROM {
    confidence: $origin_confidence,
    inserted_by: $host_doc_id
}]->(target)
```

Isso permite queries no grafo como:
- "Quais dispositivos do Código Penal foram inseridos por outras leis?"
- "Quais leis alteram o CPC?"

**Prioridade**: baixa. Pode ser implementado depois do pipeline estar rodando.

#### C) Query-Time — Filtros e Ranking

**Filtro por proveniência** (já possível com campos existentes):

```python
# Retornar só conteúdo próprio da lei (excluir material externo)
expr = 'is_external_material == false'

# Retornar só material externo de alta confiança
expr = 'is_external_material == true and origin_confidence == "high"'
```

**Ajuste de ranking** (futuro):

```python
# Reduzir score de chunks externos quando a query é sobre a lei hospedeira
if query_targets_specific_law and chunk.is_external_material:
    chunk.score *= 0.7  # penalidade de 30%
```

#### D) snippet_extractor — Informação de proveniência na resposta

Quando o snippet_extractor monta o Evidence Drawer, incluir a proveniência:

```python
# Se chunk é external, adicionar nota na resposta
if chunk.origin_type == "external" and chunk.origin_reference_name:
    attribution = (
        f"Dispositivo do {chunk.origin_reference_name} "
        f"(inserido pela {host_doc_name})"
    )
else:
    attribution = host_doc_name
```

Resultado para o usuário:
- self: "Art. 23 da Lei 14.133/2021"
- external: "Art. 337-E do Código Penal (inserido pela Lei 14.133/2021)"

**Prioridade**: média. Funciona sem isso, mas a UX melhora significativamente.

#### E) alarm_service — Alertas de qualidade

Logar quando:
- Chunk chega com `origin_confidence = "low"` → possível classificação incorreta
- Chunk chega com `origin_type = "external"` e `origin_reference = ""` → norma-alvo não identificada
- Documento tem > 30% de chunks `external` → possível consolidação ou documento atípico

**Prioridade**: baixa. Nice-to-have para monitoramento.

---

### 18.3 Resumo da Divisão

```
RunPod (IMPLEMENTAR)                    VPS (INTEGRAR)
─────────────────────                   ─────────────────────
origin_classifier.py                    Insert no Milvus: ler campos do chunk
  ├─ compute_enter_score()              Neo4j: relação ORIGINATES_FROM (futuro)
  ├─ compute_exit_score()               Query-time: filtro is_external_material
  ├─ resolve_reference()                snippet_extractor: atribuição na resposta
  ├─ classify_document()                alarm_service: alertas de qualidade
  └─ assign_origin()
Testes unitários                        Verificar que insert não descarta campos
Integração no Inspector
Integração no pipeline (pós-Reconciliator)
```

**Dependência**: VPS depende do RunPod entregar chunks com campos preenchidos.
Enquanto o RunPod não implementar, a VPS pode continuar inserindo os defaults
(`origin_type = "self"`, demais campos vazios).

---

## 19. Dependências e Performance

- **Dependências**: `re` (stdlib), zero pacotes externos
- **Performance**: O(n) no número de chunks, O(1) por chunk (regex sobre janela fixa)
- **Roda em**: RunPod CPU, junto com Reconciliator e IntegrityValidator
- **Determinístico**: mesma entrada → mesma saída, sempre
- **Testável**: unit tests com strings fixas, sem mocks
