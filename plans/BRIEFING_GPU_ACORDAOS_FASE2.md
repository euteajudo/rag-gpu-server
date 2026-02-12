# BRIEFING GPU SERVER — Pipeline de Ingestão de Acórdãos (Fase 2)

**Para:** Claude responsável pelo código do GPU Server (RunPod)  
**De:** Arquitetura de produto VectorGov  
**Data:** 2026-02-11  
**Prioridade:** Alta

---

## Visão Geral

Criar pipeline de ingestão para acórdãos do TCU. O pipeline será **duplicado do pipeline de leis (entrada 1: PyMuPDF + Regex)** e adaptado para a estrutura de acórdãos. Todas as medidas de segurança, guardrails e controles de qualidade do pipeline de leis devem estar presentes.

O pipeline antigo de acórdãos será **deletado** — começamos do zero.

---

## Arquitetura do Pipeline

Duas entradas, mesma lógica do pipeline de leis:

```
ENTRADA 1: PyMuPDF + Regex (preferencial)
PDF → PyMuPDF (extração de texto) → AcordaoParser (regex) → SpanBuilder 
→ EmbeddingService (BGE-M3) → CitationExtractor → OriginClassifier 
→ RetrievalTextBuilder → Output JSON

ENTRADA 2: Qwen3 VL + Regex (fallback para PDFs escaneados)
PDF → Qwen3 VL (OCR/extração) → AcordaoParser (regex) → SpanBuilder
→ EmbeddingService (BGE-M3) → CitationExtractor → OriginClassifier
→ RetrievalTextBuilder → Output JSON
```

O frontend terá switches para escolher qual entrada usar (mesma lógica dos switches de leis).

---

## Estrutura de um Acórdão do TCU

Todo acórdão do TCU segue esta estrutura fixa de alto nível:

```
ACÓRDÃO {numero}/{ano} – TCU – {colegiado}
├── CABEÇALHO (metadados estruturados)
│   ├── Processo: TC xxx.xxx/xxxx-x
│   ├── Classe de Assunto: VII – Representação
│   ├── Unidade: ...
│   ├── Relator: Ministro ...
│   ├── Unidade Técnica: ...
│   └── SUMÁRIO: ...
│
├── RELATÓRIO (instrução da unidade técnica — OPINATIVO)
│   ├── [Seções variáveis — detectar por heading]
│   ├── Ex: INTRODUÇÃO
│   ├── Ex: EXAME DE ADMISSIBILIDADE
│   ├── Ex: EXAME TÉCNICO
│   │   ├── I. Análise dos pressupostos...
│   │   │   ├── I.1. Perigo da demora
│   │   │   ├── I.2. Perigo da demora reverso
│   │   │   └── I.3. Plausibilidade jurídica
│   │   │       └── I.3.1. Ausência de justificativas no ETP
│   │   └── Demais alegações
│   ├── CONCLUSÃO
│   └── PROPOSTA DE ENCAMINHAMENTO
│
├── VOTO (raciocínio do relator — FUNDAMENTAÇÃO)
│   └── [Parágrafos numerados]
│
└── ACÓRDÃO (dispositivo — VINCULANTE)
    ├── 9.1. conhecer da representação...
    ├── 9.2. indeferir...
    ├── 9.3. no mérito...
    ├── 9.4. dar ciência...
    │   ├── 9.4.1. ausência no ETP...
    │   ├── 9.4.2. vedação à subcontratação...
    │   └── 9.4.3. ausência de análise de custo...
    ├── 9.5. comunicar...
    └── 9.6. arquivar...
```

### REGRA CRÍTICA: O RELATÓRIO tem estrutura variável

As seções internas do Relatório **mudam de acórdão para acórdão** dependendo do tipo de processo (representação, auditoria, tomada de contas, consulta, etc.). O parser **NÃO deve hardcodar** nomes de seções. Deve detectar headings dinamicamente.

### REGRA CRÍTICA: Peso jurídico por seção

| Seção | `authority_level` | Significado |
|-------|------------------|-------------|
| ACÓRDÃO (dispositivo) | `vinculante` | Decisão dos ministros — o que vale juridicamente |
| VOTO | `fundamentacao` | Raciocínio do relator — forma jurisprudência |
| RELATÓRIO | `opinativo` | Análise da unidade técnica — ministro pode acolher ou rejeitar |

---

## Componente 1: AcordaoParser

### Função

Extrai do texto bruto do PDF a estrutura hierárquica do acórdão: metadados do cabeçalho, seções do relatório, parágrafos do voto, itens do dispositivo.

### Extração de metadados do cabeçalho

```python
class AcordaoHeaderParser:
    """
    Extrai metadados estruturados do cabeçalho do acórdão.
    """
    
    def parse_header(self, text: str) -> dict:
        """
        Retorna:
            numero: int (ex: 2450)
            ano: int (ex: 2025)
            colegiado: str ("Plenario" | "1a_Camara" | "2a_Camara")
            processo: str (ex: "TC 018.677/2025-8")
            natureza: str (ex: "Representação")
            unidade: str (ex: "Instituto Federal de...")
            relator: str (ex: "Jorge Oliveira")
            unidade_tecnica: str (ex: "AudContratações")
            data_sessao: str (ex: "2025-10-22")
            sumario: str (texto do SUMÁRIO)
            resultado: str (ex: "Parcialmente procedente")
        """
        # Regex para número e ano do acórdão:
        # "ACÓRDÃO Nº 2450/2025" ou "ACÓRDÃO N° 2450/2025"
        acordao_match = re.search(
            r'AC[OÓ]RD[AÃ]O\s+(?:N[°ºo.]?\s*)?(\d+)/(\d{4})',
            text, re.IGNORECASE
        )
        
        # Regex para colegiado:
        # "Plenário", "1ª Câmara", "2ª Câmara", "Primeira Câmara", "Segunda Câmara"
        colegiado_match = re.search(
            r'(Plen[aá]rio|1[ªa]\s*C[aâ]mara|2[ªa]\s*C[aâ]mara|'
            r'Primeira\s+C[aâ]mara|Segunda\s+C[aâ]mara)',
            text, re.IGNORECASE
        )
        
        # Regex para processo:
        # "TC 018.677/2025-8"
        processo_match = re.search(
            r'TC\s+(\d{3}\.\d{3}/\d{4}-\d)',
            text
        )
        
        # Regex para natureza:
        # "Natureza: Representação"
        natureza_match = re.search(
            r'Natureza:\s*(.+?)(?:\n|$)',
            text
        )
        
        # Regex para relator:
        # "Relator: Ministro Jorge Oliveira" ou "5. Relator: Ministro Jorge Oliveira"
        relator_match = re.search(
            r'Relator:\s*(?:Ministro\s+)?(.+?)(?:\n|$)',
            text
        )
        
        # Regex para data da sessão:
        # "Data da Sessão: 22/10/2025" ou "em 22 de outubro de 2025"
        data_match = re.search(
            r'Data\s+da\s+Sess[aã]o:\s*(\d{1,2}/\d{1,2}/\d{4})',
            text
        )
        
        # Regex para resultado (no corpo do ACÓRDÃO):
        # "considerar a representação parcialmente procedente"
        # "considerar procedente"
        # "considerar improcedente"
        resultado_match = re.search(
            r'considerar\s+(?:a\s+\w+\s+)?'
            r'(parcialmente\s+procedente|procedente|improcedente)',
            text, re.IGNORECASE
        )
        
        # ... montar dict com valores extraídos
```

### Detecção dinâmica de seções

```python
class SectionDetector:
    """
    Detecta seções do acórdão por padrões de heading.
    NÃO hardcoda nomes — detecta dinamicamente.
    """
    
    # Seções primárias (sempre presentes)
    PRIMARY_SECTIONS = [
        (r'^\s*RELAT[OÓ]RIO\s*$', 'RELATORIO'),
        (r'^\s*VOTO\s*$', 'VOTO'),
        (r'^\s*AC[OÓ]RD[AÃ]O\s+(?:N[°ºo.]?\s*)?\d+', 'ACORDAO'),
    ]
    
    # Headings de seções dentro do RELATÓRIO (detectados dinamicamente)
    SUBSECTION_PATTERNS = [
        # Headings em caixa alta: "EXAME TÉCNICO", "CONCLUSÃO", "INTRODUÇÃO"
        (r'^([A-ZÁÀÂÃÉÊÍÓÔÕÚÇ\s]{4,})$', 'heading_upper'),
        
        # Numeração romana com título: "I. Análise dos pressupostos"
        (r'^([IVX]+)\.\s+(.+)', 'heading_roman'),
        
        # Sub-numeração: "I.1. Perigo da demora", "I.3.1. Ausência de..."
        (r'^([IVX]+\.\d+(?:\.\d+)?)\.\s+(.+)', 'heading_sub'),
        
        # Headings em negrito/sublinhado (detectados por formatação)
        (r'^_+(.+?)_+$', 'heading_underlined'),
        
        # "Demais alegações do representante" (ad hoc)
        (r'^(Demais\s+alegações.*)$', 'heading_adhoc'),
    ]
    
    # Itens do dispositivo do ACÓRDÃO
    ACORDAO_ITEM_PATTERN = re.compile(
        r'^(\d+\.\d+(?:\.\d+)?)\.\s+(.+)', re.MULTILINE
    )
    
    # Parágrafos numerados (relatório e voto)
    PARAGRAPH_PATTERN = re.compile(
        r'^(\d{1,3})\.\s+(.+?)(?=\n\d{1,3}\.\s|\Z)', re.MULTILINE | re.DOTALL
    )
```

### Geração de span_ids

```
Convenção para acórdãos:

Seção primária:
  SEC-RELATORIO                    → Seção "Relatório" inteira
  SEC-VOTO                         → Seção "Voto" inteira  
  SEC-ACORDAO                      → Seção "Acórdão" inteira

Sub-seções do relatório (dinâmicas):
  SEC-RELATORIO-INTRODUCAO         → Subseção "Introdução"
  SEC-RELATORIO-EXAME-TECNICO      → Subseção "Exame Técnico"
  SEC-RELATORIO-I.3.1              → Subseção numerada "I.3.1"
  SEC-RELATORIO-CONCLUSAO          → Subseção "Conclusão"

Parágrafos:
  PAR-RELATORIO-22                 → § 22 do Relatório
  PAR-VOTO-7                       → § 7 do Voto

Itens do dispositivo:
  ITEM-9.1                         → Item 9.1 do Acórdão
  ITEM-9.4                         → Item 9.4 (pai)
  ITEM-9.4.1                       → Sub-item 9.4.1
  ITEM-9.4.2                       → Sub-item 9.4.2
```

### Hierarquia pai-filho

```
ACORDAO-2450-2025-P (document_id)
├── SEC-RELATORIO (section)
│   ├── SEC-RELATORIO-INTRODUCAO (section)
│   │   ├── PAR-RELATORIO-1 (paragraph)
│   │   ├── PAR-RELATORIO-2 (paragraph)
│   │   └── PAR-RELATORIO-7 (paragraph)
│   ├── SEC-RELATORIO-EXAME-ADMISSIBILIDADE (section)
│   │   └── PAR-RELATORIO-8 ... PAR-RELATORIO-12
│   ├── SEC-RELATORIO-EXAME-TECNICO (section)
│   │   ├── SEC-RELATORIO-I.1 (section)
│   │   │   └── PAR-RELATORIO-15 (paragraph)
│   │   ├── SEC-RELATORIO-I.3.1 (section)
│   │   │   └── PAR-RELATORIO-18 ... PAR-RELATORIO-22
│   │   └── ...
│   ├── SEC-RELATORIO-CONCLUSAO (section)
│   └── SEC-RELATORIO-PROPOSTA (section)
├── SEC-VOTO (section)
│   ├── PAR-VOTO-1 (paragraph)
│   └── PAR-VOTO-25 (paragraph)
└── SEC-ACORDAO (section)
    ├── ITEM-9.1 (item)
    ├── ITEM-9.2 (item)
    ├── ITEM-9.4 (item — pai)
    │   ├── ITEM-9.4.1 (item)
    │   ├── ITEM-9.4.2 (item)
    │   └── ITEM-9.4.3 (item)
    └── ITEM-9.6 (item)
```

---

## Componente 2: RetrievalTextBuilder para Acórdãos

### Regra de enriquecimento

Cada chunk recebe um retrieval_text que inclui:
1. **Identificação do acórdão** (número, colegiado, relator)
2. **Natureza do trecho** (vinculante/fundamentação/opinativo)
3. **Caminho hierárquico** (section_path)
4. **Resumo do contexto** (sumário do acórdão)
5. **Texto do chunk**

### Exemplos por tipo

**Chunk de ITEM DO ACÓRDÃO (vinculante):**
```
DECISÃO VINCULANTE – Acórdão 2450/2025 – TCU – Plenário.
Relator: Min. Jorge Oliveira. Natureza: Representação. 
Resultado: Parcialmente procedente.
Processo: TC 018.677/2025-8.
Assunto: Pregão Eletrônico – Serviços contínuos de locação de veículos.
Dispositivo 9.4.1:
Ausência, no Estudo Técnico Preliminar, das estimativas de quantitativos, 
para cada campus, acompanhadas das respectivas memórias de cálculo e dos 
documentos de suporte, em afronta ao art. 18, § 1º, inciso IV, da Lei 
14.133/2021, o que compromete a rastreabilidade e a transparência do 
planejamento da contratação.
```

**Chunk de PARÁGRAFO DO VOTO (fundamentação):**
```
FUNDAMENTAÇÃO DO RELATOR – Acórdão 2450/2025 – TCU – Plenário.
Relator: Min. Jorge Oliveira. Natureza: Representação.
Assunto: Pregão Eletrônico – Serviços contínuos de locação de veículos.
Seção: Voto, § 10.
Por conseguinte, o ETP também não apresentou as estimativas de quantitativos 
por campus acompanhadas das respectivas memórias de cálculo e documentos 
de suporte, em descumprimento ao art. 18, § 1º, IV, da Lei 14.133/2021.
```

**Chunk de PARÁGRAFO DO RELATÓRIO (opinativo):**
```
ANÁLISE DA UNIDADE TÉCNICA – Acórdão 2450/2025 – TCU – Plenário.
Unidade técnica: AudContratações. Natureza: Representação.
Assunto: Pregão Eletrônico – Serviços contínuos de locação de veículos.
Seção: Relatório > Exame Técnico > I.3.1 Ausência de justificativas no ETP, § 22.
Todavia, constata-se que o Estudo Técnico Preliminar não apresentou as 
estimativas de quantitativos, para cada campus, acompanhadas das memórias 
de cálculo e dos documentos de suporte, em afronta ao art. 18, § 1º, 
inciso IV, da Lei 14.133/2021.
```

**Chunk de SEÇÃO COMPOSTA (composite):**
```
ANÁLISE DA UNIDADE TÉCNICA – Acórdão 2450/2025 – TCU – Plenário.
Seção: Relatório > Exame Técnico > I.3.1 Ausência de justificativas no ETP.
§ 18. A primeira irregularidade apontada pelo representante refere-se à 
estimativa de quantitativos sem lastro técnico...
§ 19. Nos termos do art. 18, § 1º, inciso IV, da Lei 14.133/2021...
§ 20. No caso concreto, observa-se...
§ 21. Ainda, por ser um registro de preços...
§ 22. Todavia, constata-se que o Estudo Técnico Preliminar não apresentou...
```

---

## Componente 3: CitationExtractor para Acórdãos

### Citações a extrair

Os acórdãos citam massivamente dispositivos legais. Exemplos deste acórdão:

```
"art. 18, § 1º, inciso IV, da Lei 14.133/2021"  → LEI-14.133-2021#PAR-018-1, INC-018-4
"art. 11, inc. I, da Lei 14.133/2021"            → LEI-14.133-2021#INC-011-1
"art. 5º da Lei 14.133/2021"                     → LEI-14.133-2021#ART-005
"art. 122, § 2º"                                 → LEI-14.133-2021#PAR-122-2
"art. 170, § 4º, da Lei 14.133/2021"             → LEI-14.133-2021#PAR-170-4
"art. 276 do Regimento Interno/TCU"              → (referência externa — registrar mas não linkar)
"art. 103, § 1º, da Resolução - TCU 259/2014"   → (referência externa — registrar mas não linkar)
```

Também citam outros acórdãos:
```
"Acórdão 1234/2023 – Plenário"  → ACORDAO-1234-2023-P (se estiver na base)
```

### Output do CitationExtractor

```json
{
  "citations": [
    {
      "raw_text": "art. 18, § 1º, inciso IV, da Lei 14.133/2021",
      "target_document_id": "LEI-14.133-2021",
      "target_span_id": "INC-018-4",
      "relationship": "INTERPRETA",
      "confidence": 0.95
    },
    {
      "raw_text": "art. 276 do Regimento Interno/TCU",
      "target_document_id": null,
      "target_span_id": null,
      "relationship": "CITA_EXTERNA",
      "confidence": 0.80
    }
  ]
}
```

As citações vão para o **Neo4j** como arestas, NÃO ficam no Milvus.

---

## Componente 4: OriginClassifier para Acórdãos

Nos acórdãos, o OriginClassifier é **mais simples** que nas leis. Um acórdão é sempre produção própria do TCU — não tem o problema de material externo embutido como a Lei 14.133 tem com o Código Penal.

No entanto, o acórdão pode **citar trechos de leis literalmente** (transcrevendo). Nesses casos:

```python
origin_type = "self"           # O acórdão é do TCU
# Mesmo quando transcreve um artigo de lei, é o TCU citando
# A citação vai para Neo4j, não muda o origin_type
```

O OriginClassifier para acórdãos pode ser uma versão simplificada. O que importa é que **o campo existe e está preenchido** para manter compatibilidade com o schema de leis.

---

## Componente 5: Offsets canônicos

Mesma lógica do pipeline de leis:

1. **Texto canônico** = texto integral do PDF extraído (PyMuPDF ou Qwen3 VL)
2. Cada chunk registra `canonical_start` e `canonical_end` (posição no texto canônico)
3. Hash SHA-256 do texto canônico salvo como `canonical_hash`
4. Arquivo `offsets.json` gerado e enviado ao MinIO junto com o manifesto

---

## Guardrails e Medidas de Segurança

**TODAS as medidas do pipeline de leis devem estar presentes.** Especificamente:

### Do GPU Server:

| Guardrail | Descrição | Presente em leis? |
|-----------|-----------|-------------------|
| Regex de artigos com sufixo | Captura Art. 337-E etc. | ✅ Sim — mas não se aplica a acórdãos (não tem artigos com sufixo). Manter regex de items: `9.4.1` |
| OriginClassifier | Classifica origem do material | ✅ Sim — versão simplificada para acórdãos |
| Offsets canônicos | canonical_start/end para cada chunk | ✅ Sim |
| Manifesto de ingestão | Resumo com contagens, hashes, external_material | ✅ Sim |
| Hash do documento fonte | SHA-256 do PDF | ✅ Sim |
| Validação de spans | Nenhum span com texto vazio | ✅ Sim |
| Validação de hierarquia | Todo filho tem pai válido | ✅ Sim |
| Validação de embeddings | Nenhum vetor zerado | ✅ Sim |
| enrichment_text nunca vazio | retrieval_text sempre preenchido | ✅ Sim |

### Manifesto de ingestão (output)

```json
{
  "document_id": "ACORDAO-2450-2025-P",
  "document_type": "acordao",
  "source_hash": "sha256:...",
  "canonical_hash": "sha256:...",
  "total_spans": 85,
  "span_counts": {
    "section": 12,
    "paragraph": 58,
    "item_dispositivo": 8,
    "ementa": 1
  },
  "section_types": {
    "relatorio": 45,
    "voto": 25,
    "acordao": 8,
    "ementa": 1
  },
  "authority_levels": {
    "vinculante": 8,
    "fundamentacao": 25,
    "opinativo": 45,
    "metadado": 1
  },
  "metadata": {
    "numero": 2450,
    "ano": 2025,
    "colegiado": "Plenario",
    "processo": "TC 018.677/2025-8",
    "relator": "Jorge Oliveira",
    "natureza": "Representação",
    "resultado": "Parcialmente procedente",
    "data_sessao": "2025-10-22"
  },
  "citations": {
    "total": 15,
    "to_known_documents": 10,
    "to_external": 5
  },
  "offsets_coverage": 0.94,
  "embedding_model": "BAAI/bge-m3",
  "extraction_method": "pymupdf",
  "pipeline_version": "acordaos_v2",
  "timestamp": "2025-02-11T..."
}
```

---

## Contrato GPU ↔ VPS para Acórdãos

### Endpoint de ingestão

```
POST /api/v1/ingest/acordao
Content-Type: application/json

{
  "document_id": "ACORDAO-2450-2025-P",
  "document_type": "acordao",
  "source_hash": "sha256:...",
  "canonical_hash": "sha256:...",
  "extraction_method": "pymupdf",
  "pipeline_version": "acordaos_v2",
  
  "metadata": {
    "numero": 2450,
    "ano": 2025,
    "colegiado": "Plenario",
    "processo": "TC 018.677/2025-8",
    "relator": "Jorge Oliveira",
    "unidade_tecnica": "AudContratações",
    "natureza": "Representação",
    "resultado": "Parcialmente procedente",
    "data_sessao": "2025-10-22",
    "sumario": "REPRESENTAÇÃO. PREGÃO ELETRÔNICO..."
  },
  
  "spans": [
    {
      "span_id": "ITEM-9.4.1",
      "node_id": "acordaos:ACORDAO-2450-2025-P#ITEM-9.4.1",
      "parent_node_id": "acordaos:ACORDAO-2450-2025-P#ITEM-9.4",
      "device_type": "item_dispositivo",
      "chunk_level": "device",
      "section_type": "acordao",
      "section_path": "ACORDAO > 9.4 > 9.4.1",
      "authority_level": "vinculante",
      "text": "ausência, no Estudo Técnico Preliminar, das estimativas de quantitativos...",
      "retrieval_text": "DECISÃO VINCULANTE – Acórdão 2450/2025 – TCU – Plenário...",
      "dense_vector": [0.023, -0.045, ...],
      "sparse_vector": {"indices": [...], "values": [...]},
      "canonical_start": 28456,
      "canonical_end": 28890,
      "canonical_hash": "sha256:...",
      "page_number": 11,
      
      "origin_type": "self",
      "origin_reference": null,
      "origin_reference_name": null,
      "is_external_material": false
    }
  ],
  
  "citations": [
    {
      "source_span_id": "ITEM-9.4.1",
      "target_document_id": "LEI-14.133-2021",
      "target_span_id": "INC-018-4",
      "relationship": "INTERPRETA",
      "raw_text": "art. 18, § 1º, inciso IV, da Lei 14.133/2021",
      "confidence": 0.95
    }
  ],
  
  "offsets": {
    "ITEM-9.4.1": {"start": 28456, "end": 28890},
    "PAR-RELATORIO-22": {"start": 12340, "end": 12780}
  }
}
```

### Response da VPS

```json
{
  "status": "accepted",
  "document_id": "ACORDAO-2450-2025-P",
  "task_id": "uuid-...",
  "spans_received": 85,
  "citations_received": 15,
  "gates": {
    "gate_a_text_fidelity": "pending",
    "gate_b_hierarchy": "pending",
    "gate_c_reconciliation": "pending"
  }
}
```

---

## Testes Obrigatórios (GPU)

Cada teste deve passar antes de enviar para a VPS.

### T1: Extração de metadados
```python
def test_header_extraction():
    result = parser.parse(pdf_text)
    assert result["numero"] == 2450
    assert result["ano"] == 2025
    assert result["colegiado"] == "Plenario"
    assert result["processo"] == "TC 018.677/2025-8"
    assert result["relator"] == "Jorge Oliveira"
    assert result["natureza"] == "Representação"
    assert result["resultado"] == "Parcialmente procedente"
    assert result["data_sessao"] == "2025-10-22"
```

### T2: Detecção de seções primárias
```python
def test_primary_sections():
    sections = parser.detect_sections(pdf_text)
    primary = [s["type"] for s in sections if s["level"] == "primary"]
    assert "RELATORIO" in primary
    assert "VOTO" in primary
    assert "ACORDAO" in primary
```

### T3: Itens do dispositivo
```python
def test_acordao_items():
    items = parser.extract_acordao_items(acordao_section)
    item_ids = [i["span_id"] for i in items]
    assert "ITEM-9.1" in item_ids
    assert "ITEM-9.4" in item_ids
    assert "ITEM-9.4.1" in item_ids
    assert "ITEM-9.4.2" in item_ids
    assert "ITEM-9.4.3" in item_ids
    # Hierarquia
    item_9_4_1 = next(i for i in items if i["span_id"] == "ITEM-9.4.1")
    assert item_9_4_1["parent_span_id"] == "ITEM-9.4"
```

### T4: Authority levels
```python
def test_authority_levels():
    spans = parser.parse_all(pdf_text)
    for span in spans:
        if span["section_type"] == "acordao":
            assert span["authority_level"] == "vinculante"
        elif span["section_type"] == "voto":
            assert span["authority_level"] == "fundamentacao"
        elif span["section_type"] == "relatorio":
            assert span["authority_level"] == "opinativo"
```

### T5: Nenhum span com texto vazio
```python
def test_no_empty_text():
    spans = parser.parse_all(pdf_text)
    for span in spans:
        assert len(span["text"].strip()) > 0
        assert len(span["retrieval_text"].strip()) > 0
```

### T6: Nenhum vetor zerado
```python
def test_no_zero_vectors():
    spans = parser.parse_all(pdf_text)
    for span in spans:
        assert any(v != 0.0 for v in span["dense_vector"])
```

### T7: Hierarquia válida
```python
def test_hierarchy():
    spans = parser.parse_all(pdf_text)
    node_ids = {s["node_id"] for s in spans}
    for span in spans:
        if span["parent_node_id"]:
            # Todo pai referenciado deve existir (ou ser o document root)
            assert span["parent_node_id"] in node_ids or \
                   span["parent_node_id"].endswith(span["document_id"])
```

### T8: Offsets canônicos contíguos
```python
def test_offsets_valid():
    spans = parser.parse_all(pdf_text)
    for span in spans:
        if span["canonical_start"] is not None:
            assert span["canonical_end"] > span["canonical_start"]
            assert span["canonical_start"] >= 0
```

### T9: Citações extraídas
```python
def test_citations():
    citations = extractor.extract(pdf_text)
    # Deve encontrar art. 18 da Lei 14.133
    lei_citations = [c for c in citations if c["target_document_id"] == "LEI-14.133-2021"]
    assert len(lei_citations) >= 5  # Acórdão 2450 cita vários artigos
```

### T10: Retrieval text enriquecido
```python
def test_retrieval_text_enrichment():
    spans = parser.parse_all(pdf_text)
    item_941 = next(s for s in spans if s["span_id"] == "ITEM-9.4.1")
    assert "DECISÃO VINCULANTE" in item_941["retrieval_text"]
    assert "Acórdão 2450/2025" in item_941["retrieval_text"]
    assert "Plenário" in item_941["retrieval_text"]
    
    par_voto = next(s for s in spans if s["span_id"] == "PAR-VOTO-7")
    assert "FUNDAMENTAÇÃO DO RELATOR" in par_voto["retrieval_text"]
    
    par_rel = next(s for s in spans if s["span_id"] == "PAR-RELATORIO-22")
    assert "ANÁLISE DA UNIDADE TÉCNICA" in par_rel["retrieval_text"]
```

---

## Arquivos a criar (GPU Server)

```
src/parsers/acordao_parser.py          ← AcordaoParser + SectionDetector
src/parsers/acordao_header_parser.py   ← Extração de metadados do cabeçalho
src/builders/acordao_span_builder.py   ← Monta span_ids, node_ids, hierarquia
src/builders/acordao_retrieval_text.py ← RetrievalTextBuilder para acórdãos
src/classification/acordao_origin.py   ← OriginClassifier simplificado
src/pipeline/acordao_pipeline.py       ← Orquestra o fluxo completo
tests/test_acordao_parser.py           ← T1-T10
```

O pipeline antigo de acórdãos deve ser **deletado** — não adaptar, não refatorar. Começar do zero usando a arquitetura do pipeline de leis como base.

---

## Resumo de prioridades

| # | Ação | Prioridade |
|---|------|------------|
| 1 | Deletar pipeline antigo de acórdãos | 🔴 Primeiro |
| 2 | Implementar AcordaoParser (regex de seções + itens) | 🔴 Alta |
| 3 | Implementar AcordaoHeaderParser (metadados) | 🔴 Alta |
| 4 | Implementar RetrievalTextBuilder para acórdãos | 🔴 Alta |
| 5 | Implementar CitationExtractor (citações a leis) | 🟡 Média |
| 6 | Implementar offsets canônicos | 🟡 Média |
| 7 | Implementar testes T1-T10 | 🔴 Alta |
| 8 | Gerar manifesto de ingestão | 🟡 Média |
| 9 | Endpoint /ingest/acordao no FastAPI | 🔴 Alta |
