"""
Prompts para Enriquecimento de Chunks com LLM.

===========================================================================
DEPRECATED / CÓDIGO LEGADO - NÃO USAR EM CÓDIGO NOVO
===========================================================================
Este módulo foi descontinuado em favor do retrieval_text determinístico.

Motivo da descontinuação:
- Custo: chamadas LLM por chunk
- Não-determinístico: resultados variam entre execuções
- Complexidade desnecessária para o caso de uso atual

Estratégia atual:
- retrieval_text determinístico em chunk_materializer.build_retrieval_text()
- Sem dependência de LLM para gerar embeddings

Este arquivo será removido em versão futura.
===========================================================================

Implementava a técnica de Contextual Retrieval da Anthropic,
enriquecendo chunks de documentos legais com metadados gerados por LLM.

Referência: https://www.anthropic.com/news/contextual-retrieval

Arquitetura de Enriquecimento:
=============================

    ┌─────────────────────────────────────────────────────────────────────┐
    │                    PIPELINE DE ENRIQUECIMENTO                       │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │  LegalChunk (texto)                                                 │
    │        │                                                            │
    │        ▼                                                            │
    │  ┌──────────────────────────────────────────────────────────────┐  │
    │  │  build_enrichment_prompt()                                    │  │
    │  │  ├── Monta contexto do documento                              │  │
    │  │  ├── Inclui hierarquia (cap, art)                             │  │
    │  │  └── Gera (system_prompt, user_prompt)                        │  │
    │  └──────────────────────────────────────────────────────────────┘  │
    │        │                                                            │
    │        ▼                                                            │
    │  ┌──────────────────────────────────────────────────────────────┐  │
    │  │  LLM (Qwen 3 8B)                                              │  │
    │  │  └── Gera JSON com metadados                                  │  │
    │  └──────────────────────────────────────────────────────────────┘  │
    │        │                                                            │
    │        ▼                                                            │
    │  ┌──────────────────────────────────────────────────────────────┐  │
    │  │  parse_enrichment_response()                                  │  │
    │  │  ├── Extrai JSON da resposta                                  │  │
    │  │  ├── Valida campos obrigatórios                               │  │
    │  │  └── Normaliza tipos                                          │  │
    │  └──────────────────────────────────────────────────────────────┘  │
    │        │                                                            │
    │        ▼                                                            │
    │  ┌──────────────────────────────────────────────────────────────┐  │
    │  │  build_enriched_text()                                        │  │
    │  │  └── [CONTEXTO] + texto + perguntas                           │  │
    │  └──────────────────────────────────────────────────────────────┘  │
    │        │                                                            │
    │        ▼                                                            │
    │  LegalChunk (enriquecido)                                          │
    │  ├── context_header: "Este artigo da IN 58..."                     │
    │  ├── thesis_text: "Estabelece definições..."                       │
    │  ├── thesis_type: "definicao"                                      │
    │  ├── synthetic_questions: "O que é ETP?\n..."                      │
    │  └── enriched_text: "[CONTEXTO:...] Art.3..."                      │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘

Campos Gerados pelo LLM:
=======================

    | Campo               | Max Chars | Descrição                              |
    |---------------------|-----------|----------------------------------------|
    | context_header      | 200       | Frase contextualizando o dispositivo   |
    | thesis_text         | 500       | Resumo do que o dispositivo determina  |
    | thesis_type         | -         | Classificação do conteúdo (enum)       |
    | synthetic_questions | 5 itens   | Perguntas práticas que o chunk responde|

Tipos de Tese (thesis_type):
===========================

    | Tipo        | Gatilhos no Texto                | Exemplo                      |
    |-------------|----------------------------------|------------------------------|
    | definicao   | "considera-se", "define-se"      | Conceitos e termos           |
    | procedimento| "o processo", "etapas", "deverá" | Como fazer algo              |
    | prazo       | "prazo de", "em até", "dias"     | Limites temporais            |
    | requisito   | "requisitos", "condições"        | Condições obrigatórias       |
    | competencia | "compete ao", "atribuições"      | Quem pode/deve fazer         |
    | vedacao     | "é vedado", "não poderá"         | Proibições                   |
    | excecao     | "exceto", "salvo", "dispensado"  | Casos especiais              |
    | sancao      | "multa", "penalidade"            | Consequências                |
    | disposicao  | (default)                        | Outros                       |

Modos de Processamento:
======================

    1. INDIVIDUAL (padrão)
       - build_enrichment_prompt() para um chunk
       - Maior qualidade, mais tokens

    2. BATCH (economia de tokens)
       - build_batch_enrichment_prompt() para múltiplos chunks
       - 3-5 chunks por batch
       - ~40% economia de tokens

Exemplo de enriched_text Gerado:
===============================

    ```
    [CONTEXTO: Este artigo da IN 58/2022 define os conceitos básicos
    para elaboração de ETP no âmbito federal]

    Art. 3º Para fins do disposto nesta Instrução Normativa, considera-se:
    I - Estudo Técnico Preliminar - ETP: documento constitutivo...
    II - Requisitante: agente público responsável...

    Perguntas que este trecho responde:
    - O que é ETP segundo a IN 58/2022?
    - Quem pode ser requisitante?
    - O requisitante pode acumular função de área técnica?
    - Qual é o papel do ETP no planejamento?
    - Quando o ETP é obrigatório?
    ```

Funções Disponíveis:
===================

    | Função                          | Descrição                              |
    |---------------------------------|----------------------------------------|
    | build_enrichment_prompt()       | Monta prompt para 1 chunk              |
    | build_batch_enrichment_prompt() | Monta prompt para N chunks             |
    | build_enriched_text()           | Monta texto enriquecido para embedding |
    | parse_enrichment_response()     | Parseia JSON de 1 chunk                |
    | parse_batch_enrichment_response()| Parseia JSON de N chunks              |

Prompts Disponíveis:
===================

    | Constante                       | Uso                                    |
    |---------------------------------|----------------------------------------|
    | ENRICHMENT_SYSTEM_PROMPT        | System prompt para enriquecimento      |
    | ENRICHMENT_USER_PROMPT          | User prompt (template) individual      |
    | BATCH_ENRICHMENT_SYSTEM_PROMPT  | System prompt para batch               |
    | BATCH_ENRICHMENT_USER_PROMPT    | User prompt (template) batch           |
    | CHUNK_TEMPLATE                  | Template de chunk dentro do batch      |
    | CLASSIFICATION_PROMPT           | Prompt leve só para thesis_type        |

Exemplo de Uso:
==============

    ```python
    from chunking.enrichment_prompts import (
        build_enrichment_prompt,
        parse_enrichment_response,
        build_enriched_text,
    )
    from llm import VLLMClient

    # 1. Preparar prompt
    system, user = build_enrichment_prompt(
        text="Art. 3º Para fins do disposto...",
        document_type="INSTRUÇÃO NORMATIVA",
        number="58",
        year=2022,
        issuing_body="SEGES/ME",
        chapter_number="I",
        chapter_title="DISPOSIÇÕES GERAIS",
        article_number="3",
    )

    # 2. Chamar LLM
    client = VLLMClient()
    response = client.chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
    )

    # 3. Parsear resposta
    enrichment = parse_enrichment_response(response)
    # {
    #     "context_header": "Este artigo da IN 58/2022 define...",
    #     "thesis_text": "Estabelece definições de termos...",
    #     "thesis_type": "definicao",
    #     "synthetic_questions": "O que é ETP?\\nQuem é requisitante?..."
    # }

    # 4. Montar texto enriquecido
    enriched = build_enriched_text(
        text="Art. 3º Para fins...",
        context_header=enrichment["context_header"],
        synthetic_questions=enrichment["synthetic_questions"].split("\\n"),
    )
    ```

Tratamento de Erros:
===================

    O parse_enrichment_response() trata:
    - Markdown code blocks (```json)
    - JSON embutido em texto
    - thesis_type inválido (normaliza para "disposicao")
    - synthetic_questions como lista ou string

    Levanta ValueError se:
    - Não encontrar JSON válido
    - Campo obrigatório ausente
    - Batch com número incorreto de itens

Integração com Outros Módulos:
=============================

    - law_chunker.py: Usa prompts para enriquecer LegalChunks
    - enrichment/chunk_enricher.py: ChunkEnricher usa estes prompts
    - llm/vllm_client.py: Cliente que executa os prompts

@author: Equipe VectorGov
@version: 1.0.0
@since: 21/12/2024
"""

# =============================================================================
# PROMPT DE ENRIQUECIMENTO PRINCIPAL
# =============================================================================

ENRICHMENT_SYSTEM_PROMPT = """Você é um especialista em direito administrativo brasileiro.
Sua tarefa é analisar dispositivos legais e gerar metadados estruturados para um sistema de busca semântica.

REGRAS:
1. Seja objetivo e técnico
2. Use linguagem clara, sem juridiquês desnecessário
3. Foque no conteúdo normativo, não em repetir o texto
4. Perguntas devem ser práticas, como um servidor público faria
5. Responda APENAS com JSON válido, sem explicações"""


ENRICHMENT_USER_PROMPT = """Analise este dispositivo legal e gere os metadados solicitados.

═══════════════════════════════════════════════════════════════════════════════
DOCUMENTO: {document_type} nº {number}/{year}
ÓRGÃO EMISSOR: {issuing_body}
CAPÍTULO: {chapter_number} - {chapter_title}
ARTIGO: {article_number}{article_title_suffix}
═══════════════════════════════════════════════════════════════════════════════

TEXTO DO DISPOSITIVO:
{text}

═══════════════════════════════════════════════════════════════════════════════

Gere um JSON com os seguintes campos:

{{
    "context_header": "Frase de 1-2 linhas contextualizando este dispositivo no documento (máx 200 caracteres)",
    "thesis_text": "Resumo objetivo do que este dispositivo determina ou define, sem repetir o texto literal (máx 500 caracteres)",
    "thesis_type": "escolha UM: definicao | procedimento | prazo | requisito | competencia | vedacao | excecao | sancao | disposicao",
    "synthetic_questions": [
        "Pergunta prática 1 que um servidor público faria",
        "Pergunta prática 2",
        "Pergunta prática 3",
        "Pergunta prática 4",
        "Pergunta prática 5"
    ]
}}

INSTRUÇÕES PARA CADA CAMPO:

• context_header: Situe o leitor. Ex: "Este artigo da IN 58/2022 define os conceitos básicos para elaboração de ETP no âmbito federal"

• thesis_text: Capture a essência normativa. NÃO repita o texto. Ex: "Estabelece que ETP, requisitante e área técnica são conceitos fundamentais, podendo ser exercidos pelo mesmo servidor"

• thesis_type:
  - definicao: Define conceitos, termos técnicos
  - procedimento: Estabelece como fazer algo, etapas
  - prazo: Define prazos, cronogramas
  - requisito: Estabelece condições, exigências
  - competencia: Define quem pode/deve fazer
  - vedacao: Proíbe algo, estabelece limitações
  - excecao: Estabelece exceções a regras
  - sancao: Define penalidades, consequências
  - disposicao: Disposições gerais que não se encaixam

• synthetic_questions: Perguntas que um servidor faria ao consultar a norma. Ex:
  - "O que é ETP segundo a IN 58?"
  - "Quem pode ser requisitante?"
  - "O requisitante pode ser também área técnica?"

Responda APENAS com o JSON, sem texto adicional."""


# =============================================================================
# PROMPT PARA BATCH (múltiplos chunks)
# =============================================================================

BATCH_ENRICHMENT_SYSTEM_PROMPT = """Você é um especialista em direito administrativo brasileiro.
Analise múltiplos dispositivos legais e gere metadados para cada um.
Responda com uma lista JSON contendo os metadados de cada dispositivo na ordem recebida."""


BATCH_ENRICHMENT_USER_PROMPT = """Analise os seguintes dispositivos legais e gere metadados para cada um.

═══════════════════════════════════════════════════════════════════════════════
DOCUMENTO: {document_type} nº {number}/{year}
ÓRGÃO EMISSOR: {issuing_body}
═══════════════════════════════════════════════════════════════════════════════

{chunks_text}

═══════════════════════════════════════════════════════════════════════════════

Para CADA dispositivo acima, gere um objeto JSON com:
- context_header (máx 200 chars)
- thesis_text (máx 500 chars)
- thesis_type (definicao|procedimento|prazo|requisito|competencia|vedacao|excecao|sancao|disposicao)
- synthetic_questions (lista de 5 perguntas práticas)

Responda com uma lista JSON na mesma ordem dos dispositivos:
[
    {{"context_header": "...", "thesis_text": "...", "thesis_type": "...", "synthetic_questions": [...]}},
    ...
]"""


# =============================================================================
# TEMPLATE PARA MONTAR CHUNK NO BATCH
# =============================================================================

CHUNK_TEMPLATE = """
---[ DISPOSITIVO {index} ]---
CAPÍTULO: {chapter_number} - {chapter_title}
ARTIGO: {article_number}{article_title_suffix}

{text}
"""


# =============================================================================
# PROMPT PARA CLASSIFICAÇÃO DE TIPO (mais leve)
# =============================================================================

CLASSIFICATION_PROMPT = """Classifique o tipo deste dispositivo legal:

TEXTO: {text}

Responda apenas com uma palavra:
definicao | procedimento | prazo | requisito | competencia | vedacao | excecao | sancao | disposicao"""


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def build_enrichment_prompt(
    text: str,
    document_type: str,
    number: str,
    year: int,
    issuing_body: str,
    chapter_number: str,
    chapter_title: str,
    article_number: str,
    article_title: str | None = None,
) -> tuple[str, str]:
    """
    Constrói prompt de enriquecimento para um chunk.

    Returns:
        Tuple (system_prompt, user_prompt)
    """
    article_title_suffix = f" - {article_title}" if article_title else ""

    user_prompt = ENRICHMENT_USER_PROMPT.format(
        document_type=document_type,
        number=number,
        year=year,
        issuing_body=issuing_body,
        chapter_number=chapter_number,
        chapter_title=chapter_title,
        article_number=article_number,
        article_title_suffix=article_title_suffix,
        text=text,
    )

    return ENRICHMENT_SYSTEM_PROMPT, user_prompt


def build_batch_enrichment_prompt(
    chunks: list[dict],
    document_type: str,
    number: str,
    year: int,
    issuing_body: str,
) -> tuple[str, str]:
    """
    Constrói prompt de enriquecimento em batch para múltiplos chunks.

    Args:
        chunks: Lista de dicts com keys: text, chapter_number, chapter_title,
                article_number, article_title

    Returns:
        Tuple (system_prompt, user_prompt)
    """
    chunks_text_parts = []

    for i, chunk in enumerate(chunks, 1):
        article_title_suffix = (
            f" - {chunk['article_title']}" if chunk.get("article_title") else ""
        )

        chunk_text = CHUNK_TEMPLATE.format(
            index=i,
            chapter_number=chunk.get("chapter_number", ""),
            chapter_title=chunk.get("chapter_title", ""),
            article_number=chunk.get("article_number", ""),
            article_title_suffix=article_title_suffix,
            text=chunk["text"],
        )
        chunks_text_parts.append(chunk_text)

    chunks_text = "\n".join(chunks_text_parts)

    user_prompt = BATCH_ENRICHMENT_USER_PROMPT.format(
        document_type=document_type,
        number=number,
        year=year,
        issuing_body=issuing_body,
        chunks_text=chunks_text,
    )

    return BATCH_ENRICHMENT_SYSTEM_PROMPT, user_prompt


def build_enriched_text(
    text: str,
    context_header: str,
    synthetic_questions: list[str],
) -> str:
    """
    Monta texto enriquecido para embedding.

    O texto enriquecido inclui:
    - Contexto no início
    - Texto original
    - Perguntas que o trecho responde

    Isso melhora o recall na busca semântica (Contextual Retrieval).
    """
    questions_text = "\n".join(f"- {q}" for q in synthetic_questions)

    return f"""[CONTEXTO: {context_header}]

{text}

Perguntas que este trecho responde:
{questions_text}"""


def parse_enrichment_response(response: str) -> dict:
    """
    Parseia resposta do LLM de enriquecimento.

    Args:
        response: Resposta JSON do LLM

    Returns:
        Dict com context_header, thesis_text, thesis_type, synthetic_questions
    """
    import json
    import re

    # Tenta extrair JSON da resposta
    response = response.strip()

    # Remove markdown code blocks se presentes
    if response.startswith("```"):
        # Remove ```json e ``` finais
        response = re.sub(r"^```\w*\n?", "", response)
        response = re.sub(r"\n?```$", "", response)

    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        # Tenta encontrar JSON na resposta
        match = re.search(r"\{[\s\S]*\}", response)
        if match:
            data = json.loads(match.group())
        else:
            raise ValueError(f"Não foi possível parsear resposta: {response[:200]}")

    # Valida campos obrigatórios
    required = ["context_header", "thesis_text", "thesis_type", "synthetic_questions"]
    for field in required:
        if field not in data:
            raise ValueError(f"Campo obrigatório ausente: {field}")

    # Normaliza synthetic_questions para string
    questions = data["synthetic_questions"]
    if isinstance(questions, list):
        data["synthetic_questions"] = "\n".join(questions)

    # Valida thesis_type
    valid_types = {
        "definicao", "procedimento", "prazo", "requisito",
        "competencia", "vedacao", "excecao", "sancao", "disposicao"
    }
    if data["thesis_type"] not in valid_types:
        data["thesis_type"] = "disposicao"

    return data


def parse_batch_enrichment_response(response: str, expected_count: int) -> list[dict]:
    """
    Parseia resposta do LLM de enriquecimento em batch.

    Args:
        response: Resposta JSON do LLM (lista)
        expected_count: Número esperado de itens

    Returns:
        Lista de dicts com metadados de cada chunk
    """
    import json
    import re

    response = response.strip()

    # Remove markdown code blocks
    if response.startswith("```"):
        response = re.sub(r"^```\w*\n?", "", response)
        response = re.sub(r"\n?```$", "", response)

    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        # Tenta encontrar array JSON
        match = re.search(r"\[[\s\S]*\]", response)
        if match:
            data = json.loads(match.group())
        else:
            raise ValueError(f"Não foi possível parsear resposta batch: {response[:200]}")

    if not isinstance(data, list):
        raise ValueError("Resposta deve ser uma lista JSON")

    if len(data) != expected_count:
        raise ValueError(
            f"Esperado {expected_count} itens, recebido {len(data)}"
        )

    # Valida e normaliza cada item
    results = []
    for item in data:
        normalized = parse_enrichment_response(json.dumps(item))
        results.append(normalized)

    return results
