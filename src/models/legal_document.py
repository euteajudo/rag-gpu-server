"""
Modelos Pydantic para documentos legais brasileiros.

Estes modelos definem a estrutura exata do JSON que o LLM deve gerar.
O schema Pydantic é convertido em JSON Schema e enviado ao modelo,
garantindo output estruturado e validado.

Uso:
    from models.legal_document import LegalDocument
    
    # Validar JSON do LLM
    doc = LegalDocument.model_validate(json_from_llm)
    
    # Gerar schema para prompt
    schema = LegalDocument.model_json_schema()
"""

from typing import Optional
from pydantic import BaseModel, Field


# =============================================================================
# SUB-MODELOS (de baixo para cima na hierarquia)
# =============================================================================

class SubItem(BaseModel):
    """Alínea de um inciso (a, b, c, d...)"""
    
    item_identifier: str = Field(
        ...,
        description="Letra da alínea (a, b, c, d). Apenas a letra, sem parênteses.",
        examples=["a", "b", "c"]
    )
    description: str = Field(
        ...,
        description="Texto completo da alínea."
    )


class Item(BaseModel):
    """Inciso de um artigo (I, II, III...)"""
    
    item_identifier: str = Field(
        ...,
        description="Numeral romano do inciso (I, II, III). Apenas o numeral, sem hífen.",
        examples=["I", "II", "III", "IV", "V"]
    )
    description: str = Field(
        ...,
        description="Texto completo do inciso."
    )
    sub_items: list[SubItem] = Field(
        default_factory=list,
        description="Lista de alíneas do inciso. Vazio se não houver alíneas."
    )


class Paragraph(BaseModel):
    """Parágrafo de um artigo (§ 1º, § 2º, Parágrafo único...)"""
    
    paragraph_identifier: str = Field(
        ...,
        description="Identificador do parágrafo: número (1, 2, 3) ou 'unico'. Sem símbolo §.",
        examples=["1", "2", "unico"]
    )
    content: str = Field(
        ...,
        description="Texto completo do parágrafo."
    )


class Article(BaseModel):
    """Artigo de uma lei (Art. 1º, Art. 2º...)"""
    
    article_number: str = Field(
        ...,
        description="Número do artigo. Apenas o número, sem 'Art.' ou 'º'.",
        examples=["1", "2", "10", "15"]
    )
    title: Optional[str] = Field(
        default=None,
        description="Título do artigo, se houver. Null se não houver título específico."
    )
    content: str = Field(
        ...,
        description="Texto do caput do artigo (parte principal antes dos incisos)."
    )
    items: list[Item] = Field(
        default_factory=list,
        description="Lista de incisos do artigo. Vazio se não houver incisos."
    )
    paragraphs: list[Paragraph] = Field(
        default_factory=list,
        description="Lista de parágrafos do artigo. Vazio se não houver parágrafos."
    )


class Chapter(BaseModel):
    """Capítulo de uma lei (Capítulo I, II...)"""
    
    chapter_number: Optional[str] = Field(
        default=None,
        description="Número do capítulo em romano (I, II, III). Null se não houver capítulo.",
        examples=["I", "II", "III", None]
    )
    title: str = Field(
        ...,
        description="Título do capítulo."
    )
    articles: list[Article] = Field(
        ...,
        description="Lista de artigos do capítulo. Deve conter todos os artigos.",
        min_length=1
    )


# =============================================================================
# MODELO PRINCIPAL
# =============================================================================

class PublicationDetails(BaseModel):
    """Detalhes da publicação no Diário Oficial."""
    
    source: str = Field(
        default="DIARIO OFICIAL DA UNIAO",
        description="Fonte da publicação."
    )
    publication_date: str = Field(
        ...,
        description="Data da publicação no formato YYYY-MM-DD.",
        examples=["2022-08-09"]
    )
    edition: Optional[str] = Field(
        default=None,
        description="Número da edição."
    )
    section: Optional[str] = Field(
        default=None,
        description="Seção do diário."
    )
    page: Optional[str] = Field(
        default=None,
        description="Página inicial."
    )


class LegalDocument(BaseModel):
    """
    Modelo principal para documentos legais brasileiros.
    
    Este modelo representa a estrutura completa de uma lei, decreto,
    instrução normativa ou portaria.
    """
    
    document_type: str = Field(
        ...,
        description="Tipo do documento: LEI, DECRETO, INSTRUCAO NORMATIVA, PORTARIA, etc.",
        examples=["LEI", "DECRETO", "INSTRUCAO NORMATIVA", "PORTARIA"]
    )
    issuing_body: str = Field(
        ...,
        description="Nome completo do órgão emissor."
    )
    issuing_body_acronym: Optional[str] = Field(
        default=None,
        description="Sigla do órgão emissor (ex: SEGES, ME, MF)."
    )
    number: str = Field(
        ...,
        description="Número do documento.",
        examples=["58", "14133", "8666"]
    )
    date: str = Field(
        ...,
        description="Data do documento no formato YYYY-MM-DD.",
        examples=["2022-08-08"]
    )
    ementa: str = Field(
        ...,
        description="Ementa/resumo oficial do documento."
    )
    publication_details: Optional[PublicationDetails] = Field(
        default=None,
        description="Detalhes da publicação. Null se não disponível."
    )
    chapters: list[Chapter] = Field(
        ...,
        description="Lista de capítulos do documento com todos os artigos.",
        min_length=1
    )
    signatory: Optional[str] = Field(
        default=None,
        description="Nome da autoridade que assina o documento."
    )
    
    class Config:
        """Configuração do modelo."""
        json_schema_extra = {
            "example": {
                "document_type": "INSTRUCAO NORMATIVA",
                "issuing_body": "Ministério da Economia",
                "issuing_body_acronym": "SEGES",
                "number": "58",
                "date": "2022-08-08",
                "ementa": "Dispõe sobre...",
                "chapters": [
                    {
                        "chapter_number": "I",
                        "title": "DISPOSIÇÕES PRELIMINARES",
                        "articles": [
                            {
                                "article_number": "1",
                                "title": None,
                                "content": "Esta Instrução Normativa dispõe sobre...",
                                "items": [],
                                "paragraphs": []
                            }
                        ]
                    }
                ],
                "signatory": "RENATO RIBEIRO FENILI"
            }
        }


# =============================================================================
# FUNÇÕES UTILITÁRIAS
# =============================================================================

def get_schema_for_prompt() -> str:
    """
    Retorna o JSON Schema formatado para usar no prompt do LLM.
    """
    import json
    schema = LegalDocument.model_json_schema()
    return json.dumps(schema, indent=2, ensure_ascii=False)


def get_simplified_schema() -> str:
    """
    Retorna uma versão simplificada do schema para prompts menores.
    """
    return """{
  "document_type": "INSTRUCAO NORMATIVA | LEI | DECRETO",
  "issuing_body": "Nome do órgão",
  "issuing_body_acronym": "Sigla ou null",
  "number": "Número do documento",
  "date": "YYYY-MM-DD",
  "ementa": "Resumo oficial",
  "publication_details": {
    "source": "DIARIO OFICIAL DA UNIAO",
    "publication_date": "YYYY-MM-DD",
    "edition": "string ou null",
    "section": "string ou null",
    "page": "string ou null"
  },
  "chapters": [
    {
      "chapter_number": "I, II, III ou null",
      "title": "Título do capítulo",
      "articles": [
        {
          "article_number": "1, 2, 3 (apenas número)",
          "title": "Título ou null",
          "content": "Texto do caput",
          "items": [
            {
              "item_identifier": "I, II, III (numeral romano)",
              "description": "Texto do inciso",
              "sub_items": [
                {"item_identifier": "a, b, c", "description": "Texto da alínea"}
              ]
            }
          ],
          "paragraphs": [
            {"paragraph_identifier": "1, 2, unico", "content": "Texto"}
          ]
        }
      ]
    }
  ],
  "signatory": "Nome ou null"
}"""


def validate_extraction(json_data: dict) -> tuple[bool, list[str]]:
    """
    Valida um JSON extraído contra o schema Pydantic.
    
    Returns:
        (is_valid, list_of_errors)
    """
    try:
        LegalDocument.model_validate(json_data)
        return True, []
    except Exception as e:
        errors = []
        if hasattr(e, 'errors'):
            for err in e.errors():
                loc = ' -> '.join(str(x) for x in err['loc'])
                errors.append(f"{loc}: {err['msg']}")
        else:
            errors.append(str(e))
        return False, errors


def count_articles(doc: LegalDocument) -> int:
    """Conta total de artigos no documento."""
    return sum(len(chapter.articles) for chapter in doc.chapters)


def get_article_numbers(doc: LegalDocument) -> list[int]:
    """Retorna lista de números dos artigos."""
    numbers = []
    for chapter in doc.chapters:
        for article in chapter.articles:
            try:
                numbers.append(int(article.article_number))
            except ValueError:
                pass
    return sorted(numbers)


# =============================================================================
# EXEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    # Mostrar schema
    print("=== JSON SCHEMA COMPLETO ===")
    print(get_schema_for_prompt()[:2000] + "...")
    
    print("\n=== SCHEMA SIMPLIFICADO ===")
    print(get_simplified_schema())
    
    # Exemplo de validação
    print("\n=== TESTE DE VALIDAÇÃO ===")
    
    test_json = {
        "document_type": "LEI",
        "issuing_body": "Congresso Nacional",
        "number": "14133",
        "date": "2021-04-01",
        "ementa": "Lei de Licitações",
        "chapters": [
            {
                "chapter_number": "I",
                "title": "DISPOSIÇÕES GERAIS",
                "articles": [
                    {
                        "article_number": "1",
                        "content": "Esta Lei estabelece normas gerais...",
                        "items": [],
                        "paragraphs": []
                    }
                ]
            }
        ]
    }
    
    is_valid, errors = validate_extraction(test_json)
    print(f"Válido: {is_valid}")
    if errors:
        print(f"Erros: {errors}")

