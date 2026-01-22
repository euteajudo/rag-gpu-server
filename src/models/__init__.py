"""Modelos Pydantic e utilitarios para o sistema de extracao."""

from .legal_document import (
    LegalDocument,
    Chapter,
    Article,
    Item,
    SubItem,
    Paragraph,
    PublicationDetails,
    get_schema_for_prompt,
    get_simplified_schema,
    validate_extraction,
    count_articles,
    get_article_numbers,
)

from .extraction_utils import (
    DoclingValidator,
    ExtractionValidator,
    AutoFixer,
    ElementCount,
    ValidationReport,
    validate_and_fix,
    get_few_shot_examples,
    get_few_shot_prompt,
)

__all__ = [
    # Legal Document Models
    "LegalDocument",
    "Chapter",
    "Article",
    "Item",
    "SubItem",
    "Paragraph",
    "PublicationDetails",
    "get_schema_for_prompt",
    "get_simplified_schema",
    "validate_extraction",
    "count_articles",
    "get_article_numbers",
    # Extraction Utils
    "DoclingValidator",
    "ExtractionValidator",
    "AutoFixer",
    "ElementCount",
    "ValidationReport",
    "validate_and_fix",
    "get_few_shot_examples",
    "get_few_shot_prompt",
]

