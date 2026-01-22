"""
Utilitarios de validacao e correcao para extracao de documentos legais.

Este modulo contem:
1. DoclingValidator - Valida e conta elementos no Markdown do Docling
2. ExtractionValidator - Compara extracao JSON vs Markdown original
3. AutoFixer - Corrige erros conhecidos automaticamente
4. Few-shot Examples - Exemplos no formato real do Docling

Uso:
    from models.extraction_utils import (
        DoclingValidator,
        ExtractionValidator,
        AutoFixer,
        validate_and_fix,
        get_few_shot_examples,
    )

    # 1. Validar Markdown do Docling
    docling_val = DoclingValidator(markdown)
    docling_report = docling_val.validate()

    # 2. Extrair com LLM...

    # 3. Validar e corrigir extracao
    fixed_data, report, fixes = validate_and_fix(markdown, json_data)
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# PADROES REGEX PARA DOCUMENTOS LEGAIS BRASILEIROS
# =============================================================================

PATTERNS = {
    # Artigos: "Art. 1o", "Art. 10.", "Art. 100"
    "article": r'Art\.?\s*(\d+)[ºo°.]?',

    # Paragrafos: "§ 1o", "§ 2o", "§1o", "Paragrafo unico"
    "paragraph": r'§\s*(\d+)[ºo°]?|[Pp]ar[áa]grafo\s+[úu]nico',

    # Paragrafos em lista numerada do Docling: "12. § 1o"
    "paragraph_numbered": r'^\d+\.\s*§\s*(\d+)[ºo°]?',

    # Incisos: "I -", "II -", "III -", "IV-", "V -"
    # Pode vir como bullet "- I -" ou lista "2. IV -"
    "item": r'(?:^|\n)\s*(?:-\s*)?(?:\d+\.\s*)?([IVXLC]+)\s*[-–—]',

    # Alineas: "a)", "b)", "c)"
    "sub_item": r'(?:^|\n)\s*(?:-\s*)?([a-z])\)',

    # Capitulos: "CAPITULO I", "CAPITULO II"
    "chapter": r'CAP[ÍI]TULO\s+([IVXLC]+)',
}


# =============================================================================
# DATACLASSES PARA RELATORIOS
# =============================================================================

@dataclass
class ElementCount:
    """Contagem de elementos encontrados."""
    articles: list[str] = field(default_factory=list)
    paragraphs: list[tuple[str, str]] = field(default_factory=list)  # (art, §)
    items: list[tuple[str, str]] = field(default_factory=list)  # (art, inciso)
    sub_items: list[tuple[str, str, str]] = field(default_factory=list)  # (art, inciso, alinea)
    chapters: list[str] = field(default_factory=list)

    @property
    def total_articles(self) -> int:
        return len(self.articles)

    @property
    def total_paragraphs(self) -> int:
        return len(self.paragraphs)

    @property
    def total_items(self) -> int:
        return len(self.items)

    @property
    def total_sub_items(self) -> int:
        return len(self.sub_items)

    @property
    def total_chapters(self) -> int:
        return len(self.chapters)


@dataclass
class ValidationReport:
    """Relatorio de validacao."""
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    expected: Optional[ElementCount] = None
    found: Optional[ElementCount] = None

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def summary(self) -> str:
        lines = [
            f"Valid: {self.is_valid}",
            f"Errors: {len(self.errors)}",
            f"Warnings: {len(self.warnings)}",
        ]
        if self.expected and self.found:
            lines.extend([
                f"Articles: {self.found.total_articles}/{self.expected.total_articles}",
                f"Paragraphs: {self.found.total_paragraphs}/{self.expected.total_paragraphs}",
                f"Items: {self.found.total_items}/{self.expected.total_items}",
                f"Chapters: {self.found.total_chapters}/{self.expected.total_chapters}",
            ])
        if self.errors:
            lines.append("--- ERRORS ---")
            lines.extend(self.errors[:10])
        if self.warnings:
            lines.append("--- WARNINGS ---")
            lines.extend(self.warnings[:10])
        return "\n".join(lines)


# =============================================================================
# VALIDADOR DO MARKDOWN DO DOCLING
# =============================================================================

class DoclingValidator:
    """
    Valida e conta elementos no Markdown gerado pelo Docling.

    Uso:
        validator = DoclingValidator(markdown_text)
        report = validator.validate()
        counts = validator.count_elements()
    """

    def __init__(self, markdown: str):
        self.markdown = markdown
        self._counts: Optional[ElementCount] = None

    def count_elements(self) -> ElementCount:
        """Conta todos os elementos legais no Markdown."""
        if self._counts:
            return self._counts

        counts = ElementCount()

        # Contar capitulos
        for match in re.finditer(PATTERNS["chapter"], self.markdown, re.IGNORECASE):
            counts.chapters.append(match.group(1))

        # Contar artigos
        for match in re.finditer(PATTERNS["article"], self.markdown):
            counts.articles.append(match.group(1))

        # Contar paragrafos (incluindo formato lista numerada do Docling)
        # IMPORTANTE: Distinguir paragrafos proprios de referencias a outras leis
        current_article = None
        for line in self.markdown.split('\n'):
            # Detectar artigo atual
            art_match = re.search(PATTERNS["article"], line)
            if art_match:
                current_article = art_match.group(1)

            line_stripped = line.strip()

            # § 1o no inicio da linha
            para_match = re.match(r'^§\s*(\d+)[ºo°]?', line_stripped)

            # Lista numerada do Docling: "12. § 1o"
            if not para_match:
                para_match = re.match(r'^\d+\.\s*§\s*(\d+)[ºo°]?', line_stripped)

            # Paragrafo unico
            if not para_match:
                if re.match(r'^[Pp]ar[áa]grafo\s+[úu]nico', line_stripped):
                    if current_article:
                        counts.paragraphs.append((current_article, "unico"))
                    continue

            if para_match and current_article:
                para_id = para_match.group(1)
                counts.paragraphs.append((current_article, para_id))

        # Contar incisos
        current_article = None
        for line in self.markdown.split('\n'):
            art_match = re.search(PATTERNS["article"], line)
            if art_match:
                current_article = art_match.group(1)

            item_match = re.search(PATTERNS["item"], line)
            if item_match and current_article:
                counts.items.append((current_article, item_match.group(1)))

        # Contar alineas
        current_article = None
        current_item = None
        for line in self.markdown.split('\n'):
            art_match = re.search(PATTERNS["article"], line)
            if art_match:
                current_article = art_match.group(1)
                current_item = None

            item_match = re.search(PATTERNS["item"], line)
            if item_match:
                current_item = item_match.group(1)

            sub_match = re.search(PATTERNS["sub_item"], line)
            if sub_match and current_article and current_item:
                counts.sub_items.append((current_article, current_item, sub_match.group(1)))

        self._counts = counts
        return counts

    def validate(self) -> ValidationReport:
        """Valida o Markdown do Docling."""
        report = ValidationReport(is_valid=True)
        counts = self.count_elements()
        report.expected = counts

        # Verificar se ha artigos
        if counts.total_articles == 0:
            report.add_error("Nenhum artigo encontrado no Markdown")

        # Verificar sequencia de artigos
        try:
            art_nums = sorted([int(a) for a in counts.articles])
            expected_seq = list(range(1, max(art_nums) + 1))
            missing = set(expected_seq) - set(art_nums)
            if missing:
                report.add_warning(f"Artigos possivelmente faltando no MD: {sorted(missing)}")
        except (ValueError, TypeError):
            pass

        # Verificar problemas conhecidos do Docling
        if "<!-- image -->" in self.markdown:
            report.add_warning("Markdown contem marcadores de imagem que podem afetar extracao")

        # Verificar lista numerada com § (problema comum)
        if re.search(r'^\d+\.\s*§', self.markdown, re.MULTILINE):
            report.add_warning("Paragrafos em formato lista numerada detectados (ex: '12. § 1o')")

        # Verificar Art. como item de lista (erro do Docling)
        if re.search(r'^-\s*Art\.', self.markdown, re.MULTILINE):
            report.add_error("Artigo detectado como item de lista (- Art.) - erro do Docling")

        return report

    def get_article_paragraphs_map(self) -> dict[str, list[str]]:
        """Retorna mapa de artigo -> lista de paragrafos."""
        counts = self.count_elements()
        result = {}
        for art, para in counts.paragraphs:
            if art not in result:
                result[art] = []
            result[art].append(para)
        return result

    def get_article_items_map(self) -> dict[str, list[str]]:
        """Retorna mapa de artigo -> lista de incisos."""
        counts = self.count_elements()
        result = {}
        for art, item in counts.items:
            if art not in result:
                result[art] = []
            result[art].append(item)
        return result


# =============================================================================
# VALIDADOR DA EXTRACAO JSON
# =============================================================================

class ExtractionValidator:
    """
    Compara extracao JSON com Markdown original.

    Uso:
        validator = ExtractionValidator(markdown, json_data)
        report = validator.validate()
    """

    def __init__(self, markdown: str, json_data: dict):
        self.markdown = markdown
        self.json_data = json_data
        self.docling_validator = DoclingValidator(markdown)

    def count_json_elements(self) -> ElementCount:
        """Conta elementos no JSON extraido."""
        counts = ElementCount()

        for chapter in self.json_data.get("chapters", []):
            if chapter.get("chapter_number"):
                counts.chapters.append(chapter["chapter_number"])

            for article in chapter.get("articles", []):
                art_num = article.get("article_number", "")
                counts.articles.append(art_num)

                for para in article.get("paragraphs", []):
                    counts.paragraphs.append((art_num, para.get("paragraph_identifier", "")))

                for item in article.get("items", []):
                    item_id = item.get("item_identifier", "")
                    counts.items.append((art_num, item_id))

                    for sub in item.get("sub_items", []):
                        counts.sub_items.append((art_num, item_id, sub.get("item_identifier", "")))

        return counts

    def validate(self) -> ValidationReport:
        """Valida extracao comparando com Markdown original."""
        report = ValidationReport(is_valid=True)

        expected = self.docling_validator.count_elements()
        found = self.count_json_elements()

        report.expected = expected
        report.found = found

        # Comparar artigos
        if found.total_articles != expected.total_articles:
            report.add_error(
                f"Artigos: esperado {expected.total_articles}, extraido {found.total_articles}"
            )

            expected_arts = set(expected.articles)
            found_arts = set(found.articles)
            missing = expected_arts - found_arts
            extra = found_arts - expected_arts

            if missing:
                report.add_error(f"Artigos faltando: {sorted(missing, key=lambda x: int(x) if x.isdigit() else 0)}")
            if extra:
                report.add_warning(f"Artigos extras: {sorted(extra)}")

        # Comparar paragrafos
        if found.total_paragraphs != expected.total_paragraphs:
            report.add_error(
                f"Paragrafos: esperado {expected.total_paragraphs}, extraido {found.total_paragraphs}"
            )

            expected_map = self.docling_validator.get_article_paragraphs_map()
            found_map = {}
            for art, para in found.paragraphs:
                if art not in found_map:
                    found_map[art] = []
                found_map[art].append(para)

            for art, expected_paras in expected_map.items():
                found_paras = found_map.get(art, [])
                if len(found_paras) != len(expected_paras):
                    report.add_error(
                        f"Art. {art}: esperado {len(expected_paras)} paragrafos, extraido {len(found_paras)}"
                    )

        # Comparar incisos (warning, nao erro critico)
        if found.total_items != expected.total_items:
            diff = abs(found.total_items - expected.total_items)
            if diff > 2:  # Tolerancia de 2
                report.add_warning(
                    f"Incisos: esperado ~{expected.total_items}, extraido {found.total_items}"
                )

        # Comparar capitulos
        if found.total_chapters != expected.total_chapters:
            report.add_warning(
                f"Capitulos: esperado {expected.total_chapters}, extraido {found.total_chapters}"
            )

        return report


# =============================================================================
# AUTO-FIXER PARA ERROS CONHECIDOS
# =============================================================================

class AutoFixer:
    """
    Corrige automaticamente erros conhecidos na extracao.

    Erros corrigidos:
    1. Paragrafos classificados como incisos
    2. Capitulos vazios
    3. Incisos duplicados
    4. Sub-items fantasmas
    """

    def __init__(self, json_data: dict, markdown: str):
        self.data = json_data
        self.markdown = markdown
        self.docling_validator = DoclingValidator(markdown)
        self.fixes_applied: list[str] = []

    def fix_all(self) -> dict:
        """Aplica todas as correcoes."""
        self._fix_paragraphs_as_items()
        self._fix_empty_chapters()
        self._fix_phantom_sub_items()
        self._fix_duplicate_content()
        return self.data

    def _fix_paragraphs_as_items(self):
        """
        Corrige incisos que sao na verdade paragrafos.

        Detecta quando:
        1. O Markdown original tem § mas o JSON tem inciso
        2. O item_identifier e numero (1, 2) em vez de romano (I, II)
        """
        expected_map = self.docling_validator.get_article_paragraphs_map()

        # Mapear romano para numero
        roman_to_num = {"I": "1", "II": "2", "III": "3", "IV": "4", "V": "5", "VI": "6"}

        for chapter in self.data.get("chapters", []):
            for article in chapter.get("articles", []):
                art_num = article.get("article_number", "")

                expected_paras = expected_map.get(art_num, [])
                current_paras = article.get("paragraphs", [])

                if len(expected_paras) > len(current_paras):
                    items = article.get("items", [])
                    items_to_remove = []

                    for i, item in enumerate(items):
                        identifier = item.get("item_identifier", "")
                        desc = item.get("description", "")

                        # Verificar se este inciso deveria ser paragrafo
                        should_be_para = False
                        para_num = None

                        # 1. Identificador numerico
                        if identifier.isdigit() and identifier in expected_paras:
                            should_be_para = True
                            para_num = identifier

                        # 2. Romano que corresponde a § no markdown
                        elif identifier in roman_to_num:
                            num = roman_to_num[identifier]
                            if num in expected_paras:
                                # Verificar no markdown se realmente e §
                                pattern = rf'Art\.?\s*{art_num}[ºo°.]?.*?§\s*{num}[ºo°]?'
                                if re.search(pattern, self.markdown, re.DOTALL | re.IGNORECASE):
                                    should_be_para = True
                                    para_num = num

                        if should_be_para and para_num:
                            items_to_remove.append(i)
                            current_paras.append({
                                "paragraph_identifier": para_num,
                                "content": desc
                            })
                            self.fixes_applied.append(
                                f"Art. {art_num}: Movido inciso '{identifier}' para paragrafo {para_num}"
                            )

                    # Remover items movidos
                    for i in reversed(items_to_remove):
                        items.pop(i)

                    # Ordenar paragrafos
                    current_paras.sort(
                        key=lambda p: (
                            0 if p["paragraph_identifier"] == "unico"
                            else int(p["paragraph_identifier"]) if p["paragraph_identifier"].isdigit()
                            else 999
                        )
                    )
                    article["paragraphs"] = current_paras

    def _fix_empty_chapters(self):
        """Remove capitulos sem artigos."""
        chapters = self.data.get("chapters", [])
        original_count = len(chapters)

        self.data["chapters"] = [
            ch for ch in chapters
            if ch.get("articles") and len(ch["articles"]) > 0
        ]

        removed = original_count - len(self.data["chapters"])
        if removed > 0:
            self.fixes_applied.append(f"Removidos {removed} capitulos vazios")

    def _fix_phantom_sub_items(self):
        """Remove sub_items que duplicam o texto do item pai."""
        for chapter in self.data.get("chapters", []):
            for article in chapter.get("articles", []):
                for item in article.get("items", []):
                    sub_items = item.get("sub_items", [])

                    if len(sub_items) == 1:
                        sub_desc = sub_items[0].get("description", "")[:50]
                        item_desc = item.get("description", "")

                        if sub_desc in item_desc:
                            item["sub_items"] = []
                            self.fixes_applied.append(
                                f"Art. {article.get('article_number')}, "
                                f"Inciso {item.get('item_identifier')}: "
                                f"Removido sub_item duplicado"
                            )

    def _fix_duplicate_content(self):
        """Remove duplicacoes de conteudo entre artigos."""
        seen_contents = {}

        for chapter in self.data.get("chapters", []):
            for article in chapter.get("articles", []):
                content = article.get("content", "")[:100]
                art_num = article.get("article_number", "")

                if content in seen_contents:
                    other_art = seen_contents[content]
                    if other_art != art_num:
                        self.fixes_applied.append(
                            f"AVISO: Art. {art_num} tem conteudo similar ao Art. {other_art}"
                        )
                else:
                    seen_contents[content] = art_num

    def get_report(self) -> str:
        """Retorna relatorio das correcoes aplicadas."""
        if not self.fixes_applied:
            return "Nenhuma correcao necessaria"

        lines = [f"Correcoes aplicadas: {len(self.fixes_applied)}"]
        lines.extend(self.fixes_applied)
        return "\n".join(lines)


# =============================================================================
# FEW-SHOT EXAMPLES (FORMATO REAL DO DOCLING)
# =============================================================================

FEW_SHOT_EXAMPLES = '''
=== EXEMPLO 1: Artigo com Incisos (formato bullet do Docling) ===

MARKDOWN:
Art. 3o Para fins do disposto nesta Instrucao Normativa, considera-se:

- I - Estudo Tecnico Preliminar - ETP: documento constitutivo da primeira etapa...
- II - Sistema ETP Digital: ferramenta informatizada integrante...
- III - area requisitante: unidade responsavel por identificar a necessidade...

§ 1o Os papeis de requisitante e de area tecnica poderao ser...
§ 2o A definicao dos requisitantes, das areas tecnicas...

JSON:
{
  "article_number": "3",
  "content": "Para fins do disposto nesta Instrucao Normativa, considera-se:",
  "items": [
    {"item_identifier": "I", "description": "Estudo Tecnico Preliminar - ETP: documento constitutivo da primeira etapa...", "sub_items": []},
    {"item_identifier": "II", "description": "Sistema ETP Digital: ferramenta informatizada integrante...", "sub_items": []},
    {"item_identifier": "III", "description": "area requisitante: unidade responsavel por identificar a necessidade...", "sub_items": []}
  ],
  "paragraphs": [
    {"paragraph_identifier": "1", "content": "Os papeis de requisitante e de area tecnica poderao ser..."},
    {"paragraph_identifier": "2", "content": "A definicao dos requisitantes, das areas tecnicas..."}
  ]
}

=== EXEMPLO 2: Artigo com Incisos em lista numerada (formato Docling) ===

MARKDOWN:
Art. 10. Durante a elaboracao do ETP deverao ser avaliadas:

2. I  -  a  possibilidade  de  utilizacao  de  mao  de  obra...
3. II - os servicos de manutencao a serem contratados...

<!-- image -->

4. III - os materiais de consumo e permanentes a serem...

§ 2o Caso, apos o levantamento do mercado...

JSON:
{
  "article_number": "10",
  "content": "Durante a elaboracao do ETP deverao ser avaliadas:",
  "items": [
    {"item_identifier": "I", "description": "a possibilidade de utilizacao de mao de obra...", "sub_items": []},
    {"item_identifier": "II", "description": "os servicos de manutencao a serem contratados...", "sub_items": []},
    {"item_identifier": "III", "description": "os materiais de consumo e permanentes a serem...", "sub_items": []}
  ],
  "paragraphs": [
    {"paragraph_identifier": "2", "content": "Caso, apos o levantamento do mercado..."}
  ]
}

=== EXEMPLO 3: Artigo com Alineas (sub_items) ===

MARKDOWN:
Art. 14. A elaboracao do ETP:

- I  -  e  facultada nas hipoteses dos incisos I, II, VII e VIII...
- II  -  e  dispensada na hipotese do inciso III do art. 75...:
  a) prorrogacoes dos contratos de servicos...
  b) casos de emergencia ou calamidade publica...

JSON:
{
  "article_number": "14",
  "content": "A elaboracao do ETP:",
  "items": [
    {"item_identifier": "I", "description": "e facultada nas hipoteses dos incisos I, II, VII e VIII...", "sub_items": []},
    {"item_identifier": "II", "description": "e dispensada na hipotese do inciso III do art. 75...:", "sub_items": [
      {"item_identifier": "a", "description": "prorrogacoes dos contratos de servicos..."},
      {"item_identifier": "b", "description": "casos de emergencia ou calamidade publica..."}
    ]}
  ],
  "paragraphs": []
}

=== EXEMPLO 4: Artigo com Paragrafo unico ===

MARKDOWN:
Art. 19. Esta Instrucao Normativa entra em vigor em 1o de setembro de 2022.

Paragrafo unico. Permanecem regidos pela Instrucao Normativa no 40...

JSON:
{
  "article_number": "19",
  "content": "Esta Instrucao Normativa entra em vigor em 1o de setembro de 2022.",
  "items": [],
  "paragraphs": [
    {"paragraph_identifier": "unico", "content": "Permanecem regidos pela Instrucao Normativa no 40..."}
  ]
}

=== EXEMPLO 5: Artigo SOMENTE com Paragrafos (SEM incisos) ===

MARKDOWN:
Art. 17. Os orgaos, as entidades, os dirigentes e os servidores que utilizarem o Sistema ETP Digital responderao administrativa, civil e penalmente...

§ 1o Os orgaos e as entidades assegurarao o sigilo e a integridade dos dados...

§ 2o As informacoes e os dados do Sistema ETP digital nao poderao ser comercializados...

JSON:
{
  "article_number": "17",
  "content": "Os orgaos, as entidades, os dirigentes e os servidores que utilizarem o Sistema ETP Digital responderao administrativa, civil e penalmente...",
  "items": [],
  "paragraphs": [
    {"paragraph_identifier": "1", "content": "Os orgaos e as entidades assegurarao o sigilo e a integridade dos dados..."},
    {"paragraph_identifier": "2", "content": "As informacoes e os dados do Sistema ETP digital nao poderao ser comercializados..."}
  ]
}

=== REGRAS CRITICAS ===

1. PARAGRAFO vs INCISO:
   - § 1o, § 2o, Paragrafo unico -> paragraphs[]
   - I, II, III, IV, V -> items[]
   - NUNCA confunda § com numeral romano!

2. ALINEAS:
   - a), b), c) -> sub_items[] dentro do inciso pai
   - So existem se o texto original tiver letras com parenteses

3. FORMATO DOCLING:
   - Ignore "<!-- image -->"
   - Ignore numeros de lista "2. I -" -> extraia apenas "I"
   - Ignore bullets "- I -" -> extraia apenas "I"
'''


def get_few_shot_examples() -> str:
    """Retorna exemplos few-shot para o prompt do LLM."""
    return FEW_SHOT_EXAMPLES


def get_few_shot_prompt(markdown: str, schema_str: str) -> str:
    """
    Gera prompt completo com few-shot examples.

    Args:
        markdown: Documento Markdown do Docling
        schema_str: JSON Schema como string

    Returns:
        Prompt formatado para o LLM
    """
    return f"""Voce e um especialista em extracao de documentos legais brasileiros.

{FEW_SHOT_EXAMPLES}

Agora extraia o documento abaixo seguindo os mesmos padroes:

DOCUMENTO:
{markdown}

JSON SCHEMA:
{schema_str}

Retorne APENAS o JSON extraido, sem explicacoes. /no_think"""


# =============================================================================
# FUNCAO DE CONVENIENCIA
# =============================================================================

def validate_and_fix(markdown: str, json_data: dict) -> tuple[dict, ValidationReport, str]:
    """
    Funcao de conveniencia que valida e corrige em uma chamada.

    Returns:
        (fixed_data, validation_report, fix_report)
    """
    # 1. Validar Markdown original
    docling_val = DoclingValidator(markdown)
    docling_report = docling_val.validate()

    # 2. Corrigir JSON
    fixer = AutoFixer(json_data, markdown)
    fixed_data = fixer.fix_all()
    fix_report = fixer.get_report()

    # 3. Validar extracao corrigida
    extraction_val = ExtractionValidator(markdown, fixed_data)
    final_report = extraction_val.validate()

    # Mesclar warnings do Docling
    for w in docling_report.warnings:
        final_report.add_warning(f"[Docling] {w}")
    for e in docling_report.errors:
        final_report.add_warning(f"[Docling] {e}")

    return fixed_data, final_report, fix_report


# =============================================================================
# EXEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    print("=== Modulo extraction_utils ===")
    print("Componentes disponiveis:")
    print("  - DoclingValidator: Valida Markdown do Docling")
    print("  - ExtractionValidator: Compara JSON vs Markdown")
    print("  - AutoFixer: Corrige erros automaticamente")
    print("  - get_few_shot_examples(): Retorna exemplos few-shot")
    print("  - get_few_shot_prompt(): Gera prompt com examples")
    print("  - validate_and_fix(): Valida e corrige em uma chamada")
