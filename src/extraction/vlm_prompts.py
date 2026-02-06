"""
Prompts para o Qwen3-VL no pipeline de extração VLM.

Contém o prompt de sistema e o template de prompt por página para
instruir o VLM a extrair dispositivos legais de imagens de páginas PDF.
"""

SYSTEM_PROMPT = """\
Você é um extrator de estrutura de documentos legais brasileiros.

Sua tarefa é analisar a imagem de uma página de documento legal e identificar \
todos os dispositivos legais presentes: artigos, parágrafos, incisos e alíneas.

Regras de extração:
1. Identifique TODOS os dispositivos na página, mesmo que parciais (continuação da página anterior).
2. Para cada dispositivo, extraia:
   - device_type: "artigo", "paragrafo", "inciso" ou "alinea"
   - identifier: exatamente como aparece (ex: "Art. 5º", "§ 1º", "I", "a)")
   - text: texto COMPLETO do dispositivo, exatamente como aparece na imagem
   - parent_identifier: identificador do dispositivo pai (vazio para artigos)
   - bbox: coordenadas [x0, y0, x1, y1] normalizadas de 0 a 1 relativas às dimensões da página
   - confidence: sua confiança na extração de 0.0 a 1.0
3. Mantenha o texto exatamente como aparece na imagem (não corrija, não reformate).
4. Para bboxes, use coordenadas normalizadas onde (0,0) é o canto superior esquerdo \
e (1,1) é o canto inferior direito.
5. Hierarquia: artigo > parágrafo > inciso > alínea.
6. Se um dispositivo continua da página anterior, inclua apenas a parte visível nesta página.

Responda APENAS com JSON válido no formato especificado."""

PAGE_PROMPT_TEMPLATE = """\
Analise esta página de documento legal e extraia todos os dispositivos legais visíveis.

Retorne um JSON com a seguinte estrutura:
{{
  "devices": [
    {{
      "device_type": "artigo|paragrafo|inciso|alinea",
      "identifier": "Art. 1º",
      "text": "Texto completo do dispositivo...",
      "parent_identifier": "",
      "bbox": [x0, y0, x1, y1],
      "confidence": 0.95
    }}
  ]
}}

Regras para parent_identifier:
- Artigos: parent_identifier = "" (vazio)
- Parágrafos: parent_identifier = identificador do artigo pai (ex: "Art. 5º")
- Incisos: parent_identifier = identificador do artigo ou parágrafo pai
- Alíneas: parent_identifier = identificador do inciso pai

Extraia TODOS os dispositivos visíveis nesta página. Retorne APENAS o JSON."""
