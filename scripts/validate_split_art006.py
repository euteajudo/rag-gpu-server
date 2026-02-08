#!/usr/bin/env python3
"""
Script de validação do split de artigos grandes.
Executa reingestão da Lei 14.133 e coleta evidências de que ART-006 foi splitado.

Uso: PYTHONPATH=/workspace/rag-gpu-server python scripts/validate_split_art006.py
"""

import sys
import os

# Configura PYTHONPATH para importar src como pacote
sys.path.insert(0, "/workspace/rag-gpu-server")

import logging

# Configura logging detalhado ANTES de importar módulos
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Silencia loggers verbosos
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

def main():
    print("=" * 80)
    print("VALIDAÇÃO: Split de Artigos Grandes - Lei 14.133/2021")
    print("=" * 80)
    print()

    # 1. Verifica se o PDF existe
    pdf_path = "/workspace/pdf/Lei 14.133_2021.pdf"
    if not os.path.exists(pdf_path):
        print(f"ERRO: PDF não encontrado em {pdf_path}")
        sys.exit(1)
    print(f"[OK] PDF encontrado: {pdf_path}")
    print(f"     Tamanho: {os.path.getsize(pdf_path):,} bytes")
    print()

    # 2. Importa pipeline
    print("[INFO] Importando pipeline...")
    try:
        from src.ingestion.pipeline import IngestionPipeline
        from src.ingestion.models import IngestRequest
        print("[OK] Pipeline importado")
    except Exception as e:
        print(f"[ERRO] Falha ao importar pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    print()

    # 3. Configura request
    request = IngestRequest(
        document_id="LEI-14133-2021",
        tipo_documento="LEI",
        numero="14133",
        ano=2021,
        titulo="Lei de Licitações e Contratos",
        skip_enrichment=True,  # Pula enriquecimento LLM para acelerar
        skip_embeddings=False,  # PRECISA dos embeddings para validar
    )
    print(f"[INFO] Request configurado:")
    print(f"       document_id: {request.document_id}")
    print(f"       skip_enrichment: {request.skip_enrichment}")
    print(f"       skip_embeddings: {request.skip_embeddings}")
    print()

    # 4. Lê PDF como bytes
    print("[INFO] Lendo PDF...")
    with open(pdf_path, "rb") as f:
        pdf_content = f.read()
    print(f"[OK] PDF lido: {len(pdf_content):,} bytes")
    print()

    # 5. Executa pipeline
    print("[INFO] Iniciando pipeline de ingestão...")
    print("-" * 80)

    pipeline = IngestionPipeline()
    result = pipeline.process(pdf_content, request)

    print("-" * 80)
    print()

    # 5. Analisa resultado
    print("[INFO] Analisando resultado...")
    print(f"       Status: {result.status}")
    print(f"       Total chunks: {len(result.chunks)}")
    print(f"       Tempo total: {result.total_time_seconds}s")
    print()

    if result.errors:
        print("[ERRO] Erros encontrados:")
        for err in result.errors:
            print(f"       - {err.phase}: {err.message}")
        print()

    # 6. Busca evidências do ART-006
    print("=" * 80)
    print("EVIDÊNCIA 1: Chunks relacionados ao ART-006")
    print("=" * 80)

    art006_chunks = [c for c in result.chunks if "ART-006" in c.chunk_id]
    print(f"Total de chunks com 'ART-006' no chunk_id: {len(art006_chunks)}")
    print()

    for chunk in art006_chunks:
        skip_flag = "(SKIP - não indexado)" if not chunk.dense_vector or len(chunk.dense_vector) == 0 else ""
        print(f"  - node_id: {chunk.node_id}")
        print(f"    device_type: {chunk.device_type}")
        print(f"    text_len: {len(chunk.text)} chars")
        print(f"    dense_vector: {len(chunk.dense_vector) if chunk.dense_vector else 0} dims {skip_flag}")
        print()

    # 7. Verifica split
    print("=" * 80)
    print("EVIDÊNCIA 2: Verificação do Split")
    print("=" * 80)

    parts = [c for c in art006_chunks if "-P" in c.chunk_id]
    parent = [c for c in art006_chunks if c.chunk_id == "LEI-14133-2021#ART-006"]

    print(f"Partes encontradas: {len(parts)}")
    print(f"Pai canônico encontrado: {len(parent)}")
    print()

    if parent:
        p = parent[0]
        has_vector = p.dense_vector and len(p.dense_vector) > 0
        print(f"[{'FALHA' if has_vector else 'OK'}] Pai ART-006 {'TEM' if has_vector else 'NÃO TEM'} embedding (esperado: não ter)")
        print(f"       text_len: {len(p.text)} chars")
    else:
        print("[INFO] Pai canônico não está na lista de chunks processados (esperado se filtrado)")

    print()

    # 8. Verifica partes
    print("=" * 80)
    print("EVIDÊNCIA 3: Detalhes das Partes")
    print("=" * 80)

    for part in sorted(parts, key=lambda x: x.chunk_id):
        vec_len = len(part.dense_vector) if part.dense_vector else 0
        print(f"  {part.chunk_id}:")
        print(f"    text_len: {len(part.text)} chars")
        print(f"    dense_vector: {vec_len} dims")
        print(f"    parent_chunk_id: {part.parent_chunk_id}")
        if vec_len == 1024:
            print(f"    [OK] Dimensão correta (1024)")
        elif vec_len > 0:
            print(f"    [AVISO] Dimensão inesperada ({vec_len})")
        else:
            print(f"    [FALHA] Sem embedding!")
        print()

    # 9. Resumo final
    print("=" * 80)
    print("RESUMO DA VALIDAÇÃO")
    print("=" * 80)

    # Verifica critérios
    criteria = []

    # Critério 1: ART-006 foi splitado
    if len(parts) >= 2:
        criteria.append(("ART-006 splitado em partes", True, f"{len(parts)} partes"))
    else:
        criteria.append(("ART-006 splitado em partes", False, f"Apenas {len(parts)} partes"))

    # Critério 2: Partes têm embeddings
    parts_with_embeddings = [p for p in parts if p.dense_vector and len(p.dense_vector) == 1024]
    if len(parts_with_embeddings) == len(parts) and len(parts) > 0:
        criteria.append(("Partes têm embeddings (1024d)", True, f"{len(parts_with_embeddings)}/{len(parts)}"))
    else:
        criteria.append(("Partes têm embeddings (1024d)", False, f"{len(parts_with_embeddings)}/{len(parts)}"))

    # Critério 3: Nenhuma parte excede 8000 chars
    oversized = [p for p in parts if len(p.text) > 8000]
    if len(oversized) == 0:
        criteria.append(("Nenhuma parte > 8000 chars", True, "OK"))
    else:
        criteria.append(("Nenhuma parte > 8000 chars", False, f"{len(oversized)} partes oversized"))

    # Critério 4: Pai não está nos chunks finais (foi filtrado)
    if len(parent) == 0:
        criteria.append(("Pai filtrado dos chunks finais", True, "Não encontrado (correto)"))
    else:
        p = parent[0]
        if not p.dense_vector or len(p.dense_vector) == 0:
            criteria.append(("Pai sem embedding", True, "Sem vetor (correto)"))
        else:
            criteria.append(("Pai sem embedding", False, f"Tem vetor de {len(p.dense_vector)} dims"))

    print()
    all_passed = True
    for name, passed, detail in criteria:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}: {detail}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("🎉 VALIDAÇÃO PASSOU - Split funcionando corretamente!")
    else:
        print("⚠️  VALIDAÇÃO COM PROBLEMAS - Revisar critérios que falharam")

    print()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
