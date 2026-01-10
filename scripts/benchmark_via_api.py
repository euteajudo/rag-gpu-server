"""
Benchmark Docling via API (usa modelos já carregados no servidor).
"""

import os
import time
import requests

API_URL = "http://localhost:8000"
API_KEY = "vg_gpu_internal_2025"
PDF_PATH = "/workspace/test.pdf"


def benchmark_via_api():
    """Benchmark chamando API de ingestão."""
    print("\n" + "="*60)
    print("BENCHMARK DOCLING VIA API - MODELOS PRÉ-CARREGADOS")
    print("="*60)

    if not os.path.exists(PDF_PATH):
        print(f"ERRO: PDF não encontrado: {PDF_PATH}")
        return

    with open(PDF_PATH, "rb") as f:
        pdf_content = f.read()

    print(f"PDF: {PDF_PATH} ({len(pdf_content)} bytes)")

    headers = {"X-GPU-API-Key": API_KEY}

    # Benchmark - 5 execuções
    print("\nExecutando 5 conversões via API...")
    times = []

    for i in range(5):
        start = time.perf_counter()

        response = requests.post(
            f"{API_URL}/ingest",
            headers=headers,
            files={"file": ("test.pdf", pdf_content, "application/pdf")},
            data={
                "document_id": f"BENCHMARK-{i+1}",
                "tipo_documento": "IN",
                "numero": "test",
                "ano": 2024,
                "skip_embeddings": True,  # Só queremos medir Docling
            },
            timeout=300,
        )

        elapsed = time.perf_counter() - start

        if response.status_code == 200:
            result = response.json()
            # Pega tempo do Docling especificamente
            phases = result.get("phases", [])
            docling_phase = None
            for p in phases:
                if isinstance(p, dict) and p.get("name") == "docling":
                    docling_phase = p
                    break

            docling_time = docling_phase.get("duration_seconds", elapsed) if docling_phase else elapsed

            times.append(docling_time)
            print(f"  [{i+1}] Docling: {docling_time:.3f}s | Total API: {elapsed:.2f}s")
        else:
            print(f"  [{i+1}] ERRO: {response.status_code} - {response.text[:200]}")

    if not times:
        print("Nenhuma conversão bem-sucedida!")
        return

    # Resultados
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print("\n" + "-"*60)
    print("RESULTADOS DOCLING (via API, modelos pré-carregados):")
    print(f"  Tempo médio: {avg_time:.3f}s")
    print(f"  Tempo mínimo: {min_time:.3f}s")
    print(f"  Tempo máximo: {max_time:.3f}s")
    print("-"*60)

    # Comparação
    cpu_avg = 3.49  # Do benchmark anterior
    gpu_cold = 0.94  # Do benchmark anterior (após warmup individual)
    speedup_cpu = cpu_avg / avg_time
    speedup_cold = gpu_cold / avg_time if avg_time > 0 else 0

    print(f"\n📊 Comparação:")
    print(f"   CPU: {cpu_avg:.2f}s → Speedup: {speedup_cpu:.1f}x")
    print(f"   GPU (cold): {gpu_cold:.2f}s → Similar: {speedup_cold:.1f}x")


if __name__ == "__main__":
    benchmark_via_api()
