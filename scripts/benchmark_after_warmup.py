"""
Benchmark Docling APÓS warmup (modelos já carregados na GPU).
Mostra a performance real de conversão de PDFs.
"""

import os
import sys
import time

sys.path.insert(0, '/workspace/rag-gpu-server/src')


def benchmark():
    """Benchmark com modelos já carregados."""
    from ingestion.pipeline import get_pipeline

    print("\n" + "="*60)
    print("BENCHMARK DOCLING - APÓS WARMUP")
    print("="*60)

    # Pega o pipeline (já tem o converter carregado)
    pipeline = get_pipeline()

    print(f"\nDocling já carregado: {pipeline.is_warmed_up()}")

    if not pipeline.is_warmed_up():
        print("ERRO: Docling não está carregado! Execute o warmup primeiro.")
        return

    pdf_path = "/workspace/test.pdf"
    if not os.path.exists(pdf_path):
        print(f"ERRO: PDF de teste não encontrado: {pdf_path}")
        return

    print(f"Usando PDF: {pdf_path}")
    print(f"Tamanho: {os.path.getsize(pdf_path)} bytes")

    # Benchmark - 5 execuções
    print("\nExecutando 5 conversões...")
    times = []

    for i in range(5):
        start = time.perf_counter()
        result = pipeline.docling_converter.convert(pdf_path)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        markdown = result.document.export_to_markdown()
        print(f"  [{i+1}] {elapsed:.3f}s ({len(markdown)} chars)")

    # Resultados
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print("\n" + "-"*60)
    print("RESULTADOS (modelos já carregados na GPU):")
    print(f"  Tempo médio: {avg_time:.3f}s")
    print(f"  Tempo mínimo: {min_time:.3f}s")
    print(f"  Tempo máximo: {max_time:.3f}s")
    print("-"*60)

    # Comparação com benchmark anterior (CPU)
    cpu_avg = 3.49  # Do benchmark anterior
    speedup = cpu_avg / avg_time

    print(f"\n📊 Comparação com CPU ({cpu_avg:.2f}s média):")
    print(f"   GPU após warmup: {avg_time:.3f}s")
    print(f"   Speedup: {speedup:.1f}x mais rápida")

    return times


if __name__ == "__main__":
    benchmark()
