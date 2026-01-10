"""
Comparação de performance: GPU vs CPU para Docling.
Executa ambos os benchmarks e mostra resultado comparativo.
"""

import os
import sys
import time
import tempfile
import requests

# Adiciona src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def download_test_pdf():
    """Baixa um PDF de teste."""
    # Primeiro tenta usar PDF local
    local_path = "/workspace/test.pdf"
    if os.path.exists(local_path):
        print(f"Usando PDF local: {local_path}")
        with open(local_path, 'rb') as f:
            size = len(f.read())
        print(f"Tamanho: {size} bytes")
        return local_path

    # Se não existir, baixa da internet
    url = "https://www.gov.br/compras/pt-br/acesso-a-informacao/legislacao/instrucoes-normativas/instrucao-normativa-seges-me-no-58-de-8-de-agosto-de-2022/@@download/file"

    print("Baixando PDF de teste...")
    response = requests.get(url, timeout=60, allow_redirects=True)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(response.content)
        print(f"PDF baixado: {len(response.content)} bytes")
        return f.name


def benchmark_gpu(pdf_path: str) -> dict:
    """Benchmark com GPU."""
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline

    print("\n" + "="*60)
    print("BENCHMARK GPU (CUDA)")
    print("="*60)

    pipeline_options = ThreadedPdfPipelineOptions(
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CUDA),
        layout_batch_size=64,
        table_batch_size=4,
        ocr_batch_size=4,
    )
    pipeline_options.do_ocr = False

    # Inicialização
    print("Inicializando (carregando modelos na GPU)...")
    init_start = time.perf_counter()
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=ThreadedStandardPdfPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )
    init_time = time.perf_counter() - init_start
    print(f"Inicialização: {init_time:.2f}s")

    # Warmup
    print("Warmup...")
    warmup_start = time.perf_counter()
    result = converter.convert(pdf_path)
    warmup_time = time.perf_counter() - warmup_start
    print(f"Warmup: {warmup_time:.2f}s")

    # Benchmark
    print("Executando 3 iterações...")
    times = []
    for i in range(3):
        start = time.perf_counter()
        result = converter.convert(pdf_path)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        markdown = result.document.export_to_markdown()
        print(f"  [{i+1}] {elapsed:.2f}s ({len(markdown)} chars)")

    return {
        "device": "GPU",
        "init_time": init_time,
        "warmup_time": warmup_time,
        "times": times,
        "avg_time": sum(times) / len(times),
    }


def benchmark_cpu(pdf_path: str) -> dict:
    """Benchmark com CPU."""
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions

    print("\n" + "="*60)
    print("BENCHMARK CPU")
    print("="*60)

    pipeline_options = PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
    )
    pipeline_options.do_ocr = False

    # Inicialização
    print("Inicializando (carregando modelos na CPU)...")
    init_start = time.perf_counter()
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )
    init_time = time.perf_counter() - init_start
    print(f"Inicialização: {init_time:.2f}s")

    # Warmup
    print("Warmup...")
    warmup_start = time.perf_counter()
    result = converter.convert(pdf_path)
    warmup_time = time.perf_counter() - warmup_start
    print(f"Warmup: {warmup_time:.2f}s")

    # Benchmark
    print("Executando 3 iterações...")
    times = []
    for i in range(3):
        start = time.perf_counter()
        result = converter.convert(pdf_path)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        markdown = result.document.export_to_markdown()
        print(f"  [{i+1}] {elapsed:.2f}s ({len(markdown)} chars)")

    return {
        "device": "CPU",
        "init_time": init_time,
        "warmup_time": warmup_time,
        "times": times,
        "avg_time": sum(times) / len(times),
    }


def main():
    print("\n" + "#"*60)
    print("#  DOCLING BENCHMARK: GPU vs CPU")
    print("#"*60)

    # Baixa PDF
    pdf_path = download_test_pdf()

    # Executa benchmarks
    gpu_results = benchmark_gpu(pdf_path)
    cpu_results = benchmark_cpu(pdf_path)

    # Comparação
    speedup = cpu_results["avg_time"] / gpu_results["avg_time"]

    print("\n" + "="*60)
    print("COMPARAÇÃO FINAL")
    print("="*60)
    print(f"\n{'Métrica':<25} {'GPU':>12} {'CPU':>12} {'Speedup':>12}")
    print("-"*60)
    print(f"{'Inicialização':<25} {gpu_results['init_time']:>10.2f}s {cpu_results['init_time']:>10.2f}s")
    print(f"{'Warmup':<25} {gpu_results['warmup_time']:>10.2f}s {cpu_results['warmup_time']:>10.2f}s")
    print(f"{'Tempo médio (3 iter)':<25} {gpu_results['avg_time']:>10.2f}s {cpu_results['avg_time']:>10.2f}s {speedup:>10.1f}x")
    print("-"*60)

    if speedup > 1:
        print(f"\n🚀 GPU é {speedup:.1f}x mais rápida que CPU!")
    else:
        print(f"\n⚠️ CPU foi mais rápida (speedup: {speedup:.2f}x)")

    print("\nDetalhes por iteração:")
    print(f"  GPU: {[f'{t:.2f}s' for t in gpu_results['times']]}")
    print(f"  CPU: {[f'{t:.2f}s' for t in cpu_results['times']]}")

    # Limpa PDF temporário
    if pdf_path.startswith(tempfile.gettempdir()):
        os.remove(pdf_path)

    return gpu_results, cpu_results


if __name__ == "__main__":
    main()
