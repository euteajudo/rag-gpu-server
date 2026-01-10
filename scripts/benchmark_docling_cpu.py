"""
Benchmark Docling com CPU.
Usa SimplePdfPipeline sem aceleração GPU.
"""

import os
import sys
import time
import tempfile
import requests

# Adiciona src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def download_test_pdf():
    """Baixa um PDF de teste (IN 58/2022)."""
    url = "https://www.gov.br/compras/pt-br/acesso-a-informacao/legislacao/instrucoes-normativas/instrucao-normativa-seges-me-no-58-de-8-de-agosto-de-2022/@@download/file"

    print("Baixando PDF de teste...")

    try:
        response = requests.get(url, timeout=30, allow_redirects=True)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(response.content)
            print(f"PDF baixado: {len(response.content)} bytes")
            return f.name
    except Exception as e:
        print(f"Erro ao baixar PDF: {e}")
        local_path = "/workspace/test.pdf"
        if os.path.exists(local_path):
            print(f"Usando PDF local: {local_path}")
            return local_path
        raise


def benchmark_cpu():
    """Executa benchmark com CPU."""
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions

    print("\n" + "="*60)
    print("BENCHMARK DOCLING - CPU")
    print("="*60)

    # Configuração CPU - sem aceleração GPU
    pipeline_options = PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
    )
    pipeline_options.do_ocr = False

    print("\nConfigurações:")
    print(f"  - Device: CPU")
    print(f"  - Pipeline: SimplePdfPipeline (padrão)")
    print(f"  - OCR: Desabilitado")

    # Inicializa converter
    print("\nInicializando converter (carregando modelos na CPU)...")
    init_start = time.perf_counter()

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )

    init_time = time.perf_counter() - init_start
    print(f"Tempo de inicialização: {init_time:.2f}s")

    # Baixa PDF de teste
    pdf_path = download_test_pdf()

    # Warmup - primeira conversão carrega modelos
    print("\nWarmup (primeira conversão)...")
    warmup_start = time.perf_counter()
    result = converter.convert(pdf_path)
    warmup_time = time.perf_counter() - warmup_start
    print(f"Warmup concluído: {warmup_time:.2f}s")

    # Benchmark - 3 execuções
    print("\nExecutando benchmark (3 iterações)...")
    times = []

    for i in range(3):
        start = time.perf_counter()
        result = converter.convert(pdf_path)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        markdown = result.document.export_to_markdown()
        print(f"  Iteração {i+1}: {elapsed:.2f}s ({len(markdown)} caracteres)")

    # Resultados
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print("\n" + "-"*60)
    print("RESULTADOS CPU:")
    print(f"  Tempo médio: {avg_time:.2f}s")
    print(f"  Tempo mínimo: {min_time:.2f}s")
    print(f"  Tempo máximo: {max_time:.2f}s")
    print("-"*60)

    # Limpa PDF temporário
    if pdf_path.startswith(tempfile.gettempdir()):
        os.remove(pdf_path)

    return {
        "device": "CPU",
        "init_time": init_time,
        "warmup_time": warmup_time,
        "times": times,
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
    }


if __name__ == "__main__":
    results = benchmark_cpu()
    print(f"\n✅ Benchmark CPU concluído: média {results['avg_time']:.2f}s")
