"""
Configuração global do pytest para testes do rag-gpu-server.

PR3 v2 - Hard Reset RAG Architecture

Este arquivo configura o PYTHONPATH para que os imports funcionem corretamente.
"""

import sys
from pathlib import Path

# Adiciona o diretório src ao path para que os imports funcionem
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Também adiciona o diretório raiz do projeto
root_path = Path(__file__).parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))
