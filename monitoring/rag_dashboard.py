"""
Dashboard RAG - Cliente para API em Produção

Conecta à API RAG hospedada na VPS (77.37.43.160) que usa
GPU remota no Google Cloud para embeddings, reranking e LLM.

Uso:
    streamlit run monitoring/rag_dashboard.py
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
import os

# Configuração da página
st.set_page_config(
    page_title="RAG Legal - Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

# URLs da API em produção
VPS_API_URL = os.getenv("VPS_API_URL", "http://77.37.43.160:8000")
GPU_SERVER_URL = os.getenv("GPU_SERVER_URL", "http://34.44.157.159:8000")
VLLM_URL = os.getenv("VLLM_URL", "http://34.44.157.159:8001")

# Token de autenticação (se necessário)
API_TOKEN = os.getenv("RAG_API_TOKEN", "")


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def get_headers():
    """Retorna headers com autenticação se configurada."""
    headers = {"Content-Type": "application/json"}
    if API_TOKEN:
        headers["Authorization"] = f"Bearer {API_TOKEN}"
    return headers


def check_service(url: str, endpoint: str = "/health", timeout: int = 5) -> dict:
    """Verifica status de um serviço."""
    try:
        start = time.time()
        resp = requests.get(f"{url}{endpoint}", timeout=timeout)
        latency = (time.time() - start) * 1000

        if resp.status_code == 200:
            return {
                "status": "online",
                "latency_ms": round(latency, 1),
                "data": resp.json() if resp.text else {},
            }
        return {
            "status": "error",
            "code": resp.status_code,
            "message": resp.text[:100],
        }
    except requests.exceptions.Timeout:
        return {"status": "timeout"}
    except requests.exceptions.ConnectionError:
        return {"status": "offline"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def ask_question(query: str, config: dict) -> dict:
    """Envia pergunta para a API RAG."""
    try:
        payload = {
            "query": query,
            "mode": config.get("mode", "completo"),
            "use_hyde": config.get("use_hyde", False),
            "use_reranker": config.get("use_reranker", True),
            "use_cache": config.get("use_cache", True),
            "top_k": config.get("top_k", 5),
        }

        start = time.time()
        resp = requests.post(
            f"{VPS_API_URL}/api/v1/ask",
            json=payload,
            headers=get_headers(),
            timeout=120,  # 2 minutos para queries complexas
        )
        total_time = (time.time() - start) * 1000

        if resp.status_code == 200:
            data = resp.json()
            data["_client_latency_ms"] = round(total_time, 1)
            return {"success": True, "data": data}
        elif resp.status_code == 401:
            return {"success": False, "error": "Não autorizado. Configure o token."}
        else:
            return {
                "success": False,
                "error": f"Erro {resp.status_code}: {resp.text[:200]}",
            }
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Timeout - a API demorou muito para responder."}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Não foi possível conectar à API."}
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# PÁGINA: STATUS DOS SERVIÇOS
# =============================================================================

def page_status():
    """Página de status dos serviços."""
    st.header("Status dos Serviços")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("VPS API")
        with st.spinner("Verificando..."):
            status = check_service(VPS_API_URL, "/health")

        if status["status"] == "online":
            st.success(f"Online ({status['latency_ms']}ms)")
            if status.get("data"):
                with st.expander("Detalhes"):
                    st.json(status["data"])
        elif status["status"] == "offline":
            st.error("Offline")
        elif status["status"] == "timeout":
            st.warning("Timeout")
        else:
            st.error(f"Erro: {status.get('message', 'Desconhecido')}")

    with col2:
        st.subheader("GPU Server")
        with st.spinner("Verificando..."):
            status = check_service(GPU_SERVER_URL, "/health")

        if status["status"] == "online":
            st.success(f"Online ({status['latency_ms']}ms)")
            if status.get("data"):
                with st.expander("Detalhes"):
                    st.json(status["data"])
        elif status["status"] == "offline":
            st.error("Offline")
        elif status["status"] == "timeout":
            st.warning("Timeout")
        else:
            st.error(f"Erro: {status.get('message', 'Desconhecido')}")

    with col3:
        st.subheader("vLLM")
        with st.spinner("Verificando..."):
            status = check_service(VLLM_URL, "/health")

        if status["status"] == "online":
            st.success(f"Online ({status['latency_ms']}ms)")
        elif status["status"] == "offline":
            st.error("Offline")
        elif status["status"] == "timeout":
            st.warning("Timeout")
        else:
            st.error(f"Erro: {status.get('message', 'Desconhecido')}")

    st.divider()

    # Informações de configuração
    st.subheader("Configuração")

    col1, col2 = st.columns(2)

    with col1:
        st.code(f"""
VPS_API_URL = {VPS_API_URL}
GPU_SERVER_URL = {GPU_SERVER_URL}
VLLM_URL = {VLLM_URL}
        """)

    with col2:
        st.write("**Variáveis de Ambiente:**")
        st.write("- `VPS_API_URL`: URL da API RAG")
        st.write("- `GPU_SERVER_URL`: URL do GPU Server")
        st.write("- `VLLM_URL`: URL do vLLM")
        st.write("- `RAG_API_TOKEN`: Token de autenticação")


# =============================================================================
# PÁGINA: PERGUNTAR
# =============================================================================

def page_ask():
    """Página para fazer perguntas ao sistema RAG."""
    st.header("Perguntar ao Sistema RAG")

    # Sidebar com configurações
    with st.sidebar:
        st.subheader("Configurações")

        mode = st.radio(
            "Modo",
            ["completo", "rapido"],
            help="Completo: usa HyDE e Reranker. Rápido: busca direta.",
        )

        st.divider()

        st.subheader("Opções Avançadas")

        use_hyde = st.checkbox(
            "HyDE (Query Expansion)",
            value=(mode == "completo"),
            help="Gera documentos hipotéticos para melhorar recall. Adiciona ~15-20s.",
        )

        use_reranker = st.checkbox(
            "Reranker",
            value=True,
            help="Reordena resultados com cross-encoder. Melhora precisão.",
        )

        use_cache = st.checkbox(
            "Cache Semântico",
            value=True,
            help="Usa cache para queries similares. Reduz latência.",
        )

        top_k = st.slider(
            "Top K",
            min_value=1,
            max_value=20,
            value=5,
            help="Número de chunks a recuperar.",
        )

        st.divider()

        # Token de autenticação
        st.subheader("Autenticação")
        token_input = st.text_input(
            "Token (opcional)",
            value=API_TOKEN,
            type="password",
            help="Token JWT para autenticação na API.",
        )
        if token_input != API_TOKEN:
            os.environ["RAG_API_TOKEN"] = token_input
            st.success("Token atualizado!")

    # Campo de pergunta
    query = st.text_area(
        "Digite sua pergunta:",
        placeholder="Ex: Quais são os critérios de julgamento previstos na Lei 14.133?",
        height=100,
    )

    col1, col2 = st.columns([1, 5])

    with col1:
        ask_button = st.button("Perguntar", type="primary", use_container_width=True)

    with col2:
        if mode == "rapido":
            st.info("Modo rápido: busca direta sem HyDE")
        else:
            st.info("Modo completo: HyDE + Reranker (mais preciso, mais lento)")

    # Processar pergunta
    if ask_button and query:
        config = {
            "mode": mode,
            "use_hyde": use_hyde,
            "use_reranker": use_reranker,
            "use_cache": use_cache,
            "top_k": top_k,
        }

        with st.spinner("Gerando resposta..."):
            result = ask_question(query, config)

        if result["success"]:
            data = result["data"]

            # Métricas
            st.divider()

            col1, col2, col3, col4 = st.columns(4)

            metadata = data.get("metadata", {})

            with col1:
                st.metric("Confiança", f"{data.get('confidence', 0) * 100:.1f}%")

            with col2:
                st.metric("Latência Total", f"{data.get('_client_latency_ms', 0):.0f}ms")

            with col3:
                st.metric("Retrieval", f"{metadata.get('retrieval_ms', 0):.0f}ms")

            with col4:
                st.metric("Geração", f"{metadata.get('generation_ms', 0):.0f}ms")

            # Cache hit?
            if metadata.get("from_cache"):
                st.success("Resposta recuperada do cache!")

            st.divider()

            # Resposta
            st.subheader("Resposta")
            st.markdown(data.get("answer", "Sem resposta"))

            st.divider()

            # Citações
            citations = data.get("citations", [])
            if citations:
                st.subheader(f"Citações ({len(citations)})")

                for i, citation in enumerate(citations, 1):
                    # Formata título da citação
                    title = citation.get("text", citation.get("short", f"Citação {i}"))
                    if len(title) > 80:
                        title = title[:80] + "..."

                    with st.expander(f"[{i}] {title}", expanded=(i <= 2)):
                        col1, col2 = st.columns(2)

                        with col1:
                            doc_type = citation.get("document_type", "")
                            doc_num = citation.get("document_number", "")
                            year = citation.get("year", "")
                            st.write(f"**Documento:** {doc_type} {doc_num}/{year}")

                            article = citation.get("article", "")
                            if article:
                                st.write(f"**Artigo:** {article}")

                        with col2:
                            device = citation.get("device", "")
                            device_num = citation.get("device_number", "")
                            if device:
                                st.write(f"**Dispositivo:** {device}")
                            if device_num:
                                st.write(f"**Número:** {device_num}")

                            relevance = citation.get("relevance", 0)
                            st.write(f"**Relevância:** {relevance:.2f}")

            st.divider()

            # Fontes
            sources = data.get("sources", [])
            if sources:
                st.subheader(f"Fontes ({len(sources)})")

                import pandas as pd
                sources_df = pd.DataFrame([
                    {
                        "Documento": s.get("document_id", ""),
                        "Tipo": s.get("tipo_documento", ""),
                        "Número": s.get("numero", ""),
                        "Ano": s.get("ano", ""),
                    }
                    for s in sources
                ])
                st.dataframe(sources_df, hide_index=True, use_container_width=True)

            # JSON completo (debug)
            with st.expander("Ver resposta JSON completa"):
                st.json(data)

        else:
            st.error(f"Erro: {result['error']}")


# =============================================================================
# PÁGINA: EXEMPLOS
# =============================================================================

def page_examples():
    """Página com perguntas de exemplo."""
    st.header("Perguntas de Exemplo")

    st.write("Clique em uma pergunta para testá-la:")

    examples = [
        {
            "categoria": "Lei 14.133/2021 - Licitações",
            "perguntas": [
                "Quais são os critérios de julgamento previstos na Lei 14.133?",
                "Quando é possível fazer contratação direta?",
                "Qual o prazo de vigência dos contratos administrativos?",
                "O que é o estudo técnico preliminar (ETP)?",
                "Quais são as modalidades de licitação?",
            ],
        },
        {
            "categoria": "IN 65/2021 - Pesquisa de Preços",
            "perguntas": [
                "Como fazer pesquisa de preços em contratações públicas?",
                "Qual o prazo de validade dos preços pesquisados?",
                "Quantas fontes são necessárias para pesquisa de preços?",
                "Quando a pesquisa de preços pode ser dispensada?",
            ],
        },
        {
            "categoria": "IN 58/2022 - ETP",
            "perguntas": [
                "Quando o ETP pode ser dispensado?",
                "Quem são os responsáveis pela elaboração do ETP?",
                "O que deve conter o estudo técnico preliminar?",
            ],
        },
    ]

    for category in examples:
        st.subheader(category["categoria"])

        cols = st.columns(2)
        for i, pergunta in enumerate(category["perguntas"]):
            with cols[i % 2]:
                if st.button(pergunta, key=f"ex_{hash(pergunta)}", use_container_width=True):
                    st.session_state["example_query"] = pergunta
                    st.switch_page = "Perguntar"
                    st.rerun()

        st.divider()

    # Se uma pergunta foi selecionada, mostra na sidebar
    if "example_query" in st.session_state:
        st.sidebar.success(f"Pergunta selecionada: {st.session_state['example_query'][:50]}...")
        st.sidebar.info("Vá para a página 'Perguntar' para ver a resposta.")


# =============================================================================
# PÁGINA: SOBRE
# =============================================================================

def page_about():
    """Página sobre o sistema."""
    st.header("Sobre o Sistema RAG Legal")

    st.markdown("""
    ## Arquitetura

    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Google Cloud VM (GPU)                        │
    │                    34.44.157.159                                │
    │                                                                 │
    │   GPU Server (:8000)          vLLM (:8001)                      │
    │   - BGE-M3 (embeddings)       - Qwen3-8B-AWQ                    │
    │   - BGE-Reranker              - Geração de texto                │
    └─────────────────────────────────────────────────────────────────┘
                            │
                            │ HTTP
                            ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    VPS Hostinger                                │
    │                    77.37.43.160                                 │
    │                                                                 │
    │   RAG API (:8000)                                               │
    │   - /api/v1/ask (perguntas)                                     │
    │   - /api/v1/search (busca)                                      │
    │   - Milvus (vetores)                                            │
    │   - Redis (cache)                                               │
    │   - PostgreSQL (usuários)                                       │
    └─────────────────────────────────────────────────────────────────┘
    ```

    ## Tecnologias

    | Componente | Tecnologia |
    |------------|------------|
    | Embeddings | BGE-M3 (dense 1024d + sparse) |
    | Reranker | BGE-Reranker-v2-m3 (cross-encoder) |
    | LLM | Qwen3-8B-AWQ (quantizado) |
    | Vector DB | Milvus 2.6 |
    | Cache | Redis + Cache Semântico |
    | API | FastAPI |

    ## Documentos Indexados

    - Lei 14.133/2021 (Nova Lei de Licitações)
    - IN 65/2021 (Pesquisa de Preços)
    - IN 58/2022 (Estudo Técnico Preliminar)

    ## Repositórios

    - [rag-gpu-server](https://github.com/euteajudo/rag-gpu-server) - GPU Server
    - vector_govi_2/extracao - API RAG + Extração
    """)


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Sidebar - Navegação
    st.sidebar.title("⚖️ RAG Legal")
    st.sidebar.caption("Sistema de Consulta à Legislação")

    st.sidebar.divider()

    # Menu de páginas
    page = st.sidebar.radio(
        "Navegação",
        ["Perguntar", "Status", "Exemplos", "Sobre"],
        label_visibility="collapsed",
    )

    st.sidebar.divider()

    # Indicadores rápidos de status
    st.sidebar.caption("Status dos Serviços")

    col1, col2, col3 = st.sidebar.columns(3)

    # Check rápido (sem bloquear)
    try:
        resp = requests.get(f"{VPS_API_URL}/health", timeout=2)
        col1.success("API")
    except:
        col1.error("API")

    try:
        resp = requests.get(f"{GPU_SERVER_URL}/health", timeout=2)
        col2.success("GPU")
    except:
        col2.error("GPU")

    try:
        resp = requests.get(f"{VLLM_URL}/health", timeout=2)
        col3.success("LLM")
    except:
        col3.error("LLM")

    # Renderiza página selecionada
    if page == "Perguntar":
        # Verifica se tem pergunta de exemplo
        if "example_query" in st.session_state:
            query = st.session_state.pop("example_query")
            st.session_state["prefill_query"] = query
        page_ask()
    elif page == "Status":
        page_status()
    elif page == "Exemplos":
        page_examples()
    elif page == "Sobre":
        page_about()


if __name__ == "__main__":
    main()
