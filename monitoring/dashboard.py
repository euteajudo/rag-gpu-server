"""
Dashboard de Monitoramento - VM GPU Google Cloud

Monitora métricas da VM em tempo real:
- CPU usage
- Memória
- GPU (NVIDIA)
- I/O de disco
- Uso de disco

Uso:
    streamlit run monitoring/dashboard.py
"""

import streamlit as st
import pandas as pd
import subprocess
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
import json

# Configuração da página
st.set_page_config(
    page_title="GPU Server Monitor",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configurações centralizadas (importadas de config.py)
from config import (
    VM_IP, VPS_IP, SSH_USER, SSH_KEY, SSH_PORT, VPS_SSH_KEY,
    GPU_SERVER_URL, VLLM_BASE_URL, VM_NAME, GPU_PROVIDER
)


@dataclass
class VMMetrics:
    """Métricas coletadas da VM."""
    timestamp: datetime
    # CPU
    cpu_percent: float
    cpu_count: int
    load_1min: float
    load_5min: float
    load_15min: float
    # Memória
    mem_total_gb: float
    mem_used_gb: float
    mem_free_gb: float
    mem_percent: float
    # GPU
    gpu_name: str
    gpu_temp: float
    gpu_util: float
    gpu_mem_used_mb: float
    gpu_mem_total_mb: float
    gpu_mem_percent: float
    gpu_power_draw: float
    gpu_power_limit: float
    # Disco
    disk_total_gb: float
    disk_used_gb: float
    disk_free_gb: float
    disk_percent: float
    # I/O
    io_read_mb: float
    io_write_mb: float
    io_read_count: int
    io_write_count: int
    # Rede
    net_recv_mb: float
    net_sent_mb: float
    # Status serviços
    vllm_status: str
    gpu_server_status: str


def run_ssh_command(command: str, timeout: int = 10) -> tuple[bool, str]:
    """Executa comando via SSH na VM GPU."""
    # -T: desabilita pseudo-terminal (evita MOTD)
    # -o LogLevel=ERROR: suprime mensagens de aviso
    # -p: porta SSH (diferente para RunPod)
    port_arg = f"-p {SSH_PORT}" if SSH_PORT != 22 else ""
    ssh_cmd = f'ssh -T {port_arg} -i {SSH_KEY} -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o LogLevel=ERROR {SSH_USER}@{VM_IP} "{command}"'
    try:
        result = subprocess.run(
            ssh_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def run_vps_command(command: str, timeout: int = 10) -> tuple[bool, str]:
    """Executa comando via SSH na VPS."""
    ssh_cmd = f'ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@{VPS_IP} "{command}"'
    try:
        result = subprocess.run(
            ssh_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


@dataclass
class EnrichmentStatus:
    """Status do enriquecimento."""
    task_id: str
    status: str
    total_chunks: int
    chunks_completed: int
    chunks_failed: int
    progress_percent: float
    errors: list
    started_at: Optional[str] = None


def get_enrichment_status() -> Optional[EnrichmentStatus]:
    """Busca status do enriquecimento mais recente da VPS."""
    # Busca todas as chaves de enriquecimento
    success, keys_output = run_vps_command(
        'docker exec redis redis-cli -n 2 KEYS "enrich:task:*"'
    )

    if not success or not keys_output:
        return None

    keys = keys_output.strip().split('\n')
    if not keys or keys[0] == '':
        return None

    # Busca o status mais recente (com maior progresso ou mais recente)
    best_status = None
    for key in keys:
        key = key.strip()
        if not key:
            continue

        success, data = run_vps_command(
            f'docker exec redis redis-cli -n 2 GET "{key}"'
        )

        if success and data:
            try:
                status_data = json.loads(data)
                total = status_data.get('total_chunks', 0)
                completed = status_data.get('chunks_completed', 0)
                failed = status_data.get('chunks_failed', 0)
                progress = ((completed + failed) / total * 100) if total > 0 else 0

                status = EnrichmentStatus(
                    task_id=status_data.get('task_id', key.split(':')[-1]),
                    status=status_data.get('status', 'unknown'),
                    total_chunks=total,
                    chunks_completed=completed,
                    chunks_failed=failed,
                    progress_percent=progress,
                    errors=status_data.get('errors', [])[:10],
                    started_at=status_data.get('started_at'),
                )

                # Prefere tarefas em andamento com progresso
                if best_status is None or (status.status == 'processing' and status.progress_percent > 0):
                    best_status = status

            except json.JSONDecodeError:
                continue

    return best_status


def get_celery_queue_size() -> int:
    """Retorna tamanho da fila Celery."""
    success, output = run_vps_command('docker exec redis redis-cli LLEN celery')
    if success:
        try:
            return int(output)
        except ValueError:
            return 0
    return 0


def get_celery_workers_count() -> int:
    """Retorna número de workers Celery ativos."""
    success, output = run_vps_command(
        'ps aux | grep "celery.*worker" | grep -v grep | wc -l'
    )
    if success:
        try:
            return int(output)
        except ValueError:
            return 0
    return 0


def start_enrichment() -> tuple[bool, str]:
    """Inicia novo processo de enriquecimento."""
    # Limpa fila e tasks anteriores
    run_vps_command('docker exec redis redis-cli FLUSHDB')
    run_vps_command('docker exec redis redis-cli -n 2 FLUSHDB')

    # Reinicia Celery
    run_vps_command('systemctl restart rag-celery')

    # Dispara enriquecimento via script (usa URLs do config.py)
    success, output = run_vps_command(f'''
cd /opt/rag-api && source venv/bin/activate && PYTHONPATH=/opt/rag-api python3 << 'EOF'
import os
import sys
import json
import uuid
sys.path.insert(0, "/opt/rag-api")
os.environ["MILVUS_HOST"] = "127.0.0.1"
os.environ["REDIS_HOST"] = "127.0.0.1"
os.environ["GPU_SERVER_URL"] = "{GPU_SERVER_URL}"
os.environ["VLLM_BASE_URL"] = "{VLLM_BASE_URL}"

import redis
from pymilvus import connections, Collection
from src.enrichment.tasks import enrich_chunk_task

# Conecta Redis
r = redis.Redis(host="127.0.0.1", port=6379, db=2, decode_responses=True)

# Conecta Milvus
connections.connect(alias="enrich", host="127.0.0.1", port="19530")
collection = Collection("leis_v3", using="enrich")
collection.load()

# Query chunks pendentes (sem context_header)
results = collection.query(
    expr='context_header == ""',
    output_fields=["chunk_id", "text", "device_type", "article_number", "document_id", "tipo_documento", "numero", "ano"],
    limit=10000,
)

connections.disconnect("enrich")

if not results:
    print(json.dumps({"success": True, "task_id": "none", "chunks": 0}))
else:
    # Gera task_id
    task_id = str(uuid.uuid4())[:8]

    # Salva status inicial
    status = {
        "task_id": task_id,
        "status": "processing",
        "total_chunks": len(results),
        "chunks_queued": 0,
        "chunks_completed": 0,
        "chunks_failed": 0,
        "errors": [],
    }
    r.setex(f"enrich:task:{task_id}", 86400, json.dumps(status))

    # Dispara tasks
    for chunk in results:
        try:
            enrich_chunk_task.delay(
                chunk_id=chunk["chunk_id"],
                text=chunk["text"],
                device_type=chunk.get("device_type", ""),
                article_number=chunk.get("article_number", ""),
                document_id=chunk["document_id"],
                document_type=chunk.get("tipo_documento", ""),
                number=chunk.get("numero", ""),
                year=chunk.get("ano", 0),
                enrich_task_id=task_id,
            )
            status["chunks_queued"] += 1
        except Exception as e:
            status["errors"].append(str(e)[:100])

    r.setex(f"enrich:task:{task_id}", 86400, json.dumps(status))
    print(json.dumps({"success": True, "task_id": task_id, "chunks": len(results)}))
EOF
''', timeout=60)

    if success:
        try:
            result = json.loads(output)
            if result.get('success'):
                return True, f"Enriquecimento iniciado! Task ID: {result.get('task_id')}, Chunks: {result.get('chunks')}"
            return False, "Falha ao iniciar enriquecimento"
        except json.JSONDecodeError:
            return False, f"Erro ao parsear resposta: {output}"
    return False, output


def ensure_metrics_script() -> bool:
    """Garante que o script de coleta existe na VM."""
    check_cmd = "test -f /tmp/collect_metrics.py && echo EXISTS"
    success, output = run_ssh_command(check_cmd, timeout=5)
    if success and "EXISTS" in output:
        return True

    # Script não existe, cria
    script_content = '''import json
import subprocess
import os

metrics = {}

try:
    with open('/proc/loadavg', 'r') as f:
        loads = f.read().split()
        metrics['load_1min'] = float(loads[0])
        metrics['load_5min'] = float(loads[1])
        metrics['load_15min'] = float(loads[2])
    metrics['cpu_count'] = os.cpu_count()
    metrics['cpu_percent'] = min(100, (metrics['load_1min'] / metrics['cpu_count']) * 100)
except Exception as e:
    metrics['cpu_error'] = str(e)

try:
    with open('/proc/meminfo', 'r') as f:
        meminfo = {}
        for line in f:
            parts = line.split()
            meminfo[parts[0].rstrip(':')] = int(parts[1])
    mem_total = meminfo['MemTotal'] / 1024 / 1024
    mem_available = meminfo.get('MemAvailable', meminfo['MemFree']) / 1024 / 1024
    mem_used = mem_total - mem_available
    metrics['mem_total_gb'] = round(mem_total, 2)
    metrics['mem_used_gb'] = round(mem_used, 2)
    metrics['mem_free_gb'] = round(mem_available, 2)
    metrics['mem_percent'] = round((mem_used / mem_total) * 100, 1)
except Exception as e:
    metrics['mem_error'] = str(e)

try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit', '--format=csv,noheader,nounits'], capture_output=True, text=True)
    if result.returncode == 0:
        parts = result.stdout.strip().split(', ')
        metrics['gpu_name'] = parts[0]
        metrics['gpu_temp'] = float(parts[1])
        metrics['gpu_util'] = float(parts[2])
        metrics['gpu_mem_used_mb'] = float(parts[3])
        metrics['gpu_mem_total_mb'] = float(parts[4])
        metrics['gpu_mem_percent'] = round((float(parts[3]) / float(parts[4])) * 100, 1)
        metrics['gpu_power_draw'] = float(parts[5])
        metrics['gpu_power_limit'] = float(parts[6])
except Exception as e:
    metrics['gpu_error'] = str(e)

try:
    statvfs = os.statvfs('/')
    disk_total = (statvfs.f_frsize * statvfs.f_blocks) / 1024 / 1024 / 1024
    disk_free = (statvfs.f_frsize * statvfs.f_bavail) / 1024 / 1024 / 1024
    disk_used = disk_total - disk_free
    metrics['disk_total_gb'] = round(disk_total, 2)
    metrics['disk_used_gb'] = round(disk_used, 2)
    metrics['disk_free_gb'] = round(disk_free, 2)
    metrics['disk_percent'] = round((disk_used / disk_total) * 100, 1)
except Exception as e:
    metrics['disk_error'] = str(e)

try:
    with open('/proc/diskstats', 'r') as f:
        io_read = io_write = io_read_count = io_write_count = 0
        for line in f:
            parts = line.split()
            if len(parts) >= 14:
                io_read_count += int(parts[3])
                io_read += int(parts[5])
                io_write_count += int(parts[7])
                io_write += int(parts[9])
    metrics['io_read_mb'] = round((io_read * 512) / 1024 / 1024, 2)
    metrics['io_write_mb'] = round((io_write * 512) / 1024 / 1024, 2)
    metrics['io_read_count'] = io_read_count
    metrics['io_write_count'] = io_write_count
except Exception as e:
    metrics['io_error'] = str(e)

try:
    with open('/proc/net/dev', 'r') as f:
        lines = f.readlines()[2:]
        recv_bytes = sent_bytes = 0
        for line in lines:
            parts = line.split()
            if len(parts) >= 10:
                recv_bytes += int(parts[1])
                sent_bytes += int(parts[9])
    metrics['net_recv_mb'] = round(recv_bytes / 1024 / 1024, 2)
    metrics['net_sent_mb'] = round(sent_bytes / 1024 / 1024, 2)
except Exception as e:
    metrics['net_error'] = str(e)

try:
    result = subprocess.run(['ss', '-tlnp'], capture_output=True, text=True)
    metrics['vllm_status'] = 'running' if ':8001' in result.stdout else 'stopped'
    metrics['gpu_server_status'] = 'running' if ':8000' in result.stdout else 'stopped'
except Exception as e:
    metrics['service_error'] = str(e)

print(json.dumps(metrics))
'''
    # Escapa aspas simples para bash
    escaped = script_content.replace("'", "'\\''")
    create_cmd = f"echo '{escaped}' > /tmp/collect_metrics.py"
    success, _ = run_ssh_command(create_cmd, timeout=10)
    return success


def collect_metrics() -> Optional[VMMetrics]:
    """Coleta todas as métricas da VM."""
    # Garante que o script existe
    if not ensure_metrics_script():
        st.error("Erro ao criar script de métricas na VM")
        return None

    # Executa o script
    success, output = run_ssh_command("python3 /tmp/collect_metrics.py", timeout=15)

    if not success:
        st.error(f"Erro ao coletar métricas: {output}")
        return None

    try:
        data = json.loads(output)

        return VMMetrics(
            timestamp=datetime.now(),
            # CPU
            cpu_percent=data.get('cpu_percent', 0),
            cpu_count=data.get('cpu_count', 0),
            load_1min=data.get('load_1min', 0),
            load_5min=data.get('load_5min', 0),
            load_15min=data.get('load_15min', 0),
            # Memória
            mem_total_gb=data.get('mem_total_gb', 0),
            mem_used_gb=data.get('mem_used_gb', 0),
            mem_free_gb=data.get('mem_free_gb', 0),
            mem_percent=data.get('mem_percent', 0),
            # GPU
            gpu_name=data.get('gpu_name', 'N/A'),
            gpu_temp=data.get('gpu_temp', 0),
            gpu_util=data.get('gpu_util', 0),
            gpu_mem_used_mb=data.get('gpu_mem_used_mb', 0),
            gpu_mem_total_mb=data.get('gpu_mem_total_mb', 0),
            gpu_mem_percent=data.get('gpu_mem_percent', 0),
            gpu_power_draw=data.get('gpu_power_draw', 0),
            gpu_power_limit=data.get('gpu_power_limit', 0),
            # Disco
            disk_total_gb=data.get('disk_total_gb', 0),
            disk_used_gb=data.get('disk_used_gb', 0),
            disk_free_gb=data.get('disk_free_gb', 0),
            disk_percent=data.get('disk_percent', 0),
            # I/O
            io_read_mb=data.get('io_read_mb', 0),
            io_write_mb=data.get('io_write_mb', 0),
            io_read_count=data.get('io_read_count', 0),
            io_write_count=data.get('io_write_count', 0),
            # Rede
            net_recv_mb=data.get('net_recv_mb', 0),
            net_sent_mb=data.get('net_sent_mb', 0),
            # Status
            vllm_status=data.get('vllm_status', 'unknown'),
            gpu_server_status=data.get('gpu_server_status', 'unknown'),
        )
    except json.JSONDecodeError as e:
        st.error(f"Erro ao parsear métricas: {e}\nOutput: {output}")
        return None


def get_status_color(status: str) -> str:
    """Retorna cor baseada no status."""
    if status == 'running':
        return 'green'
    elif status == 'stopped':
        return 'red'
    return 'orange'


def create_gauge_html(value: float, max_value: float, label: str, unit: str = "%", color: str = "#1f77b4") -> str:
    """Cria HTML para gauge simples."""
    percent = min(100, (value / max_value) * 100) if max_value > 0 else 0

    # Cores baseadas no valor
    if percent > 90:
        bar_color = "#ff4444"
    elif percent > 70:
        bar_color = "#ffaa00"
    else:
        bar_color = "#00cc66"

    return f"""
    <div style="margin: 10px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span style="font-weight: bold;">{label}</span>
            <span>{value:.1f}{unit}</span>
        </div>
        <div style="background: #e0e0e0; border-radius: 10px; height: 20px; overflow: hidden;">
            <div style="background: {bar_color}; width: {percent}%; height: 100%; border-radius: 10px; transition: width 0.3s;"></div>
        </div>
    </div>
    """


def render_enrichment_tab():
    """Renderiza tab de monitoramento de enriquecimento."""
    st.header("📝 Monitoramento de Enriquecimento")
    st.markdown(f"**VPS:** `{VPS_IP}` | **Última atualização:** {datetime.now().strftime('%H:%M:%S')}")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        if st.button("🚀 Iniciar Novo Enriquecimento", type="primary", use_container_width=True):
            with st.spinner("Iniciando enriquecimento..."):
                success, message = start_enrichment()
                if success:
                    st.success(message)
                else:
                    st.error(f"Erro: {message}")
                time.sleep(2)
                st.rerun()

    with col2:
        queue_size = get_celery_queue_size()
        st.metric("📬 Fila Celery", queue_size)

    with col3:
        workers = get_celery_workers_count()
        st.metric("👷 Workers Ativos", workers)

    st.divider()

    # Status do enriquecimento
    status = get_enrichment_status()

    if status is None:
        st.info("ℹ️ Nenhum enriquecimento em andamento ou concluído.")
        return

    # Métricas principais
    st.subheader(f"📊 Task: `{status.task_id}`")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        status_emoji = "🔄" if status.status == "processing" else "✅" if status.status == "completed" else "❓"
        st.metric("Status", f"{status_emoji} {status.status}")

    with col2:
        st.metric("✅ Completos", status.chunks_completed)

    with col3:
        st.metric("❌ Falhas", status.chunks_failed)

    with col4:
        st.metric("📦 Total", status.total_chunks)

    # Barra de progresso
    st.markdown("### Progresso")
    progress = min(status.progress_percent / 100, 1.0)
    st.progress(progress, text=f"{status.progress_percent:.1f}% ({status.chunks_completed + status.chunks_failed}/{status.total_chunks})")

    # Gauge visual
    remaining = status.total_chunks - status.chunks_completed - status.chunks_failed
    if status.chunks_completed > 0:
        # Estima tempo restante
        # Assumindo ~10s por chunk com 6 workers = ~1.7 chunks/segundo
        eta_seconds = remaining / 0.6  # ~0.6 chunks/s com 6 workers
        eta_minutes = eta_seconds / 60
        if eta_minutes > 60:
            eta_str = f"~{eta_minutes/60:.1f}h restantes"
        else:
            eta_str = f"~{eta_minutes:.0f}min restantes"
        st.info(f"⏱️ {eta_str} | Restam {remaining} chunks")

    # Erros
    if status.errors:
        with st.expander(f"⚠️ Últimos Erros ({len(status.errors)})"):
            for error in status.errors:
                st.code(error)

    # Started at
    if status.started_at:
        st.caption(f"Iniciado em: {status.started_at}")


def main():
    st.title("🖥️ VectorGov Infrastructure Monitor")
    provider_badge = "🟢 RunPod" if GPU_PROVIDER == "runpod" else "🔵 Google Cloud"
    st.markdown(f"**{provider_badge}** | **GPU:** `{VM_NAME}` ({VM_IP}) | **VPS:** `{VPS_IP}` | **Atualização:** {datetime.now().strftime('%H:%M:%S')}")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configurações")

        auto_refresh = st.checkbox("Auto-refresh", value=True)
        refresh_interval = st.slider("Intervalo (segundos)", 5, 60, 10)

        st.divider()

        if st.button("🔄 Atualizar Agora", use_container_width=True):
            st.rerun()

        st.divider()

        st.markdown("### 📡 Conexões")
        port_str = f" -p {SSH_PORT}" if SSH_PORT != 22 else ""
        st.code(f"# GPU Server ({GPU_PROVIDER})\nssh{port_str} {SSH_USER}@{VM_IP}")
        st.code(f"# VPS\nssh root@{VPS_IP}")

    # Tabs principais
    tab_gpu, tab_enrich = st.tabs(["🎮 GPU Server", "📝 Enriquecimento"])

    with tab_enrich:
        render_enrichment_tab()

    with tab_gpu:
        # Coleta métricas
        with st.spinner("Coletando métricas da VM GPU..."):
            metrics = collect_metrics()

        if metrics is None:
            st.error("❌ Não foi possível conectar à VM. Verifique a conexão SSH.")
        else:
            # Status dos serviços
            st.subheader("🔌 Status dos Serviços")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                status_color = "🟢" if metrics.gpu_server_status == 'running' else "🔴"
                st.metric("GPU Server (8000)", f"{status_color} {metrics.gpu_server_status}")

            with col2:
                status_color = "🟢" if metrics.vllm_status == 'running' else "🔴"
                st.metric("vLLM (8001)", f"{status_color} {metrics.vllm_status}")

            with col3:
                st.metric("GPU", metrics.gpu_name)

            with col4:
                st.metric("CPUs", metrics.cpu_count)

            st.divider()

            # Métricas principais
            col1, col2 = st.columns(2)

            # CPU
            with col1:
                st.subheader("🖥️ CPU")

                st.markdown(create_gauge_html(
                    metrics.cpu_percent, 100, "Utilização CPU", "%"
                ), unsafe_allow_html=True)

                col_load1, col_load2, col_load3 = st.columns(3)
                with col_load1:
                    st.metric("Load 1min", f"{metrics.load_1min:.2f}")
                with col_load2:
                    st.metric("Load 5min", f"{metrics.load_5min:.2f}")
                with col_load3:
                    st.metric("Load 15min", f"{metrics.load_15min:.2f}")

            # Memória
            with col2:
                st.subheader("🧠 Memória RAM")

                st.markdown(create_gauge_html(
                    metrics.mem_percent, 100, f"Utilização ({metrics.mem_used_gb:.1f} / {metrics.mem_total_gb:.1f} GB)", "%"
                ), unsafe_allow_html=True)

                col_mem1, col_mem2, col_mem3 = st.columns(3)
                with col_mem1:
                    st.metric("Total", f"{metrics.mem_total_gb:.1f} GB")
                with col_mem2:
                    st.metric("Usado", f"{metrics.mem_used_gb:.1f} GB")
                with col_mem3:
                    st.metric("Livre", f"{metrics.mem_free_gb:.1f} GB")

            st.divider()

            # GPU
            st.subheader("🎮 GPU NVIDIA")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(create_gauge_html(
                    metrics.gpu_util, 100, "Utilização GPU", "%"
                ), unsafe_allow_html=True)

            with col2:
                st.markdown(create_gauge_html(
                    metrics.gpu_mem_percent, 100,
                    f"VRAM ({metrics.gpu_mem_used_mb:.0f} / {metrics.gpu_mem_total_mb:.0f} MB)", "%"
                ), unsafe_allow_html=True)

            with col3:
                st.markdown(create_gauge_html(
                    metrics.gpu_temp, 90, "Temperatura", "°C"
                ), unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("GPU Util", f"{metrics.gpu_util:.0f}%")
            with col2:
                st.metric("VRAM", f"{metrics.gpu_mem_used_mb:.0f} MB")
            with col3:
                st.metric("Temperatura", f"{metrics.gpu_temp:.0f}°C")
            with col4:
                st.metric("Power", f"{metrics.gpu_power_draw:.0f}W / {metrics.gpu_power_limit:.0f}W")

            st.divider()

            # Disco e I/O
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("💾 Disco")

                st.markdown(create_gauge_html(
                    metrics.disk_percent, 100,
                    f"Utilização ({metrics.disk_used_gb:.1f} / {metrics.disk_total_gb:.1f} GB)", "%"
                ), unsafe_allow_html=True)

                col_d1, col_d2, col_d3 = st.columns(3)
                with col_d1:
                    st.metric("Total", f"{metrics.disk_total_gb:.1f} GB")
                with col_d2:
                    st.metric("Usado", f"{metrics.disk_used_gb:.1f} GB")
                with col_d3:
                    st.metric("Livre", f"{metrics.disk_free_gb:.1f} GB")

            with col2:
                st.subheader("📊 I/O de Disco")

                col_io1, col_io2 = st.columns(2)
                with col_io1:
                    st.metric("Leitura Total", f"{metrics.io_read_mb:,.0f} MB")
                    st.metric("Operações Read", f"{metrics.io_read_count:,}")
                with col_io2:
                    st.metric("Escrita Total", f"{metrics.io_write_mb:,.0f} MB")
                    st.metric("Operações Write", f"{metrics.io_write_count:,}")

            st.divider()

            # Rede
            st.subheader("🌐 Rede")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📥 Recebido", f"{metrics.net_recv_mb:,.0f} MB")
            with col2:
                st.metric("📤 Enviado", f"{metrics.net_sent_mb:,.0f} MB")

            # Histórico (armazenado em session_state)
            if 'metrics_history' not in st.session_state:
                st.session_state.metrics_history = []

            # Adiciona métricas ao histórico
            st.session_state.metrics_history.append({
                'timestamp': metrics.timestamp,
                'cpu_percent': metrics.cpu_percent,
                'mem_percent': metrics.mem_percent,
                'gpu_util': metrics.gpu_util,
                'gpu_mem_percent': metrics.gpu_mem_percent,
                'gpu_temp': metrics.gpu_temp,
            })

            # Mantém apenas últimos 60 pontos
            if len(st.session_state.metrics_history) > 60:
                st.session_state.metrics_history = st.session_state.metrics_history[-60:]

            # Gráfico histórico
            if len(st.session_state.metrics_history) > 1:
                st.divider()
                st.subheader("📈 Histórico")

                df = pd.DataFrame(st.session_state.metrics_history)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)

                hist_tab1, hist_tab2, hist_tab3 = st.tabs(["CPU & Memória", "GPU", "Temperatura"])

                with hist_tab1:
                    st.line_chart(df[['cpu_percent', 'mem_percent']], use_container_width=True)

                with hist_tab2:
                    st.line_chart(df[['gpu_util', 'gpu_mem_percent']], use_container_width=True)

                with hist_tab3:
                    st.line_chart(df[['gpu_temp']], use_container_width=True)

    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
