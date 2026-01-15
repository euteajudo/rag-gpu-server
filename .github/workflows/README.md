# GitHub Actions CI/CD para RunPod

Este diretório contém workflows do GitHub Actions para deploy automático no RunPod GPU Server.

## Workflow: deploy-runpod.yml

Deploy automático quando há push na branch `main`.

### Configuração dos Secrets

Configure os seguintes secrets no repositório GitHub:
**Settings → Secrets and variables → Actions → New repository secret**

| Secret | Descrição | Exemplo |
|--------|-----------|---------|
| `RUNPOD_SSH_HOST` | IP ou hostname do pod RunPod | `69.30.85.101` |
| `RUNPOD_SSH_PORT` | Porta SSH do pod | `22075` |
| `RUNPOD_SSH_KEY` | Chave privada SSH (conteúdo completo) | `-----BEGIN OPENSSH PRIVATE KEY-----...` |
| `RUNPOD_SSH_USER` | Usuário SSH (opcional, default: root) | `root` |

### Como obter as credenciais do RunPod

1. Acesse https://runpod.io/console/pods
2. Encontre o pod `vectorgov-gpu`
3. Clique em "Connect" → "SSH over exposed TCP"
4. Copie o comando SSH que aparece (ex: `ssh root@69.30.85.101 -p 22075`)
5. A chave privada está em `~/.ssh/id_ed25519` ou `~/.ssh/id_rsa`

### Triggers

- **Automático**: Push na branch `main`
- **Manual**: Via GitHub Actions UI (workflow_dispatch)

### Opções de Execução Manual

Quando executado manualmente, você pode escolher:
- `restart_services`: Se deve reiniciar o servidor após deploy (default: true)

### O que o workflow faz

1. **Checkout**: Baixa o código do repositório
2. **SSH Setup**: Configura a chave SSH
3. **Test Connection**: Verifica se consegue conectar ao RunPod
4. **Pull Code**: Executa `git pull` no RunPod
5. **Update Dependencies**: Instala/atualiza dependências Python
6. **Restart Server**: Reinicia o GPU Server (uvicorn)
7. **Verify**: Verifica se o deploy foi bem-sucedido
8. **Cleanup**: Remove a chave SSH

### Logs e Troubleshooting

- Logs do GPU Server: `/workspace/logs/gpu-server.log`
- Health check: `curl http://localhost:8000/health`

### Nota sobre IPs Dinâmicos do RunPod

O RunPod pode alterar o IP/porta do pod após reinicialização. Se o deploy falhar com erro de conexão:

1. Acesse o console do RunPod
2. Verifique as novas credenciais SSH
3. Atualize os secrets no GitHub

### Segurança

- A chave SSH é usada apenas durante o deploy
- É removida após a execução (mesmo em caso de falha)
- `StrictHostKeyChecking` está desabilitado devido aos IPs dinâmicos do RunPod
