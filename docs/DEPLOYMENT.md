# 🚀 Deployment Guide - Whisper Stream

Guia completo para deployment do Whisper Stream em produção.

## Visão Geral

- **Servidor Python**: Deploy em servidor com GPU (recomendado)
- **Cliente**: Roda localmente, conecta ao servidor remoto

## Deployment do Servidor Python

### Opção 1: VPS com GPU NVIDIA (Recomendado)

#### Providers:
- **Hetzner**: GPUs dedicadas, bom preço
- **DigitalOcean**: GPU Droplets
- **Vultr**: High Frequency GPU
- **OVH**: Bare metal com GPU

#### Setup:

```bash
# 1. Conectar ao servidor
ssh user@seu-servidor.com

# 2. Instalar CUDA (Ubuntu/Debian)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda

# Verificar instalação
nvidia-smi

# 3. Instalar Python e uv
sudo apt-get install python3.10 python3-pip
curl -LsSf https://astral.sh/uv/install.sh | sh

# 4. Clonar projeto
git clone https://github.com/seu-usuario/whisper-stream.git
cd whisper-stream

# 5. Instalar dependências (CUDA)
uv pip install -e ".[cuda]"

# 6. Configurar
cp server-config.example.yaml server-config.yaml
nano server-config.yaml
# Ajustar:
# - host: "0.0.0.0"
# - backend: "cuda"

# 7. Criar systemd service
sudo nano /etc/systemd/system/whisper-stream.service
```

**whisper-stream.service:**
```ini
[Unit]
Description=Whisper Stream Server
After=network.target

[Service]
Type=simple
User=whisper
WorkingDirectory=/home/whisper/whisper-stream
ExecStart=/home/whisper/.local/bin/uv run python -m server.main --config server-config.yaml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# 8. Iniciar serviço
sudo systemctl daemon-reload
sudo systemctl enable whisper-stream
sudo systemctl start whisper-stream

# Verificar status
sudo systemctl status whisper-stream

# Logs
sudo journalctl -u whisper-stream -f
```

#### Firewall:

```bash
# Abrir porta 9090
sudo ufw allow 9090/tcp
sudo ufw enable
```

#### Nginx Reverse Proxy (Opcional):

```nginx
# /etc/nginx/sites-available/whisper-stream
upstream whisper {
    server localhost:9090;
}

server {
    listen 80;
    server_name whisper.seu-dominio.com;

    location / {
        proxy_pass http://whisper;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/whisper-stream /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# SSL com Let's Encrypt
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d whisper.seu-dominio.com
```

---

### Opção 2: Cloud GPU (Vast.ai / RunPod / Lambda Labs)

#### Vast.ai Example:

1. **Criar conta** em [vast.ai](https://vast.ai)

2. **Selecionar instância**:
   - GPU: RTX 3090 ou melhor
   - VRAM: >= 8GB
   - Storage: 20GB
   - Template: PyTorch (já tem CUDA)

3. **Conectar via SSH**:
```bash
ssh -p PORT root@IP -L 9090:localhost:9090
```

4. **Setup**:
```bash
# Já tem CUDA instalado, apenas instalar projeto
git clone https://github.com/seu-usuario/whisper-stream.git
cd whisper-stream
pip install -e ".[cuda]"

# Configurar e rodar
cp server-config.example.yaml server-config.yaml
python -m server.main --config server-config.yaml
```

5. **Manter rodando**:
```bash
# Usar screen ou tmux
screen -S whisper
python -m server.main --config server-config.yaml
# Ctrl+A, D para detach

# Ou usar nohup
nohup python -m server.main --config server-config.yaml > server.log 2>&1 &
```

---

### Opção 3: Apple Silicon (Mac Mini / Mac Studio)

Deploy local em Mac com M1/M2/M3 para uso interno.

```bash
# Instalar Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Instalar Python
brew install python@3.10

# Instalar uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup projeto
git clone https://github.com/seu-usuario/whisper-stream.git
cd whisper-stream

# Instalar com MLX
uv pip install -e ".[mlx]"

# Configurar
cp server-config.example.yaml server-config.yaml
# backend: "mlx"
# host: "0.0.0.0" (se quiser aceitar conexões de rede local)

# Rodar
uv run python -m server.main --config server-config.yaml
```

**Manter rodando com launchd:**

```xml
<!-- ~/Library/LaunchAgents/com.whisper-stream.plist -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.whisper-stream</string>
    <key>ProgramArguments</key>
    <array>
        <string>/opt/homebrew/bin/python3</string>
        <string>-m</string>
        <string>server.main</string>
        <string>--config</string>
        <string>server-config.yaml</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/seu-usuario/whisper-stream</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/whisper-stream.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/whisper-stream-error.log</string>
</dict>
</plist>
```

```bash
launchctl load ~/Library/LaunchAgents/com.whisper-stream.plist
launchctl start com.whisper-stream
```

---

### Opção 4: Docker

#### Dockerfile (CPU):

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar projeto
COPY . /app

# Instalar dependências Python
RUN pip install uv
RUN uv pip install --system -e .

# Expor porta
EXPOSE 9090

# Configuração padrão
ENV CONFIG_FILE=server-config.yaml

# Comando
CMD ["python", "-m", "server.main", "--config", "server-config.yaml"]
```

#### Dockerfile (CUDA):

```dockerfile
FROM nvidia/cuda:12.0-base-ubuntu22.04

# Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install uv
RUN uv pip install --system -e ".[cuda]"

EXPOSE 9090
CMD ["python3", "-m", "server.main", "--config", "server-config.yaml"]
```

#### docker-compose.yml:

```yaml
version: '3.8'

services:
  whisper-server:
    build: .
    ports:
      - "9090:9090"
    volumes:
      - ./server-config.yaml:/app/server-config.yaml:ro
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
```

**Rodar:**
```bash
docker-compose up -d
docker-compose logs -f
```

---

## Deployment do Cliente

Cliente roda **localmente** e conecta ao servidor remoto.

### Setup Cliente:

```bash
# 1. Instalar Bun
curl -fsSL https://bun.sh/install | bash

# 2. Clonar (ou baixar só o cliente)
git clone https://github.com/seu-usuario/whisper-stream.git
cd whisper-stream

# 3. Instalar dependências
bun install

# 4. Instalar sox
# macOS:
brew install sox

# Linux:
sudo apt-get install sox

# 5. Configurar
cp config.example.yaml config.yaml
nano config.yaml
```

**config.yaml:**
```yaml
server:
  # Apontar para servidor remoto
  url: "ws://seu-servidor.com:9090/ws"
  health_url: "http://seu-servidor.com:9090/health"

  # Ou com SSL (se configurou nginx + certbot)
  # url: "wss://whisper.seu-dominio.com/ws"
  # health_url: "https://whisper.seu-dominio.com/health"
```

```bash
# 6. Rodar
bun start
```

---

## Monitoramento

### Logs do Servidor:

```bash
# Systemd
sudo journalctl -u whisper-stream -f

# Docker
docker-compose logs -f whisper-server

# Arquivo
tail -f logs/server_*.log
```

### Métricas:

**Endpoints úteis:**
- `GET /health` - Status do servidor
- `GET /info` - Informações do backend
- `GET /stats` - Estatísticas (conexões ativas)

**Exemplo de script de monitoramento:**
```bash
#!/bin/bash
# monitor.sh

while true; do
  status=$(curl -s http://localhost:9090/health | jq -r '.status')
  if [ "$status" != "healthy" ]; then
    echo "⚠️ Server unhealthy!"
    # Enviar alerta (email, slack, etc)
  fi
  sleep 60
done
```

---

## Segurança

### 1. Firewall:

Apenas expor porta necessária:
```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 9090/tcp
sudo ufw enable
```

### 2. Rate Limiting (Nginx):

```nginx
limit_req_zone $binary_remote_addr zone=whisper:10m rate=10r/s;

location / {
    limit_req zone=whisper burst=20;
    # ...
}
```

### 3. Auth (Futuro):

Adicionar API key no servidor:
```yaml
# server-config.yaml
security:
  require_auth: true
  api_key: "seu-token-secreto"
```

---

## Custos Estimados

### GPU Cloud:

| Provider | GPU | VRAM | $/hora | $/mês (24/7) |
|----------|-----|------|--------|--------------|
| Vast.ai | RTX 3090 | 24GB | $0.25 | $180 |
| RunPod | RTX 4090 | 24GB | $0.69 | $500 |
| Lambda Labs | A10 | 24GB | $0.60 | $432 |

### VPS (sem GPU):

CPU only, mais lento:
- Hetzner CPX31: €12/mês
- DigitalOcean Basic: $24/mês

### Apple Silicon (Local):

- Mac Mini M2: $599 (one-time)
- Sem custo mensal, uso local

---

## Troubleshooting

### Servidor não inicia:

```bash
# Verificar logs
sudo journalctl -u whisper-stream -n 50

# Testar manualmente
uv run python -m server.main --config server-config.yaml

# Verificar CUDA
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Cliente não conecta:

```bash
# Testar conectividade
curl http://servidor:9090/health

# Testar WebSocket
npm install -g wscat
wscat -c ws://servidor:9090/ws

# Verificar firewall
sudo ufw status
```

### Performance ruim:

- Usar modelo menor (`tiny` ou `base`)
- Verificar se GPU está sendo usada (logs do servidor)
- Aumentar `chunk_duration` no cliente
- Desabilitar VAD se causar problemas

---

## Backup e Updates

### Backup configurações:

```bash
# Fazer backup de configs
tar -czf whisper-backup.tar.gz \
  server-config.yaml \
  config.yaml \
  models/
```

### Update servidor:

```bash
cd whisper-stream
git pull
uv pip install -e ".[cuda]"  # ou mlx
sudo systemctl restart whisper-stream
```

---

**Última atualização**: 2025-01-17
