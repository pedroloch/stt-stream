# 🚀 Deploy no RunPod - Whisper Stream

Guia completo para fazer deploy do Whisper Stream no RunPod com GPU de alta performance usando modelo large-v3.

## 🎯 Visão Geral

Este guia mostra como deployar o servidor Python no RunPod usando:
- **Modelo**: large-v3 (melhor qualidade de transcrição)
- **GPU**: RTX 4090 ou A100
- **Deploy**: Docker (fácil e reproduzível)
- **Modo**: 24/7 com volume persistente

## ⚡ Quick Start (5 minutos)

```bash
# 1. Build e push da imagem Docker
export DOCKER_USERNAME="seu-usuario-dockerhub"
./runpod-deploy.sh all

# 2. Criar pod no RunPod (veja instruções abaixo)

# 3. Atualizar cliente local
echo "WHISPER_SERVER_URL=ws://SEU-POD-IP:9090/ws" > .env
bun start
```

## 📋 Pré-requisitos

### Local (sua máquina)
- Docker instalado
- Conta no Docker Hub (para hospedar a imagem)
- Conta no RunPod (https://www.runpod.io/)

### No RunPod
- Créditos na conta (~$50 para começar)
- Nada mais! O Docker faz tudo.

## 🔧 Passo a Passo Detalhado

### 1. Preparar Imagem Docker

#### 1.1. Configurar Docker Username

Edite `runpod-deploy.sh` ou defina variável de ambiente:

```bash
export DOCKER_USERNAME="seu-usuario-dockerhub"
```

#### 1.2. Build da Imagem

```bash
./runpod-deploy.sh build
```

Isso vai:
- Build da imagem usando `Dockerfile.runpod`
- Instalar CUDA, Python, faster-whisper
- Copiar configuração otimizada (`server-config.runpod.yaml`)
- Tag: `seu-usuario/whisper-stream-runpod:latest`

**Tempo estimado**: 5-10 minutos na primeira vez

#### 1.3. Push para Docker Hub

```bash
# Login no Docker Hub (se necessário)
docker login

# Push
./runpod-deploy.sh push
```

**Tempo estimado**: 2-5 minutos (upload ~2GB)

#### 1.4. Ou Fazer Tudo de Uma Vez

```bash
./runpod-deploy.sh all
```

---

### 2. Criar Pod no RunPod

#### 2.1. Acessar RunPod

1. Acesse https://www.runpod.io/
2. Login ou crie conta
3. Adicione créditos (mínimo $10, recomendado $50)

#### 2.2. Criar Novo Pod

1. **Clique em "Deploy"** no menu

2. **Selecione GPU**:
   - **Recomendado**: RTX 4090 (24GB VRAM)
     - Custo: ~$0.60-0.80/hora
     - Performance excelente para large-v3
   - **Alternativa**: RTX A6000 (48GB VRAM)
     - Custo: ~$0.80-1.10/hora
     - Overkill mas mais disponível
   - **Budget**: RTX 3090 (24GB VRAM)
     - Custo: ~$0.50-0.70/hora
     - Funciona mas um pouco mais lento

3. **Template**: Selecione "PyTorch" ou "Custom"

4. **Container Configuration**:
   ```
   Docker Image: seu-usuario/whisper-stream-runpod:latest
   Docker Command: (deixe vazio, usa CMD do Dockerfile)
   ```

5. **Volume (IMPORTANTE)**:
   - Container Path: `/app/models`
   - Volume Size: `20 GB`
   - Isso mantém os modelos entre restarts!

6. **Port Mapping**:
   - Container Port: `9090`
   - Type: `HTTP` (ou TCP)
   - Protocolo: TCP

7. **Environment Variables** (opcional):
   ```
   CUDA_VISIBLE_DEVICES=0
   ```

#### 2.3. Deploy

1. Clique em "Deploy" ou "Deploy On-Demand Pod"
2. Aguarde pod inicializar (~2-5 minutos)
3. **IMPORTANTE**: Na primeira vez, o modelo large-v3 será baixado (~3GB)
   - Isso demora ~5-10 minutos extras
   - Depois fica salvo no volume persistente

---

### 3. Verificar Pod

#### 3.1. Obter IP do Pod

No painel do RunPod, copie o **Public IP** ou **External IP** do pod.

Exemplo: `167.99.123.456`

#### 3.2. Testar Health Endpoint

```bash
curl http://167.99.123.456:9090/health
```

**Esperado**:
```json
{
  "status": "healthy",
  "processor_ready": true
}
```

#### 3.3. Ver Informações do Servidor

```bash
curl http://167.99.123.456:9090/info
```

Você deve ver:
```json
{
  "processor": {
    "model": "large-v3",
    "backend_type": "cuda",
    "hardware": {
      "type": "cuda",
      "device_name": "NVIDIA GeForce RTX 4090"
    }
  }
}
```

#### 3.4. Ver Logs

No painel RunPod:
1. Clique no pod
2. Aba "Logs"
3. Você deve ver:
   ```
   ✅ Servidor rodando em ws://0.0.0.0:9090
   Modelo: large-v3
   Backend: cuda
   ```

---

### 4. Configurar Cliente Local

#### 4.1. Criar .env

```bash
cd /path/to/whisper-stream
echo "WHISPER_SERVER_URL=ws://167.99.123.456:9090/ws" > .env
```

#### 4.2. Atualizar config.yaml

Ou edite `config.yaml`:

```yaml
server:
  url: "ws://167.99.123.456:9090/ws"
  health_url: "http://167.99.123.456:9090/health"
```

#### 4.3. Rodar Cliente

```bash
bun start
```

**Esperado**:
```
🔍 Verificando servidor...
✅ Servidor está online
🔌 Conectando ao WebSocket...
✅ WebSocket conectado
🎙️  Iniciando captura de áudio...
Fale no microfone - transcrições aparecerão abaixo:
```

---

## 💰 Custos e Estimativas

### GPU RTX 4090 (Recomendada)

| Uso | Horas/mês | Custo/hora | Total/mês |
|-----|-----------|------------|-----------|
| 24/7 | 720 | $0.70 | $504 |
| 12h/dia | 360 | $0.70 | $252 |
| 8h/dia | 240 | $0.70 | $168 |
| Sob demanda (100h) | 100 | $0.70 | $70 |

### GPU RTX 3090 (Budget)

| Uso | Horas/mês | Custo/hora | Total/mês |
|-----|-----------|------------|-----------|
| 24/7 | 720 | $0.50 | $360 |
| 12h/dia | 360 | $0.50 | $180 |
| Sob demanda (100h) | 100 | $0.50 | $50 |

**Dica**: Use "Spot Instances" para economizar ~50% (mas podem ser interrompidas)

---

## 🔒 Segurança e Melhores Práticas

### 1. Adicionar Autenticação (Recomendado)

Edite `server-config.runpod.yaml` antes do build:

```yaml
security:
  require_auth: true
  api_key: "seu-token-secreto-aqui"  # Gere com: openssl rand -hex 32
```

No cliente, adicione header:
```typescript
const ws = new WebSocket(url, {
  headers: { 'Authorization': `Bearer ${API_KEY}` }
});
```

### 2. Restringir CORS

```yaml
server:
  cors_enabled: true
  allowed_origins: ["https://seu-dominio.com"]  # Não use "*" em produção
```

### 3. Rate Limiting

```yaml
security:
  rate_limit_enabled: true
  max_requests_per_minute: 60
```

### 4. Firewall do RunPod

No painel RunPod:
- Configure "Allowed IPs" se souber os IPs dos clientes
- Use VPN ou Tailscale para acesso privado

---

## 🔧 Troubleshooting

### Pod não inicia

**Problema**: Pod fica em "Initializing"

**Soluções**:
1. Verifique logs do pod
2. Confirme que a imagem Docker existe no Docker Hub
3. Tente outro template (PyTorch, CUDA 12.1)

### Modelo não baixa

**Problema**: Erro "Failed to download model"

**Soluções**:
1. Verifique conexão internet do pod
2. Aumentar timeout no Dockerfile
3. Use outro modelo (medium, large)

### GPU não detectada

**Problema**: Backend usa CPU ao invés de CUDA

**Soluções**:
1. Verifique que selecionou GPU no RunPod
2. Confirme que usou `--gpus all` (já está no Dockerfile)
3. Veja logs para mensagens de erro CUDA

### Conexão recusada do cliente

**Problema**: Cliente não conecta ao WebSocket

**Soluções**:
1. Verifique IP do pod (pode mudar ao reiniciar)
2. Confirme porta 9090 está mapeada
3. Teste com `curl http://IP:9090/health` primeiro
4. Verifique firewall local

### Performance ruim

**Problema**: Transcrição muito lenta

**Soluções**:
1. Verifique se está usando GPU (logs devem mostrar CUDA)
2. Reduza beam_size e best_of no config:
   ```yaml
   beam_size: 1
   best_of: 1
   ```
3. Use modelo menor (large ao invés de large-v3)
4. Upgrade para GPU mais potente

---

## 📊 Monitoramento

### Ver Stats do Pod

```bash
curl http://SEU-POD-IP:9090/stats
```

**Resposta**:
```json
{
  "connections": 1,
  "total_connections": 5,
  "uptime": 3600,
  "transcriptions": 42
}
```

### Logs em Tempo Real

No terminal:
```bash
# Instalar runpodctl (CLI do RunPod)
pip install runpodctl

# Login
runpodctl config

# Ver logs
runpodctl logs POD_ID --follow
```

### Prometheus/Grafana (Avançado)

Adicione endpoint `/metrics` no servidor para Prometheus scraping.

---

## 🚀 Deploy Automatizado com CI/CD

### GitHub Actions

Crie `.github/workflows/deploy-runpod.yml`:

```yaml
name: Deploy to RunPod

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Login to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}

      - name: Build and push
        run: |
          docker build -f Dockerfile.runpod -t ${{ secrets.DOCKER_USERNAME }}/whisper-stream-runpod:latest .
          docker push ${{ secrets.DOCKER_USERNAME }}/whisper-stream-runpod:latest

      - name: Restart RunPod (via API)
        run: |
          # Usar RunPod API para reiniciar pod com nova imagem
          # https://docs.runpod.io/reference/restart-pod
```

---

## 🔄 Atualizar Servidor

### Atualizar Código

```bash
# 1. Fazer mudanças no código local

# 2. Rebuild e push
./runpod-deploy.sh all

# 3. No RunPod, restart o pod
# Painel > Pod > Actions > Restart
```

O pod vai puxar a nova imagem automaticamente.

### Atualizar Modelo

Edite `server-config.runpod.yaml`:

```yaml
whisper:
  model: "large-v3-turbo"  # Modelo mais recente
```

Rebuild e redeploy.

---

## 🎓 Próximos Passos

### Opcional: Domínio Customizado

1. Configure Cloudflare Tunnel ou Nginx reverse proxy
2. Aponte domínio para IP do pod
3. Configure SSL com Let's Encrypt

### Opcional: Load Balancing

Para múltiplos clientes:
1. Deploy múltiplos pods
2. Use load balancer (Cloudflare, Nginx)
3. Distribua conexões entre pods

### Opcional: Auto-scaling

Use RunPod Serverless para escalar automaticamente baseado em demanda.

---

## 📚 Recursos

- [RunPod Documentation](https://docs.runpod.io/)
- [Docker Hub](https://hub.docker.com/)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [Whisper Models](https://github.com/openai/whisper#available-models-and-languages)

---

## 🆘 Suporte

**Problemas?**
1. Verifique logs do pod primeiro
2. Teste health endpoint
3. Veja issues no GitHub
4. Discord do RunPod (suporte de infra)

---

**Última atualização**: 2025-01-17
**Autor**: Whisper Stream Team
