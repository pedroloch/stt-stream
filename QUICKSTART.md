# 🚀 Quick Start - Whisper Stream

Guia rápido para começar a usar o Whisper Stream.

## ⚡ Instalação Rápida

### 1. Python + Servidor

```bash
# Clone o repositório
git clone <seu-repo>
cd whisper-stream

# Instalar com uv (recomendado)
uv pip install -e .

# **IMPORTANTE**: Instalar backend específico
# Apple Silicon (M1/M2/M3):
uv pip install mlx-whisper

# NVIDIA GPU:
uv pip install torch  # ou cuda extra

# CPU only: já funciona com instalação base
```

### 2. Cliente (Opcional)

```bash
# Instalar Bun
curl -fsSL https://bun.sh/install | bash

# Instalar sox (para captura de áudio)
# macOS:
brew install sox

# Linux:
sudo apt-get install sox

# Instalar dependências do cliente
bun install
```

## 🧪 Testar Instalação

```bash
# 1. Testar detecção de hardware
python scripts/check-hardware.py

# 2. Testar instalação Python
python scripts/test-install.sh

# 3. Rodar testes unitários
pytest tests/unit/ -v
```

## 🎯 Rodar Servidor

### Opção 1: Com Arquivo de Config

```bash
# Copiar config de exemplo
cp server-config.example.yaml server-config.yaml

# Editar se necessário
nano server-config.yaml

# Iniciar servidor
python -m server.main --config server-config.yaml
```

### Opção 2: Com Argumentos CLI

```bash
# Apple Silicon
python -m server.main --backend mlx --model tiny --language pt

# NVIDIA GPU
python -m server.main --backend cuda --model base --language en

# CPU
python -m server.main --backend cpu --model tiny --language pt
```

**O servidor estará disponível em**: `ws://localhost:9090/ws`

## 🎤 Rodar Cliente

```bash
# Em outro terminal
cp config.example.yaml config.yaml
bun start
```

## 🔍 Verificar que Está Funcionando

### 1. Health Check

```bash
curl http://localhost:9090/health
```

**Esperado**:
```json
{
  "status": "healthy",
  "processor_ready": true
}
```

### 2. Informações do Servidor

```bash
curl http://localhost:9090/info
```

### 3. Testar WebSocket

```bash
# Instalar wscat
npm install -g wscat

# Conectar
wscat -c ws://localhost:9090/ws
```

## ⚙️ Configuração Básica

### Servidor (server-config.yaml)

```yaml
server:
  host: "0.0.0.0"  # ou "localhost" para apenas local
  port: 9090

whisper:
  model: "tiny"     # tiny, base, small, medium, large
  language: "pt"    # pt, en, es, etc
  backend: "auto"   # auto, mlx, cuda, cpu
```

### Cliente (config.yaml)

```yaml
server:
  url: "ws://localhost:9090/ws"  # URL do servidor

display:
  show_partial: true
  colors:
    final: "green"
    partial: "cyan"
```

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'mlx_whisper'"

```bash
uv pip install mlx-whisper
```

### "Module 'mlx_whisper' has no attribute 'load_model'"

Certifique-se de que tem a versão mais recente do código (após commit de correção do MLX).

### "CUDA not available"

Verifique drivers NVIDIA:
```bash
nvidia-smi
```

### Cliente não conecta

1. Certifique-se que o servidor está rodando
2. Verifique a URL em `config.yaml`
3. Teste com `curl http://localhost:9090/health`

## 📚 Próximos Passos

- Leia [ARCHITECTURE.md](docs/ARCHITECTURE.md) para entender o design
- Veja [API.md](docs/API.md) para criar seu próprio cliente
- Consulte [DEPLOYMENT.md](docs/DEPLOYMENT.md) para produção
- Leia [TESTING.md](docs/TESTING.md) para contribuir

## 💡 Dicas

**Modelos mais rápidos**: Use `tiny` ou `base` para baixa latência

**Melhor qualidade**: Use `medium` ou `large` (requer GPU)

**Produção**: Deploy o servidor Python em máquina com GPU separada

**Development**: Use `--verbose` para ver logs detalhados

---

**Precisa de ajuda?** Veja a documentação completa em `docs/`
