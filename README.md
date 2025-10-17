# 🎤 Whisper Stream - Real-Time Transcription

Sistema de transcrição em tempo real usando OpenAI Whisper com arquitetura cliente-servidor separada.

## 🏗️ Arquitetura

Este projeto é dividido em dois componentes independentes:

### Cliente (Bun/TypeScript)
- Captura áudio do microfone
- Interface de terminal elegante
- Conecta a servidor Python remoto via WebSocket
- **Agnóstico ao hardware** - não precisa de GPU

### Servidor (Python)
- Servidor WebSocket independente
- Processamento Whisper com detecção automática de hardware
- Suporta Apple Silicon (MLX), NVIDIA (CUDA) e CPU
- **Pode ser deployado separadamente** com GPU

## 🚀 Quick Start

### Servidor Python

```bash
# 1. Instalar dependências
cd /path/to/whisper-stream
poetry install

# Para Apple Silicon:
poetry install -E mlx

# Para NVIDIA GPU:
poetry install -E cuda

# 2. Copiar e configurar
cp server-config.example.yaml server-config.yaml
# Edite server-config.yaml conforme necessário

# 3. Iniciar servidor
poetry run python -m server.main --config server-config.yaml
```

### Cliente Bun

```bash
# 1. Instalar dependências
bun install

# 2. Copiar e configurar
cp config.example.yaml config.yaml
# Edite config.yaml (especialmente server.url se o servidor estiver remoto)

# 3. Iniciar cliente
bun start
```

## ⚙️ Configuração

### Cliente (config.yaml)

```yaml
server:
  url: "ws://localhost:9090/ws"  # Ou URL remota
audio:
  sample_rate: 16000
  device: "default"
display:
  show_partial: true
  colors:
    final: "green"
    partial: "cyan"
```

### Servidor (server-config.yaml)

```yaml
server:
  host: "0.0.0.0"
  port: 9090
whisper:
  model: "base"
  language: "pt"
  backend: "auto"  # Detecta automaticamente
```

## 📋 Status do Projeto

- [x] Estrutura inicial
- [ ] Servidor Python implementado
- [ ] Cliente Bun implementado
- [ ] Testado e documentado

## 📄 Licença

MIT License - veja [LICENSE](LICENSE)

---

**Documentação completa em desenvolvimento...**
