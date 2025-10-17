# 🎙️ Especificação Técnica: Whisper Real-Time Transcription System

## 📋 Visão Geral

Sistema de transcrição em tempo real usando OpenAI Whisper, com arquitetura cliente-servidor via WebSocket. Interface de linha de comando simples com `bun start` que exibe transcrições enquanto você fala.

---

## 🎯 Objetivos

### Objetivo Principal

Criar um sistema plug-and-play de transcrição em tempo real que:

- Funciona com um único comando: `bun start`
- Mostra transcrições aparecendo dinamicamente no terminal
- Suporta múltiplas plataformas (macOS, Linux com CUDA)
- É altamente configurável via arquivo de configuração

### Requisitos Funcionais

#### RF01 - Comando Único de Inicialização

- **Descrição**: Usuário deve poder iniciar todo o sistema com `bun start`
- **Critério**: Servidor Python e cliente TypeScript iniciam automaticamente
- **Prioridade**: Alta

#### RF02 - Transcrição em Tempo Real

- **Descrição**: Áudio capturado do microfone é transcrito continuamente
- **Critério**: Latência < 3 segundos com modelo base
- **Prioridade**: Alta

#### RF03 - Visualização no Terminal

- **Descrição**: Transcrições aparecem no terminal de forma legível
- **Critério**:
  - Transcrições parciais em uma cor
  - Transcrições finais em outra cor
  - Auto-scroll
  - Timestamps opcionais
- **Prioridade**: Alta

#### RF04 - Configuração Flexível

- **Descrição**: Sistema configurável via arquivo YAML/JSON
- **Critério**: Permitir configurar:
  - Modelo Whisper (tiny, base, small, medium, large)
  - Idioma
  - Backend (MLX, CUDA, CPU)
  - Porta do servidor
  - Sample rate
  - VAD on/off
- **Prioridade**: Alta

#### RF05 - Detecção Automática de Hardware

- **Descrição**: Sistema detecta automaticamente GPU disponível
- **Critério**:
  - Detecta Apple Silicon → usa MLX
  - Detecta NVIDIA GPU → usa CUDA
  - Fallback para CPU
- **Prioridade**: Média

#### RF06 - Gestão de Processo

- **Descrição**: Cliente gerencia ciclo de vida do servidor
- **Critério**:
  - Inicia servidor automaticamente
  - Para servidor ao sair (Ctrl+C)
  - Reconecta em caso de queda
- **Prioridade**: Alta

### Requisitos Não-Funcionais

#### RNF01 - Performance

- Latência máxima: 3 segundos (modelo base)
- Uso de memória: < 2GB RAM (modelo base)
- CPU: Uso otimizado com threads configuráveis

#### RNF02 - Usabilidade

- Setup em < 5 minutos
- Documentação clara
- Mensagens de erro úteis
- Logs informativos

#### RNF03 - Portabilidade

- macOS (Intel e Apple Silicon)
- Linux (CUDA)
- Python 3.10+
- Bun 1.0+

#### RNF04 - Confiabilidade

- Reconexão automática
- Tratamento de erros robusto
- Graceful shutdown

---

## 🏗️ Arquitetura

### Visão Geral da Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                         USUÁRIO                                 │
│                    $ bun start                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BUN ORCHESTRATOR                             │
│  - Lê configuração (config.yaml)                               │
│  - Detecta hardware (GPU/CPU)                                  │
│  - Inicia servidor Python                                      │
│  - Inicia cliente de captura                                   │
│  - Gerencia processos                                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
┌───────────────────────────┐   ┌────────────────────────────┐
│   SERVIDOR PYTHON         │   │   CLIENTE BUN/TYPESCRIPT   │
│   (Whisper WebSocket)     │   │   (Audio Capture)          │
│                           │   │                            │
│  - Recebe áudio via WS    │◄──┤  - Captura microfone       │
│  - Processa com Whisper   │   │  - Envia via WebSocket     │
│  - Retorna transcrição    │──►│  - Exibe no terminal       │
│                           │   │                            │
│  Backend:                 │   │  UI:                       │
│  • MLX (Apple Silicon)    │   │  • Terminal elegante       │
│  • CUDA (NVIDIA)          │   │  • Cores/formatação        │
│  • CPU (fallback)         │   │  • Timestamps              │
└───────────────────────────┘   └────────────────────────────┘
         │                               │
         │                               │
         ▼                               ▼
┌──────────────────┐            ┌─────────────────┐
│  Whisper Model   │            │   Microfone     │
│  (*.bin)         │            │   (sox/Web API) │
└──────────────────┘            └─────────────────┘
```

### Componentes

#### 1. Orchestrator (Bun/TypeScript)

**Responsabilidades:**

- Entry point do sistema (`bun start`)
- Leitura de configuração
- Detecção de hardware
- Gerenciamento de processos (iniciar/parar servidor)
- Coordenação entre componentes

**Arquivos:**

- `src/index.ts` - Entry point
- `src/orchestrator.ts` - Lógica de orquestração
- `src/config.ts` - Gerenciamento de configuração
- `src/hardware-detector.ts` - Detecção de hardware

#### 2. Servidor WebSocket (Python)

**Responsabilidades:**

- Receber áudio via WebSocket
- Processar com Whisper (whisper_streaming)
- Gerenciar múltiplos clientes
- Retornar transcrições

**Arquivos:**

- `server/main.py` - Entry point do servidor
- `server/websocket_handler.py` - Lógica WebSocket
- `server/whisper_processor.py` - Wrapper do Whisper
- `server/backends/` - Backends específicos (MLX, CUDA, CPU)

#### 3. Cliente de Captura (Bun/TypeScript)

**Responsabilidades:**

- Capturar áudio do microfone
- Enviar para servidor via WebSocket
- Receber e exibir transcrições
- Interface de terminal elegante

**Arquivos:**

- `src/client/audio-capture.ts` - Captura de áudio
- `src/client/websocket-client.ts` - Cliente WebSocket
- `src/client/terminal-ui.ts` - Interface do terminal
- `src/client/transcription-display.ts` - Exibição formatada

---

## 📂 Estrutura de Diretórios

```
whisper-realtime/
├── README.md                    # Documentação principal
├── PROJECT_SPEC.md             # Esta especificação
├── LICENSE                     # MIT License
├── .gitignore                  # Git ignore
│
├── config.yaml                 # Configuração principal
├── config.example.yaml         # Exemplo de configuração
│
├── package.json                # Dependências Bun/Node
├── tsconfig.json               # Config TypeScript
├── bun.lockb                   # Lock file do Bun
│
├── pyproject.toml              # Dependências Python (Poetry)
├── poetry.lock                 # Lock file Python
├── requirements.txt            # Fallback pip
│
├── scripts/                    # Scripts auxiliares
│   ├── install.sh             # Instalação completa
│   ├── setup-python.sh        # Setup Python
│   ├── setup-bun.sh           # Setup Bun
│   ├── download-models.sh     # Download modelos
│   └── test-hardware.sh       # Testar hardware
│
├── src/                        # Código TypeScript
│   ├── index.ts               # Entry point (bun start)
│   ├── orchestrator.ts        # Orquestração
│   ├── config.ts              # Gestão de config
│   ├── hardware-detector.ts   # Detecção hardware
│   ├── process-manager.ts     # Gestão de processos
│   │
│   ├── client/                # Cliente de captura
│   │   ├── audio-capture.ts
│   │   ├── websocket-client.ts
│   │   ├── terminal-ui.ts
│   │   └── transcription-display.ts
│   │
│   ├── types/                 # Type definitions
│   │   ├── config.ts
│   │   ├── transcription.ts
│   │   └── websocket.ts
│   │
│   └── utils/                 # Utilitários
│       ├── logger.ts
│       ├── errors.ts
│       └── platform.ts
│
├── server/                     # Servidor Python
│   ├── __init__.py
│   ├── main.py                # Entry point servidor
│   ├── config.py              # Config do servidor
│   ├── websocket_handler.py   # Handler WebSocket
│   ├── whisper_processor.py   # Processador Whisper
│   │
│   ├── backends/              # Backends específicos
│   │   ├── __init__.py
│   │   ├── base.py           # Interface base
│   │   ├── mlx_backend.py    # Backend MLX (Apple)
│   │   ├── cuda_backend.py   # Backend CUDA (NVIDIA)
│   │   └── cpu_backend.py    # Backend CPU (fallback)
│   │
│   └── utils/                 # Utilitários Python
│       ├── __init__.py
│       ├── logger.py
│       └── audio.py
│
├── models/                     # Modelos Whisper (gitignored)
│   └── .gitkeep
│
├── logs/                       # Logs (gitignored)
│   └── .gitkeep
│
├── tests/                      # Testes
│   ├── unit/
│   │   ├── test_config.ts
│   │   ├── test_hardware.ts
│   │   └── test_client.py
│   │
│   └── integration/
│       ├── test_e2e.ts
│       └── test_websocket.ts
│
└── docs/                       # Documentação adicional
    ├── ARCHITECTURE.md
    ├── CONFIGURATION.md
    ├── DEPLOYMENT.md
    ├── TROUBLESHOOTING.md
    └── API.md
```

---

## ⚙️ Configuração

### config.yaml

```yaml
# Configuração do Sistema de Transcrição

# Configurações do Servidor
server:
  host: "0.0.0.0"
  port: 9090
  max_clients: 5
  auto_start: true
  log_level: "info" # debug, info, warning, error

# Configurações do Whisper
whisper:
  # Modelo: tiny, base, small, medium, large, large-v3, large-v3-turbo
  model: "base"

  # Idioma (auto para detecção automática)
  language: "pt"

  # Backend: auto, mlx, cuda, cpu
  # auto = detecta automaticamente baseado no hardware
  backend: "auto"

  # Configurações avançadas
  device: "auto" # auto, cuda, cpu, mps
  compute_type: "float16" # float16, int8, int8_float16

  # VAD (Voice Activity Detection)
  use_vad: true
  vad_threshold: 0.5

  # Buffer settings
  min_chunk_size: 1.0 # segundos
  buffer_trimming: "segment" # segment ou sentence

# Configurações de Áudio
audio:
  sample_rate: 16000
  channels: 1
  chunk_duration: 1000 # milliseconds
  device: "default" # default ou device ID específico

# Configurações do Cliente
client:
  auto_reconnect: true
  reconnect_delay: 2000 # milliseconds
  max_reconnect_attempts: 5

# Configurações de Display
display:
  show_partial: true
  show_timestamps: true
  timestamp_format: "HH:mm:ss"

  # Cores (suporta: black, red, green, yellow, blue, magenta, cyan, white)
  colors:
    partial: "cyan"
    final: "green"
    timestamp: "gray"
    error: "red"
    info: "yellow"

  # Formatação
  clear_screen: false
  auto_scroll: true
  max_lines: 100

# Configurações de Performance
performance:
  threads: 4 # Número de threads para Whisper
  batch_size: 1

# Paths
paths:
  models_dir: "./models"
  logs_dir: "./logs"
  cache_dir: "~/.cache/whisper-realtime"

# Debug
debug:
  save_audio: false
  audio_output_dir: "./debug/audio"
  verbose: false
  profile: false
```

---

## 🔧 Implementação Detalhada

### 1. Entry Point (src/index.ts)

```typescript
#!/usr/bin/env bun

/**
 * Entry point do sistema de transcrição
 * Comando: bun start
 */

import { Orchestrator } from "./orchestrator";
import { Logger } from "./utils/logger";
import { loadConfig } from "./config";
import chalk from "chalk";

const logger = new Logger("main");

async function main() {
  try {
    // Banner
    console.clear();
    console.log(
      chalk.bold.cyan("╔═══════════════════════════════════════════════╗")
    );
    console.log(
      chalk.bold.cyan("║   🎤 Whisper Real-Time Transcription         ║")
    );
    console.log(
      chalk.bold.cyan("╚═══════════════════════════════════════════════╝")
    );
    console.log();

    // Carregar configuração
    logger.info("Carregando configuração...");
    const config = await loadConfig();

    // Criar orchestrator
    const orchestrator = new Orchestrator(config);

    // Inicializar sistema
    await orchestrator.initialize();

    // Iniciar transcrição
    await orchestrator.start();

    // Handle Ctrl+C
    process.on("SIGINT", async () => {
      console.log("\n");
      logger.info("Encerrando...");
      await orchestrator.stop();
      process.exit(0);
    });

    // Manter processo vivo
    await new Promise(() => {});
  } catch (error) {
    logger.error("Erro fatal:", error);
    process.exit(1);
  }
}

main();
```

### 2. Orchestrator (src/orchestrator.ts)

```typescript
import { Config } from "./types/config";
import { HardwareDetector } from "./hardware-detector";
import { ProcessManager } from "./process-manager";
import { AudioCapture } from "./client/audio-capture";
import { WebSocketClient } from "./client/websocket-client";
import { TranscriptionDisplay } from "./client/transcription-display";
import { Logger } from "./utils/logger";

export class Orchestrator {
  private config: Config;
  private hardware: HardwareInfo;
  private processManager: ProcessManager;
  private audioCapture: AudioCapture;
  private wsClient: WebSocketClient;
  private display: TranscriptionDisplay;
  private logger: Logger;

  constructor(config: Config) {
    this.config = config;
    this.logger = new Logger("orchestrator");
    this.processManager = new ProcessManager();
  }

  async initialize(): Promise<void> {
    // 1. Detectar hardware
    this.logger.info("Detectando hardware...");
    const detector = new HardwareDetector();
    this.hardware = await detector.detect();

    this.logger.info(`Hardware: ${this.hardware.type}`);

    // 2. Ajustar configuração baseado no hardware
    this.adjustConfigForHardware();

    // 3. Verificar dependências
    await this.checkDependencies();

    // 4. Iniciar servidor Python
    if (this.config.server.auto_start) {
      await this.startServer();
    }

    // 5. Aguardar servidor estar pronto
    await this.waitForServer();

    // 6. Inicializar componentes do cliente
    this.initializeClient();
  }

  async start(): Promise<void> {
    this.logger.info("Iniciando transcrição...");

    // Conectar ao servidor
    await this.wsClient.connect();

    // Iniciar captura de áudio
    await this.audioCapture.start();

    this.logger.success("Sistema pronto! Fale no microfone...");
  }

  async stop(): Promise<void> {
    // Parar captura
    await this.audioCapture?.stop();

    // Desconectar WebSocket
    await this.wsClient?.disconnect();

    // Parar servidor
    if (this.config.server.auto_start) {
      await this.processManager.stopServer();
    }

    this.logger.info("Sistema encerrado");
  }

  private adjustConfigForHardware(): void {
    if (this.config.whisper.backend === "auto") {
      switch (this.hardware.type) {
        case "apple_silicon":
          this.config.whisper.backend = "mlx";
          break;
        case "cuda":
          this.config.whisper.backend = "cuda";
          break;
        default:
          this.config.whisper.backend = "cpu";
      }
    }

    this.logger.info(`Backend selecionado: ${this.config.whisper.backend}`);
  }

  private async startServer(): Promise<void> {
    this.logger.info("Iniciando servidor Python...");

    const serverConfig = {
      host: this.config.server.host,
      port: this.config.server.port,
      model: this.config.whisper.model,
      language: this.config.whisper.language,
      backend: this.config.whisper.backend,
      use_vad: this.config.whisper.use_vad,
    };

    await this.processManager.startServer(serverConfig);
  }

  private async waitForServer(): Promise<void> {
    this.logger.info("Aguardando servidor...");

    const maxAttempts = 30;
    let attempts = 0;

    while (attempts < maxAttempts) {
      try {
        const response = await fetch(
          `http://localhost:${this.config.server.port}/health`
        );
        if (response.ok) {
          this.logger.success("Servidor pronto!");
          return;
        }
      } catch (error) {
        // Servidor ainda não está pronto
      }

      await Bun.sleep(1000);
      attempts++;
    }

    throw new Error("Timeout aguardando servidor");
  }

  private initializeClient(): void {
    // Display
    this.display = new TranscriptionDisplay(this.config.display);

    // WebSocket Client
    this.wsClient = new WebSocketClient(
      this.config.server.host,
      this.config.server.port,
      {
        onTranscription: (text, isFinal) => {
          this.display.addTranscription(text, isFinal);
        },
        onError: (error) => {
          this.logger.error("WebSocket error:", error);
        },
      }
    );

    // Audio Capture
    this.audioCapture = new AudioCapture(this.config.audio, (audioData) => {
      this.wsClient.sendAudio(audioData);
    });
  }

  private async checkDependencies(): Promise<void> {
    // Verificar Python
    const pythonCheck = await $`python3 --version`.quiet();
    if (pythonCheck.exitCode !== 0) {
      throw new Error("Python 3 não encontrado");
    }

    // Verificar sox (para captura de áudio)
    const soxCheck = await $`sox --version`.quiet();
    if (soxCheck.exitCode !== 0) {
      this.logger.warn("sox não encontrado - usando fallback");
    }

    this.logger.success("Dependências verificadas");
  }
}
```

### 3. Hardware Detector (src/hardware-detector.ts)

```typescript
import { exec } from "child_process";
import { promisify } from "util";

const execAsync = promisify(exec);

export interface HardwareInfo {
  type: "apple_silicon" | "cuda" | "cpu";
  details: {
    gpu?: string;
    memory?: string;
    cores?: number;
  };
}

export class HardwareDetector {
  async detect(): Promise<HardwareInfo> {
    // Detectar Apple Silicon
    if (await this.isAppleSilicon()) {
      return {
        type: "apple_silicon",
        details: await this.getAppleInfo(),
      };
    }

    // Detectar CUDA
    if (await this.hasCUDA()) {
      return {
        type: "cuda",
        details: await this.getCUDAInfo(),
      };
    }

    // Fallback para CPU
    return {
      type: "cpu",
      details: await this.getCPUInfo(),
    };
  }

  private async isAppleSilicon(): Promise<boolean> {
    try {
      const { stdout } = await execAsync("uname -m");
      return stdout.trim() === "arm64" && process.platform === "darwin";
    } catch {
      return false;
    }
  }

  private async hasCUDA(): Promise<boolean> {
    try {
      const { stdout } = await execAsync(
        "nvidia-smi --query-gpu=name --format=csv,noheader"
      );
      return stdout.length > 0;
    } catch {
      return false;
    }
  }

  private async getAppleInfo(): Promise<any> {
    try {
      const { stdout } = await execAsync("sysctl -n hw.model");
      return {
        gpu: "Apple Silicon",
        model: stdout.trim(),
      };
    } catch {
      return {};
    }
  }

  private async getCUDAInfo(): Promise<any> {
    try {
      const { stdout } = await execAsync(
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
      );
      const [gpu, memory] = stdout.trim().split(",");
      return { gpu: gpu.trim(), memory: memory.trim() };
    } catch {
      return {};
    }
  }

  private async getCPUInfo(): Promise<any> {
    const cores = require("os").cpus().length;
    return { cores };
  }
}
```

### 4. Terminal UI (src/client/terminal-ui.ts)

```typescript
import chalk from "chalk";
import ora from "ora";

export interface DisplayConfig {
  show_partial: boolean;
  show_timestamps: boolean;
  timestamp_format: string;
  colors: {
    partial: string;
    final: string;
    timestamp: string;
    error: string;
    info: string;
  };
  clear_screen: boolean;
  auto_scroll: boolean;
  max_lines: number;
}

export class TranscriptionDisplay {
  private config: DisplayConfig;
  private lines: string[] = [];
  private spinner: any;

  constructor(config: DisplayConfig) {
    this.config = config;

    if (config.clear_screen) {
      console.clear();
    }

    this.printHeader();
  }

  private printHeader(): void {
    console.log(chalk.bold.cyan("═".repeat(60)));
    console.log(chalk.bold.cyan("  🎤 Transcrição em Tempo Real"));
    console.log(chalk.bold.cyan("═".repeat(60)));
    console.log();
  }

  addTranscription(text: string, isFinal: boolean): void {
    // Parar spinner se existir
    if (this.spinner) {
      this.spinner.stop();
      this.spinner = null;
    }

    const timestamp = this.config.show_timestamps ? this.formatTimestamp() : "";

    const color = isFinal
      ? this.getColor(this.config.colors.final)
      : this.getColor(this.config.colors.partial);

    const icon = isFinal ? "✅" : "🎤";

    const line = `${timestamp}${icon} ${color(text)}`;

    if (isFinal) {
      // Transcrição final - adicionar às linhas
      this.lines.push(line);
      console.log(line);

      // Limitar número de linhas
      if (this.lines.length > this.config.max_lines) {
        this.lines.shift();
        if (this.config.auto_scroll) {
          this.redraw();
        }
      }
    } else {
      // Transcrição parcial - substituir linha atual
      if (this.config.show_partial) {
        process.stdout.write("\r\x1b[K" + line);
      }
    }
  }

  showListening(): void {
    this.spinner = ora({
      text: "Aguardando fala...",
      color: "cyan",
    }).start();
  }

  showError(message: string): void {
    const color = this.getColor(this.config.colors.error);
    console.log(`\n❌ ${color(message)}`);
  }

  showInfo(message: string): void {
    const color = this.getColor(this.config.colors.info);
    console.log(`ℹ️  ${color(message)}`);
  }

  private formatTimestamp(): string {
    const now = new Date();
    const hours = String(now.getHours()).padStart(2, "0");
    const minutes = String(now.getMinutes()).padStart(2, "0");
    const seconds = String(now.getSeconds()).padStart(2, "0");

    const timestamp = `[${hours}:${minutes}:${seconds}] `;
    const color = this.getColor(this.config.colors.timestamp);

    return color(timestamp);
  }

  private getColor(colorName: string): (text: string) => string {
    const colors: Record<string, any> = {
      black: chalk.black,
      red: chalk.red,
      green: chalk.green,
      yellow: chalk.yellow,
      blue: chalk.blue,
      magenta: chalk.magenta,
      cyan: chalk.cyan,
      white: chalk.white,
      gray: chalk.gray,
    };

    return colors[colorName] || chalk.white;
  }

  private redraw(): void {
    if (this.config.clear_screen) {
      console.clear();
      this.printHeader();
      this.lines.forEach((line) => console.log(line));
    }
  }
}
```

### 5. Servidor Python (server/main.py)

```python
#!/usr/bin/env python3
"""
Servidor WebSocket para transcrição Whisper
"""

import asyncio
import sys
import argparse
from pathlib import Path

from config import ServerConfig
from websocket_handler import WebSocketHandler
from whisper_processor import WhisperProcessor
from utils.logger import setup_logger

logger = setup_logger(__name__)


async def health_check(request):
    """Health check endpoint"""
    return web.Response(text='OK')


async def main(config: ServerConfig):
    # Inicializar processador Whisper
    logger.info(f"Inicializando Whisper (modelo: {config.model}, backend: {config.backend})")
    processor = WhisperProcessor(config)
    await processor.initialize()

    # Criar handler WebSocket
    handler = WebSocketHandler(processor, config)

    # Iniciar servidor
    from aiohttp import web
    app = web.Application()
    app.router.add_get('/health', health_check)
    app.router.add_get('/ws', handler.handle_websocket)

    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, config.host, config.port)
    await site.start()

    logger.info(f"✅ Servidor rodando em ws://{config.host}:{config.port}")
    logger.info(f"Backend: {config.backend} | VAD: {config.use_vad}")

    # Manter rodando
    try:
        await asyncio.Future()
    except KeyboardInterrupt:
        logger.info("Encerrando servidor...")
        await runner.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=9090)
    parser.add_argument('--model', default='base')
    parser.add_argument('--language', default='pt')
    parser.add_argument('--backend', default='auto')
    parser.add_argument('--use-vad', action='store_true', default=True)
    parser.add_argument('--config', help='Path to config file')

    args = parser.parse_args()

    # Criar configuração
    if args.config:
        config = ServerConfig.from_file(args.config)
    else:
        config = ServerConfig(
            host=args.host,
            port=args.port,
            model=args.model,
            language=args.language,
            backend=args.backend,
            use_vad=args.use_vad
        )

    try:
        asyncio.run(main(config))
    except KeyboardInterrupt:
        logger.info("Servidor encerrado")
        sys.exit(0)
```

---

## 📦 package.json

```json
{
  "name": "whisper-realtime",
  "version": "1.0.0",
  "description": "Sistema de transcrição em tempo real com Whisper",
  "type": "module",
  "scripts": {
    "start": "bun run src/index.ts",
    "dev": "bun --watch run src/index.ts",
    "build": "bun build src/index.ts --outdir dist --target bun",
    "test": "bun test",
    "install:python": "cd server && poetry install",
    "install:all": "bun install && bun run install:python",
    "setup": "bun run scripts/install.sh",
    "clean": "rm -rf dist logs/*.log"
  },
  "dependencies": {
    "ws": "^8.16.0",
    "chalk": "^5.3.0",
    "ora": "^8.0.1",
    "yaml": "^2.3.4",
    "zod": "^3.22.4"
  },
  "devDependencies": {
    "@types/bun": "latest",
    "@types/ws": "^8.5.10",
    "@types/node": "^20.11.5",
    "bun-types": "latest"
  },
  "peerDependencies": {
    "typescript": "^5.3.3"
  }
}
```

---

## 🐍 pyproject.toml

```toml
[tool.poetry]
name = "whisper-realtime-server"
version = "1.0.0"
description = "Servidor WebSocket para transcrição Whisper"
authors = ["Your Name <you@example.com>"]

[tool.poetry.dependencies]
python = "^3.10"
aiohttp = "^3.9.0"
websockets = "^12.0"
numpy = "^1.24.0"
librosa = "^0.10.0"
soundfile = "^0.12.0"

# Whisper streaming
whisper-streaming = {git = "https://github.com/ufal/whisper_streaming.git"}

# Backends (opcionais)
mlx-whisper = {version = "^0.3.0", optional = true, markers = "sys_platform == 'darwin' and platform_machine == 'arm64'"}
faster-whisper = {version = "^1.0.0", optional = true}

[tool.poetry.extras]
mlx = ["mlx-whisper"]
cuda = ["faster-whisper"]

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
pytest-asyncio = "^0.21.0"
black = "^23.12.0"
ruff = "^0.1.9"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

---

## 🚀 Scripts de Instalação

### scripts/install.sh

```bash
#!/bin/bash
set -e

echo "🚀 Instalando Whisper Real-Time Transcription"
echo ""

# Detectar sistema operacional
OS="$(uname -s)"
ARCH="$(uname -m)"

echo "📊 Sistema: $OS $ARCH"
echo ""

# 1. Instalar sox
echo "📦 Instalando sox..."
if [[ "$OS" == "Darwin" ]]; then
    brew install sox
elif [[ "$OS" == "Linux" ]]; then
    sudo apt-get update && sudo apt-get install -y sox
fi

# 2. Verificar Python
echo "🐍 Verificando Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 não encontrado. Instale Python 3.10+"
    exit 1
fi

# 3. Instalar Poetry
echo "📚 Instalando Poetry..."
if ! command -v poetry &> /dev/null; then
    curl -sSL https://install.python-poetry.org | python3 -
fi

# 4. Instalar dependências Python
echo "📦 Instalando dependências Python..."
cd server
poetry install

# Instalar backend específico baseado no hardware
if [[ "$ARCH" == "arm64" ]] && [[ "$OS" == "Darwin" ]]; then
    echo "🍎 Apple Silicon detectado - instalando MLX..."
    poetry install -E mlx
elif command -v nvidia-smi &> /dev/null; then
    echo "🎮 NVIDIA GPU detectada - instalando CUDA..."
    poetry install -E cuda
else
    echo "💻 CPU mode"
fi

cd ..

# 5. Instalar Bun
echo "🥟 Verificando Bun..."
if ! command -v bun &> /dev/null; then
    echo "Instalando Bun..."
    curl -fsSL https://bun.sh/install | bash
fi

# 6. Instalar dependências Bun
echo "📦 Instalando dependências Bun..."
bun install

# 7. Copiar config exemplo
if [ ! -f config.yaml ]; then
    cp config.example.yaml config.yaml
    echo "✅ config.yaml criado"
fi

echo ""
echo "✅ Instalação completa!"
echo ""
echo "Para iniciar, execute:"
echo "  bun start"
echo ""
```

---

## 📖 README.md

````markdown
# 🎤 Whisper Real-Time Transcription

Sistema de transcrição em tempo real usando OpenAI Whisper com interface de terminal elegante.

## ✨ Features

- 🚀 **Comando único**: `bun start` e pronto!
- ⚡ **Baixa latência**: < 3 segundos
- 🎯 **Auto-detecção**: Detecta hardware (Apple Silicon, CUDA, CPU)
- ⚙️ **Configurável**: Tudo via `config.yaml`
- 🎨 **Terminal elegante**: Cores, timestamps, auto-scroll
- 🔄 **Robusto**: Reconexão automática, tratamento de erros

## 🚀 Quick Start

### 1. Instalação

```bash
# Clone o repositório
git clone https://github.com/user/whisper-realtime.git
cd whisper-realtime

# Instale tudo
./scripts/install.sh
```
````

### 2. Uso

```bash
# Simplesmente rode
bun start

# Fale no microfone e veja a transcrição aparecer!
```

## ⚙️ Configuração

Edite `config.yaml`:

```yaml
whisper:
  model: "base" # tiny, base, small, medium, large
  language: "pt" # pt, en, es, etc
  backend: "auto" # auto, mlx, cuda, cpu

audio:
  sample_rate: 16000
  device: "default"

display:
  show_partial: true
  show_timestamps: true
  colors:
    final: "green"
    partial: "cyan"
```

## 🖥️ Requisitos

- Python 3.10+
- Bun 1.0+
- sox (audio capture)
- macOS ou Linux (CUDA opcional)

## 📚 Documentação

- [Arquitetura](docs/ARCHITECTURE.md)
- [Configuração Avançada](docs/CONFIGURATION.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [API](docs/API.md)

## 📄 Licença

MIT License - veja [LICENSE](LICENSE)

```

---

## ✅ Checklist de Implementação

### Fase 1: Setup Inicial
- [ ] Criar estrutura de diretórios
- [ ] Configurar package.json e pyproject.toml
- [ ] Criar arquivos de configuração
- [ ] Implementar sistema de logging
- [ ] Scripts de instalação

### Fase 2: Core Backend
- [ ] Implementar detector de hardware
- [ ] Criar abstrações de backends (MLX, CUDA, CPU)
- [ ] Implementar processador Whisper
- [ ] Servidor WebSocket Python
- [ ] Health check endpoint

### Fase 3: Core Frontend
- [ ] Orchestrator TypeScript
- [ ] Process manager
- [ ] Config loader com validação
- [ ] WebSocket client
- [ ] Audio capture

### Fase 4: UI/UX
- [ ] Terminal UI com cores
- [ ] Display de transcrições
- [ ] Progress indicators
- [ ] Error handling visual
- [ ] Keyboard shortcuts

### Fase 5: Testing
- [ ] Testes unitários (TypeScript)
- [ ] Testes unitários (Python)
- [ ] Testes de integração
- [ ] Testes E2E
- [ ] Performance testing

### Fase 6: Documentação
- [ ] README completo
- [ ] Documentação de arquitetura
- [ ] Guia de configuração
- [ ] Troubleshooting guide
- [ ] API documentation

### Fase 7: Polish
- [ ] CI/CD setup
- [ ] Release scripts
- [ ] Docker support
- [ ] Benchmarks
- [ ] Demo video

---

## 🎯 Critérios de Aceitação

### Para Entrega Mínima (MVP)

✅ **Funcional**
- [ ] `bun start` inicia sistema completo
- [ ] Transcrição aparece no terminal enquanto fala
- [ ] Suporta macOS e Linux com CUDA
- [ ] Configurável via config.yaml

✅ **Qualidade**
- [ ] Latência < 3s com modelo base
- [ ] Graceful shutdown (Ctrl+C)
- [ ] Logs informativos
- [ ] Tratamento de erros básico

✅ **Documentação**
- [ ] README com quick start
- [ ] Config exemplo comentado
- [ ] Script de instalação funcional

### Para Entrega Completa

Todos itens do MVP +

✅ **Funcionalidades Avançadas**
- [ ] Auto-reconexão
- [ ] Suporte múltiplos modelos
- [ ] VAD configurável
- [ ] Keyboard shortcuts

✅ **Qualidade**
- [ ] Cobertura de testes > 80%
- [ ] Performance otimizada
- [ ] Memory leaks detectados/corrigidos
- [ ] Profiling e benchmarks

✅ **DevEx**
- [ ] Hot reload (modo dev)
- [ ] Debug mode
- [ ] Logs estruturados
- [ ] Error messages úteis

---

## 📈 Roadmap Futuro

### v1.1
- [ ] Interface web opcional
- [ ] Suporte múltiplas línguas simultâneas
- [ ] Export de transcrições (txt, srt, vtt)
- [ ] Estatísticas (WPM, accuracy)

### v1.2
- [ ] Suporte Docker
- [ ] Deploy scripts (Railway, Render)
- [ ] API REST opcional
- [ ] Dashboard web

### v2.0
- [ ] Speaker diarization
- [ ] Punctuation restoration
- [ ] Real-time translation
- [ ] Plugin system

---

**Data de Criação**: 2025-10-17
**Versão**: 1.0.0
**Status**: Especificação Completa
**Próximo Passo**: Implementação
```
