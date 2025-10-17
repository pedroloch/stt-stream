#!/usr/bin/env bun
/**
 * Whisper Stream Client - Versão Melhorada
 *
 * Cliente com UI profissional:
 * - Header fixo com status
 * - Chat descendo
 * - Filtro de duplicatas
 * - Modo debug
 * - Tipagem TypeScript completa
 *
 * Uso:
 *   bun start              # Normal
 *   DEBUG=1 bun start      # Com debug
 */

import { WebSocket } from "ws";
import chalk from "chalk";
import { spawn, type ChildProcess } from "child_process";
import { readFileSync, existsSync } from "fs";
import { parse } from "yaml";
import type {
  Config,
  AnyWebSocketMessage,
  TranscriptionMessage,
  ConnectedMessage,
  ErrorMessage,
  ConnectionStatus,
  AudioStatus,
  MessageHistoryItem,
} from "./types";

class WhisperStreamClient {
  private config: Config;
  private ws: WebSocket | null = null;
  private audioProcess: ChildProcess | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;

  // Estado
  private connectionStatus: ConnectionStatus = ConnectionStatus.Disconnected;
  private audioStatus: AudioStatus = AudioStatus.Inactive;

  // Filtro de duplicatas
  private lastPartialText = "";
  private lastFinalText = "";

  // Histórico de mensagens
  private messageHistory: MessageHistoryItem[] = [];
  private maxHistory = 50;

  // Debug mode
  private debugMode: boolean;

  constructor(configPath: string = "config.yaml") {
    // Carregar config
    if (existsSync(configPath)) {
      const configFile = readFileSync(configPath, "utf-8");
      this.config = parse(configFile) as Config;
    } else {
      // Config padrão se arquivo não existe
      console.log(chalk.yellow(`⚠️  Config file not found: ${configPath}`));
      console.log(chalk.yellow("Using default configuration..."));
      this.config = {
        server: {
          url: "ws://localhost:9090/ws",
          health_url: "http://localhost:9090/health",
        },
        audio: {
          sample_rate: 16000,
          channels: 1,
        },
        display: {
          show_partial: true,
          show_timestamps: true,
          colors: {
            partial: "cyan",
            final: "green",
            error: "red",
            info: "yellow",
            debug: "gray",
            header: "cyan",
          },
          max_history: 50,
          word_wrap_width: 80,
        },
      };
    }

    // Debug mode: ler de env ou config
    this.debugMode =
      process.env.DEBUG === "1" ||
      process.env.DEBUG === "true" ||
      this.config.debug?.enabled ||
      false;

    // Aplicar configurações
    this.maxHistory = this.config.display.max_history || 50;
  }

  async start() {
    // Limpar terminal e mostrar header
    this.renderUI();

    // Verificar health do servidor
    await this.checkServerHealth();

    // Conectar WebSocket
    await this.connectWebSocket();

    // Iniciar captura de áudio
    this.startAudioCapture();

    // Handle Ctrl+C
    process.on("SIGINT", () => this.stop());
  }

  // ============================================
  // UI Rendering
  // ============================================

  private renderUI() {
    console.clear();
    this.renderHeader();
    this.renderHistory();
  }

  private renderHeader() {
    const headerColor = this.getChalkColor(
      this.config.display.colors.header || "cyan"
    );

    console.log(headerColor.bold("═".repeat(70)));
    console.log(headerColor.bold("🎤 WHISPER STREAM CLIENT"));
    console.log(
      chalk.gray(
        `Server: ${this.config.server.url} | Debug: ${this.debugMode ? "ON" : "OFF"}`
      )
    );

    // Status line
    let statusLine = "";

    // Connection status
    switch (this.connectionStatus) {
      case ConnectionStatus.Connected:
        statusLine += chalk.green("✅ Conectado");
        break;
      case ConnectionStatus.Connecting:
        statusLine += chalk.yellow("🔄 Conectando...");
        break;
      case ConnectionStatus.Reconnecting:
        statusLine +=
          chalk.yellow(`🔄 Reconectando (${this.reconnectAttempts})...`);
        break;
      case ConnectionStatus.Disconnected:
        statusLine += chalk.red("❌ Desconectado");
        break;
      case ConnectionStatus.Error:
        statusLine += chalk.red("⚠️ Erro");
        break;
    }

    statusLine += " | ";

    // Audio status
    switch (this.audioStatus) {
      case AudioStatus.Active:
        statusLine += chalk.green("🎙️ Áudio Ativo");
        break;
      case AudioStatus.Starting:
        statusLine += chalk.yellow("🎙️ Iniciando...");
        break;
      case AudioStatus.Inactive:
        statusLine += chalk.gray("🎙️ Inativo");
        break;
      case AudioStatus.Error:
        statusLine += chalk.red("🎙️ Erro");
        break;
    }

    console.log(statusLine);
    console.log(headerColor.bold("═".repeat(70)));
    console.log();
  }

  private renderHistory() {
    // Mostrar últimas N mensagens
    const recentMessages = this.messageHistory.slice(-this.maxHistory);

    if (recentMessages.length === 0) {
      console.log(
        chalk.gray(
          "Aguardando transcrições... Fale no microfone para começar."
        )
      );
      console.log();
    } else {
      recentMessages.forEach((msg) => this.renderMessage(msg));
    }
  }

  private renderMessage(msg: MessageHistoryItem) {
    const time = msg.timestamp.toLocaleTimeString();
    const confStr = msg.confidence
      ? ` ${chalk.gray(`(${(msg.confidence * 100).toFixed(0)}%)`)}`
      : "";

    // Word wrap
    const wrapped = this.wordWrap(
      msg.text,
      this.config.display.word_wrap_width || 80
    );

    const color = msg.is_final
      ? this.getChalkColor(this.config.display.colors.final)
      : this.getChalkColor(this.config.display.colors.partial);

    const icon = msg.is_final ? "" : "🎤 ";

    wrapped.forEach((line, i) => {
      if (i === 0) {
        // Primeira linha: timestamp + texto
        if (this.config.display.show_timestamps) {
          console.log(
            chalk.gray(`[${time}] `) + icon + color(line) + (i === 0 ? confStr : "")
          );
        } else {
          console.log(icon + color(line) + (i === 0 ? confStr : ""));
        }
      } else {
        // Linhas seguintes: indentadas
        const indent = this.config.display.show_timestamps
          ? " ".repeat(11 + icon.length)
          : " ".repeat(icon.length);
        console.log(indent + color(line));
      }
    });
  }

  private wordWrap(text: string, width: number): string[] {
    const words = text.split(" ");
    const lines: string[] = [];
    let currentLine = "";

    for (const word of words) {
      if ((currentLine + word).length > width) {
        if (currentLine) {
          lines.push(currentLine.trim());
          currentLine = word + " ";
        } else {
          // Palavra muito longa, quebrar de qualquer jeito
          lines.push(word);
        }
      } else {
        currentLine += word + " ";
      }
    }

    if (currentLine.trim()) {
      lines.push(currentLine.trim());
    }

    return lines.length > 0 ? lines : [text];
  }

  private updatePartialTranscription(text: string) {
    // Limpar linha anterior
    process.stdout.write("\r" + " ".repeat(100) + "\r");

    // Mostrar nova transcrição parcial
    const wrapped = this.wordWrap(
      text,
      this.config.display.word_wrap_width || 80
    );
    const color = this.getChalkColor(this.config.display.colors.partial);

    if (wrapped.length > 0) {
      process.stdout.write("🎤 " + color(wrapped[0]));
    }
  }

  // ============================================
  // Server Communication
  // ============================================

  private async checkServerHealth() {
    try {
      this.connectionStatus = ConnectionStatus.Connecting;
      this.renderHeader();

      const response = await fetch(this.config.server.health_url);

      if (response.ok) {
        const data = await response.json();
        this.log("info", `Servidor está online (${data.status})`);
      } else {
        this.log("warning", "Servidor respondeu mas não está pronto");
      }
    } catch (error) {
      this.log("error", "Não foi possível conectar ao servidor");
      console.log(chalk.red(`   URL: ${this.config.server.health_url}`));
      console.log(
        chalk.yellow("\n💡 Certifique-se de que o servidor Python está rodando:")
      );
      console.log(
        chalk.gray("   python -m server.main --config server-config.yaml")
      );
      process.exit(1);
    }
  }

  private async connectWebSocket() {
    return new Promise<void>((resolve, reject) => {
      this.connectionStatus = ConnectionStatus.Connecting;
      this.renderHeader();

      this.ws = new WebSocket(this.config.server.url);

      this.ws.on("open", () => {
        this.connectionStatus = ConnectionStatus.Connected;
        this.reconnectAttempts = 0;
        this.renderUI();
        resolve();
      });

      this.ws.on("message", (data: Buffer) => {
        this.handleMessage(data);
      });

      this.ws.on("error", (error) => {
        this.log("error", `WebSocket error: ${error.message}`);
        this.connectionStatus = ConnectionStatus.Error;
        reject(error);
      });

      this.ws.on("close", () => {
        this.log("warning", "WebSocket desconectado");
        this.connectionStatus = ConnectionStatus.Disconnected;
        this.handleDisconnect();
      });
    });
  }

  private handleMessage(data: Buffer) {
    try {
      const message: AnyWebSocketMessage = JSON.parse(data.toString());

      // Debug mode: mostrar JSON completo
      if (this.debugMode) {
        this.logDebug("Message received:", message);
      }

      switch (message.type) {
        case "connected":
          this.handleConnected(message as ConnectedMessage);
          break;

        case "transcription":
          this.handleTranscription(message as TranscriptionMessage);
          break;

        case "error":
          this.handleError(message as ErrorMessage);
          break;

        case "pong":
          // Resposta ao ping (ignorar no debug mode normal)
          if (this.debugMode) {
            this.logDebug("Pong received");
          }
          break;

        default:
          this.logDebug(`Unknown message type: ${message.type}`, message);
      }
    } catch (error) {
      this.log("error", `Erro ao processar mensagem: ${error}`);
    }
  }

  private handleConnected(message: ConnectedMessage) {
    this.log("success", message.message);

    if (message.session_id) {
      this.logDebug(`Session ID: ${message.session_id}`);
    }
  }

  private handleTranscription(message: TranscriptionMessage) {
    const { text, is_final, confidence, timestamp } = message;

    // Filtro: ignorar se vazio ou muito curto
    if (!text || text.trim().length < 2) {
      this.logDebug("Ignored empty/short transcription");
      return;
    }

    // Filtro de duplicatas
    if (!this.shouldShowMessage(text, is_final)) {
      this.logDebug("Ignored duplicate transcription");
      return;
    }

    // Adicionar ao histórico se final
    if (is_final) {
      // Limpar linha parcial antes de adicionar final
      process.stdout.write("\r" + " ".repeat(100) + "\r");

      const historyItem: MessageHistoryItem = {
        timestamp: new Date(timestamp),
        text,
        is_final: true,
        confidence,
      };

      this.messageHistory.push(historyItem);

      // Renderizar só a nova mensagem
      this.renderMessage(historyItem);
    } else if (this.config.display.show_partial) {
      // Transcrição parcial: atualizar linha
      this.updatePartialTranscription(text);
    }
  }

  private handleError(message: ErrorMessage) {
    this.log("error", message.message);

    if (message.code) {
      this.logDebug(`Error code: ${message.code}`);
    }
  }

  private shouldShowMessage(text: string, is_final: boolean): boolean {
    if (is_final) {
      // Mensagem final: verificar se é diferente da última final
      if (text.trim() === this.lastFinalText.trim()) {
        return false;
      }
      this.lastFinalText = text;
      // Resetar parcial quando temos uma final nova
      this.lastPartialText = "";
      return true;
    } else {
      // Mensagem parcial: verificar se é diferente da última parcial
      if (text.trim() === this.lastPartialText.trim()) {
        return false;
      }
      this.lastPartialText = text;
      return true;
    }
  }

  // ============================================
  // Audio Capture
  // ============================================

  private startAudioCapture() {
    this.audioStatus = AudioStatus.Starting;
    this.renderHeader();

    // Usar sox para capturar áudio do microfone
    const soxArgs = [
      "-d", // Default input (microfone)
      "-t",
      "raw", // Formato raw
      "-r",
      this.config.audio.sample_rate.toString(),
      "-c",
      this.config.audio.channels.toString(),
      "-b",
      "16", // 16-bit
      "-e",
      "signed-integer",
      "-", // Output para stdout
    ];

    this.audioProcess = spawn("sox", soxArgs);

    if (!this.audioProcess.stdout) {
      this.log("error", "Erro ao iniciar captura de áudio");
      this.audioStatus = AudioStatus.Error;
      return;
    }

    this.audioStatus = AudioStatus.Active;
    this.renderUI();

    // Enviar áudio em chunks
    const chunkSize = this.config.audio.sample_rate * 2; // 2 bytes per sample
    let buffer = Buffer.alloc(0);

    this.audioProcess.stdout.on("data", (data: Buffer) => {
      buffer = Buffer.concat([buffer, data]);

      while (buffer.length >= chunkSize) {
        const chunk = buffer.subarray(0, chunkSize);
        buffer = buffer.subarray(chunkSize);

        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(chunk);
        }
      }
    });

    this.audioProcess.stderr?.on("data", (data) => {
      const msg = data.toString();
      if (!msg.includes("WARN") && this.debugMode) {
        this.logDebug(`[sox] ${msg}`);
      }
    });

    this.audioProcess.on("error", (error) => {
      this.log("error", `Erro no processo de áudio: ${error.message}`);
      this.audioStatus = AudioStatus.Error;

      if (
        error.message.includes("ENOENT") ||
        error.message.includes("Executable not found")
      ) {
        console.log(
          chalk.yellow("\n💡 sox não está instalado. Instale com:")
        );
        console.log(chalk.gray("   macOS: brew install sox"));
        console.log(chalk.gray("   Linux: sudo apt-get install sox"));
        console.log(
          chalk.gray("\nDepois de instalar, rode novamente: bun start")
        );
        process.exit(1);
      }
    });

    this.audioProcess.on("exit", (code) => {
      this.log("warning", `Processo de áudio encerrado (código: ${code})`);
      this.audioStatus = AudioStatus.Inactive;
    });
  }

  // ============================================
  // Reconnection
  // ============================================

  private handleDisconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      this.connectionStatus = ConnectionStatus.Reconnecting;
      this.renderHeader();

      setTimeout(() => {
        this.connectWebSocket().catch(() => {
          // handleDisconnect será chamado novamente
        });
      }, 2000);
    } else {
      this.log("error", "Número máximo de tentativas de reconexão atingido");
      this.stop();
    }
  }

  // ============================================
  // Logging & Utilities
  // ============================================

  private log(
    level: "info" | "success" | "warning" | "error",
    message: string
  ) {
    const prefix = {
      info: chalk.blue("ℹ️"),
      success: chalk.green("✅"),
      warning: chalk.yellow("⚠️"),
      error: chalk.red("❌"),
    }[level];

    console.log(`${prefix} ${message}`);
  }

  private logDebug(message: string, data?: any) {
    if (!this.debugMode) return;

    const debugColor = this.getChalkColor(
      this.config.display.colors.debug || "gray"
    );

    console.log(debugColor("─".repeat(70)));
    console.log(debugColor(`[DEBUG] ${message}`));

    if (data) {
      console.log(debugColor(JSON.stringify(data, null, 2)));
    }

    console.log(debugColor("─".repeat(70)));
  }

  private getChalkColor(colorName: string): typeof chalk.white {
    const colors: Record<string, typeof chalk.white> = {
      cyan: chalk.cyan,
      green: chalk.green,
      yellow: chalk.yellow,
      red: chalk.red,
      blue: chalk.blue,
      magenta: chalk.magenta,
      white: chalk.white,
      gray: chalk.gray,
    };
    return colors[colorName] || chalk.white;
  }

  // ============================================
  // Cleanup
  // ============================================

  private stop() {
    console.log(chalk.yellow("\n\n🛑 Encerrando cliente...\n"));

    // Parar captura de áudio
    if (this.audioProcess) {
      this.audioProcess.kill();
      this.audioProcess = null;
    }

    // Fechar WebSocket
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    console.log(chalk.green("✅ Cliente encerrado"));
    process.exit(0);
  }
}

// ============================================
// Main
// ============================================

async function main() {
  const client = new WhisperStreamClient();
  await client.start();
}

main().catch((error) => {
  console.error(chalk.red(`❌ Erro fatal: ${error.message}`));
  process.exit(1);
});
