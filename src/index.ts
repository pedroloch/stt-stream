#!/usr/bin/env bun
/**
 * Whisper Stream Client - Exemplo simples
 *
 * Cliente minimalista que demonstra como consumir o servidor Python.
 * Captura áudio do microfone e exibe transcrições no terminal.
 *
 * Uso:
 *   bun start
 */

import { WebSocket } from "ws";
import chalk from "chalk";
import { spawn, type ChildProcess } from "child_process";
import { readFileSync, existsSync } from "fs";
import { parse } from "yaml";

interface Config {
  server: {
    url: string;
    health_url: string;
  };
  audio: {
    sample_rate: number;
    channels: number;
  };
  display: {
    show_partial: boolean;
    show_timestamps: boolean;
    colors: {
      partial: string;
      final: string;
      error: string;
      info: string;
    };
  };
}

class WhisperStreamClient {
  private config: Config;
  private ws: WebSocket | null = null;
  private audioProcess: ChildProcess | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;

  constructor(configPath: string = "config.yaml") {
    // Carregar config
    if (existsSync(configPath)) {
      const configFile = readFileSync(configPath, "utf-8");
      this.config = parse(configFile);
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
          },
        },
      };
    }
  }

  async start() {
    this.printBanner();

    // Verificar health do servidor
    await this.checkServerHealth();

    // Conectar WebSocket
    await this.connectWebSocket();

    // Iniciar captura de áudio
    this.startAudioCapture();

    // Handle Ctrl+C
    process.on("SIGINT", () => this.stop());
  }

  private printBanner() {
    console.clear();
    console.log(chalk.bold.cyan("╔═══════════════════════════════════════════════╗"));
    console.log(chalk.bold.cyan("║   🎤 Whisper Stream Client                   ║"));
    console.log(chalk.bold.cyan("╚═══════════════════════════════════════════════╝"));
    console.log();
  }

  private async checkServerHealth() {
    try {
      console.log(chalk.blue("🔍 Verificando servidor..."));
      const response = await fetch(this.config.server.health_url);

      if (response.ok) {
        const data = await response.json();
        console.log(chalk.green("✅ Servidor está online"));
        console.log(chalk.gray(`   Status: ${data.status}`));
      } else {
        console.log(chalk.yellow("⚠️  Servidor respondeu mas não está pronto"));
      }
    } catch (error) {
      console.log(chalk.red("❌ Não foi possível conectar ao servidor"));
      console.log(chalk.red(`   URL: ${this.config.server.health_url}`));
      console.log(chalk.yellow("\n💡 Certifique-se de que o servidor Python está rodando:"));
      console.log(chalk.gray("   python -m server.main --config server-config.yaml"));
      process.exit(1);
    }
  }

  private async connectWebSocket() {
    return new Promise<void>((resolve, reject) => {
      console.log(chalk.blue(`🔌 Conectando ao WebSocket...`));
      console.log(chalk.gray(`   ${this.config.server.url}`));

      this.ws = new WebSocket(this.config.server.url);

      this.ws.on("open", () => {
        console.log(chalk.green("✅ WebSocket conectado"));
        this.reconnectAttempts = 0;
        resolve();
      });

      this.ws.on("message", (data: Buffer) => {
        this.handleMessage(data);
      });

      this.ws.on("error", (error) => {
        console.log(chalk.red(`❌ WebSocket error: ${error.message}`));
        reject(error);
      });

      this.ws.on("close", () => {
        console.log(chalk.yellow("\n⚠️  WebSocket desconectado"));
        this.handleDisconnect();
      });
    });
  }

  private handleMessage(data: Buffer) {
    try {
      const message = JSON.parse(data.toString());

      switch (message.type) {
        case "connected":
          console.log(chalk.green(`\n✅ ${message.message}`));
          console.log(chalk.gray("═".repeat(60)));
          console.log(chalk.bold.white("Fale no microfone - transcrições aparecerão abaixo:"));
          console.log(chalk.gray("═".repeat(60)));
          console.log();
          break;

        case "transcription":
          this.displayTranscription(message);
          break;

        case "error":
          console.log(chalk.red(`\n❌ Erro: ${message.message}`));
          break;

        case "pong":
          // Resposta ao ping
          break;

        default:
          console.log(chalk.gray(`[Debug] ${message.type}: ${JSON.stringify(message)}`));
      }
    } catch (error) {
      console.log(chalk.red(`Erro ao processar mensagem: ${error}`));
    }
  }

  private displayTranscription(message: any) {
    const { text, is_final, confidence, timestamp } = message;

    // Timestamp
    let output = "";
    if (this.config.display.show_timestamps) {
      const time = new Date(timestamp).toLocaleTimeString();
      output += chalk.gray(`[${time}] `);
    }

    // Ícone
    const icon = is_final ? "✅" : "🎤";
    output += `${icon} `;

    // Texto com cor
    const color = is_final
      ? this.config.display.colors.final
      : this.config.display.colors.partial;

    const colorFn = this.getChalkColor(color);
    output += colorFn(text);

    // Confiança
    if (is_final && confidence) {
      const confPercent = (confidence * 100).toFixed(0);
      output += chalk.gray(` (${confPercent}%)`);
    }

    // Imprimir
    if (is_final) {
      console.log(output);
    } else if (this.config.display.show_partial) {
      // Transcrição parcial - sobrescreve linha
      process.stdout.write("\r" + " ".repeat(100) + "\r" + output);
    }
  }

  private getChalkColor(colorName: string): (text: string) => string {
    const colors: Record<string, any> = {
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

  private startAudioCapture() {
    console.log(chalk.blue("\n🎙️  Iniciando captura de áudio..."));

    // Usar sox para capturar áudio do microfone
    // Formato: raw PCM, int16, mono, 16kHz
    const soxArgs = [
      "-d",                                    // Default input (microfone)
      "-t", "raw",                            // Formato raw
      "-r", this.config.audio.sample_rate.toString(), // Sample rate
      "-c", this.config.audio.channels.toString(),     // Channels (mono)
      "-b", "16",                             // 16-bit
      "-e", "signed-integer",                 // Signed integer
      "-",                                     // Output para stdout
    ];

    this.audioProcess = spawn("sox", soxArgs);

    if (!this.audioProcess.stdout) {
      console.log(chalk.red("❌ Erro ao iniciar captura de áudio"));
      return;
    }

    console.log(chalk.green("✅ Captura de áudio iniciada"));

    // Enviar áudio em chunks de ~1 segundo
    const chunkSize = this.config.audio.sample_rate * 2; // 16-bit = 2 bytes por sample
    let buffer = Buffer.alloc(0);

    this.audioProcess.stdout.on("data", (data: Buffer) => {
      buffer = Buffer.concat([buffer, data]);

      // Quando tivermos um chunk completo, enviar
      while (buffer.length >= chunkSize) {
        const chunk = buffer.subarray(0, chunkSize);
        buffer = buffer.subarray(chunkSize);

        // Enviar via WebSocket
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(chunk);
        }
      }
    });

    this.audioProcess.stderr?.on("data", (data) => {
      // Sox pode gerar warnings que podemos ignorar
      const msg = data.toString();
      if (!msg.includes("WARN")) {
        console.log(chalk.gray(`[sox] ${msg}`));
      }
    });

    this.audioProcess.on("error", (error) => {
      console.log(chalk.red(`\n❌ Erro no processo de áudio: ${error.message}`));
      if (error.message.includes("ENOENT")) {
        console.log(chalk.yellow("\n💡 sox não está instalado. Instale com:"));
        console.log(chalk.gray("   macOS: brew install sox"));
        console.log(chalk.gray("   Linux: sudo apt-get install sox"));
        process.exit(1);
      }
    });

    this.audioProcess.on("exit", (code) => {
      console.log(chalk.yellow(`\n⚠️  Processo de áudio encerrado (código: ${code})`));
    });
  }

  private handleDisconnect() {
    // Tentar reconectar
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(
        chalk.yellow(
          `🔄 Tentando reconectar (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`
        )
      );

      setTimeout(() => {
        this.connectWebSocket().catch(() => {
          // Se falhar, handleDisconnect será chamado novamente
        });
      }, 2000);
    } else {
      console.log(chalk.red("\n❌ Número máximo de tentativas de reconexão atingido"));
      this.stop();
    }
  }

  private stop() {
    console.log(chalk.yellow("\n\n🛑 Encerrando cliente..."));

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

// Main
async function main() {
  const client = new WhisperStreamClient();
  await client.start();
}

main().catch((error) => {
  console.error(chalk.red(`❌ Erro fatal: ${error.message}`));
  process.exit(1);
});
