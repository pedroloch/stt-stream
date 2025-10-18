#!/usr/bin/env bun
/**
 * Test script para API Batch - POST /v1/transcribe
 *
 * Testa transcrição batch usando Bun native APIs
 *
 * Uso:
 *   bun test-batch-api.ts
 *   bun test-batch-api.ts --model faster-whisper --language en
 *   bun test-batch-api.ts --file audios/outro-audio.mp3
 */

import { parse } from "yaml";
import chalk from "chalk";

interface Config {
  server: {
    runpod_id?: string;
    url: string;
    health_url: string;
  };
}

interface TranscribeOptions {
  file: string;
  model?: string;
  modelSize?: string;
  language?: string;
  task?: "transcribe" | "translate";
  responseFormat?: "json" | "verbose_json" | "text" | "srt" | "vtt";
  returnMetrics?: boolean;
  enableDiarization?: boolean;
  diarizationBackend?: string;
  numSpeakers?: number;
}

/**
 * Converte segundos para formato mm:ss
 */
function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
}

/**
 * Formata resultado da transcrição em Markdown legível
 */
function formatMarkdown(result: any, options: TranscribeOptions): string {
  const lines: string[] = [];
  const date = new Date().toLocaleString("pt-BR");

  // Cabeçalho
  lines.push(`# 🎙️ Transcrição - ${date}`);
  lines.push("");

  // Metadata
  lines.push("## 📊 Metadata");
  lines.push("");

  if (result.duration_sec) {
    lines.push(`- **Duração:** ${formatTime(result.duration_sec)}`);
  }
  lines.push(`- **Idioma:** ${result.language || options.language || "auto"}`);
  lines.push(`- **Modelo:** ${options.model || "auto"}/${options.modelSize || "base"}`);

  // Speakers detectados
  const speakers = new Set<string>();
  result.segments?.forEach((seg: any) => {
    if (seg.speaker_id) speakers.add(seg.speaker_id);
  });

  if (speakers.size > 0) {
    lines.push(`- **Speakers:** ${speakers.size} detectados (${Array.from(speakers).join(", ")})`);
    if (result.backend_info?.diarization_backend) {
      lines.push(`- **Diarization:** ${result.backend_info.diarization_backend}`);
    }
  }

  // Performance
  if (result.metrics) {
    lines.push(`- **Processamento:** ${result.metrics.processing_time_sec.toFixed(2)}s`);
    lines.push(`- **RTF:** ${result.metrics.real_time_factor.toFixed(3)}x`);
  }

  lines.push("");
  lines.push("---");
  lines.push("");

  // Transcrição por speaker
  lines.push("## 💬 Transcrição");
  lines.push("");

  let currentSpeaker: string | null = null;
  let currentTime = 0;

  for (const seg of result.segments || []) {
    const speaker = seg.speaker_id || "UNKNOWN";
    const time = formatTime(seg.start);

    // Nova seção se mudou de speaker
    if (speaker !== currentSpeaker) {
      if (currentSpeaker !== null) {
        lines.push(""); // Espaço entre speakers
      }
      lines.push(`### ${time} - **${speaker}**`);
      lines.push("");
      currentSpeaker = speaker;
      currentTime = seg.start;
    }

    // Texto do segmento
    lines.push(seg.text.trim());
    lines.push("");
  }

  lines.push("---");
  lines.push("");

  // Detalhes (opcional - só se tiver words)
  const hasWords = result.segments?.some((seg: any) => seg.words && seg.words.length > 0);

  if (hasWords) {
    lines.push("## 📝 Detalhes por Segmento");
    lines.push("");
    lines.push("_Timestamps palavra por palavra com confiança_");
    lines.push("");

    for (const seg of result.segments || []) {
      const timeStart = formatTime(seg.start);
      const timeEnd = formatTime(seg.end);
      const speaker = seg.speaker_id || "UNKNOWN";

      lines.push(`### ${timeStart} - ${timeEnd} | ${speaker}`);
      lines.push("");
      lines.push(`> ${seg.text.trim()}`);
      lines.push("");

      if (seg.words && seg.words.length > 0) {
        lines.push("**Palavras:**");

        for (const word of seg.words) {
          const wordTime = formatTime(word.start);
          const confidence = word.probability ? `${(word.probability * 100).toFixed(0)}%` : "N/A";
          const speakerTag = word.speaker_id ? ` [${word.speaker_id}]` : "";
          lines.push(`- \`${wordTime}\` ${word.word} (${confidence})${speakerTag}`);
        }

        lines.push("");
      }
    }
  }

  lines.push("---");
  lines.push("");
  lines.push(`_Gerado por Whisper Stream - ${date}_`);

  return lines.join("\n");
}

async function testBatchAPI(options: TranscribeOptions) {
  console.log(chalk.blue.bold("\n🎙️  Whisper Stream - Teste API Batch\n"));

  // 1. Carregar config
  const configFile = Bun.file("config.example.yaml");
  const configText = await configFile.text();
  const config = parse(configText) as Config;

  // 2. Construir URL da API batch
  let baseUrl: string;

  if (config.server.runpod_id && config.server.runpod_id.trim()) {
    // RunPod
    const podId = config.server.runpod_id.trim();
    baseUrl = `https://${podId}-9090.proxy.runpod.net`;
    console.log(chalk.gray(`📡 Servidor: RunPod (${podId})`));
  } else {
    // Localhost ou custom
    baseUrl = config.server.health_url.replace("/health", "");
    console.log(chalk.gray(`📡 Servidor: ${baseUrl}`));
  }

  const apiUrl = `${baseUrl}/v1/transcribe`;

  // 3. Verificar se arquivo existe
  const audioFile = Bun.file(options.file);
  const fileExists = await audioFile.exists();

  if (!fileExists) {
    console.error(chalk.red(`❌ Arquivo não encontrado: ${options.file}`));
    console.log(chalk.gray(`   Procure em: ${process.cwd()}/${options.file}`));
    process.exit(1);
  }

  const fileSizeMB = audioFile.size / (1024 * 1024);
  console.log(chalk.gray(`📁 Arquivo: ${options.file} (${fileSizeMB.toFixed(2)} MB)`));

  // 4. Criar FormData
  console.log(chalk.gray("📤 Enviando request...\n"));

  const formData = new FormData();
  formData.append("file", audioFile, options.file.split("/").pop());

  // Adicionar parâmetros
  if (options.model) formData.append("model", options.model);
  if (options.modelSize) formData.append("model_size", options.modelSize);
  if (options.language) formData.append("language", options.language);
  if (options.task) formData.append("task", options.task);
  if (options.responseFormat) formData.append("response_format", options.responseFormat);
  if (options.returnMetrics !== undefined) formData.append("return_metrics", String(options.returnMetrics));
  if (options.enableDiarization !== undefined) formData.append("enable_diarization", String(options.enableDiarization));
  if (options.diarizationBackend) formData.append("diarization_backend", options.diarizationBackend);
  if (options.numSpeakers !== undefined) formData.append("num_speakers", String(options.numSpeakers));

  console.log(chalk.gray("Parâmetros:"));
  console.log(chalk.gray(`  model: ${options.model || "faster-whisper"}`));
  console.log(chalk.gray(`  language: ${options.language || "pt"}`));
  console.log(chalk.gray(`  task: ${options.task || "transcribe"}`));
  console.log(chalk.gray(`  response_format: ${options.responseFormat || "verbose_json"}`));
  if (options.enableDiarization) {
    console.log(chalk.gray(`  diarization: enabled`));
    if (options.diarizationBackend) {
      console.log(chalk.gray(`  diarization_backend: ${options.diarizationBackend}`));
    }
    if (options.numSpeakers) {
      console.log(chalk.gray(`  num_speakers: ${options.numSpeakers}`));
    }
  }
  console.log();

  // 5. Enviar request
  const startTime = Date.now();

  try {
    const response = await fetch(apiUrl, {
      method: "POST",
      body: formData,
    });

    const elapsedSec = (Date.now() - startTime) / 1000;

    // 6. Processar resposta
    if (!response.ok) {
      const errorData = await response.json();
      console.error(chalk.red(`\n❌ Erro ${response.status}: ${response.statusText}`));
      console.error(chalk.red(JSON.stringify(errorData, null, 2)));
      process.exit(1);
    }

    const result = await response.json();

    // 7. Mostrar resultado
    console.log(chalk.green.bold("✅ Transcrição completa!\n"));

    console.log(chalk.cyan("=" .repeat(60)));
    console.log(chalk.cyan.bold("📝 TEXTO TRANSCRITO"));
    console.log(chalk.cyan("=" .repeat(60)));
    console.log(chalk.white(result.text));
    console.log(chalk.cyan("=" .repeat(60)));

    console.log();

    // Metadata
    console.log(chalk.yellow("📊 METADATA:"));
    console.log(chalk.gray(`  Idioma detectado: ${result.language}`));
    console.log(chalk.gray(`  Duração do áudio: ${result.duration.toFixed(1)}s`));
    console.log(chalk.gray(`  Segmentos: ${result.segments.length}`));
    console.log(chalk.gray(`  Backend: ${result.backend_info.model} (${result.backend_info.model_size})`));
    console.log(chalk.gray(`  Device: ${result.backend_info.device}`));

    // Metrics
    if (result.metrics) {
      console.log();
      console.log(chalk.yellow("⚡ PERFORMANCE:"));
      console.log(chalk.gray(`  Tempo de processamento: ${result.metrics.processing_time_sec.toFixed(2)}s`));
      console.log(chalk.gray(`  Real-Time Factor (RTF): ${result.metrics.real_time_factor.toFixed(3)}x`));
      console.log(chalk.gray(`  Tempo de carregamento: ${result.metrics.model_load_time_sec?.toFixed(2) || "N/A"}s`));
      console.log(chalk.gray(`  Tempo de transcrição: ${result.metrics.transcription_time_sec?.toFixed(2) || "N/A"}s`));
    }

    // Segmentos detalhados (primeiros 3)
    if (result.segments && result.segments.length > 0) {
      console.log();
      console.log(chalk.yellow("📄 SEGMENTOS (primeiros 3):"));

      const segmentsToShow = result.segments.slice(0, 3);

      for (const seg of segmentsToShow) {
        const timeRange = `[${seg.start.toFixed(1)}s - ${seg.end.toFixed(1)}s]`;
        console.log(chalk.gray(`  ${timeRange} ${seg.text}`));

        // Mostrar algumas palavras se existirem
        if (seg.words && seg.words.length > 0) {
          const wordsPreview = seg.words.slice(0, 5).map((w: any) =>
            `${w.word}(${w.probability?.toFixed(2) || "?"})`)
.join(" ");
          console.log(chalk.gray(`    Words: ${wordsPreview}${seg.words.length > 5 ? "..." : ""}`));
        }
      }

      if (result.segments.length > 3) {
        console.log(chalk.gray(`  ... e mais ${result.segments.length - 3} segmentos`));
      }
    }

    // Diarization (se habilitado)
    if (result.backend_info?.diarization_backend) {
      console.log();
      console.log(chalk.magenta("🎤 DIARIZATION (Speaker Identification):"));
      console.log();

      // Agrupar por speaker e mostrar timeline
      const speakers = new Set<string>();
      result.segments.forEach((seg: any) => {
        if (seg.speaker_id) speakers.add(seg.speaker_id);
      });

      console.log(chalk.magenta(`  Speakers detectados: ${speakers.size}`));
      console.log(chalk.gray(`  Backend: ${result.backend_info.diarization_backend}`));

      if (result.metrics?.diarization_time_sec) {
        console.log(chalk.gray(`  Tempo de diarization: ${result.metrics.diarization_time_sec.toFixed(2)}s`));
      }

      console.log();
      console.log(chalk.magenta("  Timeline:"));

      // Mostrar timeline de quem falou quando
      for (const seg of result.segments) {
        if (seg.speaker_id) {
          const timeRange = `[${seg.start.toFixed(1)}s - ${seg.end.toFixed(1)}s]`;
          const speakerColor = seg.speaker_id === "SPEAKER_00" ? chalk.cyan :
                               seg.speaker_id === "SPEAKER_01" ? chalk.green :
                               seg.speaker_id === "SPEAKER_02" ? chalk.yellow :
                               chalk.blue;

          console.log(speakerColor(`  ${timeRange} ${seg.speaker_id}: "${seg.text}"`));
        } else {
          const timeRange = `[${seg.start.toFixed(1)}s - ${seg.end.toFixed(1)}s]`;
          console.log(chalk.gray(`  ${timeRange} [unknown]: "${seg.text}"`));
        }
      }
    }

    console.log();
    console.log(chalk.green(`✅ Request completo em ${elapsedSec.toFixed(2)}s`));
    console.log();

    // Salvar resultado completo em arquivo JSON
    const timestamp = Date.now();
    const outputFile = `output-${timestamp}.json`;
    await Bun.write(outputFile, JSON.stringify(result, null, 2));
    console.log(chalk.gray(`💾 JSON salvo em: ${outputFile}`));

    // Salvar Markdown (se tiver segmentos)
    if (result.segments && result.segments.length > 0) {
      const mdFile = `output-${timestamp}.md`;
      const markdown = formatMarkdown(result, options);
      await Bun.write(mdFile, markdown);
      console.log(chalk.gray(`📄 Markdown salvo em: ${mdFile}`));

      // Preview do Markdown (primeiras linhas)
      const previewLines = markdown.split("\n").slice(0, 25);
      console.log();
      console.log(chalk.cyan("📖 Preview do Markdown:"));
      console.log(chalk.gray("─".repeat(60)));
      console.log(previewLines.join("\n"));
      if (markdown.split("\n").length > 25) {
        console.log(chalk.gray(`\n... (${markdown.split("\n").length - 25} linhas restantes no arquivo)`));
      }
      console.log(chalk.gray("─".repeat(60)));
    }

    console.log();

  } catch (error) {
    console.error(chalk.red("\n❌ Erro ao fazer request:"));
    console.error(chalk.red(String(error)));
    process.exit(1);
  }
}

// Parse CLI arguments
const args = process.argv.slice(2);
const options: TranscribeOptions = {
  file: "audios/audio-test.mp3", // Default
};

// Processar argumentos
for (let i = 0; i < args.length; i++) {
  const arg = args[i];
  const value = args[i + 1];

  switch (arg) {
    case "--file":
    case "-f":
      options.file = value;
      i++;
      break;
    case "--model":
    case "-m":
      options.model = value;
      i++;
      break;
    case "--model-size":
      options.modelSize = value;
      i++;
      break;
    case "--language":
    case "-l":
      options.language = value;
      i++;
      break;
    case "--task":
    case "-t":
      options.task = value as "transcribe" | "translate";
      i++;
      break;
    case "--format":
      options.responseFormat = value as any;
      i++;
      break;
    case "--no-metrics":
      options.returnMetrics = false;
      break;
    case "--diarization":
      options.enableDiarization = true;
      break;
    case "--diarization-backend":
      options.diarizationBackend = value;
      i++;
      break;
    case "--num-speakers":
      options.numSpeakers = parseInt(value);
      i++;
      break;
    case "--help":
    case "-h":
      console.log(`
Uso: bun test-batch-api.ts [opções]

Opções:
  -f, --file <path>              Caminho do arquivo de áudio (default: audios/audio-test.mp3)
  -m, --model <backend>          Backend (faster-whisper, mlx, whisperx, auto)
  --model-size <size>            Tamanho do modelo (tiny, base, small, medium, large, distil-large-v3)
  -l, --language <lang>          Idioma (pt, en, es, fr, etc)
  -t, --task <task>              Tarefa (transcribe, translate)
  --format <format>              Formato (json, verbose_json, text, srt, vtt)
  --no-metrics                   Não retornar métricas
  --diarization                  Habilitar diarization
  --diarization-backend <name>   Backend de diarization (pyannote, sortformer, auto)
  --num-speakers <n>             Número de speakers (para pyannote)
  -h, --help                     Mostrar ajuda

Exemplos:
  bun test-batch-api.ts
  bun test-batch-api.ts --file audios/outro.mp3
  bun test-batch-api.ts --model faster-whisper --language en
  bun test-batch-api.ts --diarization
  bun test-batch-api.ts --diarization --diarization-backend pyannote --num-speakers 2
      `);
      process.exit(0);
  }
}

// Executar teste
testBatchAPI(options);
