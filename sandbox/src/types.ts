/**
 * Tipos TypeScript para Whisper Stream Client
 */

// ============================================
// WebSocket Messages
// ============================================

/**
 * Mensagem base do WebSocket
 */
export interface WebSocketMessage {
  type: string;
}

/**
 * Mensagem de conexão bem-sucedida
 */
export interface ConnectedMessage extends WebSocketMessage {
  type: "connected";
  message: string;
  session_id?: string;
  server_version?: string;
}

/**
 * Mensagem de transcrição
 */
export interface TranscriptionMessage extends WebSocketMessage {
  type: "transcription";
  text: string;
  is_final: boolean;
  language: string;
  confidence: number;
  timestamp: string;
  segments?: TranscriptionSegment[];
  is_sentence_end?: boolean; // ⭐ Flag de fim de frase (detecção de pausa)
}

/**
 * Segmento de transcrição com timecodes
 */
export interface TranscriptionSegment {
  start: number;
  end: number;
  text: string;
}

/**
 * Mensagem de erro
 */
export interface ErrorMessage extends WebSocketMessage {
  type: "error";
  message: string;
  code?: string;
}

/**
 * Mensagem de pong (resposta ao ping)
 */
export interface PongMessage extends WebSocketMessage {
  type: "pong";
  timestamp?: string;
}

/**
 * Union type de todas as mensagens possíveis
 */
export type AnyWebSocketMessage =
  | ConnectedMessage
  | TranscriptionMessage
  | ErrorMessage
  | PongMessage;

// ============================================
// Configuration
// ============================================

/**
 * Configuração do servidor
 */
export interface ServerConfig {
  url: string;
  health_url: string;
}

/**
 * Configuração de áudio
 */
export interface AudioConfig {
  sample_rate: number;
  channels: number;
}

/**
 * Configuração de cores do display
 */
export interface DisplayColors {
  partial: string;
  final: string;
  error: string;
  info: string;
  debug?: string;
  header?: string;
}

/**
 * Modo de visualização de parciais
 */
export type PartialMode = "inline" | "minimal" | "full";

/**
 * Configuração de display
 */
export interface DisplayConfig {
  partial_mode?: PartialMode;  // Modo de visualização (inline, minimal, full)
  show_partial?: boolean;       // Deprecated: use partial_mode
  show_timestamps: boolean;
  colors: DisplayColors;
  max_history?: number;
  word_wrap_width?: number;
}

/**
 * Configuração de debug
 */
export interface DebugConfig {
  enabled: boolean;
  show_json: boolean;
  show_audio_stats: boolean;
}

/**
 * Configuração completa do cliente
 */
export interface Config {
  server: ServerConfig;
  audio: AudioConfig;
  display: DisplayConfig;
  debug?: DebugConfig;
}

// ============================================
// Application State
// ============================================

/**
 * Estado da conexão (substitui enum por const as const)
 */
export const ConnectionStatus = {
  Disconnected: "disconnected",
  Connecting: "connecting",
  Connected: "connected",
  Reconnecting: "reconnecting",
  Error: "error",
} as const;

export type ConnectionStatus = typeof ConnectionStatus[keyof typeof ConnectionStatus];

/**
 * Estado do áudio (substitui enum por const as const)
 */
export const AudioStatus = {
  Inactive: "inactive",
  Starting: "starting",
  Active: "active",
  Error: "error",
} as const;

export type AudioStatus = typeof AudioStatus[keyof typeof AudioStatus];

/**
 * Estado da aplicação
 */
export interface AppState {
  connectionStatus: ConnectionStatus;
  audioStatus: AudioStatus;
  reconnectAttempts: number;
  lastError?: string;
  transcriptionCount: number;
  startTime: Date;
}

/**
 * Histórico de mensagens para display
 */
export interface MessageHistoryItem {
  timestamp: Date;
  text: string;
  is_final: boolean;
  confidence?: number;
}
