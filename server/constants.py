"""
Constantes para Whisper Stream

Centraliza magic numbers e strings hardcoded para facilitar manutenção
e evitar erros de digitação.
"""

# ============================================
# Audio Processing
# ============================================

# PCM Audio
PCM_INT16_MAX = 32768.0
"""Valor máximo de PCM int16 para normalização"""

DEFAULT_SAMPLE_RATE = 16000
"""Taxa de amostragem padrão em Hz"""

DEFAULT_CHANNELS = 1
"""Número de canais de áudio (mono)"""

DEFAULT_CHUNK_SIZE_SECONDS = 1.0
"""Duração padrão de chunks de áudio em segundos"""

# ============================================
# WebSocket
# ============================================

WS_MAX_MESSAGE_SIZE = 10 * 1024 * 1024
"""Tamanho máximo de mensagem WebSocket (10MB)"""

WS_PING_INTERVAL = 30
"""Intervalo entre pings em segundos"""

WS_PING_TIMEOUT = 10
"""Timeout para resposta de ping em segundos"""

# ============================================
# Rate Limiting
# ============================================

DEFAULT_RATE_LIMIT_REQUESTS = 100
"""Número máximo de requisições por janela de tempo"""

DEFAULT_RATE_LIMIT_WINDOW = 60
"""Janela de tempo para rate limiting em segundos"""

# ============================================
# Retry Logic
# ============================================

DEFAULT_MAX_RETRIES = 3
"""Número máximo de tentativas em caso de erro"""

DEFAULT_RETRY_DELAY = 1.0
"""Delay entre tentativas em segundos"""

DEFAULT_RETRY_BACKOFF = 2.0
"""Fator de backoff exponencial"""

# ============================================
# Timeouts
# ============================================

DEFAULT_INITIALIZATION_TIMEOUT = 60
"""Timeout para inicialização de backend em segundos"""

DEFAULT_TRANSCRIPTION_TIMEOUT = 30
"""Timeout para transcrição de chunk em segundos"""

DEFAULT_HEALTH_CHECK_TIMEOUT = 5
"""Timeout para health check em segundos"""

# ============================================
# Cache & Context
# ============================================

DEFAULT_CONTEXT_LENGTH = 500
"""Comprimento máximo de contexto em caracteres"""

DEFAULT_CACHE_SIZE = 100
"""Tamanho máximo do cache (itens)"""

DEFAULT_CACHE_TTL = 3600
"""Time-to-live do cache em segundos (1 hora)"""

# ============================================
# Transcription
# ============================================

MIN_AUDIO_DURATION = 0.1
"""Duração mínima de áudio para processar (segundos)"""

MAX_AUDIO_DURATION = 30.0
"""Duração máxima de áudio por chunk (segundos)"""

MIN_TEXT_LENGTH = 2
"""Comprimento mínimo de texto para considerar válido"""

DEFAULT_CONFIDENCE_THRESHOLD = 0.5
"""Threshold padrão de confiança para aceitar transcrição"""

# ============================================
# Server
# ============================================

DEFAULT_SERVER_HOST = "0.0.0.0"
"""Host padrão do servidor"""

DEFAULT_SERVER_PORT = 9090
"""Porta padrão do servidor"""

DEFAULT_MAX_CLIENTS = 5
"""Número máximo de clientes conectados simultaneamente"""

DEFAULT_IDLE_TIMEOUT = 300
"""Timeout de inatividade em segundos (5 minutos)"""

# ============================================
# Logging
# ============================================

LOG_MAX_MESSAGE_LENGTH = 200
"""Comprimento máximo de mensagem em log"""

LOG_ROTATION_SIZE = 10 * 1024 * 1024
"""Tamanho máximo de arquivo de log antes de rotação (10MB)"""

LOG_ROTATION_COUNT = 5
"""Número de arquivos de log a manter"""

# ============================================
# Performance
# ============================================

DEFAULT_BATCH_SIZE = 1
"""Tamanho padrão de batch para processamento"""

DEFAULT_THREADS = 4
"""Número padrão de threads (quando não auto)"""

# ============================================
# Model Paths
# ============================================

DEFAULT_MODELS_DIR = "./models"
"""Diretório padrão para modelos baixados"""

DEFAULT_CACHE_DIR = "~/.cache/whisper-stream"
"""Diretório padrão para cache"""

DEFAULT_LOGS_DIR = "./logs"
"""Diretório padrão para logs"""
