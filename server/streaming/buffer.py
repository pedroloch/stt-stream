"""
Streaming Buffer com LocalAgreement Policy

Implementação CORRETA baseada em whisper_streaming (UFAL):
https://github.com/ufal/whisper_streaming/blob/main/whisper_online.py

Mudanças principais da implementação anterior:
- Usa HypothesisBuffer com word-level timestamps
- Mantém buffer_time_offset para timestamps absolutos
- Context window otimizado (apenas texto scrolled away)
- Trimming conservador (buffer > threshold, não imediato)
- Detecta e pula segmentos de silêncio (no_speech_prob)
"""

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from ..backends.base import WhisperBackend
from ..constants import DEFAULT_SAMPLE_RATE
from ..models.result import TranscriptionResult, Word
from .hypothesis_buffer import HypothesisBuffer

logger = logging.getLogger(__name__)


@dataclass
class BufferConfig:
    """Configuração do buffer de streaming"""

    min_chunk_size: float = 1.0  # segundos (tamanho mínimo para processar)
    buffer_trimming: str = "segment"  # "segment" ou "sentence"
    buffer_trimming_sec: float = 10.0  # Trim quando buffer > 10s (balanço performance/accuracy)
    agreement_threshold: int = 2  # Não usado mais (HypothesisBuffer sempre n=2)
    max_buffer_size: float = 20.0  # segundos (limite hard de memória - força trim se exceder)
    # Detecção de pausa (fim de frase)
    pause_detection_enabled: bool = True  # Detectar pausas longas
    pause_threshold_sec: float = 2.5  # Pausa > 2.5s = possível fim de frase
    auto_punctuate_on_pause: bool = True  # Adicionar '.' se sem pontuação


class StreamingBuffer:
    """
    Buffer inteligente para streaming de áudio com LocalAgreement

    Acumula chunks, re-transcreve com overlap, usa HypothesisBuffer
    para confirmar apenas texto estável palavra por palavra.

    Diferença chave vs. implementação anterior:
    - Word-level LocalAgreement (não string-based)
    - buffer_time_offset tracking (timestamps absolutos)
    - Context window correto (apenas scrolled away)
    - Trimming conservador (não imediato)
    """

    def __init__(self, backend: WhisperBackend, config: BufferConfig | None = None):
        """
        Args:
            backend: Backend de transcrição (faster-whisper, MLX, etc)
            config: Configuração do buffer
        """
        self.backend = backend
        self.config = config or BufferConfig()

        # Buffer de áudio
        self.audio_buffer = np.array([], dtype=np.float32)

        # ⭐ NOVO: HypothesisBuffer com word-level timestamps
        self.hypothesis = HypothesisBuffer()

        # ⭐ NOVO: Offset temporal do buffer (tempo absoluto)
        # Quando fazemos trim, incrementamos este valor
        # Exemplo: buffer começa em 0s, trim 5s → buffer_time_offset = 5s
        self.buffer_time_offset: float = 0.0

        # Estado
        self.total_duration = 0.0  # segundos de áudio processado (scrolled away)
        self.chunk_count = 0

        # Lista de palavras confirmadas (para prompt/context)
        # Formato: lista de Word objects
        self.commited_words: list[Word] = []

        # ⭐ Detecção de pausa: tracking temporal
        self.last_audio_time: float = 0.0  # Timestamp do último chunk recebido
        self.pause_detected_flag: bool = False  # Flag de pausa detectada

        # ⭐ NOVO: Stall detection - tracking de último commit
        self.time_of_last_commit: float = 0.0  # Timestamp absoluto do último commit

        logger.info(
            f"StreamingBuffer inicializado: "
            f"min_chunk={self.config.min_chunk_size}s, "
            f"buffer_trimming={self.config.buffer_trimming}, "
            f"trim_threshold={self.config.buffer_trimming_sec}s, "
            f"pause_detection={'ON' if self.config.pause_detection_enabled else 'OFF'}"
        )

    async def add_chunk(self, audio_chunk: np.ndarray):
        """
        Adiciona chunk de áudio ao buffer

        Args:
            audio_chunk: Array numpy (float32, mono, 16kHz)
        """
        # ⭐ DETECÇÃO DE PAUSA: Verificar gap temporal
        import time
        current_time = time.time()

        if self.config.pause_detection_enabled and self.last_audio_time > 0:
            gap = current_time - self.last_audio_time

            # Pausa longa detectada?
            if gap > self.config.pause_threshold_sec and len(self.audio_buffer) > 0:
                logger.info(
                    f"⏸️  Pausa detectada: {gap:.2f}s (threshold: {self.config.pause_threshold_sec}s)"
                )
                self.pause_detected_flag = True

        self.last_audio_time = current_time

        # Validar formato
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # Normalizar se necessário
        if np.abs(audio_chunk).max() > 1.0:
            audio_chunk = audio_chunk / np.abs(audio_chunk).max()

        # Adicionar ao buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
        self.chunk_count += 1

        logger.debug(
            f"Chunk #{self.chunk_count} adicionado: "
            f"{len(audio_chunk)/DEFAULT_SAMPLE_RATE:.2f}s, "
            f"buffer total: {len(self.audio_buffer)/DEFAULT_SAMPLE_RATE:.2f}s, "
            f"buffer_offset: {self.buffer_time_offset:.2f}s"
        )

        # Verificar limite de buffer (segurança)
        buffer_duration = len(self.audio_buffer) / DEFAULT_SAMPLE_RATE
        if buffer_duration > self.config.max_buffer_size:
            logger.warning(
                f"Buffer muito grande ({buffer_duration:.1f}s > {self.config.max_buffer_size}s), "
                f"forçando trim"
            )
            await self._force_trim()

    def _get_prompt(self) -> str:
        """
        Retorna prompt otimizado para Whisper

        CRÍTICO: Retorna apenas texto que foi REMOVIDO do buffer (scrolled away).
        NÃO passa texto que ainda está no buffer!

        Isso evita desalinhamento entre contexto e áudio.

        Returns:
            Últimas 200 chars do texto scrolled away
        """
        # Encontrar palavras que estão FORA do buffer atual
        # (palavras confirmadas com end_time <= buffer_time_offset)
        scrolled_away_words = [
            word for word in self.commited_words
            if word.end <= self.buffer_time_offset
        ]

        if not scrolled_away_words:
            # Nenhum texto scrolled away ainda
            # Usar prompt padrão do idioma (se configurado)
            if self.backend.language and self.backend.language != "auto":
                language_prompts = {
                    "pt": "Olá, como vai? Este é um texto em português do Brasil.",
                    "en": "Hello, how are you? This is a text in English.",
                    "es": "Hola, ¿cómo estás? Este es un texto en español.",
                    "fr": "Bonjour, comment allez-vous? Ceci est un texte en français.",
                }
                return language_prompts.get(self.backend.language, "")
            return ""

        # Concatenar texto (words já incluem espaços, então concatenar sem espaço adicional)
        full_text = "".join(w.word for w in scrolled_away_words).strip()

        # Últimas 200 chars apenas (limite do Whisper)
        prompt = full_text[-200:] if len(full_text) > 200 else full_text

        logger.debug(
            f"Prompt ({len(prompt)} chars): '{prompt[:50]}...' "
            f"({len(scrolled_away_words)} palavras scrolled away)"
        )

        return prompt

    async def process(self) -> TranscriptionResult | None:
        """
        Processa buffer atual e retorna transcrição (parcial ou final)

        Fluxo:
        1. Verifica se buffer >= min_chunk_size
        2. Transcreve buffer completo com prompt correto
        3. Extrai palavras com timestamps
        4. HypothesisBuffer.insert() + flush()
        5. Se confirmou palavras → retorna FINAL (apenas texto novo)
        6. Senão → retorna PARCIAL (preview)
        7. Trim se buffer > threshold

        Returns:
            TranscriptionResult se tiver resultado, None se buffer muito pequeno
        """
        # Verificar se temos áudio suficiente
        duration = len(self.audio_buffer) / DEFAULT_SAMPLE_RATE
        if duration < self.config.min_chunk_size:
            logger.debug(
                f"Buffer pequeno ({duration:.2f}s < {self.config.min_chunk_size}s), aguardando"
            )
            return None

        # Obter prompt (apenas texto scrolled away!)
        prompt = self._get_prompt()

        # Transcrever buffer completo
        logger.info(
            f"📊 Transcrevendo buffer: {duration:.2f}s "
            f"(offset: {self.buffer_time_offset:.2f}s, "
            f"confirmadas: {len(self.commited_words)} palavras)"
        )

        result = await self.backend.transcribe_chunk(
            self.audio_buffer,
            context=prompt,  # ⭐ Apenas texto scrolled away
        )

        # Se vazio, retornar None
        if not result.text:
            logger.debug("Transcrição vazia")
            return None

        # ⭐ Extrair palavras com timestamps
        if not result.words:
            logger.warning("Backend não retornou words! LocalAgreement não funcionará corretamente.")
            # Fallback: retornar transcrição sem LocalAgreement
            return result

        logger.debug(f"Extraídas {len(result.words)} palavras com timestamps")

        # ⭐ HypothesisBuffer: insert + flush (passa Word objects diretamente!)
        self.hypothesis.insert(result.words, self.buffer_time_offset)
        confirmed_words = self.hypothesis.flush()

        # ⭐ STALL DETECTION: Detectar se buffer travou (sem commits por muito tempo)
        current_buffer_end_time = self.buffer_time_offset + (len(self.audio_buffer) / DEFAULT_SAMPLE_RATE)
        buffer_duration = len(self.audio_buffer) / DEFAULT_SAMPLE_RATE

        if not confirmed_words and buffer_duration > self.config.buffer_trimming_sec:
            # Não houve commits E buffer está grande
            time_since_last_commit = current_buffer_end_time - self.time_of_last_commit

            if time_since_last_commit > self.config.buffer_trimming_sec:
                # STALL DETECTADO! Sem commits por muito tempo
                logger.warning(
                    f"⚠️ STALL DETECTADO: {time_since_last_commit:.1f}s sem commits "
                    f"(buffer: {buffer_duration:.1f}s). Resetando buffer."
                )

                # Reset completo do buffer
                removed_duration = len(self.audio_buffer) / DEFAULT_SAMPLE_RATE
                self.buffer_time_offset += removed_duration
                self.total_duration += removed_duration
                self.audio_buffer = np.array([], dtype=np.float32)

                # Reset do HypothesisBuffer
                self.hypothesis.reset()

                # Atualizar tempo de último commit (para evitar reset loops)
                self.time_of_last_commit = current_buffer_end_time

                logger.info(f"Buffer resetado: novo offset={self.buffer_time_offset:.2f}s")

                # Retornar None (sem resultado neste ciclo)
                return None

        # ⭐ DETECÇÃO DE PAUSA: Verificar se devemos forçar confirmação
        is_sentence_end = False
        if self.pause_detected_flag:
            # Pausa foi detectada - verificar se é fim de frase
            is_sentence_end = self._check_sentence_end(result.text)

            if is_sentence_end:
                logger.info(f"🔚 Fim de frase detectado após pausa ('{result.text[-20:] if len(result.text) > 20 else result.text}')")

                # Se não confirmou palavras pelo LocalAgreement, forçar confirmação de TUDO
                if not confirmed_words:
                    logger.info("Forçando confirmação de todo buffer devido à pausa")
                    # Usar todas as palavras como confirmadas
                    confirmed_words = result.words

                # Auto-punctuate se configurado E sem pontuação final
                if self.config.auto_punctuate_on_pause:
                    last_word = confirmed_words[-1] if confirmed_words else None
                    if last_word and not last_word.word.rstrip().endswith((".", "!", "?", "...")):
                        # Criar nova palavra com pontuação
                        from ..models.result import Word
                        confirmed_words[-1] = Word(
                            word=last_word.word.rstrip() + ".",
                            start=last_word.start,
                            end=last_word.end,
                            probability=last_word.probability,
                            is_filler=last_word.is_filler,
                            is_punctuation=last_word.is_punctuation,
                            speaker_id=last_word.speaker_id,
                            language=last_word.language,
                        )
                        logger.debug("Adicionado '.' ao final do texto")
            else:
                logger.info(f"Pausa detectada mas texto continua (reticências ou vírgula)")

            # Limpar flag
            self.pause_detected_flag = False

        if confirmed_words:
            # ✅ TEXTO CONFIRMADO!
            # NOTE: Whisper words já incluem espaços (ex: ' olá', ' tudo')
            # Por isso concatenamos sem espaço adicional e depois strip()
            confirmed_text = "".join(w.word for w in confirmed_words).strip()

            logger.info(
                f"✅ Confirmado ({len(confirmed_words)} palavras): '{confirmed_text}'"
            )
            logger.debug(f"   Palavras: {[w.word for w in confirmed_words]}")

            # Adicionar ao histórico de palavras confirmadas
            self.commited_words.extend(confirmed_words)

            # ⭐ Atualizar tempo de último commit (para stall detection)
            self.time_of_last_commit = confirmed_words[-1].end

            # ⭐ Se for fim de frase (pausa), limpar buffer completamente
            if is_sentence_end:
                logger.info("Limpando buffer devido a fim de frase")
                # Limpar áudio
                removed_duration = len(self.audio_buffer) / DEFAULT_SAMPLE_RATE
                self.buffer_time_offset += removed_duration
                self.total_duration += removed_duration
                self.audio_buffer = np.array([], dtype=np.float32)
                # Limpar HypothesisBuffer
                self.hypothesis.reset()
                logger.debug(f"Buffer limpo: offset={self.buffer_time_offset:.2f}s")
            else:
                # ⭐ Trim conservador normal: apenas se buffer > threshold
                buffer_dur = len(self.audio_buffer) / DEFAULT_SAMPLE_RATE
                if buffer_dur > self.config.buffer_trimming_sec:
                    logger.debug(
                        f"Buffer grande ({buffer_dur:.2f}s > {self.config.buffer_trimming_sec}s), "
                        f"fazendo trim"
                    )
                    await self._trim_buffer_conservative()

            # Retornar resultado FINAL (apenas texto novo confirmado!)
            return TranscriptionResult(
                text=confirmed_text,
                is_final=True,
                confidence=result.confidence,
                language=result.language,
                timestamp=datetime.now(),
                segments=result.segments,  # Manter segmentos originais
                words=result.words,
                is_sentence_end=is_sentence_end,  # ⭐ Flag de fim de frase
            )
        else:
            # ⏳ TEXTO PARCIAL (ainda não estável)
            logger.debug(f"⏳ Parcial: '{result.text[:50]}...'")

            return TranscriptionResult(
                text=result.text,
                is_final=False,
                confidence=result.confidence,
                language=result.language,
                timestamp=datetime.now(),
                segments=result.segments,
                words=result.words,
            )

    def _check_sentence_end(self, text: str) -> bool:
        """
        Verifica se texto termina com pontuação de fim de frase

        Usado para detecção inteligente de pausa:
        - '.' ou '!' ou '?' → FIM de frase
        - '...' ou ',' → NÃO é fim (continua)
        - Sem pontuação → Presumir FIM

        Args:
            text: Texto a verificar

        Returns:
            True se for fim de frase, False se continuar
        """
        text = text.strip()
        if not text:
            return False

        # Verificar pontuação final
        if text.endswith("..."):
            # Reticências = NÃO é fim (pensamento contínuo)
            logger.debug(f"Texto termina com '...' → NÃO é fim de frase")
            return False

        if text.endswith(","):
            # Vírgula = NÃO é fim
            logger.debug(f"Texto termina com ',' → NÃO é fim de frase")
            return False

        if text.endswith((".", "!", "?")):
            # Pontuação final = FIM de frase
            logger.debug(f"Texto termina com '{text[-1]}' → É fim de frase")
            return True

        # Sem pontuação = Presumir FIM (vamos adicionar '.')
        logger.debug(f"Texto sem pontuação final → Presumir fim de frase")
        return True

    async def _trim_buffer_conservative(self):
        """
        Trim conservador do buffer

        Remove áudio confirmado até o timestamp da última palavra confirmada,
        mas mantém overlap (não remove tudo imediatamente).

        Estratégia:
        - Encontrar timestamp da última palavra confirmada
        - Trim até esse ponto (mas buffer_time_offset será atualizado)
        - Remover palavras confirmadas que foram scrolled away
        """
        if not self.commited_words:
            logger.debug("Sem palavras confirmadas, não é possível trim")
            return

        # Última palavra confirmada
        last_confirmed_word = self.commited_words[-1]
        trim_time = last_confirmed_word.end  # end time

        # Trim time RELATIVO ao buffer atual
        trim_time_relative = trim_time - self.buffer_time_offset

        if trim_time_relative <= 0:
            logger.debug("Trim time inválido, pulando trim")
            return

        # Converter para samples
        trim_samples = int(trim_time_relative * DEFAULT_SAMPLE_RATE)

        if trim_samples >= len(self.audio_buffer):
            # Trim completo (todo o buffer)
            removed_duration = len(self.audio_buffer) / DEFAULT_SAMPLE_RATE
            self.buffer_time_offset += removed_duration
            self.total_duration += removed_duration
            self.audio_buffer = np.array([], dtype=np.float32)

            logger.info(
                f"Buffer completamente trimmed: "
                f"removido {removed_duration:.2f}s, "
                f"buffer_offset agora: {self.buffer_time_offset:.2f}s"
            )
        else:
            # Trim parcial
            removed_duration = trim_samples / DEFAULT_SAMPLE_RATE
            self.audio_buffer = self.audio_buffer[trim_samples:]
            self.buffer_time_offset += removed_duration
            self.total_duration += removed_duration

            logger.info(
                f"Buffer trimmed: "
                f"removido {removed_duration:.2f}s, "
                f"restante {len(self.audio_buffer)/DEFAULT_SAMPLE_RATE:.2f}s, "
                f"buffer_offset agora: {self.buffer_time_offset:.2f}s"
            )

        # Remover palavras scrolled away do HypothesisBuffer
        self.hypothesis.pop_commited(self.buffer_time_offset)

    async def _force_trim(self):
        """
        Forçar trim quando buffer muito grande (evitar OOM)

        Remove tudo exceto últimos 15s.
        """
        max_samples = int(15.0 * DEFAULT_SAMPLE_RATE)
        if len(self.audio_buffer) > max_samples:
            removed_samples = len(self.audio_buffer) - max_samples
            removed_duration = removed_samples / DEFAULT_SAMPLE_RATE

            self.audio_buffer = self.audio_buffer[-max_samples:]
            self.buffer_time_offset += removed_duration
            self.total_duration += removed_duration

            logger.warning(
                f"Buffer forçadamente trimmed: "
                f"removido {removed_duration:.2f}s, "
                f"buffer_offset agora: {self.buffer_time_offset:.2f}s"
            )

            # Remover palavras scrolled away
            self.hypothesis.pop_commited(self.buffer_time_offset)

    def reset(self):
        """Reset para nova sessão"""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.hypothesis.reset()
        self.commited_words.clear()
        self.buffer_time_offset = 0.0
        self.total_duration = 0.0
        self.chunk_count = 0
        # Reset detecção de pausa
        self.last_audio_time = 0.0
        self.pause_detected_flag = False
        logger.info("StreamingBuffer resetado")

    def get_stats(self) -> dict:
        """Estatísticas do buffer"""
        return {
            "buffer_duration_s": len(self.audio_buffer) / DEFAULT_SAMPLE_RATE,
            "buffer_time_offset_s": self.buffer_time_offset,
            "total_processed_s": self.total_duration,
            "chunks_received": self.chunk_count,
            "commited_words_count": len(self.commited_words),
            "hypothesis_stats": self.hypothesis.get_stats(),
        }


# DEPRECATED: Mantido por compatibilidade, mas não usado mais
class LocalAgreementPolicy:
    """
    DEPRECATED: Use HypothesisBuffer diretamente

    Mantido apenas para compatibilidade com imports existentes.
    """

    def __init__(self, n: int = 2):
        logger.warning(
            "LocalAgreementPolicy está deprecated. "
            "StreamingBuffer agora usa HypothesisBuffer diretamente."
        )
        self.n = n
