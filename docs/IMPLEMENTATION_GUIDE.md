# 🔧 Guia de Implementação Técnica - Whisper Stream

**Data**: 2025-10-18
**Versão**: 1.0
**Objetivo**: Guia detalhado de implementação de cada componente

---

## 🎯 FASE 0: Streaming Inteligente (CRÍTICO)

Esta é a **fundação** de tudo. Sem streaming inteligente, os outros backends não fazem diferença.

### **0.1. StreamingBuffer com LocalAgreement**

#### **Arquitetura**

```
┌─────────────────────────────────────────────────┐
│              WebSocket Handler                  │
│  (recebe chunks de áudio do cliente)           │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│            StreamingBuffer                      │
│                                                 │
│  ┌──────────────────────────────────────────┐  │
│  │  Audio Buffer (numpy array)              │  │
│  │  [chunk1, chunk2, chunk3, ...]          │  │
│  └──────────────────────────────────────────┘  │
│                   │                             │
│                   ▼                             │
│  ┌──────────────────────────────────────────┐  │
│  │  LocalAgreementPolicy                    │  │
│  │  - History: [trans1, trans2, trans3]    │  │
│  │  - Check if n=2 consecutive agree       │  │
│  └──────────────────────────────────────────┘  │
│                   │                             │
│                   ▼                             │
│  ┌──────────────────────────────────────────┐  │
│  │  Buffer Trimming                         │  │
│  │  - Remove confirmed audio                │  │
│  │  - Strategy: segment or sentence        │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
                   │
                   ▼
         TranscriptionResult
         (partial or final)
```

#### **Implementação**

**Arquivo**: `server/streaming/buffer.py`

```python
"""
Streaming Buffer com LocalAgreement Policy

Inspirado no whisper_streaming (UFAL) mas adaptado para nossa arquitetura.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
import numpy as np
import logging

from ..models.result import TranscriptionResult, Segment
from ..backends.base import WhisperBackend
from ..constants import DEFAULT_SAMPLE_RATE

logger = logging.getLogger(__name__)


@dataclass
class BufferConfig:
    """Configuração do buffer de streaming"""

    min_chunk_size: float = 1.0  # segundos
    buffer_trimming: str = "segment"  # "segment" ou "sentence"
    agreement_threshold: int = 2  # número de concordâncias necessárias
    max_buffer_size: float = 30.0  # segundos (limite de memória)


class LocalAgreementPolicy:
    """
    LocalAgreement-n Policy

    Confirma transcrição quando n updates consecutivos concordam no prefixo.

    Exemplo (n=2):
        Update 1: "olá mundo como você"
        Update 2: "olá mundo como você está"
        → Confirma: "olá mundo como você" (prefixo comum)

        Update 3: "olá mundo como você está hoje"
        → Confirma: "olá mundo como você está" (novo prefixo)
    """

    def __init__(self, n: int = 2):
        """
        Args:
            n: Número de concordâncias consecutivas necessárias
        """
        self.n = n
        self.history: List[str] = []
        self.last_confirmed: str = ""

    def check(self, new_transcript: str) -> tuple[bool, Optional[str]]:
        """
        Verifica se deve confirmar transcrição

        Args:
            new_transcript: Nova transcrição do buffer atual

        Returns:
            (should_confirm, confirmed_text)
            - should_confirm: True se deve confirmar
            - confirmed_text: Texto confirmado (apenas o novo, não repetir)
        """
        # Adicionar ao histórico
        self.history.append(new_transcript)

        # Manter apenas últimas n transcrições
        if len(self.history) > self.n:
            self.history.pop(0)

        # Precisa de n transcrições para confirmar
        if len(self.history) < self.n:
            logger.debug(f"LocalAgreement: {len(self.history)}/{self.n} histórico")
            return False, None

        # Extrair prefixo comum
        common_prefix = self._longest_common_prefix(self.history)

        # Se prefixo vazio, não confirmar
        if not common_prefix:
            logger.debug("LocalAgreement: sem prefixo comum")
            return False, None

        # Remover texto já confirmado anteriormente
        new_confirmed = common_prefix[len(self.last_confirmed):].strip()

        # Se não há texto novo, não confirmar
        if not new_confirmed:
            logger.debug("LocalAgreement: sem texto novo")
            return False, None

        # Confirmar!
        self.last_confirmed = common_prefix
        logger.info(f"LocalAgreement: confirmado '{new_confirmed}'")
        return True, new_confirmed

    def _longest_common_prefix(self, transcripts: List[str]) -> str:
        """
        Encontra maior prefixo em comum entre transcrições

        Estratégia: comparar palavra por palavra

        Example:
            ["olá mundo como você", "olá mundo como você está"]
            → "olá mundo como você"
        """
        if not transcripts:
            return ""

        # Dividir em palavras
        words_lists = [t.split() for t in transcripts]

        # Encontrar menor lista
        min_len = min(len(wl) for wl in words_lists)

        common_words = []
        for i in range(min_len):
            word = words_lists[0][i]

            # Verificar se todas concordam nessa palavra
            if all(wl[i] == word for wl in words_lists):
                common_words.append(word)
            else:
                # Primeira divergência, parar
                break

        return " ".join(common_words)

    def reset(self):
        """Reset para nova sessão"""
        self.history.clear()
        self.last_confirmed = ""


class StreamingBuffer:
    """
    Buffer inteligente para streaming de áudio

    Acumula chunks, processa com re-transcrição, e usa LocalAgreement
    para confirmar texto estável.
    """

    def __init__(
        self,
        backend: WhisperBackend,
        config: Optional[BufferConfig] = None
    ):
        """
        Args:
            backend: Backend de transcrição (faster-whisper, MLX, etc)
            config: Configuração do buffer
        """
        self.backend = backend
        self.config = config or BufferConfig()

        # Buffer de áudio
        self.audio_buffer = np.array([], dtype=np.float32)

        # LocalAgreement policy
        self.agreement = LocalAgreementPolicy(n=self.config.agreement_threshold)

        # Estado
        self.confirmed_text = ""
        self.total_duration = 0.0  # segundos de áudio processado
        self.chunk_count = 0

        logger.info(
            f"StreamingBuffer criado: "
            f"min_chunk={self.config.min_chunk_size}s, "
            f"agreement_n={self.config.agreement_threshold}, "
            f"trimming={self.config.buffer_trimming}"
        )

    async def add_chunk(self, audio_chunk: np.ndarray):
        """
        Adiciona chunk de áudio ao buffer

        Args:
            audio_chunk: Array numpy (float32, mono, 16kHz)
        """
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
            f"buffer total: {len(self.audio_buffer)/DEFAULT_SAMPLE_RATE:.2f}s"
        )

        # Verificar limite de buffer
        buffer_duration = len(self.audio_buffer) / DEFAULT_SAMPLE_RATE
        if buffer_duration > self.config.max_buffer_size:
            logger.warning(
                f"Buffer muito grande ({buffer_duration:.1f}s), "
                f"forçando trim"
            )
            await self._force_trim()

    def _get_context_window(self, full_text: str, max_words: int = 100) -> str:
        """
        Retorna últimas N palavras para contexto

        TÉCNICA CRÍTICA: Passar texto confirmado como initial_prompt melhora:
        - Coerência (nomes próprios, termos técnicos)
        - Accuracy (-15% WER segundo whisper_streaming)
        - Capitalização e formatação

        Whisper tem limite de ~224 tokens no initial_prompt.
        Passar muito texto pode degradar performance.

        Args:
            full_text: Texto completo confirmado
            max_words: Máximo de palavras no contexto (default: 100)

        Returns:
            Últimas max_words palavras
        """
        if not full_text:
            return ""

        words = full_text.split()

        if len(words) <= max_words:
            return full_text

        # Retornar últimas N palavras
        context = " ".join(words[-max_words:])

        logger.debug(
            f"Context window: {len(words)} total → {max_words} palavras usadas"
        )

        return context

    async def process(self) -> Optional[TranscriptionResult]:
        """
        Processa buffer atual e retorna transcrição (parcial ou final)

        Returns:
            TranscriptionResult se tiver resultado, None se buffer muito pequeno
        """
        # Verificar se temos áudio suficiente
        duration = len(self.audio_buffer) / DEFAULT_SAMPLE_RATE
        if duration < self.config.min_chunk_size:
            logger.debug(f"Buffer pequeno ({duration:.2f}s < {self.config.min_chunk_size}s), aguardando")
            return None

        # Obter contexto otimizado (últimas N palavras apenas)
        # IMPORTANTE: Não passar TODO o texto - degrada performance!
        context = self._get_context_window(self.confirmed_text, max_words=100)

        # Transcrever buffer completo com contexto
        logger.debug(f"Transcrevendo buffer: {duration:.2f}s")
        logger.debug(f"Context: '{context[:80]}...' ({len(context.split())} palavras)")

        result = await self.backend.transcribe_chunk(
            self.audio_buffer,
            context=context  # ⭐ Context otimizado (últimas 100 palavras)
        )

        # Se vazio, retornar None
        if not result.text:
            logger.debug("Transcrição vazia")
            return None

        # LocalAgreement policy
        should_confirm, confirmed = self.agreement.check(result.text)

        if should_confirm and confirmed:
            # Texto confirmado!
            logger.info(f"✅ Confirmado: '{confirmed}'")

            # Atualizar contexto
            self.confirmed_text += " " + confirmed
            self.confirmed_text = self.confirmed_text.strip()

            # Trim buffer (remover áudio confirmado)
            await self._trim_buffer(result)

            # Retornar resultado FINAL
            return TranscriptionResult(
                text=confirmed,
                is_final=True,
                confidence=result.confidence,
                language=result.language,
                timestamp=datetime.now(),
                segments=result.segments,  # Manter segmentos originais
            )
        else:
            # Texto ainda não estável, retornar PARCIAL
            logger.debug(f"⏳ Parcial: '{result.text}'")

            return TranscriptionResult(
                text=result.text,
                is_final=False,
                confidence=result.confidence,
                language=result.language,
                timestamp=datetime.now(),
                segments=result.segments,
            )

    async def _trim_buffer(self, result: TranscriptionResult):
        """
        Remove áudio confirmado do buffer

        Estratégias:
        - "segment": corta no timestamp do último segmento
        - "sentence": corta no fim de frase (mais conservador)
        """
        if self.config.buffer_trimming == "segment":
            await self._trim_buffer_segment(result)
        elif self.config.buffer_trimming == "sentence":
            await self._trim_buffer_sentence(result)
        else:
            logger.warning(f"Trimming strategy '{self.config.buffer_trimming}' desconhecida")

    async def _trim_buffer_segment(self, result: TranscriptionResult):
        """
        Corta buffer no timestamp do último segmento confirmado
        """
        if not result.segments:
            logger.debug("Sem segmentos, não é possível trim")
            return

        # Último segmento
        last_segment = result.segments[-1]
        trim_time = last_segment.end

        # Converter para samples
        trim_samples = int(trim_time * DEFAULT_SAMPLE_RATE)

        # Trim (manter apenas áudio após trim_time)
        if trim_samples < len(self.audio_buffer):
            removed_duration = trim_samples / DEFAULT_SAMPLE_RATE
            self.audio_buffer = self.audio_buffer[trim_samples:]
            self.total_duration += removed_duration

            logger.info(
                f"Buffer trimmed: removido {removed_duration:.2f}s, "
                f"restante {len(self.audio_buffer)/DEFAULT_SAMPLE_RATE:.2f}s"
            )
        else:
            # Trim completo
            self.total_duration += len(self.audio_buffer) / DEFAULT_SAMPLE_RATE
            self.audio_buffer = np.array([], dtype=np.float32)
            logger.info("Buffer completamente trimmed")

    async def _trim_buffer_sentence(self, result: TranscriptionResult):
        """
        Corta buffer no fim de frase (ponto, interrogação, exclamação)

        Mais conservador: espera frase completa antes de trim.
        Latência maior mas menos risco de cortar mid-sentence.
        """
        confirmed = self.agreement.last_confirmed

        # Encontrar último caractere de pontuação
        last_punct_idx = max(
            confirmed.rfind('.'),
            confirmed.rfind('?'),
            confirmed.rfind('!'),
        )

        if last_punct_idx < 0:
            logger.debug("Sem pontuação, não é possível trim conservador")
            return

        # Texto até pontuação
        confirmed_until_punct = confirmed[:last_punct_idx + 1]

        # Estimar timestamp (aproximado via proporção)
        # TODO: melhorar com word-level timestamps
        proportion = len(confirmed_until_punct) / len(result.text) if result.text else 0
        buffer_duration = len(self.audio_buffer) / DEFAULT_SAMPLE_RATE
        trim_time = buffer_duration * proportion

        # Trim
        trim_samples = int(trim_time * DEFAULT_SAMPLE_RATE)
        if trim_samples < len(self.audio_buffer):
            self.audio_buffer = self.audio_buffer[trim_samples:]
            logger.info(f"Buffer trimmed (sentence): removido {trim_time:.2f}s")

    async def _force_trim(self):
        """
        Forçar trim quando buffer muito grande (evitar OOM)
        """
        # Manter apenas últimos 15s
        max_samples = int(15.0 * DEFAULT_SAMPLE_RATE)
        if len(self.audio_buffer) > max_samples:
            removed = len(self.audio_buffer) - max_samples
            self.audio_buffer = self.audio_buffer[-max_samples:]
            logger.warning(
                f"Buffer forçadamente trimmed: "
                f"removido {removed/DEFAULT_SAMPLE_RATE:.2f}s"
            )

    def reset(self):
        """Reset para nova sessão"""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.agreement.reset()
        self.confirmed_text = ""
        self.total_duration = 0.0
        self.chunk_count = 0
        logger.info("StreamingBuffer reset")

    def get_stats(self) -> dict:
        """Estatísticas do buffer"""
        return {
            "buffer_duration_s": len(self.audio_buffer) / DEFAULT_SAMPLE_RATE,
            "total_processed_s": self.total_duration,
            "chunks_received": self.chunk_count,
            "confirmed_text_len": len(self.confirmed_text),
        }
```

#### **Testes**

**Arquivo**: `tests/unit/test_streaming_buffer.py`

```python
"""
Testes para StreamingBuffer e LocalAgreementPolicy
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock

from server.streaming.buffer import (
    StreamingBuffer,
    LocalAgreementPolicy,
    BufferConfig,
)
from server.models.result import TranscriptionResult, Segment
from server.constants import DEFAULT_SAMPLE_RATE


class TestLocalAgreementPolicy:
    """Testes para LocalAgreementPolicy"""

    def test_agreement_threshold_not_reached(self):
        """Test: não confirmar se não atingiu n concordâncias"""
        policy = LocalAgreementPolicy(n=2)

        # Primeiro update
        should_confirm, confirmed = policy.check("olá mundo")

        assert should_confirm is False
        assert confirmed is None

    def test_agreement_threshold_reached(self):
        """Test: confirmar quando n concordâncias atingidas"""
        policy = LocalAgreementPolicy(n=2)

        # Update 1
        policy.check("olá mundo")

        # Update 2 (concorda com prefixo)
        should_confirm, confirmed = policy.check("olá mundo como")

        assert should_confirm is True
        assert confirmed == "olá mundo"

    def test_agreement_no_common_prefix(self):
        """Test: não confirmar se não há prefixo comum"""
        policy = LocalAgreementPolicy(n=2)

        policy.check("olá mundo")
        should_confirm, confirmed = policy.check("bom dia")

        assert should_confirm is False
        assert confirmed is None

    def test_agreement_incremental_confirmation(self):
        """Test: confirmação incremental (não repetir texto)"""
        policy = LocalAgreementPolicy(n=2)

        # Update 1-2: confirma "olá mundo"
        policy.check("olá mundo")
        should_confirm1, confirmed1 = policy.check("olá mundo como")

        assert confirmed1 == "olá mundo"

        # Update 3: confirma "como você" (incremental)
        should_confirm2, confirmed2 = policy.check("olá mundo como você")

        assert should_confirm2 is True
        assert confirmed2 == "como você"  # Apenas o novo

    def test_longest_common_prefix(self):
        """Test: encontrar prefixo comum corretamente"""
        policy = LocalAgreementPolicy(n=2)

        prefix = policy._longest_common_prefix([
            "olá mundo como você",
            "olá mundo como você está",
            "olá mundo como você está hoje"
        ])

        assert prefix == "olá mundo como você"

    def test_reset(self):
        """Test: reset limpa histórico"""
        policy = LocalAgreementPolicy(n=2)

        policy.check("olá")
        policy.check("olá mundo")

        policy.reset()

        assert len(policy.history) == 0
        assert policy.last_confirmed == ""


@pytest.mark.asyncio
class TestStreamingBuffer:
    """Testes para StreamingBuffer"""

    @pytest.fixture
    def mock_backend(self):
        """Mock de backend de transcrição"""
        backend = AsyncMock()
        backend.transcribe_chunk = AsyncMock()
        return backend

    @pytest.fixture
    def buffer(self, mock_backend):
        """Fixture de StreamingBuffer"""
        config = BufferConfig(
            min_chunk_size=1.0,
            agreement_threshold=2,
        )
        return StreamingBuffer(backend=mock_backend, config=config)

    async def test_add_chunk_accumulates_audio(self, buffer):
        """Test: add_chunk acumula áudio no buffer"""
        # Chunk de 1 segundo
        chunk = np.random.randn(DEFAULT_SAMPLE_RATE).astype(np.float32)

        await buffer.add_chunk(chunk)

        assert len(buffer.audio_buffer) == DEFAULT_SAMPLE_RATE

    async def test_process_buffer_too_small(self, buffer, mock_backend):
        """Test: não processar se buffer muito pequeno"""
        # Chunk de 0.5s (menor que min_chunk_size=1.0)
        chunk = np.random.randn(DEFAULT_SAMPLE_RATE // 2).astype(np.float32)

        await buffer.add_chunk(chunk)
        result = await buffer.process()

        # Não deve transcrever
        assert result is None
        mock_backend.transcribe_chunk.assert_not_called()

    async def test_process_returns_partial(self, buffer, mock_backend):
        """Test: retornar transcrição parcial quando não estável"""
        # Chunk de 1 segundo
        chunk = np.random.randn(DEFAULT_SAMPLE_RATE).astype(np.float32)
        await buffer.add_chunk(chunk)

        # Mock: backend retorna transcrição
        mock_backend.transcribe_chunk.return_value = TranscriptionResult(
            text="olá mundo",
            is_final=True,
            confidence=0.95,
            language="pt",
            timestamp=None,
        )

        # Primeiro processo (não atinge threshold)
        result = await buffer.process()

        assert result is not None
        assert result.is_final is False  # Parcial
        assert result.text == "olá mundo"

    async def test_process_returns_final_on_agreement(self, buffer, mock_backend):
        """Test: retornar final quando LocalAgreement confirmado"""
        # Chunk de 1 segundo
        chunk = np.random.randn(DEFAULT_SAMPLE_RATE).astype(np.float32)

        # Update 1
        await buffer.add_chunk(chunk)
        mock_backend.transcribe_chunk.return_value = TranscriptionResult(
            text="olá mundo",
            is_final=True,
            confidence=0.95,
            language="pt",
            timestamp=None,
            segments=[Segment(start=0.0, end=1.0, text="olá mundo")],
        )
        result1 = await buffer.process()
        assert result1.is_final is False  # Parcial

        # Update 2 (concorda)
        await buffer.add_chunk(chunk)
        mock_backend.transcribe_chunk.return_value = TranscriptionResult(
            text="olá mundo como",
            is_final=True,
            confidence=0.95,
            language="pt",
            timestamp=None,
            segments=[
                Segment(start=0.0, end=1.0, text="olá mundo"),
                Segment(start=1.0, end=2.0, text="como"),
            ],
        )
        result2 = await buffer.process()

        # Agora deve confirmar
        assert result2.is_final is True
        assert result2.text == "olá mundo"

    async def test_buffer_trimming(self, buffer, mock_backend):
        """Test: buffer é trimmed após confirmação"""
        # Adicionar chunk
        chunk = np.random.randn(DEFAULT_SAMPLE_RATE).astype(np.float32)
        await buffer.add_chunk(chunk)

        # Mock com segmento
        mock_backend.transcribe_chunk.return_value = TranscriptionResult(
            text="olá",
            is_final=True,
            confidence=0.95,
            language="pt",
            timestamp=None,
            segments=[Segment(start=0.0, end=0.5, text="olá")],
        )

        # Processar duas vezes (confirmar)
        await buffer.process()
        await buffer.add_chunk(chunk)
        await buffer.process()

        # Buffer deve ter sido trimmed (removeu 0.5s)
        expected_samples = DEFAULT_SAMPLE_RATE * 2 - int(0.5 * DEFAULT_SAMPLE_RATE)
        assert len(buffer.audio_buffer) == pytest.approx(expected_samples, abs=100)

    async def test_reset(self, buffer):
        """Test: reset limpa buffer"""
        chunk = np.random.randn(DEFAULT_SAMPLE_RATE).astype(np.float32)
        await buffer.add_chunk(chunk)

        buffer.reset()

        assert len(buffer.audio_buffer) == 0
        assert buffer.confirmed_text == ""
```

---

### **0.2. VAD-based Chunking**

#### **Conceito**

Cortar chunks em **pausas naturais de fala**, não em intervalos fixos.

**Problema sem VAD**:
```
"Olá mun | do como v | ocê está?"
          ↑          ↑
     Corte ruim  Corte ruim
```

**Com VAD**:
```
"Olá mundo" [pausa] "como você está?"
           ↑
     Corte natural
```

#### **Implementação**

**Arquivo**: `server/streaming/vad.py`

```python
"""
Voice Activity Detection para chunking inteligente
"""

import numpy as np
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class VADChunker:
    """
    Chunker baseado em VAD (Voice Activity Detection)

    Usa Silero VAD para detectar pausas e cortar chunks naturalmente.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_silence_duration: float = 0.3,  # segundos
        min_speech_duration: float = 0.5,  # segundos
    ):
        """
        Args:
            threshold: Threshold de confiança VAD (0-1)
            min_silence_duration: Mínimo de silêncio para considerar pausa
            min_speech_duration: Mínimo de fala para considerar speech
        """
        self.threshold = threshold
        self.min_silence_duration = min_silence_duration
        self.min_speech_duration = min_speech_duration

        # Lazy load do modelo
        self._model = None

    def _load_model(self):
        """Load Silero VAD model"""
        if self._model is None:
            try:
                import torch
                # Silero VAD model
                self._model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False,
                )
                logger.info("Silero VAD model loaded")
            except Exception as e:
                logger.error(f"Failed to load VAD model: {e}")
                raise

    def detect_speech(self, audio: np.ndarray, sample_rate: int = 16000) -> List[Tuple[float, float]]:
        """
        Detecta segmentos de fala no áudio

        Args:
            audio: Array numpy (float32, mono)
            sample_rate: Sample rate

        Returns:
            Lista de tuplas (start_time, end_time) em segundos
        """
        self._load_model()

        import torch

        # Converter para torch tensor
        audio_tensor = torch.from_numpy(audio)

        # Get speech timestamps
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            self._model,
            threshold=self.threshold,
            sampling_rate=sample_rate,
            min_silence_duration_ms=int(self.min_silence_duration * 1000),
            min_speech_duration_ms=int(self.min_speech_duration * 1000),
        )

        # Converter para segundos
        segments = [
            (ts['start'] / sample_rate, ts['end'] / sample_rate)
            for ts in speech_timestamps
        ]

        logger.debug(f"VAD detected {len(segments)} speech segments")
        return segments

    def find_best_split_point(
        self,
        audio: np.ndarray,
        target_duration: float = 5.0,
        sample_rate: int = 16000
    ) -> int:
        """
        Encontra melhor ponto de corte (em samples) baseado em pausas

        Args:
            audio: Áudio completo
            target_duration: Duração alvo do chunk (segundos)
            sample_rate: Sample rate

        Returns:
            Index (em samples) do melhor ponto de corte
        """
        # Detectar segmentos de fala
        segments = self.detect_speech(audio, sample_rate)

        if not segments:
            # Sem fala detectada, retornar target padrão
            return int(target_duration * sample_rate)

        # Converter target para samples
        target_samples = int(target_duration * sample_rate)

        # Encontrar segmento mais próximo do target
        best_split = target_samples
        min_diff = float('inf')

        for start, end in segments:
            end_samples = int(end * sample_rate)

            # Se este segmento termina perto do target
            diff = abs(end_samples - target_samples)
            if diff < min_diff and end_samples <= len(audio):
                min_diff = diff
                best_split = end_samples

        logger.debug(
            f"VAD best split: {best_split/sample_rate:.2f}s "
            f"(target: {target_duration:.2f}s)"
        )

        return best_split


# Helper function (do Silero VAD)
def get_speech_timestamps(
    audio: "torch.Tensor",
    model,
    threshold: float = 0.5,
    sampling_rate: int = 16000,
    min_silence_duration_ms: int = 300,
    min_speech_duration_ms: int = 500,
):
    """
    Get speech timestamps from audio

    (Implementação do Silero VAD - simplificada)
    """
    # Validar áudio
    if len(audio) < sampling_rate * 0.1:  # < 100ms
        return []

    # Get speech probabilities
    speech_probs = []
    window_size = int(sampling_rate * 0.03)  # 30ms windows

    for i in range(0, len(audio), window_size):
        chunk = audio[i:i + window_size]
        if len(chunk) < window_size:
            continue

        # Get probability
        with torch.no_grad():
            prob = model(chunk.unsqueeze(0), sampling_rate).item()
        speech_probs.append((i, prob))

    # Find speech segments
    timestamps = []
    is_speech = False
    speech_start = 0

    for i, (sample_idx, prob) in enumerate(speech_probs):
        if prob > threshold and not is_speech:
            # Speech start
            speech_start = sample_idx
            is_speech = True
        elif prob <= threshold and is_speech:
            # Speech end
            speech_end = sample_idx
            is_speech = False

            # Check duration
            duration_ms = (speech_end - speech_start) / sampling_rate * 1000
            if duration_ms >= min_speech_duration_ms:
                timestamps.append({
                    'start': speech_start,
                    'end': speech_end,
                })

    # Last segment
    if is_speech:
        timestamps.append({
            'start': speech_start,
            'end': len(audio),
        })

    return timestamps
```

---

### **0.3. Integração no WebSocket Handler**

**Arquivo**: `server/websocket_handler.py` (modificações)

```python
# Adicionar no início
from .streaming.buffer import StreamingBuffer, BufferConfig

class WebSocketHandler:
    def __init__(self, processor: WhisperProcessor):
        self.processor = processor
        self.streaming_buffers = {}  # {ws_id: StreamingBuffer}

    async def handle_websocket(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        # ID único para este cliente
        ws_id = id(ws)

        # Criar StreamingBuffer para este cliente
        backend = self.processor.backend  # Assumindo que já existe
        buffer_config = BufferConfig(
            min_chunk_size=1.0,
            agreement_threshold=2,
            buffer_trimming="segment",
        )
        streaming_buffer = StreamingBuffer(backend, buffer_config)
        self.streaming_buffers[ws_id] = streaming_buffer

        try:
            # Enviar mensagem de conexão
            await self._send_connected(ws)

            # Loop de processamento
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    # Áudio recebido
                    audio_data = msg.data

                    # Converter para float32
                    audio_np = self.audio_converter.pcm_int16_to_float32(audio_data)

                    # Adicionar ao buffer
                    await streaming_buffer.add_chunk(audio_np)

                    # Processar buffer
                    result = await streaming_buffer.process()

                    if result:
                        # Enviar transcrição (parcial ou final)
                        await self._send_transcription(ws, result)

        finally:
            # Cleanup
            if ws_id in self.streaming_buffers:
                self.streaming_buffers[ws_id].reset()
                del self.streaming_buffers[ws_id]

        return ws
```

---

## 📝 Resumo de Arquivos a Criar

### **FASE 0**

```
server/
├── streaming/                    # ⭐ NOVO
│   ├── __init__.py
│   ├── buffer.py                # StreamingBuffer + LocalAgreement
│   └── vad.py                   # VAD chunking
│
tests/
└── unit/
    ├── test_streaming_buffer.py # ⭐ NOVO
    ├── test_vad_chunking.py     # ⭐ NOVO
    └── test_local_agreement.py  # ⭐ NOVO (pode ser parte de test_streaming_buffer.py)
```

---

## 🎯 Checklist de Implementação FASE 0

### **Semana 1**

- [ ] Criar `server/streaming/__init__.py`
- [ ] Implementar `LocalAgreementPolicy` em `buffer.py`
- [ ] Testes unitários de `LocalAgreementPolicy`
- [ ] Implementar `StreamingBuffer` (sem VAD primeiro)
- [ ] Testes de `StreamingBuffer`
- [ ] Integrar no `WebSocketHandler` (básico)

### **Semana 2**

- [ ] Implementar `VADChunker` em `vad.py`
- [ ] Testes de VAD
- [ ] Integrar VAD no `StreamingBuffer` (opcional)
- [ ] Benchmarking: latência end-to-end
- [ ] Documentar em `docs/STREAMING.md`
- [ ] Code review e ajustes

### **Acceptance Criteria**

✅ **Funcional**:
- LocalAgreement funcionando (n=2)
- Buffer trimming implementado
- Transcrições parciais + finais retornadas
- VAD integrado (ou pelo menos opcional)

✅ **Testes**:
- Coverage > 80% em `streaming/`
- Testes passando

✅ **Performance**:
- Latência < 2s (primeira palavra)
- Sem cortes mid-word (ou mínimo possível)

✅ **Documentação**:
- `docs/STREAMING.md` completo
- Docstrings em todas funções
- README atualizado

---

## ⭐ Técnicas Críticas Implementadas

### **1. Context Window (Initial Prompt)**

**O QUE É**: Passar últimas N palavras confirmadas como `initial_prompt` para próxima transcrição

**POR QUE FUNCIONA**:
- Whisper é autoregressivo (como GPT)
- Contexto melhora coerência e accuracy
- Nomes próprios e termos técnicos ficam consistentes
- **Impacto**: -15% WER, +31% accuracy em nomes próprios (segundo paper)

**IMPLEMENTAÇÃO**:
```python
# ❌ ERRADO: passar TODO o texto
context = self.confirmed_text  # Pode ter 1000+ palavras!

# ✅ CORRETO: últimas N palavras apenas
context = self._get_context_window(self.confirmed_text, max_words=100)
```

**TUNNING**:
- **50 palavras**: fala casual
- **100 palavras**: recomendado (sweet spot) ⭐
- **200 palavras**: termos técnicos/médicos
- **500+**: degrada performance ❌

**EXEMPLO**:
```
Chunk 1: "Olá, meu nome é Maria Silva" → confirmado
Chunk 2: [áudio: "trabalho na empresa X"]
  + context: "Olá, meu nome é Maria Silva"
  → "Trabalho na empresa X" ✅ (mantém capitalização)

SEM context:
  → "trabalho na empresa x" ❌
```

### **2. Distil-Whisper em PT**

**IMPORTANTE**: Distil-Whisper **FUNCIONA EM PT** mesmo sendo treinado só em inglês!

**POR QUE**: Herda pesos do Whisper Large-v3 (multilingual)

**USO**:
```yaml
# Recomendado para streaming rápido:
whisper:
  backend: faster-whisper
  model: distil-large-v3  # ✅ 6x mais rápido, funciona em PT!
  language: pt
```

**TRADE-OFF**:
- WER: ~10-12% (vs. ~8% Large-v3) ⚠️ um pouco pior
- Velocidade: **6x mais rápido** ✅
- **Vale a pena** quando velocidade > accuracy absoluta

## 🔧 Dicas de Implementação

### **1. Começar Simples**

Não implementar tudo de uma vez:
1. LocalAgreement primeiro (sem VAD)
2. Context window (crítico!)
3. Testar com faster-whisper
4. Adicionar VAD depois

### **2. Debugging**

Adicionar logs detalhados:
```python
logger.info(f"✅ Confirmado: '{confirmed}'")
logger.debug(f"⏳ Parcial: '{result.text}'")
logger.debug(f"Buffer: {duration:.2f}s, histórico: {len(self.history)}")
```

### **3. Tunning de Parâmetros**

Experimentar:
- `agreement_threshold`: 2 vs. 3
- `min_chunk_size`: 0.5s vs. 1.0s vs. 2.0s
- `buffer_trimming`: segment vs. sentence

### **4. Testes com Áudio Real**

Usar áudios de teste variados:
- Fala lenta vs. rápida
- Com pausas vs. sem pausas
- Ruído vs. limpo
- Multi-speaker

---

## 📚 Referências

- **whisper_streaming (UFAL)**: https://github.com/ufal/whisper_streaming
- **Paper**: "Turning Whisper into Real-Time Transcription System"
- **Silero VAD**: https://github.com/snakers4/silero-vad

---

**Próximo**: Após FASE 0 completa, seguir para FASE 1 (Validação e Testes)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
