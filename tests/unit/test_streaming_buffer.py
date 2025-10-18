"""
Testes para StreamingBuffer e LocalAgreementPolicy
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

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

        # Update 3: confirma apenas "como" (prefixo estável entre últimas 2)
        # Histórico: ["olá mundo como", "olá mundo como você"]
        # Prefixo comum: "olá mundo como"
        should_confirm2, confirmed2 = policy.check("olá mundo como você")

        assert should_confirm2 is True
        assert confirmed2 == "como"  # Apenas o novo ("olá mundo como" - "olá mundo" = "como")

        # Update 4: confirma "você" agora
        # Histórico: ["olá mundo como você", "olá mundo como você está"]
        # Prefixo comum: "olá mundo como você"
        should_confirm3, confirmed3 = policy.check("olá mundo como você está")

        assert should_confirm3 is True
        assert confirmed3 == "você"  # Novo incremental

    def test_longest_common_prefix(self):
        """Test: encontrar prefixo comum corretamente"""
        policy = LocalAgreementPolicy(n=2)

        prefix = policy._longest_common_prefix(
            [
                "olá mundo como você",
                "olá mundo como você está",
                "olá mundo como você está hoje",
            ]
        )

        assert prefix == "olá mundo como você"

    def test_longest_common_prefix_empty(self):
        """Test: prefixo vazio quando sem concordância"""
        policy = LocalAgreementPolicy(n=2)

        prefix = policy._longest_common_prefix(["olá mundo", "bom dia"])

        assert prefix == ""

    def test_longest_common_prefix_empty_list(self):
        """Test: lista vazia retorna string vazia"""
        policy = LocalAgreementPolicy(n=2)

        prefix = policy._longest_common_prefix([])

        assert prefix == ""

    def test_reset(self):
        """Test: reset limpa histórico"""
        policy = LocalAgreementPolicy(n=2)

        policy.check("olá")
        policy.check("olá mundo")

        policy.reset()

        assert len(policy.history) == 0
        assert policy.last_confirmed == ""

    def test_history_size_limit(self):
        """Test: histórico mantém apenas n elementos"""
        policy = LocalAgreementPolicy(n=2)

        policy.check("primeiro")
        policy.check("segundo")
        policy.check("terceiro")

        # Deve ter apenas 2 elementos
        assert len(policy.history) == 2
        assert policy.history == ["segundo", "terceiro"]


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

    async def test_add_chunk_converts_dtype(self, buffer):
        """Test: add_chunk converte dtype para float32"""
        # Chunk int16
        chunk = np.random.randint(-32768, 32767, DEFAULT_SAMPLE_RATE, dtype=np.int16)

        await buffer.add_chunk(chunk)

        assert buffer.audio_buffer.dtype == np.float32

    async def test_add_chunk_normalizes_audio(self, buffer):
        """Test: add_chunk normaliza áudio > 1.0"""
        # Chunk com valores > 1.0
        chunk = np.random.randn(DEFAULT_SAMPLE_RATE).astype(np.float32) * 10

        await buffer.add_chunk(chunk)

        # Deve estar normalizado
        assert np.abs(buffer.audio_buffer).max() <= 1.0

    async def test_add_chunk_increments_counter(self, buffer):
        """Test: add_chunk incrementa contador"""
        chunk = np.random.randn(DEFAULT_SAMPLE_RATE).astype(np.float32)

        await buffer.add_chunk(chunk)
        await buffer.add_chunk(chunk)

        assert buffer.chunk_count == 2

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
            timestamp=datetime.now(),
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
            timestamp=datetime.now(),
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
            timestamp=datetime.now(),
            segments=[
                Segment(start=0.0, end=1.0, text="olá mundo"),
                Segment(start=1.0, end=2.0, text="como"),
            ],
        )
        result2 = await buffer.process()

        # Agora deve confirmar
        assert result2.is_final is True
        assert result2.text == "olá mundo"

    async def test_process_uses_context_window(self, buffer, mock_backend):
        """Test: context window é passado para backend"""
        chunk = np.random.randn(DEFAULT_SAMPLE_RATE).astype(np.float32)

        # Simular texto confirmado anterior
        buffer.confirmed_text = "contexto anterior importante"

        await buffer.add_chunk(chunk)
        mock_backend.transcribe_chunk.return_value = TranscriptionResult(
            text="novo texto",
            is_final=True,
            confidence=0.95,
            language="pt",
            timestamp=datetime.now(),
        )

        await buffer.process()

        # Verificar que context foi passado
        call_args = mock_backend.transcribe_chunk.call_args
        assert call_args.kwargs["context"] == "contexto anterior importante"

    async def test_process_empty_transcription(self, buffer, mock_backend):
        """Test: retornar None se transcrição vazia"""
        chunk = np.random.randn(DEFAULT_SAMPLE_RATE).astype(np.float32)
        await buffer.add_chunk(chunk)

        # Mock: transcrição vazia
        mock_backend.transcribe_chunk.return_value = TranscriptionResult(
            text="",
            is_final=True,
            confidence=0.0,
            language="pt",
            timestamp=datetime.now(),
        )

        result = await buffer.process()

        assert result is None

    async def test_buffer_trimming_segment(self, buffer, mock_backend):
        """Test: buffer é trimmed após confirmação (segment mode)"""
        # Adicionar chunk de 1s
        chunk = np.random.randn(DEFAULT_SAMPLE_RATE).astype(np.float32)
        await buffer.add_chunk(chunk)

        # Mock primeiro processo: "olá" (0.5s)
        mock_backend.transcribe_chunk.return_value = TranscriptionResult(
            text="olá",
            is_final=True,
            confidence=0.95,
            language="pt",
            timestamp=datetime.now(),
            segments=[Segment(start=0.0, end=0.5, text="olá")],
        )

        # Processar primeira vez (parcial, não confirma)
        result1 = await buffer.process()
        assert result1.is_final is False
        # Buffer ainda com 1s
        assert len(buffer.audio_buffer) == DEFAULT_SAMPLE_RATE

        # Adicionar mais 1s
        await buffer.add_chunk(chunk)
        # Buffer agora tem 2s

        # Mock segundo processo: "olá mundo" (confirma "olá" com 1.0s end)
        mock_backend.transcribe_chunk.return_value = TranscriptionResult(
            text="olá mundo",
            is_final=True,
            confidence=0.95,
            language="pt",
            timestamp=datetime.now(),
            segments=[
                Segment(start=0.0, end=1.0, text="olá mundo"),
            ],
        )
        result2 = await buffer.process()

        # Deve confirmar "olá" e trimmar buffer
        assert result2.is_final is True
        assert result2.text == "olá"

        # Buffer trimmed: removeu 1.0s, restou 1.0s
        expected_samples = DEFAULT_SAMPLE_RATE  # 1s
        assert len(buffer.audio_buffer) == pytest.approx(expected_samples, abs=100)

    async def test_force_trim_on_max_buffer(self, buffer):
        """Test: force trim quando buffer muito grande"""
        # Adicionar muito áudio
        large_chunk = np.random.randn(int(35 * DEFAULT_SAMPLE_RATE)).astype(np.float32)

        await buffer.add_chunk(large_chunk)

        # Buffer deve ter sido trimmed para 15s
        max_duration = 15.0
        expected_samples = int(max_duration * DEFAULT_SAMPLE_RATE)
        assert len(buffer.audio_buffer) == expected_samples

    async def test_get_context_window_limits_words(self, buffer):
        """Test: context window limita número de palavras"""
        # 150 palavras
        full_text = " ".join([f"palavra{i}" for i in range(150)])

        context = buffer._get_context_window(full_text, max_words=100)

        # Deve ter apenas 100 palavras
        assert len(context.split()) == 100

        # Deve ser as últimas 100
        assert context.startswith("palavra50")

    async def test_get_context_window_returns_full_if_small(self, buffer):
        """Test: context window retorna tudo se < max_words"""
        full_text = "apenas cinco palavras aqui total"

        context = buffer._get_context_window(full_text, max_words=100)

        assert context == full_text

    async def test_get_context_window_empty_text(self, buffer):
        """Test: context window com texto vazio"""
        context = buffer._get_context_window("", max_words=100)

        assert context == ""

    async def test_reset(self, buffer):
        """Test: reset limpa buffer"""
        chunk = np.random.randn(DEFAULT_SAMPLE_RATE).astype(np.float32)
        await buffer.add_chunk(chunk)
        buffer.confirmed_text = "texto confirmado"

        buffer.reset()

        assert len(buffer.audio_buffer) == 0
        assert buffer.confirmed_text == ""
        assert buffer.chunk_count == 0
        assert buffer.total_duration == 0.0

    async def test_get_stats(self, buffer):
        """Test: get_stats retorna estatísticas corretas"""
        chunk = np.random.randn(DEFAULT_SAMPLE_RATE).astype(np.float32)
        await buffer.add_chunk(chunk)
        buffer.confirmed_text = "texto teste"

        stats = buffer.get_stats()

        assert stats["buffer_duration_s"] == pytest.approx(1.0, abs=0.01)
        assert stats["chunks_received"] == 1
        assert stats["confirmed_text_len"] == len("texto teste")


class TestBufferConfig:
    """Testes para BufferConfig"""

    def test_default_values(self):
        """Test: valores padrão do config"""
        config = BufferConfig()

        assert config.min_chunk_size == 1.0
        assert config.buffer_trimming == "segment"
        assert config.agreement_threshold == 2
        assert config.max_buffer_size == 30.0

    def test_custom_values(self):
        """Test: valores customizados"""
        config = BufferConfig(
            min_chunk_size=2.0,
            buffer_trimming="sentence",
            agreement_threshold=3,
            max_buffer_size=60.0,
        )

        assert config.min_chunk_size == 2.0
        assert config.buffer_trimming == "sentence"
        assert config.agreement_threshold == 3
        assert config.max_buffer_size == 60.0
