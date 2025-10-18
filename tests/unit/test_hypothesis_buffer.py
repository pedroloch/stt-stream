"""
Testes unitários para HypothesisBuffer

Valida a implementação do LocalAgreement-n com word-level timestamps.
"""

import pytest

from server.streaming.hypothesis_buffer import HypothesisBuffer, WordWithTimestamp


class TestHypothesisBuffer:
    """Testes para HypothesisBuffer"""

    def test_init(self):
        """Testa inicialização"""
        buffer = HypothesisBuffer()

        assert buffer.commited_in_buffer == []
        assert buffer.buffer == []
        assert buffer.new == []
        assert buffer.last_commited_time == 0.0
        assert buffer.last_commited_word is None

    def test_insert_filters_old_words(self):
        """Testa que insert filtra palavras antes de last_commited_time"""
        buffer = HypothesisBuffer()
        buffer.last_commited_time = 2.0

        # Palavras: algumas antes de 2.0s, algumas depois
        words = [
            (0.5, 1.0, "old"),
            (1.5, 1.8, "old2"),
            (2.1, 2.5, "new"),  # ← Apenas esta deve passar
            (2.6, 3.0, "new2"),  # ← E esta
        ]

        buffer.insert(words, offset=0.0)

        # Apenas palavras DEPOIS de 2.0s (com tolerância de 0.1s)
        assert len(buffer.new) == 2
        assert buffer.new[0][2] == "new"
        assert buffer.new[1][2] == "new2"

    def test_insert_with_offset(self):
        """Testa que insert ajusta timestamps com offset"""
        buffer = HypothesisBuffer()

        # Palavras relativas (como vêm do backend)
        words = [
            (0.0, 0.5, "hello"),
            (0.6, 1.0, "world"),
        ]

        # Offset de 5.0s (buffer já processou 5s antes)
        buffer.insert(words, offset=5.0)

        # Timestamps devem ser absolutos
        assert buffer.new[0][0] == 5.0  # start
        assert buffer.new[0][1] == 5.5  # end
        assert buffer.new[1][0] == 5.6
        assert buffer.new[1][1] == 6.0

    def test_flush_no_agreement(self):
        """Testa flush sem concordância (primeira transcrição)"""
        buffer = HypothesisBuffer()

        words = [(0.0, 0.5, "hello"), (0.6, 1.0, "world")]
        buffer.insert(words, offset=0.0)

        # Primeira vez: não há buffer anterior para comparar
        commit = buffer.flush()

        assert commit == []  # Nada confirmado
        assert buffer.buffer == words  # Nova transcrição vira buffer
        assert buffer.new == []

    def test_flush_with_full_agreement(self):
        """Testa flush com concordância total"""
        buffer = HypothesisBuffer()

        # Primeira transcrição
        words1 = [(0.0, 0.5, "hello"), (0.6, 1.0, "world")]
        buffer.insert(words1, offset=0.0)
        buffer.flush()

        # Segunda transcrição: IGUAL
        words2 = [(0.0, 0.5, "hello"), (0.6, 1.0, "world")]
        buffer.insert(words2, offset=0.0)
        commit = buffer.flush()

        # Deve confirmar TODAS as palavras
        assert len(commit) == 2
        assert commit[0][2] == "hello"
        assert commit[1][2] == "world"

        # last_commited_time deve ser end da última palavra
        assert buffer.last_commited_time == 1.0

    def test_flush_with_partial_agreement(self):
        """Testa flush com concordância parcial (caso típico)"""
        buffer = HypothesisBuffer()

        # Primeira transcrição
        words1 = [(0.0, 0.5, "hello"), (0.6, 1.0, "world")]
        buffer.insert(words1, offset=0.0)
        buffer.flush()

        # Segunda transcrição: começa igual mas estende
        words2 = [
            (0.0, 0.5, "hello"),
            (0.6, 1.0, "world"),
            (1.1, 1.5, "how"),  # ← Nova palavra
        ]
        buffer.insert(words2, offset=0.0)
        commit = buffer.flush()

        # Deve confirmar prefixo comum: ["hello", "world"]
        assert len(commit) == 2
        assert commit[0][2] == "hello"
        assert commit[1][2] == "world"

        # Buffer agora contém apenas a nova palavra
        assert len(buffer.buffer) == 1
        assert buffer.buffer[0][2] == "how"

    def test_flush_with_divergence(self):
        """Testa flush quando transcrições divergem"""
        buffer = HypothesisBuffer()

        # Primeira transcrição
        words1 = [(0.0, 0.5, "hello"), (0.6, 1.0, "world")]
        buffer.insert(words1, offset=0.0)
        buffer.flush()

        # Segunda transcrição: começa diferente
        words2 = [(0.0, 0.5, "hi"), (0.6, 1.0, "there")]  # ← Tudo diferente
        buffer.insert(words2, offset=0.0)
        commit = buffer.flush()

        # Nenhuma concordância
        assert commit == []

        # Buffer agora contém a nova transcrição
        assert len(buffer.buffer) == 2
        assert buffer.buffer[0][2] == "hi"

    def test_pop_commited(self):
        """Testa remoção de palavras scrolled away"""
        buffer = HypothesisBuffer()

        # Simular palavras confirmadas
        buffer.commited_in_buffer = [
            (0.0, 0.5, "old1"),
            (0.6, 1.0, "old2"),
            (1.1, 1.5, "recent"),
            (1.6, 2.0, "new"),
        ]

        # Remover palavras até 1.0s (scrolled away após trim)
        buffer.pop_commited(1.0)

        # Apenas palavras com end > 1.0s devem permanecer
        assert len(buffer.commited_in_buffer) == 2
        assert buffer.commited_in_buffer[0][2] == "recent"
        assert buffer.commited_in_buffer[1][2] == "new"

    def test_remove_duplicates_1gram(self):
        """Testa remoção de duplicatas (1-gram)"""
        buffer = HypothesisBuffer()

        # Simular palavras confirmadas
        buffer.commited_in_buffer = [
            (0.0, 0.5, "hello"),
            (0.6, 1.0, "world"),
        ]

        # Nova transcrição com "world" duplicada
        buffer.new = [
            (0.6, 1.0, "world"),  # ← Duplicata (última palavra confirmada)
            (1.1, 1.5, "how"),
        ]

        buffer._remove_duplicates()

        # "world" deve ter sido removida
        assert len(buffer.new) == 1
        assert buffer.new[0][2] == "how"

    def test_remove_duplicates_2gram(self):
        """Testa remoção de duplicatas (2-gram)"""
        buffer = HypothesisBuffer()

        # Simular palavras confirmadas
        buffer.commited_in_buffer = [
            (0.0, 0.5, "hello"),
            (0.6, 1.0, "world"),
        ]

        # Nova transcrição com "hello world" duplicado
        buffer.new = [
            (0.0, 0.5, "hello"),  # ← Duplicata
            (0.6, 1.0, "world"),  # ← Duplicata
            (1.1, 1.5, "how"),
        ]

        buffer._remove_duplicates()

        # "hello world" devem ter sido removidos (2-gram match)
        assert len(buffer.new) == 1
        assert buffer.new[0][2] == "how"

    def test_reset(self):
        """Testa reset completo"""
        buffer = HypothesisBuffer()

        # Preencher com dados
        buffer.commited_in_buffer = [(0.0, 0.5, "test")]
        buffer.buffer = [(0.6, 1.0, "test2")]
        buffer.last_commited_time = 1.0
        buffer.last_commited_word = "test"

        buffer.reset()

        # Tudo deve estar limpo
        assert buffer.commited_in_buffer == []
        assert buffer.buffer == []
        assert buffer.new == []
        assert buffer.last_commited_time == 0.0
        assert buffer.last_commited_word is None

    def test_complete(self):
        """Testa retorno do buffer atual (incomplete)"""
        buffer = HypothesisBuffer()

        buffer.buffer = [
            (0.0, 0.5, "hello"),
            (0.6, 1.0, "world"),
        ]

        incomplete = buffer.complete()

        # Deve retornar cópia do buffer
        assert len(incomplete) == 2
        assert incomplete[0][2] == "hello"

        # Não deve modificar buffer original
        assert buffer.buffer == incomplete

    def test_get_stats(self):
        """Testa estatísticas"""
        buffer = HypothesisBuffer()

        buffer.commited_in_buffer = [(0.0, 0.5, "test")]
        buffer.buffer = [(0.6, 1.0, "test2")]
        buffer.last_commited_time = 0.5
        buffer.last_commited_word = "test"

        stats = buffer.get_stats()

        assert stats["commited_count"] == 1
        assert stats["buffer_count"] == 1
        assert stats["new_count"] == 0
        assert stats["last_commited_time"] == 0.5
        assert stats["last_commited_word"] == "test"

    def test_realistic_streaming_scenario(self):
        """Testa cenário realista de streaming"""
        buffer = HypothesisBuffer()

        # Update 1: "olá mundo"
        words1 = [(0.0, 0.5, "olá"), (0.6, 1.0, "mundo")]
        buffer.insert(words1, offset=0.0)
        commit1 = buffer.flush()
        assert commit1 == []  # Primeira vez, não confirma

        # Update 2: "olá mundo como você"
        words2 = [
            (0.0, 0.5, "olá"),
            (0.6, 1.0, "mundo"),
            (1.1, 1.5, "como"),
            (1.6, 2.0, "você"),
        ]
        buffer.insert(words2, offset=0.0)
        commit2 = buffer.flush()

        # Deve confirmar prefixo comum: "olá mundo"
        assert len(commit2) == 2
        assert commit2[0][2] == "olá"
        assert commit2[1][2] == "mundo"
        assert buffer.last_commited_time == 1.0

        # Update 3: "olá mundo como você está"
        # (mas buffer já trimmed até 1.0s, então offset = 1.0)
        words3 = [
            (0.0, 0.5, "como"),  # Relativo ao buffer
            (0.6, 1.0, "você"),
            (1.1, 1.5, "está"),
        ]
        buffer.insert(words3, offset=1.0)
        commit3 = buffer.flush()

        # Deve confirmar "como você"
        assert len(commit3) == 2
        assert commit3[0][2] == "como"
        assert commit3[1][2] == "você"
        assert buffer.last_commited_time == 2.0

        # Total confirmado: "olá mundo como você"
        all_commited = commit2 + commit3
        text = " ".join(w[2] for w in all_commited)
        assert text == "olá mundo como você"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
