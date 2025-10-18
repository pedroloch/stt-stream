"""
Testes unitários para HypothesisBuffer

Valida a implementação do LocalAgreement-n com word-level timestamps.
"""

import pytest

from server.streaming.hypothesis_buffer import HypothesisBuffer
from server.models.result import Word


def make_word(start: float, end: float, text: str, probability: float = 0.9) -> Word:
    """Helper para criar Word objects nos testes"""
    return Word(
        word=text,
        start=start,
        end=end,
        probability=probability,
        is_filler=False,
        is_punctuation=False,
        speaker_id=None,
        language=None,
    )


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
        assert buffer.confidence_threshold == 0.95

    def test_insert_filters_old_words(self):
        """Testa que insert filtra palavras antes de last_commited_time"""
        buffer = HypothesisBuffer()
        buffer.last_commited_time = 2.0

        # Palavras: algumas antes de 2.0s, algumas depois
        words = [
            make_word(0.5, 1.0, "old"),
            make_word(1.5, 1.8, "old2"),
            make_word(2.1, 2.5, "new"),  # ← Apenas esta deve passar
            make_word(2.6, 3.0, "new2"),  # ← E esta
        ]

        buffer.insert(words, offset=0.0)

        # Apenas palavras DEPOIS de 2.0s (com tolerância de 0.1s)
        assert len(buffer.new) == 2
        assert buffer.new[0].word == "new"
        assert buffer.new[1].word == "new2"

    def test_insert_with_offset(self):
        """Testa que insert ajusta timestamps com offset"""
        buffer = HypothesisBuffer()

        # Palavras relativas (como vêm do backend)
        words = [
            make_word(0.0, 0.5, "hello"),
            make_word(0.6, 1.0, "world"),
        ]

        # Offset de 5.0s (buffer já processou 5s antes)
        buffer.insert(words, offset=5.0)

        # Timestamps devem ser absolutos
        assert buffer.new[0].start == 5.0
        assert buffer.new[0].end == 5.5
        assert buffer.new[1].start == 5.6
        assert buffer.new[1].end == 6.0

    def test_flush_no_agreement(self):
        """Testa flush sem concordância (primeira transcrição)"""
        buffer = HypothesisBuffer()

        words = [make_word(0.0, 0.5, "hello"), make_word(0.6, 1.0, "world")]
        buffer.insert(words, offset=0.0)

        # Primeira vez: não há buffer anterior para comparar
        commit = buffer.flush()

        # Com confidence 0.9 < 0.95, não faz early commit
        # Sem buffer anterior, não há LocalAgreement
        # Portanto: commit deve ser vazio, words vão para buffer
        assert commit == []
        assert len(buffer.buffer) == 2
        assert buffer.buffer[0].word == "hello"
        assert buffer.buffer[1].word == "world"

    def test_flush_with_full_agreement(self):
        """Testa flush com concordância total"""
        buffer = HypothesisBuffer()

        # Primeira transcrição
        words1 = [make_word(0.0, 0.5, "hello"), make_word(0.6, 1.0, "world")]
        buffer.insert(words1, offset=0.0)
        buffer.flush()

        # Segunda transcrição: IGUAL
        words2 = [make_word(0.0, 0.5, "hello"), make_word(0.6, 1.0, "world")]
        buffer.insert(words2, offset=0.0)
        commit = buffer.flush()

        # Deve confirmar TODAS as palavras
        assert len(commit) == 2
        assert commit[0].word == "hello"
        assert commit[1].word == "world"

        # last_commited_time deve ser end da última palavra
        assert buffer.last_commited_time == 1.0

    def test_flush_with_partial_agreement(self):
        """Testa flush com concordância parcial (caso típico)"""
        buffer = HypothesisBuffer()

        # Primeira transcrição
        words1 = [make_word(0.0, 0.5, "hello"), make_word(0.6, 1.0, "world")]
        buffer.insert(words1, offset=0.0)
        buffer.flush()

        # Segunda transcrição: começa igual mas estende
        words2 = [
            make_word(0.0, 0.5, "hello"),
            make_word(0.6, 1.0, "world"),
            make_word(1.1, 1.5, "how"),  # ← Nova palavra
        ]
        buffer.insert(words2, offset=0.0)
        commit = buffer.flush()

        # Deve confirmar prefixo comum: ["hello", "world"]
        assert len(commit) == 2
        assert commit[0].word == "hello"
        assert commit[1].word == "world"

        # Buffer agora contém apenas a nova palavra
        assert len(buffer.buffer) == 1
        assert buffer.buffer[0].word == "how"

    def test_flush_with_divergence(self):
        """Testa flush quando transcrições divergem"""
        buffer = HypothesisBuffer()

        # Primeira transcrição
        words1 = [make_word(0.0, 0.5, "hello"), make_word(0.6, 1.0, "world")]
        buffer.insert(words1, offset=0.0)
        buffer.flush()

        # Segunda transcrição: começa diferente
        words2 = [make_word(0.0, 0.5, "hi"), make_word(0.6, 1.0, "there")]  # ← Tudo diferente
        buffer.insert(words2, offset=0.0)
        commit = buffer.flush()

        # Nenhuma concordância (confidence=0.9 < 0.95, então não faz early commit)
        assert commit == []

        # Buffer agora contém a nova transcrição
        assert len(buffer.buffer) == 2
        assert buffer.buffer[0].word == "hi"

    def test_confidence_early_commit(self):
        """⭐ NOVO: Testa confidence-based early commit"""
        buffer = HypothesisBuffer(confidence_threshold=0.95)

        # Primeira transcrição (baixa confiança, não commita)
        words1 = [make_word(0.0, 0.5, "hello", probability=0.9)]
        buffer.insert(words1, offset=0.0)
        commit1 = buffer.flush()
        assert len(commit1) == 0  # Não commita (< 0.95)

        # Segunda transcrição com ALTA confiança
        words2 = [make_word(0.0, 0.5, "hello", probability=0.98)]  # ← Alta confiança!
        buffer.insert(words2, offset=0.0)
        commit2 = buffer.flush()

        # Deve fazer early commit imediatamente!
        assert len(commit2) == 1
        assert commit2[0].word == "hello"
        assert commit2[0].probability == 0.98

        # Verificar stats
        assert buffer.stats_early_commits == 1
        assert buffer.stats_agreement_commits == 0

    def test_confidence_early_commit_mixed(self):
        """⭐ NOVO: Testa mix de early commit + LocalAgreement"""
        buffer = HypothesisBuffer(confidence_threshold=0.95)

        # Setup: primeira transcrição baixa confiança
        words1 = [
            make_word(0.0, 0.5, "hello", probability=0.9),
            make_word(0.6, 1.0, "world", probability=0.9)
        ]
        buffer.insert(words1, offset=0.0)
        buffer.flush()

        # Segunda transcrição: mistura de confiânças
        words2 = [
            make_word(0.0, 0.5, "hello", probability=0.98),  # Alta → early commit
            make_word(0.6, 1.0, "world", probability=0.92),  # Baixa → LocalAgreement
        ]
        buffer.insert(words2, offset=0.0)
        commit = buffer.flush()

        # "hello" deve ter early commit, "world" via LocalAgreement
        assert len(commit) == 2
        assert buffer.stats_early_commits == 1  # "hello"
        assert buffer.stats_agreement_commits == 1  # "world"

    def test_pop_commited(self):
        """Testa remoção de palavras scrolled away"""
        buffer = HypothesisBuffer()

        # Simular palavras confirmadas
        buffer.commited_in_buffer = [
            make_word(0.0, 0.5, "old1"),
            make_word(0.6, 1.0, "old2"),
            make_word(1.1, 1.5, "recent"),
            make_word(1.6, 2.0, "new"),
        ]

        # Remover palavras até 1.0s (scrolled away após trim)
        buffer.pop_commited(1.0)

        # Apenas palavras com end > 1.0s devem permanecer
        assert len(buffer.commited_in_buffer) == 2
        assert buffer.commited_in_buffer[0].word == "recent"
        assert buffer.commited_in_buffer[1].word == "new"

    def test_remove_duplicates_1gram(self):
        """Testa remoção de duplicatas (1-gram)"""
        buffer = HypothesisBuffer()

        # Simular palavras confirmadas
        buffer.commited_in_buffer = [
            make_word(0.0, 0.5, "hello"),
            make_word(0.6, 1.0, "world"),
        ]

        # Nova transcrição com "world" duplicada
        buffer.new = [
            make_word(0.6, 1.0, "world"),  # ← Duplicata (última palavra confirmada)
            make_word(1.1, 1.5, "how"),
        ]

        buffer._remove_duplicates()

        # "world" deve ter sido removida
        assert len(buffer.new) == 1
        assert buffer.new[0].word == "how"

    def test_remove_duplicates_2gram(self):
        """Testa remoção de duplicatas (2-gram)"""
        buffer = HypothesisBuffer()

        # Simular palavras confirmadas
        buffer.commited_in_buffer = [
            make_word(0.0, 0.5, "hello"),
            make_word(0.6, 1.0, "world"),
        ]

        # Nova transcrição com "hello world" duplicado
        buffer.new = [
            make_word(0.0, 0.5, "hello"),  # ← Duplicata
            make_word(0.6, 1.0, "world"),  # ← Duplicata
            make_word(1.1, 1.5, "how"),
        ]

        buffer._remove_duplicates()

        # "hello world" devem ter sido removidos (2-gram match)
        assert len(buffer.new) == 1
        assert buffer.new[0].word == "how"

    def test_reset(self):
        """Testa reset completo"""
        buffer = HypothesisBuffer()

        # Preencher com dados
        buffer.commited_in_buffer = [make_word(0.0, 0.5, "test")]
        buffer.buffer = [make_word(0.6, 1.0, "test2")]
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
            make_word(0.0, 0.5, "hello"),
            make_word(0.6, 1.0, "world"),
        ]

        incomplete = buffer.complete()

        # Deve retornar cópia do buffer
        assert len(incomplete) == 2
        assert incomplete[0].word == "hello"

        # Não deve modificar buffer original
        assert len(buffer.buffer) == len(incomplete)

    def test_get_stats(self):
        """Testa estatísticas"""
        buffer = HypothesisBuffer()

        buffer.commited_in_buffer = [make_word(0.0, 0.5, "test")]
        buffer.buffer = [make_word(0.6, 1.0, "test2")]
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
        words1 = [make_word(0.0, 0.5, "olá"), make_word(0.6, 1.0, "mundo")]
        buffer.insert(words1, offset=0.0)
        commit1 = buffer.flush()
        # Primeira vez: com confidence 0.9 < 0.95, não faz early commit
        # Sem buffer anterior, não há agreement
        # Então commit deve ser vazio OU ter early commits se conf alta

        # Update 2: "olá mundo como você"
        words2 = [
            make_word(0.0, 0.5, "olá"),
            make_word(0.6, 1.0, "mundo"),
            make_word(1.1, 1.5, "como"),
            make_word(1.6, 2.0, "você"),
        ]
        buffer.insert(words2, offset=0.0)
        commit2 = buffer.flush()

        # Deve confirmar prefixo comum: "olá mundo"
        assert len(commit2) == 2
        assert commit2[0].word == "olá"
        assert commit2[1].word == "mundo"
        assert buffer.last_commited_time == 1.0

        # Update 3: "olá mundo como você está"
        # (mas buffer já trimmed até 1.0s, então offset = 1.0)
        words3 = [
            make_word(0.0, 0.5, "como"),  # Relativo ao buffer
            make_word(0.6, 1.0, "você"),
            make_word(1.1, 1.5, "está"),
        ]
        buffer.insert(words3, offset=1.0)
        commit3 = buffer.flush()

        # Deve confirmar "como você"
        assert len(commit3) == 2
        assert commit3[0].word == "como"
        assert commit3[1].word == "você"
        assert buffer.last_commited_time == 2.0

        # Total confirmado: "olá mundo como você"
        all_commited = commit2 + commit3
        text = " ".join(w.word for w in all_commited)
        assert text == "olá mundo como você"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
