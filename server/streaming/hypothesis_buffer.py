"""
HypothesisBuffer - LocalAgreement com Word-Level Timestamps

Implementação baseada em whisper_streaming (UFAL):
https://github.com/ufal/whisper_streaming/blob/main/whisper_online.py

Esta é a implementação CORRETA do algoritmo LocalAgreement-n.
"""

import logging
from typing import Literal

logger = logging.getLogger(__name__)


# Type alias para palavras com timestamps
# (start_time, end_time, word_text)
WordWithTimestamp = tuple[float, float, str]


class HypothesisBuffer:
    """
    Buffer de hipóteses para LocalAgreement-n Policy

    Mantém 3 estados:
    - commited_in_buffer: Palavras confirmadas que ainda estão no audio buffer
    - buffer: Última transcrição (palavras com timestamps)
    - new: Nova transcrição (palavras com timestamps)

    Algoritmo:
    1. insert(): Filtra palavras novas (> last_commited_time) e remove duplicatas
    2. flush(): Compara buffer vs new palavra por palavra, retorna prefixo comum
    3. Repete até estabilizar
    """

    def __init__(self, logfile=None):
        """
        Inicializa o buffer de hipóteses

        Args:
            logfile: Arquivo de log (opcional, para compatibilidade)
        """
        # Palavras confirmadas (ainda no buffer)
        self.commited_in_buffer: list[WordWithTimestamp] = []

        # Última transcrição
        self.buffer: list[WordWithTimestamp] = []

        # Nova transcrição
        self.new: list[WordWithTimestamp] = []

        # Timestamp da última palavra confirmada
        self.last_commited_time: float = 0.0

        # Última palavra confirmada (para debugging)
        self.last_commited_word: str | None = None

        logger.info("HypothesisBuffer inicializado")

    def insert(self, new_words: list[WordWithTimestamp], offset: float):
        """
        Insere nova transcrição e filtra palavras já confirmadas

        CRÍTICO: Filtra apenas palavras DEPOIS de last_commited_time!
        Isso evita re-processar texto já confirmado.

        Args:
            new_words: Lista de palavras com timestamps [(start, end, text), ...]
                       Timestamps são RELATIVOS ao chunk transcrito
            offset: Offset temporal (buffer_time_offset) para converter para absoluto
        """
        # Converter timestamps para absolutos
        new_words_absolute = [
            (start + offset, end + offset, text)
            for start, end, text in new_words
        ]

        # FILTRO CRÍTICO: Apenas palavras DEPOIS do último confirmado
        # Tolerância de 0.1s para compensar imprecisões de timestamp
        self.new = [
            (start, end, text)
            for start, end, text in new_words_absolute
            if start > self.last_commited_time - 0.1
        ]

        logger.debug(
            f"insert: {len(new_words)} palavras originais -> "
            f"{len(self.new)} palavras novas (filtradas por timestamp)"
        )

        # Remove duplicatas via n-gram matching
        if len(self.new) >= 1 and self.commited_in_buffer:
            self._remove_duplicates()

    def _remove_duplicates(self):
        """
        Remove duplicatas comparando n-gramas (1 a 5 palavras)

        Se as últimas N palavras confirmadas são iguais às primeiras N
        palavras da nova transcrição, remove as N palavras duplicadas.

        Exemplo:
            commited: [..., "mundo", "como", "você"]
            new: ["como", "você", "está"]

            2-gram match: ["como", "você"]
            Remove "como", "você" de new
            new = ["está"]
        """
        cn = len(self.commited_in_buffer)
        nn = len(self.new)

        # Testar n-gramas de 1 a 5 palavras (ou menor se não houver palavras suficientes)
        for i in range(1, min(min(cn, nn), 5) + 1):
            # Últimas i palavras confirmadas
            commited_ngram = " ".join(
                self.commited_in_buffer[-j][2] for j in range(1, i + 1)
            )[::-1]  # Reverse (pq pegamos de trás pra frente)

            # Primeiras i palavras novas
            new_ngram = " ".join(self.new[j][2] for j in range(i))

            if commited_ngram == new_ngram:
                # Match! Remover i palavras duplicadas
                removed = []
                for j in range(i):
                    word = self.new.pop(0)
                    removed.append(repr(word[2]))

                logger.debug(
                    f"Removidas {i} palavras duplicadas (n-gram match): "
                    f"{' '.join(removed)}"
                )
                break

    def flush(self) -> list[WordWithTimestamp]:
        """
        LocalAgreement: Retorna prefixo comum entre buffer e new

        Compara palavra por palavra até encontrar divergência.
        Palavras que concordam são confirmadas (commited).

        Returns:
            Lista de palavras confirmadas [(start, end, text), ...]
        """
        commit: list[WordWithTimestamp] = []

        # Comparar palavra por palavra
        while self.new and self.buffer:
            new_word = self.new[0]
            buffer_word = self.buffer[0]

            # Verificar se TEXTO da palavra é igual
            if new_word[2] == buffer_word[2]:
                # CONCORDÂNCIA! Confirmar palavra
                commit.append(new_word)

                # Atualizar estado
                self.last_commited_word = new_word[2]
                self.last_commited_time = new_word[1]  # end time

                # Remover de ambos
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                # DIVERGÊNCIA! Parar comparação
                break

        # Atualizar buffers
        self.buffer = self.new.copy()  # Nova transcrição vira anterior
        self.new = []  # Limpar nova

        # Adicionar palavras confirmadas ao commited_in_buffer
        self.commited_in_buffer.extend(commit)

        if commit:
            logger.info(
                f"✅ LocalAgreement confirmou {len(commit)} palavras: "
                f"'{' '.join(w[2] for w in commit)}'"
            )
        else:
            logger.debug("LocalAgreement: sem concordância (aguardando próximo update)")

        return commit

    def pop_commited(self, time: float):
        """
        Remove palavras confirmadas que estão antes do timestamp especificado

        Usado quando fazemos trim do audio buffer.
        Remove palavras que foram "scrolled away" do buffer.

        Args:
            time: Timestamp de corte (buffer_time_offset após trim)
        """
        removed_count = 0
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)
            removed_count += 1

        if removed_count > 0:
            logger.debug(f"pop_commited: removidas {removed_count} palavras scrolled away")

    def complete(self) -> list[WordWithTimestamp]:
        """
        Retorna buffer atual (palavras ainda não confirmadas)

        Útil para mostrar preview/parcial ao usuário.

        Returns:
            Lista de palavras no buffer (não confirmadas ainda)
        """
        return self.buffer.copy()

    def reset(self):
        """Reseta buffer para nova sessão"""
        self.commited_in_buffer.clear()
        self.buffer.clear()
        self.new.clear()
        self.last_commited_time = 0.0
        self.last_commited_word = None
        logger.info("HypothesisBuffer resetado")

    def get_stats(self) -> dict:
        """Retorna estatísticas do buffer"""
        return {
            "commited_count": len(self.commited_in_buffer),
            "buffer_count": len(self.buffer),
            "new_count": len(self.new),
            "last_commited_time": self.last_commited_time,
            "last_commited_word": self.last_commited_word,
        }
