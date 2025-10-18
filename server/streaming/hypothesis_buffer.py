"""
HypothesisBuffer - LocalAgreement com Word-Level Timestamps

Implementação baseada em whisper_streaming (UFAL):
https://github.com/ufal/whisper_streaming/blob/main/whisper_online.py

Esta é a implementação CORRETA do algoritmo LocalAgreement-n.

Melhorias vs. whisper_streaming:
- Confidence-based early commit: palavras com alta confiança (>95%) são
  confirmadas imediatamente sem esperar agreement
"""

import logging

from ..models.result import Word

logger = logging.getLogger(__name__)


# Type alias para compatibilidade (deprecated - use Word diretamente)
WordWithTimestamp = tuple[float, float, str]


class HypothesisBuffer:
    """
    Buffer de hipóteses para LocalAgreement-n Policy com Confidence Early Commit

    Mantém 3 estados:
    - commited_in_buffer: Palavras confirmadas que ainda estão no audio buffer
    - buffer: Última transcrição (palavras com timestamps)
    - new: Nova transcrição (palavras com timestamps)

    Algoritmo:
    1. insert(): Filtra palavras novas (> last_commited_time) e remove duplicatas
    2. flush(): Compara buffer vs new palavra por palavra, retorna prefixo comum
        - Se word.probability > confidence_threshold → commit imediatamente (early commit!)
        - Senão, usa LocalAgreement tradicional
    3. Repete até estabilizar

    Args:
        confidence_threshold: Threshold para early commit (default: 0.95)
            Palavras com confidence acima desse valor são confirmadas imediatamente
        logfile: Arquivo de log (opcional, para compatibilidade)
    """

    def __init__(self, confidence_threshold: float = 0.95, logfile=None):
        """
        Inicializa o buffer de hipóteses

        Args:
            confidence_threshold: Threshold de confiança para early commit (0-1)
            logfile: Arquivo de log (opcional, para compatibilidade)
        """
        # Palavras confirmadas (ainda no buffer)
        self.commited_in_buffer: list[Word] = []

        # Última transcrição
        self.buffer: list[Word] = []

        # Nova transcrição
        self.new: list[Word] = []

        # Timestamp da última palavra confirmada
        self.last_commited_time: float = 0.0

        # Última palavra confirmada (para debugging)
        self.last_commited_word: str | None = None

        # ⭐ NOVO: Threshold para confidence-based early commit
        self.confidence_threshold = confidence_threshold

        # Stats para monitoramento
        self.stats_early_commits = 0  # Commits por alta confiança
        self.stats_agreement_commits = 0  # Commits por LocalAgreement

        logger.info(
            f"HypothesisBuffer inicializado (confidence_threshold={confidence_threshold})"
        )

    def insert(self, new_words: list[Word], offset: float):
        """
        Insere nova transcrição e filtra palavras já confirmadas

        CRÍTICO: Filtra apenas palavras DEPOIS de last_commited_time!
        Isso evita re-processar texto já confirmado.

        Args:
            new_words: Lista de objetos Word com timestamps
                       Timestamps são RELATIVOS ao chunk transcrito
            offset: Offset temporal (buffer_time_offset) para converter para absoluto
        """
        # Converter timestamps para absolutos (criar novos Word objects)
        new_words_absolute = [
            Word(
                word=w.word,
                start=w.start + offset,
                end=w.end + offset,
                probability=w.probability,
                is_filler=w.is_filler,
                is_punctuation=w.is_punctuation,
                speaker_id=w.speaker_id,
                language=w.language,
            )
            for w in new_words
        ]

        # FILTRO CRÍTICO: Apenas palavras DEPOIS do último confirmado
        # Tolerância de 0.1s para compensar imprecisões de timestamp
        self.new = [
            w
            for w in new_words_absolute
            if w.start > self.last_commited_time - 0.1
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

        # Testar n-gramas de 5 a 1 palavras (do maior para o menor)
        # Testar do maior para o menor garante que removemos o overlap máximo
        for i in range(min(min(cn, nn), 5), 0, -1):
            # Últimas i palavras confirmadas
            commited_ngram = " ".join(
                word.word for word in self.commited_in_buffer[-i:]
            )

            # Primeiras i palavras novas
            new_ngram = " ".join(
                word.word for word in self.new[:i]
            )

            if commited_ngram == new_ngram:
                # Match! Remover i palavras duplicadas
                removed = [self.new[j].word for j in range(i)]

                # Remover as palavras duplicadas
                self.new = self.new[i:]

                logger.debug(
                    f"Removidas {i} palavras duplicadas (n-gram match): "
                    f"{repr(' '.join(removed))}"
                )
                break

    def flush(self) -> list[Word]:
        """
        LocalAgreement com Confidence Early Commit

        Algoritmo híbrido:
        1. Para cada palavra nova, verificar confidence
        2. Se confidence > threshold → commit imediatamente (early commit!)
        3. Senão, usar LocalAgreement tradicional (comparar com buffer)

        Retorna prefixo comum entre buffer e new.

        Returns:
            Lista de palavras confirmadas (Word objects)
        """
        commit: list[Word] = []

        # ⭐ NOVO: Processar palavra por palavra com confidence check
        while self.new:
            new_word = self.new[0]

            # ⭐ CONFIDENCE EARLY COMMIT: Alta confiança = commit imediato
            if new_word.probability >= self.confidence_threshold:
                # Commit imediatamente sem esperar agreement!
                commit.append(new_word)

                # Atualizar estado
                self.last_commited_word = new_word.word
                self.last_commited_time = new_word.end  # end time

                # Remover da fila
                self.new.pop(0)

                # Remover do buffer também se existir
                if self.buffer and self.buffer[0].word == new_word.word:
                    self.buffer.pop(0)

                # Stats
                self.stats_early_commits += 1

                logger.debug(
                    f"⚡ Early commit (conf={new_word.probability:.3f}): '{new_word.word}'"
                )
                continue

            # Se não tem buffer, parar (aguardar próxima transcrição)
            if not self.buffer:
                break

            buffer_word = self.buffer[0]

            # LocalAgreement tradicional: verificar se TEXTO concorda
            if new_word.word == buffer_word.word:
                # CONCORDÂNCIA! Confirmar palavra
                commit.append(new_word)

                # Atualizar estado
                self.last_commited_word = new_word.word
                self.last_commited_time = new_word.end  # end time

                # Remover de ambos
                self.buffer.pop(0)
                self.new.pop(0)

                # Stats
                self.stats_agreement_commits += 1
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
                f"✅ Confirmadas {len(commit)} palavras: "
                f"'{' '.join(w.word for w in commit)}' "
                f"(early: {self.stats_early_commits}, agreement: {self.stats_agreement_commits})"
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
        while self.commited_in_buffer and self.commited_in_buffer[0].end <= time:
            self.commited_in_buffer.pop(0)
            removed_count += 1

        if removed_count > 0:
            logger.debug(f"pop_commited: removidas {removed_count} palavras scrolled away")

    def complete(self) -> list[Word]:
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
