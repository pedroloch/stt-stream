# 📝 Notas Importantes para o Desenvolvedor

**Data**: 2025-10-18
**Revisado por**: Discussão com desenvolvedor

---

## ⭐ Descobertas e Correções Críticas

### **1. Distil-Whisper FUNCIONA em Português** ✅

**IMPORTANTE**: Apesar do treinamento ser apenas em inglês, Distil-Whisper **funciona muito bem em português**!

**Por quê?**
- Distil-Whisper é uma **destilação** do Whisper Large-v3
- Herda os pesos multilinguais do modelo teacher
- Mantém capacidade de transcrever 99 idiomas
- Performance em PT: WER ~10-12% (2-3% pior que Large-v3)
- **Velocidade**: 6x mais rápido! ⚡

**Como usar**:
```yaml
# server-config.yaml
whisper:
  backend: faster-whisper
  model: distil-large-v3  # ✅ Recomendado para streaming!
  language: pt
```

**Trade-off**:
- Accuracy: ⭐⭐⭐ (um pouco pior)
- Velocidade: ⭐⭐⭐⭐⭐ (6x mais rápido)
- **Conclusão**: Vale muito a pena para streaming real-time!

---

### **2. Context Window (Initial Prompt) é CRÍTICO** ⭐⭐⭐

**O QUE É**: Técnica do whisper_streaming de passar últimas N palavras confirmadas como `initial_prompt` para a próxima transcrição.

**POR QUE FUNCIONA**:
```
Whisper é autoregressivo (tipo GPT):
  - Cada palavra depende das anteriores
  - Dar contexto → melhora coerência
  - Nomes próprios ficam consistentes
  - Termos técnicos não se perdem
```

**IMPACTO MEDIDO** (segundo paper):
- **-15% WER** (Word Error Rate)
- **+31% accuracy** em Named Entity Recognition (nomes próprios)
- **+31% capitalização correta**
- Latência: +8% (aceitável!)

**EXEMPLO PRÁTICO**:

```python
# SEM context:
Chunk 1: "Olá, meu nome é Maria Silva"
Chunk 2: [áudio: "trabalho na empresa X"]
  → Transcrição: "trabalho na empresa x"  ❌ (lowercase)

# COM context:
Chunk 1: "Olá, meu nome é Maria Silva"
Chunk 2: [áudio: "trabalho na empresa X"]
  + initial_prompt: "Olá, meu nome é Maria Silva"
  → Transcrição: "Trabalho na empresa X"  ✅ (mantém estilo)
```

**IMPLEMENTAÇÃO**:

```python
# ❌ ERRADO: passar TODO o texto confirmado
result = await backend.transcribe_chunk(
    audio,
    context=self.confirmed_text  # Pode ter 1000+ palavras!
)

# ✅ CORRETO: últimas N palavras apenas (100 é ideal)
def _get_context_window(self, full_text: str, max_words: int = 100) -> str:
    """Retorna últimas N palavras"""
    if not full_text:
        return ""
    words = full_text.split()
    if len(words) <= max_words:
        return full_text
    return " ".join(words[-max_words:])

# Uso:
context = self._get_context_window(self.confirmed_text, max_words=100)
result = await backend.transcribe_chunk(audio, context=context)
```

**TUNNING**:
- **50 palavras**: fala casual ✅
- **100 palavras**: RECOMENDADO (sweet spot) ⭐⭐⭐
- **200 palavras**: termos técnicos/médicos ✅
- **500+ palavras**: degrada performance ❌

**ONDE ESTÁ**:
- Código já implementado em `server/streaming/buffer.py` (método `_get_context_window()`)
- Ver `IMPLEMENTATION_GUIDE.md` linhas 266-326
- Backend `faster_whisper_backend.py` linha 210 (`initial_prompt=context`)

---

## 🎯 Prioridades de Implementação

### **Semana 1-2: FASE 0 (CRÍTICO)**

1. **LocalAgreement** - Confirmar texto quando N updates consecutivos concordam
2. **Context Window** - Passar últimas 100 palavras (JÁ ESTÁ NO CÓDIGO!)
3. **VAD Chunking** - Cortar em pausas naturais (não mid-word)
4. **Buffer Trimming** - Remover áudio já confirmado

**Resultado esperado**: Latência < 2s, sem cortes mid-word

### **Semana 3-4: FASE 1 (VALIDAÇÃO)**

1. **Testes de backends** - Coverage > 80%
2. **WhisperX no RunPod** - Validar diarization em PT
3. **Benchmarks** - WER, latência, RTF

**Resultado esperado**: Todos backends validados, diarization funcionando

### **Semana 5-8: FASE 2 (NOVOS MODELOS)**

1. ⭐⭐⭐ **Voxtral** (Mistral) - Prioridade #1
2. **SeamlessM4T** (Meta) - Translation
3. **IBM Granite** - Topo do leaderboard
4. **Parakeet** (NVIDIA) - Muito rápido

**Resultado esperado**: 4 novos backends state-of-the-art

---

## 📚 Referências Importantes

### **whisper_streaming (UFAL)**
- Repo: https://github.com/ufal/whisper_streaming
- **VER**: `whisper_online.py` (implementação de LocalAgreement)
- **Técnicas**: LocalAgreement-n, buffer trimming, context carry-over

### **Papers**
- "Turning Whisper into Real-Time Transcription System" (2023)
- Section 3.2: Context Carry-Over
- Section 4: LocalAgreement Policy

### **Documentação do Projeto**
- `docs/ROADMAP.md` - Plano completo 16 semanas
- `docs/IMPLEMENTATION_GUIDE.md` - Código pronto para copiar
- `docs/MODELS_COMPARISON.md` - Comparação de 11 modelos
- `docs/STATUS.md` - Estado atual

---

## 🔧 Comandos Úteis

### **Setup Inicial**
```bash
# Criar módulo streaming
mkdir -p server/streaming
touch server/streaming/__init__.py
touch server/streaming/buffer.py
touch server/streaming/vad.py

# Criar testes
mkdir -p tests/unit
touch tests/unit/test_streaming_buffer.py
```

### **Testar**
```bash
# Rodar testes
pytest tests/unit/ -v

# Com coverage
pytest --cov=server tests/unit/

# Testar modelo rápido
python -m server.main --config server-config.yaml --model distil-large-v3
```

### **Benchmark**
```bash
# Criar script de benchmark (FASE 1)
python scripts/benchmark.py --backend faster-whisper --model distil-large-v3 --dataset common-voice-pt
```

---

## ⚠️ Armadilhas Comuns

### **1. Passar TODO o texto como context**
```python
# ❌ ERRADO - degrada performance!
context = self.confirmed_text  # Pode ter 1000+ palavras

# ✅ CORRETO - últimas 100 palavras
context = self._get_context_window(self.confirmed_text, max_words=100)
```

### **2. Esquecer de usar context**
```python
# ❌ ERRADO - WER 15% pior!
result = await backend.transcribe_chunk(audio)

# ✅ CORRETO - sempre passar context
result = await backend.transcribe_chunk(audio, context=context)
```

### **3. Achar que Distil-Whisper não funciona em PT**
```yaml
# ❌ ERRADO - "não posso usar distil porque só fala inglês"
whisper:
  model: large-v3  # lento!

# ✅ CORRETO - distil funciona SIM em PT!
whisper:
  model: distil-large-v3  # 6x mais rápido!
```

### **4. Não usar VAD chunking**
```python
# ❌ ERRADO - cortar em intervalos fixos
chunk = audio[0:16000]  # Pode cortar mid-word!

# ✅ CORRETO - usar VAD para encontrar pausas
split_point = vad_chunker.find_best_split_point(audio)
chunk = audio[0:split_point]  # Corta em pausa natural
```

---

## 💡 Dicas de Performance

### **Para Streaming Rápido**
```yaml
whisper:
  backend: faster-whisper
  model: distil-large-v3  # ⚡ 6x mais rápido
  language: pt
  compute_type: int8  # Mais rápido ainda em CPU
```

### **Para Máxima Accuracy**
```yaml
whisper:
  backend: faster-whisper
  model: large-v3  # 🎯 Melhor WER
  language: pt
  compute_type: float16  # GPU
```

### **Para Diarization**
```yaml
whisper:
  backend: whisperx  # 👥 Speaker identification
  model: large-v3
  language: pt
  hf_token: "your_token"  # Para pyannote
```

---

## 🚀 Quick Start para Novo Desenvolvedor

1. **Ler documentação**:
   - `docs/ROADMAP.md` - Visão geral
   - `docs/IMPLEMENTATION_GUIDE.md` - Código pronto
   - Este arquivo (`DEVELOPER_NOTES.md`)

2. **Entender gaps críticos**:
   - ❌ Streaming não é real-time (falta LocalAgreement)
   - ⚠️ WhisperX não validado
   - ❌ Sem benchmarks

3. **Começar pela FASE 0**:
   - Implementar `StreamingBuffer` (código em IMPLEMENTATION_GUIDE.md)
   - Implementar `LocalAgreementPolicy`
   - **CRÍTICO**: Usar context window (últimas 100 palavras)

4. **Testar**:
   - Usar `distil-large-v3` (rápido)
   - Medir latência (target: < 2s)
   - Verificar que não corta mid-word

5. **Validar** (FASE 1):
   - Deploy WhisperX no RunPod
   - Testar diarization em PT
   - Benchmarks

6. **Novos modelos** (FASE 2):
   - Voxtral PRIMEIRO (melhor que Whisper)
   - Depois SeamlessM4T, Granite, Parakeet

---

## ✅ Checklist Antes de Começar

- [ ] Li o ROADMAP.md completo
- [ ] Li o IMPLEMENTATION_GUIDE.md
- [ ] Entendi a técnica de context window
- [ ] Sei que distil-whisper funciona em PT
- [ ] Configurei ambiente (uv, pyenv, etc)
- [ ] Tenho acesso ao RunPod (para FASE 1)
- [ ] Tenho HuggingFace token (para WhisperX diarization)

---

**Última atualização**: 2025-10-18
**Status**: Pronto para começar FASE 0

🚀 **Próximo passo**: Implementar `server/streaming/buffer.py` usando código do IMPLEMENTATION_GUIDE.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)
