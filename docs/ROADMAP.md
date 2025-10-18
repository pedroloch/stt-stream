# 🗺️ Whisper Stream - Roadmap Completo 2025

**Data de Criação**: 2025-10-18
**Versão**: 2.0
**Objetivo**: Sistema de STT streaming open-source competitivo com Deepgram/AssemblyAI

---

## 📊 Matriz Completa de Modelos STT (2025)

### ✅ **Modelos Implementados**

| Modelo | Streaming | Batch | Diarização | Word TS | Translation | PT | EN | Outros | M1 | CUDA | CPU | Status |
|--------|-----------|-------|------------|---------|-------------|----|----|--------|----|----|-----|--------|
| **faster-whisper** | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | 99 | ✅ | ✅ | ✅ | ✅ Funcionando |
| **distil-whisper-v3** | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | 99 | ✅ | ✅ | ✅ | ✅ Via faster-whisper (6x rápido!) |
| **MLX** | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | 99 | ✅ | ❌ | ❌ | ✅ Funcionando |
| **WhisperX** | ⚠️ | ✅ | ✅ (batch) | ✅✅ | ❌ | ✅ | ✅ | 99 | ❌ | ✅ | ❌ | ⏳ Não validado |
| **CrisperWhisper** | ✅ | ✅ | ❌ | ✅ | ❌ | ❓ | ✅ | DE | ✅ | ✅ | ❌ | ⏳ Não validado PT |
| **CPU Backend** | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | 99 | ✅ | ✅ | ✅ | ✅ Funcionando |
| **CUDA Backend** | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | 99 | ❌ | ✅ | ❌ | ✅ Funcionando |

**Legenda:**
- ✅ = Suportado e funcionando
- ✅✅ = Suportado com qualidade superior
- ⚠️ = Suportado com limitações
- ❌ = Não suportado
- ❓ = Não testado/incerto
- ⏳ = Implementado mas não validado

---

### 🆕 **Modelos a Implementar - Alta Prioridade**

| Modelo | Streaming | Batch | Diarização | Word TS | Translation | PT | EN | Outros | M1 | CUDA | CPU | WER (EN) | RTF | Licença |
|--------|-----------|-------|------------|---------|-------------|----|----|--------|----|----|-----|----------|-----|---------|
| **Voxtral 24B** | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | 6 lang | ✅ | ✅ | ⚠️ | ~5-6% | ? | Apache 2.0 |
| **Voxtral 3B** | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | 6 lang | ✅ | ✅ | ✅ | ~6-7% | ? | Apache 2.0 |
| **SeamlessM4T v2** | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | 100+ | ❌ | ✅ | ⚠️ | N/A | ~0.2x | CC-BY-NC 4.0 |
| **IBM Granite 8B** | ⚠️ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | 7 lang | ❓ | ✅ | ⚠️ | ~5.85% | ? | Apache 2.0 |
| **Parakeet TDT 0.6B** | ✅ | ✅ | ❌ | ❓ | ❌ | ✅ | ✅ | 25 lang | ❓ | ✅ | ✅ | 6.05% | 3380x | Apache 2.0 |

**Idiomas suportados:**
- Voxtral: PT, EN, ES, FR, DE, NL, IT, HI
- SeamlessM4T: 100+ idiomas
- IBM Granite: EN, FR, ES, IT, DE, PT, JA, ZH
- Parakeet: 25 idiomas europeus (incluindo PT)

---

### 🔬 **Modelos Experimentais - Pesquisa**

| Modelo | Streaming | Batch | Diarização | Word TS | PT | EN | Outros | M1 | CUDA | CPU | Latência | Licença | Notas |
|--------|-----------|-------|------------|---------|----|----|--------|----|----|-----|----------|---------|-------|
| **Kyutai STT 2.6B** | ✅✅ | ✅ | ❌ | ✅ | ❌ | ✅ | FR | ✅ | ✅ | ✅ | 2.5s | CC-BY 4.0 | **Streaming nativo** |
| **Kyutai STT 1B** | ✅✅ | ✅ | ❌ | ✅ | ❌ | ✅ | FR | ✅ | ✅ | ✅ | 0.5s | CC-BY 4.0 | **VAD semântico** |
| **Moshi** | ✅✅ | ❌ | ❌ | ❌ | ❌ | ✅ | FR | ✅ | ✅ | ❌ | 160ms | CC-BY 4.0 | **Full-duplex** |
| **Canary 1B v2** | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | 25 lang | ❓ | ✅ | ⚠️ | N/A | CC-BY-NC 4.0 | **Batch only** |
| **Wav2vec2-PT** | ✅ | ✅ | ❌ | ❓ | ✅ | ❌ | - | ✅ | ✅ | ✅ | ? | Apache 2.0 | Fine-tuned PT |

**Notas importantes:**
- **Kyutai STT**: Arquitetura "Delayed Streams Modeling" - MUITO interessante para estudar
- **Moshi**: Full-duplex (fala enquanto escuta) - caso de uso diferente
- **Canary**: Ótimo WER mas **sem streaming** (só batch)
- **Wav2vec2**: Base para fine-tuning customizado

---

### ✅ **Modelos Já Disponíveis (via faster-whisper)**

| Modelo | Status | Notas |
|--------|--------|-------|
| **Distil-Whisper Large-v3** | ✅ Funcionando | ✅ **FUNCIONA EM PT!** Herda pesos do Large-v3. WER ~10-12% PT (2-3% pior que Large-v3), mas **6x mais rápido**. Recomendado para streaming! |

**CORREÇÃO IMPORTANTE**: Distil-Whisper foi treinado apenas em inglês, MAS **funciona em português** porque herda os pesos multilinguais do Whisper Large-v3. Testado e confirmado funcionando bem em PT.

### ❌ **Modelos Descartados**

| Modelo | Motivo |
|--------|--------|
| **SpeechT5** | ❌ Mais focado em TTS, não é STT state-of-the-art |
| **Deepgram Nova-3** | ❌ Closed-source (mas usar como benchmark) |
| **AssemblyAI Slam-1** | ❌ Closed-source, EN-only streaming |

---

## 🎯 Comparação: Modelos vs. Deepgram/AssemblyAI

### **Deepgram Nova-3 (Benchmark)**
- ✅ WER: 6.84% (streaming), 5-6% (batch)
- ✅ Latência: ~0.2-0.3x RTF (muito rápido)
- ✅ Streaming: code-switching 10 idiomas (incluindo PT)
- ❌ Closed-source
- ❌ Custo: $0.0077/min

### **AssemblyAI Slam-1 (Benchmark)**
- ✅ WER: ~5-6% (batch)
- ✅ Features: sentiment, PII detection, chapters
- ❌ Streaming: EN-only (outros idiomas em 2025)
- ❌ Closed-source

### **Nossa Vantagem Competitiva:**

| Feature | Deepgram | AssemblyAI | Whisper Stream |
|---------|----------|------------|----------------|
| **Self-hosted** | ❌ | ❌ | ✅ |
| **Open-source** | ❌ | ❌ | ✅ |
| **PT optimizado** | ⚠️ | ⚠️ | ✅ |
| **Diarization** | ❌ | ✅ | ✅ (WhisperX) |
| **Translation** | ❌ | ❌ | ✅ (SeamlessM4T) |
| **LGPD compliant** | ⚠️ | ⚠️ | ✅ |
| **Custo** | $0.0077/min | $0.005/min | ~$0.0001/min (GPU) |

---

## 🗂️ Roadmap em Fases (ATUALIZADO 2025-10-18)

### **FASE 0: API Batch - Fundação** ⭐⭐⭐ MÁXIMA PRIORIDADE
**Duração**: 2-3 semanas
**Objetivo**: Endpoint `/v1/transcribe` para batch processing compatível com OpenAI Whisper-1

**Por quê primeiro?**
- Permite comparação de modelos (benchmarking)
- Use case importante: transcrever arquivos gravados
- Base para adicionar novos modelos incrementalmente
- Streaming já funciona (pode esperar refinamento)

#### Tasks:

**0.1. API Batch Endpoint** ⭐ CRÍTICO
- [ ] Criar `server/api/batch.py`
- [ ] Endpoint `POST /v1/transcribe`
- [ ] Suporte a multipart/form-data (upload de arquivo)
- [ ] Parâmetros: model, language, task, response_format, etc
- [ ] Validação de arquivo (formato, tamanho, duração)
- [ ] **Tests**: `tests/integration/test_batch_api.py`
- **Acceptance**: Upload de MP3/WAV, retorna JSON

**0.2. Backend Dispatcher**
- [ ] Criar `server/backends/dispatcher.py`
- [ ] Lógica de seleção de backend por capabilities
- [ ] Auto-select baseado em plataforma se `model="auto"`
- [ ] Fallback se backend preferido não disponível
- [ ] **Tests**: `tests/unit/test_dispatcher.py`
- **Acceptance**: Seleciona backend correto por task/platform

**0.3. Batch Processor Pipeline**
- [ ] Criar `server/processors/batch_processor.py`
- [ ] Pipeline: preparar áudio → transcrever → post-process → format
- [ ] Suporte a diarization (post-processamento)
- [ ] Output formats: JSON, verbose_json, text
- [ ] Métricas de processing (tempo, RTF)
- [ ] **Tests**: `tests/unit/test_batch_processor.py`
- **Acceptance**: Pipeline completo end-to-end

**0.4. Output Formatters**
- [ ] Criar `server/formats/` (json, srt, vtt)
- [ ] `json_formatter.py`: formato simples
- [ ] `verbose_json_formatter.py`: com segments, words, metrics
- [ ] `srt_formatter.py`: legendas SubRip (.srt)
- [ ] `vtt_formatter.py`: WebVTT (.vtt)
- [ ] **Tests**: `tests/unit/test_formatters.py`
- **Acceptance**: Cada formato válido e testado

**0.5. Documentação OpenAPI**
- [ ] Schema OpenAPI 3.0 em `docs/openapi.yaml`
- [ ] Documentar todos parâmetros e responses
- [ ] Exemplos de uso (curl, Python, JavaScript)
- [ ] Integrar com FastAPI/aiohttp (auto-docs)
- **Acceptance**: Documentação navegável em `/docs`

**Deliverables**:
- ✅ Endpoint `/v1/transcribe` funcionando
- ✅ Suporte: faster-whisper, MLX, WhisperX
- ✅ Formats: JSON, verbose_json, text, SRT, VTT
- ✅ Diarization via post-processamento
- ✅ Documentação API completa
- ✅ Testes com coverage > 80%

**Riscos**:
- Upload de arquivos grandes (timeout, memória)
- Processamento longo pode travar servidor (precisa async queue)

**Mitigação**:
- Limitar tamanho de arquivo (100MB)
- Limitar duração (1 hora)
- Timeout configurável (10 min default)

---

### **FASE 0.5: Streaming Refinamento** ⭐ IMPORTANTE (mas não bloqueante)
**Duração**: 1-2 semanas
**Objetivo**: Melhorar streaming existente (já funciona, mas pode melhorar)

**Nota**: Pode ser feito em paralelo com outros desenvolvimentos

#### Tasks:

**0.5.1. StreamingBuffer Aprimoramento**
- [ ] Revisar `server/streaming/buffer.py`
- [ ] Tune LocalAgreement (n=2 confirmado como ideal?)
- [ ] Melhorar buffer trimming (segment vs sentence)
- [ ] **Tests**: `tests/unit/test_streaming_buffer.py`
- **Acceptance**: Latência < 2s, sem cortes de palavras

**0.5.2. VAD-based Chunking**
- [ ] Integrar Silero VAD (opcional)
- [ ] Chunk boundaries em pausas naturais
- [ ] Configurável via config
- [ ] **Tests**: `tests/unit/test_vad_chunking.py`
- **Acceptance**: Chunks mais naturais

**Deliverables**:
- ✅ Streaming mais suave e inteligente
- ✅ Pausas naturais respeitadas
- ✅ Testes cobrindo edge cases

---

### **FASE 1: Validação de Modelos Existentes** ⭐ IMPORTANTE
**Duração**: 1-2 semanas
**Objetivo**: Garantir que modelos implementados funcionam corretamente

#### Tasks:

**1.1. WhisperX Validation** ⚠️ CRÍTICO
- [ ] Deploy no RunPod com GPU
- [ ] Testar diarization com áudio PT (2-3 speakers)
- [ ] Medir DER (Diarization Error Rate)
- [ ] Comparar word timestamps vs faster-whisper
- [ ] Documentar em `docs/WHISPERX_VALIDATION.md`
- **Acceptance**:
  - DER < 15% em PT
  - Word timestamps funcionando
  - Decisão: manter ou descartar

**1.2. CrisperWhisper Validation**
- [ ] Testar com áudio verbatim PT
- [ ] Validar detecção de fillers ("né", "tipo", "então")
- [ ] Comparar WER com faster-whisper
- [ ] Decisão: manter ou descartar

**1.3. SeamlessM4T Completion**
- [ ] Completar implementação em `seamless_backend.py`
- [ ] Testar S2T (Speech-to-Text) PT→PT
- [ ] Testar translation PT→EN
- [ ] Documentar limitações

**Deliverables**:
- ✅ Todos backends validados e documentados
- ✅ Decisão clara: keep vs deprecate
- ✅ Benchmarks iniciais de WER/DER

---

### **FASE 1: Validação e Testes** ⭐ CRÍTICO
**Duração**: 2 semanas
**Objetivo**: Validar modelos existentes e fechar gaps de testes

#### Tasks:

**1.1. Testes de Backends Existentes**
- [ ] `tests/unit/test_faster_whisper_backend.py`
- [ ] `tests/unit/test_mlx_backend.py`
- [ ] `tests/unit/test_whisperx_backend.py` ⚠️
- [ ] `tests/unit/test_crisper_backend.py` ⚠️
- [ ] Mock de modelos pesados
- **Acceptance**: Coverage > 80% em todos backends

**1.2. Deploy e Validação RunPod** ⭐
- [ ] Criar `Dockerfile.cuda` otimizado
- [ ] Script `scripts/runpod-deploy.sh`
- [ ] Deploy WhisperX no RunPod
- [ ] Testar diarization com áudio PT (reunião 2-3 pessoas)
- [ ] Medir DER (Diarization Error Rate)
- [ ] Medir latência end-to-end
- [ ] Documentar em `docs/RUNPOD_VALIDATION.md`
- **Acceptance**:
  - DER < 15%
  - Latência < 3s
  - Diarization funcional em PT

**1.3. Validação CrisperWhisper em PT**
- [ ] Testar com áudio PT (verbatim)
- [ ] Validar detecção de fillers PT ("né", "tipo", "então")
- [ ] Comparar WER com faster-whisper
- [ ] Documentar limitações
- **Acceptance**: Decidir se vale manter ou descartar

**1.4. Testes de Integração**
- [ ] `tests/integration/test_end_to_end.py`
  - Start server → connect → send audio → verify result
- [ ] `tests/integration/test_multiple_clients.py`
- [ ] `tests/integration/test_long_audio.py` (30+ min)
- **Acceptance**: E2E flow testado

**1.5. Benchmarking Inicial**
- [ ] Benchmark faster-whisper (base, medium, large-v3, distil-large-v3)
- [ ] Benchmark MLX (Mac M1)
- [ ] Benchmark WhisperX + diarization (RunPod)
- [ ] Comparar com Deepgram (WER, latency)
- [ ] Publicar resultados em `docs/BENCHMARKS.md`
- **Acceptance**:
  - WER < 10% (PT)
  - Latência < 2s (streaming)
  - RTF > 5x (batch)

**Deliverables**:
- ✅ Todos backends validados
- ✅ WhisperX funcionando no RunPod
- ✅ Benchmarks publicados
- ✅ Testes com coverage > 80%

**Riscos**:
- WhisperX pode ter problemas com PT (variante europeia vs. brasileira)
- RunPod pode ser caro para testes longos

---

### **FASE 2: Novos Backends - State-of-the-Art 2025** ⭐ ALTO IMPACTO
**Duração**: 3-4 semanas
**Objetivo**: Adicionar modelos 2025 state-of-the-art

#### Tasks:

**2.1. Backend Voxtral (Mistral)** ⭐⭐⭐
**Prioridade**: ALTÍSSIMA (diferencial competitivo)

- [ ] Pesquisar API oficial Mistral vs. Hugging Face
- [ ] Implementar `server/backends/voxtral_backend.py`
  - Suporte a Voxtral 24B (CUDA)
  - Suporte a Voxtral 3B (M1 + CUDA + CPU)
  - Streaming via API dedicada
  - Word-level timestamps
- [ ] Criar `VoxtralBackend.INFO`
  ```python
  INFO = BackendInfo(
      name="voxtral",
      supported_platforms={Platform.LINUX_CUDA, Platform.MACOS_APPLE_SILICON, Platform.LINUX_CPU},
      capabilities={
          Capability.TRANSCRIPTION,
          Capability.WORD_TIMESTAMPS,
          Capability.STREAMING,
      },
      supported_languages={"pt", "en", "es", "fr", "de", "nl", "it", "hi"},
      model_sizes={"3b", "24b"},
  )
  ```
- [ ] Registrar no `BackendRegistry`
- [ ] Testes: `tests/unit/test_voxtral_backend.py`
- [ ] Benchmark vs. Whisper Large-v3
  - WER (PT e EN)
  - Latência
  - Memory usage
- [ ] Documentar em `docs/backends/VOXTRAL.md`
- **Acceptance**:
  - WER melhor que Whisper Large-v3
  - Latência < 1.5s
  - Funciona em M1 e CUDA

**2.2. Backend Parakeet (NVIDIA)** ⭐⭐
**Prioridade**: ALTA (muito rápido, 25 idiomas)

- [ ] Pesquisar implementação (NeMo? Hugging Face?)
- [ ] Implementar `server/backends/parakeet_backend.py`
  - Parakeet TDT 0.6B v3
  - Suporte a 25 idiomas europeus (incluindo PT)
  - Otimizado para velocidade (RTF 3380x)
- [ ] Criar `ParakeetBackend.INFO`
- [ ] Testes
- [ ] Benchmark foco em **velocidade**
  - RTF esperado > 100x
  - Testar em CPU vs. GPU
- [ ] Documentar
- **Acceptance**:
  - RTF > 100x (CPU)
  - WER aceitável (< 12% PT)
  - Deploy CPU viável

**2.3. Backend IBM Granite Speech** ⭐⭐
**Prioridade**: ALTA (topo do leaderboard HF)

- [ ] Implementar `server/backends/granite_backend.py`
  - IBM Granite Speech 3.3 8B
  - Suporte a PT + translation (PT→EN)
  - 7 idiomas
- [ ] Criar `GraniteBackend.INFO`
- [ ] Testes
- [ ] Benchmark
  - WER (deve ser ~5.85% EN, validar PT)
  - Translation quality (BLEU score)
- [ ] Documentar
- **Acceptance**:
  - WER competitivo
  - Translation funcional
  - Memória aceitável (8B params)

**2.4. Backend SeamlessM4T (Meta)** ⭐⭐⭐
**Prioridade**: ALTA (tradução simultânea)

- [ ] Implementar `server/backends/seamless_backend.py`
  - SeamlessM4T v2 Large
  - Speech-to-Text + Translation
  - Múltiplos idiomas alvo configuráveis
- [ ] **Pesquisar**: SeamlessStreaming (EMMA) vs. SeamlessM4T v2
  - Se SeamlessStreaming: implementar streaming com EMMA
  - Se SeamlessM4T v2: chunks maiores (5-10s)
- [ ] Criar `SeamlessM4TBackend.INFO`
  ```python
  capabilities={
      Capability.TRANSCRIPTION,
      Capability.TRANSLATION,  # ⭐ Feature única
      Capability.STREAMING,
  }
  ```
- [ ] Testes
- [ ] Benchmark
  - Latência (aceitável até 3-5s para translation)
  - Translation quality (BLEU: PT→EN, PT→ES)
  - Memory usage (modelo é pesado ~10GB)
- [ ] Documentar caso de uso: "Reuniões multilíngues"
- **Acceptance**:
  - PT→EN translation com BLEU > 30
  - Streaming funcional (mesmo que chunks maiores)
  - Múltiplos idiomas alvo simultâneos

**2.5. Testes Comparativos**
- [ ] Benchmark TODOS os novos backends
- [ ] Matriz comparativa atualizada
- [ ] Publicar em `docs/BENCHMARKS.md`
- [ ] Criar tabela markdown para README

**Deliverables**:
- ✅ 4 novos backends implementados
- ✅ Benchmarks comparativos
- ✅ Documentação completa
- ✅ Voxtral como flagship model

**Riscos**:
- Voxtral recém-lançado (pode ter bugs)
- SeamlessM4T é muito pesado (pode não rodar em GPU pequena)
- Granite pode não ter boa implementação pública

---

### **FASE 3: Features Premium** ⭐ DIFERENCIAL
**Duração**: 3-4 semanas
**Objetivo**: Features que empresas pagam

#### Tasks:

**3.1. Redact / PII Detection** ⭐⭐⭐
**Prioridade**: ALTÍSSIMA (compliance LGPD/GDPR)

- [ ] Pesquisar: Presidio (Microsoft) vs. alternativas
- [ ] Implementar `server/redact/detector.py`
  ```python
  class PIIDetector:
      def detect(self, text: str, language: str) -> List[PIIEntity]
      def redact(self, text: str, entities: List[PIIEntity]) -> str
  ```
- [ ] Entidades suportadas:
  - CPF (PT)
  - CNPJ (PT)
  - RG (PT)
  - Email
  - Telefone (BR)
  - Endereço
  - Nomes próprios
  - Cartão de crédito
- [ ] Integrar no pipeline de transcrição (opt-in via config)
- [ ] Configurável por capability
  ```yaml
  redact:
    enabled: true
    entities: ["cpf", "email", "phone", "names"]
    strategy: "replace"  # replace, mask, remove
  ```
- [ ] Testes: `tests/unit/test_redact.py`
  - Validar detecção de CPF: "123.456.789-00"
  - Validar nomes próprios
  - Validar emails
- [ ] Benchmark: precision/recall
- [ ] Documentar em `docs/features/REDACT.md`
- **Acceptance**:
  - Precision > 95% (poucos falsos positivos)
  - Recall > 90% (detecta maioria das entidades)
  - Latência < 50ms adicional

**3.2. Diarização Híbrida (Streaming)** 🔬
**Prioridade**: EXPERIMENTAL (pesquisa)

**Problema**: WhisperX diarization é batch-only

**Solução**: Hybrid approach
- [ ] Implementar `server/diarization/hybrid.py`
  ```python
  class HybridDiarization:
      """
      Diarization para streaming via speaker embeddings

      Estratégia:
      1. Extrair embedding de cada chunk (pyannote)
      2. Comparar com speakers conhecidos (cosine similarity)
      3. Se novo speaker, adicionar ao cache
      4. Re-diarizar buffer acumulado a cada 30s
      """
      def __init__(self):
          self.speaker_embeddings = {}
          self.buffer = []

      async def process_chunk(self, audio_chunk, transcript):
          # Extrair embedding
          embedding = extract_speaker_embedding(audio_chunk)

          # Match com speakers existentes
          speaker_id = self.match_speaker(embedding, threshold=0.75)

          # Se não encontrou, novo speaker
          if not speaker_id:
              speaker_id = f"SPEAKER_{len(self.speaker_embeddings)}"
              self.speaker_embeddings[speaker_id] = embedding

          # Adicionar ao buffer
          self.buffer.append((audio_chunk, transcript, speaker_id))

          # Re-diarizar a cada 30s
          if self.should_re_diarize():
              await self.re_diarize_buffer()

          return speaker_id
  ```
- [ ] Pesquisar: Sortformer (WhisperLiveKit) como alternativa
- [ ] Testes com áudio multi-speaker
- [ ] Benchmark: DER vs. WhisperX batch
- [ ] Documentar em `docs/features/HYBRID_DIARIZATION.md`
- **Acceptance**:
  - DER < 20% (aceitável para streaming)
  - Latência < 3s
  - Funcional em tempo real

**3.3. Verbatim Transcription** ⭐
**Prioridade**: MÉDIA (nicho mas interessante)

- [ ] Validar CrisperWhisper em PT (se FASE 1 mostrou viabilidade)
- [ ] OU: Fine-tune Whisper para fillers PT
  - Dataset: Common Voice PT + manual annotation de fillers
  - Fillers PT: "né", "tipo", "então", "ehm", "ah"
- [ ] Integrar detecção de fillers no pipeline
- [ ] Configurável via capability
  ```yaml
  verbatim:
    enabled: true
    detect_fillers: true
    detect_stutters: true
  ```
- [ ] Testes
- [ ] Documentar caso de uso: "Transcrição jurídica, médica"
- **Acceptance**:
  - Detecta fillers comuns PT (> 80% recall)
  - Não degrada WER

**Deliverables**:
- ✅ Redact/PII funcional
- ✅ Diarização híbrida (mesmo que experimental)
- ✅ Verbatim se viável
- ✅ Documentação de features premium

**Riscos**:
- Diarização híbrida pode não atingir qualidade necessária
- Redact pode ter muitos falsos positivos em PT

---

### **FASE 4: Pesquisa e Experimentação** 🔬
**Duração**: 2-3 semanas (paralelo com outras fases)
**Objetivo**: Investigar tecnologias de ponta

#### Tasks:

**4.1. Kyutai STT - Delayed Streams Modeling** 🔬
**Objetivo**: Aprender arquitetura de streaming nativa

- [ ] Estudar paper: "Delayed Streams Modeling"
- [ ] Implementar backend experimental (EN/FR apenas)
  ```python
  # server/backends/kyutai_backend.py
  INFO = BackendInfo(
      name="kyutai-stt",
      supported_platforms={Platform.MACOS_APPLE_SILICON, Platform.LINUX_CUDA},
      capabilities={
          Capability.TRANSCRIPTION,
          Capability.WORD_TIMESTAMPS,
          Capability.STREAMING,  # ⭐ Streaming nativo
          Capability.VAD,  # ⭐ VAD semântico
      },
      supported_languages={"en", "fr"},  # PT não suportado ainda
      model_sizes={"1b", "2.6b"},
  )
  ```
- [ ] Testar com EN
- [ ] Benchmark: latência (deve ser ~0.5-2.5s)
- [ ] **Extrair insights** para aplicar em outros backends:
  - Como fazer chunking incremental
  - Como gerenciar context entre chunks
  - VAD semântico vs. VAD acústico
- [ ] Documentar aprendizados em `docs/research/KYUTAI_INSIGHTS.md`
- **Acceptance**:
  - Entender arquitetura Delayed Streams
  - Aplicar técnicas em backends PT

**4.2. Moshi - Full-Duplex** 🔬
**Objetivo**: Pesquisa (caso de uso diferente)

- [ ] Estudar arquitetura Moshi
- [ ] **Avaliar**: Vale implementar? (use case é diálogo, não transcrição)
- [ ] Se sim: protótipo experimental
- [ ] Documentar em `docs/research/MOSHI.md`
- **Decisão**: Implementar ou descartar

**4.3. Canary (NVIDIA)** 🔬
**Objetivo**: Avaliar para batch processing

- [ ] Implementar backend Canary (batch-only)
- [ ] Testar WER em PT (European vs. Brazilian)
- [ ] Comparar com Whisper Large-v3
- [ ] **Decisão**: Vale usar para batch? Ou Voxtral é suficiente?
- **Acceptance**:
  - Se WER melhor que Whisper: implementar
  - Se não: descartar

**4.4. Fine-tuning Customizado** 🔬
**Objetivo**: Modelos especializados PT-BR

- [ ] Pesquisar datasets PT-BR:
  - Common Voice PT
  - MLS (Multilingual LibriSpeech)
  - CORAA (Corpus of Annotated Audios)
- [ ] Fine-tune Whisper Medium/Large em PT-BR
- [ ] Fine-tune para domínios específicos:
  - Médico
  - Jurídico
  - Call center
- [ ] Benchmark vs. base model
- [ ] Documentar em `docs/research/FINE_TUNING.md`
- **Acceptance**:
  - WER melhoria > 10%
  - Modelo customizado para domínio

**Deliverables**:
- ✅ Insights de Kyutai aplicados
- ✅ Decisão sobre Moshi/Canary
- ✅ Fine-tuning PT-BR (se viável)
- ✅ Documentação de pesquisa

**Riscos**:
- Tempo gasto em pesquisa pode atrasar features
- Fine-tuning requer recursos computacionais

---

### **FASE 5: Produção e Escala** ⭐
**Duração**: 4-6 semanas
**Objetivo**: Production-ready, multi-tenant, escala

#### Tasks:

**5.1. Monitoring e Observability**
- [ ] Prometheus metrics
  - Transcription requests/s
  - Latency (p50, p95, p99)
  - WER (se ground truth disponível)
  - GPU utilization
  - Memory usage
- [ ] Grafana dashboards
- [ ] Alert system (PagerDuty, Slack)
- [ ] Health checks avançados
- [ ] Documentar em `docs/MONITORING.md`
- **Acceptance**: Dashboard funcional, alerts configurados

**5.2. Batching e Throughput**
- [ ] GPU batching (processar múltiplos clientes simultaneamente)
- [ ] Queue management (Redis/RabbitMQ)
- [ ] Load balancing (múltiplas GPUs)
- [ ] Rate limiting por cliente
- [ ] Documentar em `docs/SCALING.md`
- **Acceptance**: > 10 conexões simultâneas sem degradação

**5.3. Multi-tenancy e Auth**
- [ ] API key authentication
- [ ] Rate limiting por tenant
- [ ] Usage tracking
- [ ] Billing integration (opcional)
- [ ] Documentar em `docs/AUTH.md`
- **Acceptance**: Multi-tenant seguro

**5.4. Docker e CI/CD**
- [ ] Multi-stage Dockerfiles otimizados
  - `Dockerfile.cuda` (produção GPU)
  - `Dockerfile.cpu` (edge/dev)
  - `Dockerfile.mlx` (Mac deployment)
- [ ] Docker Compose para dev
- [ ] GitHub Actions CI/CD
  - Testes automáticos
  - Build de images
  - Push para Docker Hub
- [ ] Deployment automático (RunPod, Vast.ai)
- [ ] Documentar em `docs/CICD.md`
- **Acceptance**: Deploy < 5 minutos

**5.5. HTTP Endpoints (Batch)** 🔙
**Prioridade**: BAIXA (deixar por último)

- [ ] Implementar `server/http_handler.py`
  - `POST /transcribe` (sync)
  - `POST /transcribe/async` (job-based)
  - `GET /transcribe/status/{job_id}`
  - `POST /diarize` (WhisperX batch)
  - `POST /translate` (SeamlessM4T batch)
- [ ] Job queue (Celery + Redis)
- [ ] Webhook callbacks
- [ ] Testes: `tests/integration/test_http_endpoints.py`
- [ ] Documentar em `docs/HTTP_API.md`
- **Acceptance**: API REST funcional para batch

**Deliverables**:
- ✅ Sistema production-ready
- ✅ Monitoring completo
- ✅ CI/CD automatizado
- ✅ Multi-tenant seguro
- ✅ HTTP API (opcional)

---

## 📋 Plano de Testes Detalhado

### **Estrutura de Testes**

```
tests/
├── unit/                          # Testes unitários
│   ├── test_config.py            # ✅ Existente (17 tests)
│   ├── test_hardware_detector.py # ✅ Existente (14 tests)
│   ├── test_streaming_buffer.py  # ⏳ FASE 0
│   ├── test_vad_chunking.py      # ⏳ FASE 0
│   ├── test_local_agreement.py   # ⏳ FASE 0
│   ├── test_redact.py            # ⏳ FASE 3
│   ├── backends/
│   │   ├── test_faster_whisper_backend.py  # ⏳ FASE 1
│   │   ├── test_mlx_backend.py             # ⏳ FASE 1
│   │   ├── test_whisperx_backend.py        # ⏳ FASE 1
│   │   ├── test_crisper_backend.py         # ⏳ FASE 1
│   │   ├── test_voxtral_backend.py         # ⏳ FASE 2
│   │   ├── test_parakeet_backend.py        # ⏳ FASE 2
│   │   ├── test_granite_backend.py         # ⏳ FASE 2
│   │   └── test_seamless_backend.py        # ⏳ FASE 2
│   └── test_backend_registry.py  # ⏳ FASE 1
│
├── integration/                   # Testes de integração
│   ├── test_end_to_end.py        # ⏳ FASE 1
│   ├── test_websocket_streaming.py # ⏳ FASE 0
│   ├── test_multiple_clients.py  # ⏳ FASE 1
│   ├── test_long_audio.py        # ⏳ FASE 1
│   ├── test_http_endpoints.py    # ⏳ FASE 5
│   └── test_diarization_e2e.py   # ⏳ FASE 1 (RunPod)
│
├── benchmarks/                    # Benchmarks
│   ├── test_wer.py               # ⏳ FASE 1
│   ├── test_latency.py           # ⏳ FASE 1
│   ├── test_rtf.py               # ⏳ FASE 1
│   └── test_memory.py            # ⏳ FASE 1
│
└── fixtures/                      # Dados de teste
    ├── audio/
    │   ├── pt_clean_1min.wav
    │   ├── pt_noisy_30s.wav
    │   ├── pt_multispeaker_2min.wav
    │   └── en_clean_1min.wav
    └── expected/
        └── transcriptions.json
```

### **Coverage Goals**

| Módulo | Target Coverage | Status |
|--------|----------------|--------|
| `server/backends/` | 80% | ⏳ 0% |
| `server/streaming/` | 90% | ⏳ 0% (não existe) |
| `server/redact/` | 85% | ⏳ 0% (não existe) |
| `server/config.py` | 100% | ✅ 100% |
| `server/websocket_handler.py` | 75% | ⏳ 0% |
| `server/whisper_processor.py` | 80% | ⏳ 0% |
| **Overall** | **80%** | ⏳ ~15% |

### **Datasets de Teste**

**Common Voice PT (Brazilian)**
- Download: https://commonvoice.mozilla.org/pt
- Usage: WER benchmark, fine-tuning
- Splits: train, dev, test

**MLS (Multilingual LibriSpeech) PT**
- Download: https://www.openslr.org/94/
- Usage: Long-form audio tests

**Custom Test Set**
- Reunião 2-3 pessoas (diarization)
- Call center (ruído, sotaques)
- Médico/Jurídico (vocabulário específico)
- Áudio sintético (TTS) para casos específicos

---

## 🧪 Ideias Experimentais

### **1. Diarização Híbrida com Tracking de Embeddings**

**Conceito**: Combinar pyannote embeddings com tracking online

```python
# Pseudocódigo
class OnlineDiarization:
    def __init__(self):
        self.embeddings_cache = {}  # {speaker_id: embedding_history}
        self.current_speaker = None

    async def process_chunk(self, audio):
        # Extrair embedding do chunk
        embedding = extract_embedding(audio)

        # Comparar com histórico (weighted average dos últimos 5 chunks)
        similarities = {}
        for speaker_id, history in self.embeddings_cache.items():
            avg_embedding = np.mean(history[-5:], axis=0)
            sim = cosine_similarity(embedding, avg_embedding)
            similarities[speaker_id] = sim

        # Match se similaridade > threshold
        if similarities and max(similarities.values()) > 0.75:
            speaker_id = max(similarities, key=similarities.get)
        else:
            # Novo speaker
            speaker_id = f"SPEAKER_{len(self.embeddings_cache)}"

        # Atualizar histórico
        if speaker_id not in self.embeddings_cache:
            self.embeddings_cache[speaker_id] = []
        self.embeddings_cache[speaker_id].append(embedding)

        # Re-diarizar a cada 30s (refinamento)
        if self.should_refine():
            await self.refine_diarization()

        return speaker_id
```

**Vantagens**:
- Funciona em streaming
- Baixa latência (~100ms adicional)

**Desvantagens**:
- Menos preciso que pyannote batch
- Precisa tuning de threshold

---

### **2. Ensemble de Modelos**

**Conceito**: Combinar outputs de múltiplos modelos para melhor accuracy

```python
class EnsembleBackend:
    def __init__(self):
        self.backends = [
            VoxtralBackend(model="24b"),
            FasterWhisperBackend(model="large-v3"),
            GraniteBackend(),
        ]

    async def transcribe_chunk(self, audio):
        # Transcrever com todos
        results = await asyncio.gather(*[
            b.transcribe_chunk(audio) for b in self.backends
        ])

        # Voting ou WER-weighted average
        final_text = self.combine_results(results)

        return TranscriptionResult(text=final_text, ...)

    def combine_results(self, results):
        # Estratégia 1: Voting (palavra mais comum)
        # Estratégia 2: WER-weighted (backend com melhor WER histórico)
        # Estratégia 3: ROVER (Recognizer Output Voting Error Reduction)
        pass
```

**Casos de uso**:
- Aplicações críticas (médico, jurídico)
- Quando latência não é problema

---

### **3. Active Learning para Fine-tuning Contínuo**

**Conceito**: Coletar feedback do usuário e re-treinar modelo

```python
class ActiveLearning:
    def __init__(self):
        self.correction_buffer = []

    async def collect_correction(self, audio, predicted, corrected):
        """Usuário corrige transcrição"""
        self.correction_buffer.append({
            "audio": audio,
            "predicted": predicted,
            "corrected": corrected,
        })

        # A cada 1000 correções, re-treinar
        if len(self.correction_buffer) >= 1000:
            await self.trigger_finetuning()

    async def trigger_finetuning(self):
        # Fine-tune Whisper com dados corrigidos
        # Deploy novo modelo
        # Limpar buffer
        pass
```

**Vantagens**:
- Melhora contínua
- Adaptação a domínio específico

**Desvantagens**:
- Complexo de implementar
- Requer infraestrutura de ML

---

### **4. Code-Switching Detection**

**Conceito**: Detectar quando usuário muda de idioma mid-sentence

```python
class CodeSwitchingDetector:
    async def detect_language_change(self, segments):
        """
        Detecta mudanças de idioma entre segmentos

        Exemplo:
        - Segment 1: "Bom dia, tudo bem?" (PT)
        - Segment 2: "Yes, everything is fine" (EN)
        - Segment 3: "Ótimo, vamos começar" (PT)
        """
        for i in range(len(segments) - 1):
            lang1 = detect_language(segments[i].text)
            lang2 = detect_language(segments[i+1].text)

            if lang1 != lang2:
                # Marcar code-switching
                segments[i+1].is_code_switching = True
                segments[i+1].previous_language = lang1
```

**Use case**: Reuniões internacionais, call centers bilíngues

---

## 📊 Métricas de Sucesso

### **Técnicas**

| Métrica | Target | Atual | Benchmark (Deepgram) |
|---------|--------|-------|----------------------|
| **WER (PT)** | < 8% | ❓ | 6.84% |
| **WER (EN)** | < 7% | ❓ | 5-6% |
| **Latência (streaming)** | < 2s | ❓ | 0.2-0.3x RTF |
| **RTF (batch)** | > 10x | ❓ | ~20x |
| **DER (diarization)** | < 15% | ❓ | N/A |
| **Uptime** | > 99% | ❓ | 99.9% |
| **Coverage (testes)** | > 80% | 15% | N/A |

### **Produto**

| Métrica | Target | Atual |
|---------|--------|-------|
| **Backends implementados** | 8+ | 5 |
| **Idiomas suportados (full)** | PT + EN | PT + EN (99 parcial) |
| **Features premium** | 3+ | 0 |
| **Documentação completa** | 100% | 60% |
| **Deploy time** | < 5 min | ❓ |

### **Negócio**

| Métrica | Target | Notas |
|---------|--------|-------|
| **Custo por minuto** | < $0.001 | RunPod: ~$0.30/hr = $0.0001/min |
| **Self-hosted viável** | ✅ | Docker + GPU |
| **LGPD compliant** | ✅ | 100% local |
| **Competitive advantage** | 3+ features | Translation, Redact, PT-optimized |

---

## 🚨 Riscos e Mitigações

| Risco | Probabilidade | Impacto | Mitigação |
|-------|--------------|---------|-----------|
| **WhisperX diarization não funciona em PT** | Média | Alto | Plano B: Hybrid diarization |
| **Voxtral bugs (recém-lançado)** | Alta | Médio | Manter faster-whisper como fallback |
| **RunPod custo alto** | Baixa | Médio | Usar spot instances, monitorar custos |
| **LocalAgreement difícil de tune** | Média | Médio | Iteração com diferentes valores de n |
| **SeamlessM4T muito lento** | Média | Baixo | Chunks maiores, ou descartar |
| **Falta de tempo para tudo** | Alta | Alto | **Priorizar fases 0-2** |

---

## 📅 Timeline Estimado

### **Sprint 1-2 (Semanas 1-4): FUNDAÇÃO**
- FASE 0: Streaming inteligente
- FASE 1: Validação e testes
- **Entrega**: Streaming funcional, WhisperX validado, benchmarks

### **Sprint 3-4 (Semanas 5-8): NOVOS MODELOS**
- FASE 2: Voxtral, Parakeet, Granite, SeamlessM4T
- **Entrega**: 4 novos backends, benchmarks comparativos

### **Sprint 5-6 (Semanas 9-12): FEATURES PREMIUM**
- FASE 3: Redact, Diarização híbrida, Verbatim
- **Entrega**: Features empresariais funcionais

### **Sprint 7-8 (Semanas 13-16): PRODUÇÃO**
- FASE 4: Pesquisa (paralelo)
- FASE 5: Monitoring, CI/CD, multi-tenancy
- **Entrega**: Sistema production-ready

### **Total**: 16 semanas (~4 meses)

---

## 🎯 Prioridades Absolutas (Não Negociáveis)

1. ✅ **FASE 0**: Streaming inteligente com LocalAgreement
2. ✅ **FASE 1**: WhisperX validado no RunPod
3. ✅ **FASE 2.1**: Backend Voxtral
4. ✅ **FASE 3.1**: Redact/PII
5. ✅ **Testes**: Coverage > 80%

**Tudo o resto é nice-to-have.**

---

## 📚 Documentação a Criar

- [ ] `docs/STREAMING.md` - Algoritmo de streaming inteligente
- [ ] `docs/BENCHMARKS.md` - Resultados comparativos
- [ ] `docs/RUNPOD_VALIDATION.md` - Validação WhisperX
- [ ] `docs/backends/VOXTRAL.md` - Guia Voxtral
- [ ] `docs/backends/PARAKEET.md` - Guia Parakeet
- [ ] `docs/backends/GRANITE.md` - Guia Granite
- [ ] `docs/backends/SEAMLESS.md` - Guia SeamlessM4T
- [ ] `docs/features/REDACT.md` - Redact/PII
- [ ] `docs/features/HYBRID_DIARIZATION.md` - Diarização híbrida
- [ ] `docs/research/KYUTAI_INSIGHTS.md` - Aprendizados Kyutai
- [ ] `docs/research/FINE_TUNING.md` - Fine-tuning PT-BR
- [ ] `docs/MONITORING.md` - Observability
- [ ] `docs/SCALING.md` - Scaling strategy
- [ ] `docs/CICD.md` - CI/CD pipeline
- [ ] `docs/HTTP_API.md` - HTTP endpoints (Fase 5)

---

## 🎓 Notas para o Próximo Desenvolvedor

### **Contexto**

Este projeto visa competir com Deepgram/AssemblyAI oferecendo:
1. **Self-hosted** (100% controle dos dados)
2. **Open-source** (zero vendor lock-in)
3. **PT-optimizado** (foco em português brasileiro)
4. **Features únicas**: Translation, Redact, LGPD compliance
5. **Custo**: 10-100x mais barato

### **Arquitetura Atual (2025-10-18)**

✅ **Funcionando**:
- Backend Registry com platform detection
- 5 backends implementados (faster-whisper, MLX, WhisperX, Crisper, CPU/CUDA)
- WebSocket server (aiohttp)
- Cliente Bun de exemplo
- Refatorações recentes: SRP, performance, type safety

⚠️ **Gaps Críticos**:
- **Streaming não é real-time** (falta LocalAgreement)
- **WhisperX não validado** (precisa testar diarization em PT)
- **Sem testes de backends** (coverage ~15%)
- **Sem benchmarks** (não sabemos WER/latência)

### **Prioridades**

Se você tem **1 semana**:
→ Implementar LocalAgreement (FASE 0.1)

Se você tem **1 mês**:
→ FASE 0 + FASE 1 completas

Se você tem **3 meses**:
→ FASE 0-2 (incluindo Voxtral)

Se você tem **6 meses**:
→ FASE 0-5 completa

### **Tecnologias Chave 2025**

- **Voxtral**: Melhor que Whisper, PT nativo, streaming
- **Kyutai STT**: Streaming nativo (inspiração arquitetural)
- **WhisperX**: Melhor diarization (mas batch-only)
- **SeamlessM4T**: Translation (feature única)

### **Benchmarks Meta**

- WER < 8% (PT) para ser competitivo
- Latência < 2s (streaming) para boa UX
- DER < 15% (diarization) para ser útil

---

**Última atualização**: 2025-10-18
**Status**: Roadmap completo, pronto para implementação
**Próximo passo**: FASE 0.1 - Implementar StreamingBuffer

🤖 Generated with [Claude Code](https://claude.com/claude-code)
