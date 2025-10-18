# 🔬 Comparação Completa de Modelos STT (2025)

**Data**: 2025-10-18
**Objetivo**: Matriz de decisão para seleção de modelos

---

## 📊 Matriz Comparativa - Visão Geral

### **Legenda**

| Símbolo | Significado |
|---------|-------------|
| ⭐⭐⭐ | Excelente / State-of-the-art |
| ⭐⭐ | Bom / Competitivo |
| ⭐ | Aceitável / Básico |
| ❌ | Não suportado |
| ❓ | Desconhecido / Não testado |
| 🔬 | Experimental / Pesquisa |

---

## 🎯 Modelos por Prioridade de Implementação

### **PRIORIDADE 1: Implementar AGORA** ⭐⭐⭐

#### **1. Voxtral (Mistral AI)**

| Categoria | Avaliação | Detalhes |
|-----------|-----------|----------|
| **WER (EN)** | ⭐⭐⭐ | ~5-6% (melhor que Whisper Large-v3) |
| **WER (PT)** | ⭐⭐⭐ | Suporte nativo PT (multilingual) |
| **Streaming** | ⭐⭐⭐ | API dedicada para streaming |
| **Latência** | ⭐⭐ | Boa (não especificado, mas otimizado) |
| **Velocidade** | ⭐⭐ | Boa (não especificado RTF) |
| **Word Timestamps** | ⭐⭐⭐ | Sim |
| **Diarization** | ❌ | Não (precisa adicionar separado) |
| **Translation** | ❌ | Não (apenas transcrição) |
| **Licença** | ⭐⭐⭐ | Apache 2.0 (comercial OK) |
| **Plataformas** | ⭐⭐⭐ | M1, CUDA, CPU (ambos os modelos) |
| **Tamanhos** | ⭐⭐⭐ | 3B (edge), 24B (produção) |
| **Idiomas** | ⭐⭐ | 8 idiomas (PT, EN, ES, FR, DE, NL, IT, HI) |
| **Maturidade** | ⭐ | Recém-lançado (julho 2025) - pode ter bugs |

**Casos de Uso**:
- ✅ Transcrição streaming PT/EN
- ✅ Deploy em M1 (Voxtral 3B)
- ✅ Deploy em GPU (Voxtral 24B)
- ✅ Alternativa superior ao Whisper Large-v3

**Diferenciais**:
- 🏆 **Melhor WER que Whisper** (state-of-the-art open)
- 🏆 **PT nativo** (não é fine-tune)
- 🏆 **Streaming API** dedicada
- 🏆 **Apache 2.0** (pode revender)

**Implementação**: FASE 2.1 (2-3 dias)

---

#### **2. SeamlessM4T v2 (Meta)**

| Categoria | Avaliação | Detalhes |
|-----------|-----------|----------|
| **WER (EN)** | ⭐⭐ | Competitivo mas não SOTA |
| **WER (PT)** | ⭐⭐ | Suporte nativo multilingual |
| **Streaming** | ⭐⭐ | SeamlessStreaming (EMMA) ou chunks maiores |
| **Latência** | ⭐ | ~0.2x RTF mas para translation pode ser 3-5s |
| **Velocidade** | ⭐⭐ | Boa para o que faz |
| **Word Timestamps** | ⭐ | Limitado |
| **Diarization** | ❌ | Não |
| **Translation** | ⭐⭐⭐ | **Sim! (feature única)** |
| **Licença** | ⭐ | CC-BY-NC 4.0 (não comercial!) |
| **Plataformas** | ⭐ | CUDA principalmente (modelo pesado) |
| **Tamanhos** | ⭐ | Large (~10GB) |
| **Idiomas** | ⭐⭐⭐ | 100+ idiomas |
| **Maturidade** | ⭐⭐⭐ | Maduro (Meta) |

**Casos de Uso**:
- ✅ **Reuniões multilíngues** (PT→EN, PT→ES simultâneo)
- ✅ Call center internacional
- ✅ Tradução de legendas
- ❌ Não para revenda (licença NC)

**Diferenciais**:
- 🏆 **Translation simultânea** (ninguém mais tem)
- 🏆 **100+ idiomas**
- ⚠️ **Licença não comercial**

**Implementação**: FASE 2.4 (3-5 dias)

---

### **PRIORIDADE 2: Implementar DEPOIS** ⭐⭐

#### **3. Parakeet TDT 0.6B (NVIDIA)**

| Categoria | Avaliação | Detalhes |
|-----------|-----------|----------|
| **WER (EN)** | ⭐⭐⭐ | 6.05% (topo do leaderboard HF) |
| **WER (PT)** | ❓ | Suportado mas sem benchmark específico |
| **Streaming** | ⭐⭐⭐ | Sim (streaming nativo) |
| **Latência** | ⭐⭐ | Boa |
| **Velocidade** | ⭐⭐⭐ | **RTF 3380x** (extremamente rápido!) |
| **Word Timestamps** | ❓ | Não confirmado |
| **Diarization** | ❌ | Não |
| **Translation** | ❌ | Não |
| **Licença** | ⭐⭐⭐ | Apache 2.0 |
| **Plataformas** | ⭐⭐ | CUDA, CPU (pequeno - 600M params) |
| **Tamanhos** | ⭐⭐⭐ | 0.6B (muito leve!) |
| **Idiomas** | ⭐⭐⭐ | 25 idiomas europeus (incluindo PT) |
| **Maturidade** | ⭐⭐⭐ | Maduro (NVIDIA NeMo) |

**Casos de Uso**:
- ✅ **Deploy CPU** (modelo pequeno)
- ✅ **Edge devices** (RPi, Jetson)
- ✅ **Batch processing rápido** (56 min de áudio em 1s)
- ✅ Fallback rápido para M1/CPU

**Diferenciais**:
- 🏆 **Velocidade absurda** (3380x RTF)
- 🏆 **Modelo pequeno** (roda em CPU)
- ⚠️ **PT não é foco principal** (benchmarks são EN)

**Implementação**: FASE 2.2 (2-3 dias)

---

#### **4. IBM Granite Speech 8B**

| Categoria | Avaliação | Detalhes |
|-----------|-----------|----------|
| **WER (EN)** | ⭐⭐⭐ | ~5.85% (topo do leaderboard HF) |
| **WER (PT)** | ⭐⭐ | Suportado + translation |
| **Streaming** | ⭐ | Batch principalmente |
| **Latência** | ❓ | Não especificado |
| **Velocidade** | ❓ | Não especificado |
| **Word Timestamps** | ⭐⭐⭐ | Sim |
| **Diarization** | ❌ | Não |
| **Translation** | ⭐⭐⭐ | **Sim! (PT→EN, PT→ES, etc)** |
| **Licença** | ⭐⭐⭐ | Apache 2.0 |
| **Plataformas** | ⭐ | CUDA (modelo grande - 8B) |
| **Tamanhos** | ⭐ | 8B |
| **Idiomas** | ⭐⭐ | EN, FR, ES, IT, DE, PT, JA, ZH |
| **Maturidade** | ⭐⭐⭐ | Maduro (IBM Research) |

**Casos de Uso**:
- ✅ **Batch transcription** de alta qualidade
- ✅ **Translation** (alternativa ao SeamlessM4T com licença comercial)
- ✅ Enterprise deployments

**Diferenciais**:
- 🏆 **Topo do leaderboard HF**
- 🏆 **Translation + Apache 2.0** (vs. SeamlessM4T NC)
- ⚠️ **Modelo grande** (8B params)

**Implementação**: FASE 2.3 (2-3 dias)

---

### **PRIORIDADE 3: Pesquisa e Experimental** 🔬

#### **5. Kyutai STT (Delayed Streams Modeling)**

| Categoria | Avaliação | Detalhes |
|-----------|-----------|----------|
| **WER (EN)** | ⭐⭐ | Competitivo (não SOTA) |
| **WER (PT)** | ❌ | **Não suportado** (EN/FR apenas) |
| **Streaming** | ⭐⭐⭐ | **Streaming nativo** (arquitetura DSM) |
| **Latência** | ⭐⭐⭐ | **0.5s (1B) / 2.5s (2.6B)** |
| **Velocidade** | ⭐⭐⭐ | RTF 3x (64 conexões simultâneas) |
| **Word Timestamps** | ⭐⭐⭐ | Sim |
| **Diarization** | ❌ | Não |
| **Translation** | ❌ | Não |
| **Licença** | ⭐⭐⭐ | CC-BY 4.0 |
| **Plataformas** | ⭐⭐⭐ | M1 (MLX), CUDA, Rust |
| **Tamanhos** | ⭐⭐⭐ | 1B, 2.6B |
| **Idiomas** | ❌ | **EN, FR apenas** |
| **Maturidade** | ⭐⭐ | Recente (2025) |

**Casos de Uso**:
- 🔬 **Pesquisa** de arquitetura streaming
- 🔬 **Aprendizado** de técnicas DSM
- ❌ **Não usar para PT** (sem suporte)

**Diferenciais**:
- 🏆 **Arquitetura streaming nativa** (inspiração!)
- 🏆 **Latência ultra-baixa** (0.5s)
- 🏆 **VAD semântico** integrado
- ⚠️ **Sem PT**

**Implementação**: FASE 4.1 (pesquisa - 1 semana)

---

#### **6. Canary 1B v2 (NVIDIA)**

| Categoria | Avaliação | Detalhes |
|-----------|-----------|----------|
| **WER (EN)** | ⭐⭐⭐ | 5.63% (topo do leaderboard) |
| **WER (PT)** | ⭐⭐ | Suportado (European PT nos dados) |
| **Streaming** | ❌ | **Batch-only** |
| **Latência** | N/A | Batch |
| **Velocidade** | ⭐⭐ | Boa para batch |
| **Word Timestamps** | ⭐⭐⭐ | Sim |
| **Diarization** | ❌ | Não |
| **Translation** | ⭐⭐⭐ | **Sim! (multilingual AST)** |
| **Licença** | ⭐ | CC-BY-NC 4.0 (não comercial) |
| **Plataformas** | ⭐⭐ | CUDA principalmente |
| **Tamanhos** | ⭐⭐⭐ | 1B |
| **Idiomas** | ⭐⭐⭐ | 25 idiomas europeus (incluindo PT) |
| **Maturidade** | ⭐⭐⭐ | Maduro (NVIDIA NeMo) |

**Casos de Uso**:
- ✅ **Batch transcription** de altíssima qualidade
- ✅ **Translation** (AST - Any-to-Any)
- ❌ **Não para streaming**
- ❌ **Não comercial**

**Diferenciais**:
- 🏆 **Melhor WER** (5.63%)
- 🏆 **Translation multilingual**
- ⚠️ **Sem streaming**
- ⚠️ **Licença NC**

**Implementação**: FASE 4.3 (pesquisa - avaliar vs. Voxtral)

---

#### **7. Moshi (Kyutai)**

| Categoria | Avaliação | Detalhes |
|-----------|-----------|----------|
| **WER (EN)** | ❓ | Não é foco (é modelo de diálogo) |
| **WER (PT)** | ❌ | Não suportado |
| **Streaming** | ⭐⭐⭐ | **Full-duplex** (160ms latência) |
| **Latência** | ⭐⭐⭐ | **160ms teórico, 200ms prático** |
| **Velocidade** | ⭐⭐⭐ | Real-time |
| **Word Timestamps** | ❌ | Não (foco em diálogo) |
| **Diarization** | ❌ | Não |
| **Translation** | ❌ | Não |
| **Licença** | ⭐⭐⭐ | CC-BY 4.0 |
| **Plataformas** | ⭐⭐⭐ | M1, CUDA |
| **Tamanhos** | ⭐⭐ | ~7B (modelo LLM) |
| **Idiomas** | ❌ | EN, FR apenas |
| **Maturidade** | ⭐ | Experimental (2024) |

**Casos de Uso**:
- 🔬 **Pesquisa** de diálogo full-duplex
- 🔬 **Voice assistants** (fala enquanto escuta)
- ❌ **Não para transcrição pura**

**Diferenciais**:
- 🏆 **Full-duplex** (único!)
- 🏆 **Latência ultra-baixa**
- ⚠️ **Caso de uso diferente** (diálogo, não STT)

**Implementação**: FASE 4.2 (pesquisa - avaliar viabilidade)

---

### **PRIORIDADE 4: Manter Existentes** ✅

#### **8. Faster-Whisper** (ATUAL)

| Categoria | Avaliação | Detalhes |
|-----------|-----------|----------|
| **WER (EN)** | ⭐⭐ | ~7-8% (Whisper Large-v3) |
| **WER (PT)** | ⭐⭐ | ~8-10% (99 idiomas) |
| **Streaming** | ⭐⭐ | Sim (chunk-by-chunk) |
| **Latência** | ⭐⭐ | Boa (~1-2s) |
| **Velocidade** | ⭐⭐⭐ | Distil: 6x mais rápido |
| **Word Timestamps** | ⭐⭐⭐ | Sim (word-level) |
| **Diarization** | ❌ | Não (usar WhisperX) |
| **Translation** | ❌ | Não |
| **Licença** | ⭐⭐⭐ | MIT |
| **Plataformas** | ⭐⭐⭐ | M1, CUDA, CPU |
| **Tamanhos** | ⭐⭐⭐ | tiny, base, medium, large, distil-large-v3 |
| **Idiomas** | ⭐⭐⭐ | 99 idiomas |
| **Maturidade** | ⭐⭐⭐ | Muito maduro |

**Status**: ✅ **Implementado e funcionando**

**Manter porque**:
- ✅ Universal (funciona em tudo)
- ✅ Maduro e estável
- ✅ Fallback confiável
- ✅ Distil-large-v3 muito rápido

---

#### **9. WhisperX** (ATUAL - NÃO VALIDADO)

| Categoria | Avaliação | Detalhes |
|-----------|-----------|----------|
| **WER (EN)** | ⭐⭐ | ~7-8% (base Whisper) |
| **WER (PT)** | ❓ | Não testado ainda |
| **Streaming** | ⭐ | **Batch-only para diarization** |
| **Latência** | ⭐⭐ | ~70x RT (batch) |
| **Velocidade** | ⭐⭐⭐ | 70x real-time |
| **Word Timestamps** | ⭐⭐⭐ | **Precisão superior** (wav2vec2) |
| **Diarization** | ⭐⭐⭐ | **Sim! (pyannote)** |
| **Translation** | ❌ | Não |
| **Licença** | ⭐⭐ | BSD-4 (permissivo) |
| **Plataformas** | ⭐ | **CUDA apenas** |
| **Tamanhos** | ⭐⭐⭐ | tiny, base, medium, large |
| **Idiomas** | ⭐⭐⭐ | 99 idiomas |
| **Maturidade** | ⭐⭐⭐ | Maduro |

**Status**: ⚠️ **Implementado mas NÃO validado**

**Validar ASAP** (FASE 1):
- ⚠️ Testar diarization em PT
- ⚠️ Medir DER
- ⚠️ Deploy no RunPod

**Manter porque**:
- ✅ **Única opção de diarization open-source**
- ✅ Word timestamps mais precisos
- ⚠️ Batch-only (não streaming real)

---

#### **10. MLX Whisper** (ATUAL)

| Categoria | Avaliação | Detalhes |
|-----------|-----------|----------|
| **WER (EN)** | ⭐⭐ | ~7-8% (Whisper) |
| **WER (PT)** | ⭐⭐ | ~8-10% |
| **Streaming** | ⭐⭐ | Sim |
| **Latência** | ⭐⭐⭐ | Excelente (M1 otimizado) |
| **Velocidade** | ⭐⭐⭐ | Muito rápido (Apple Silicon) |
| **Word Timestamps** | ⭐⭐⭐ | Sim |
| **Diarization** | ❌ | Não |
| **Translation** | ❌ | Não |
| **Licença** | ⭐⭐⭐ | MIT |
| **Plataformas** | ⭐⭐ | **M1 apenas** |
| **Tamanhos** | ⭐⭐⭐ | tiny, base, medium, large |
| **Idiomas** | ⭐⭐⭐ | 99 idiomas |
| **Maturidade** | ⭐⭐⭐ | Maduro (Apple MLX) |

**Status**: ✅ **Implementado e funcionando**

**Manter porque**:
- ✅ Melhor opção para Mac M1
- ✅ Otimizado para Apple Silicon
- ✅ Rápido e eficiente

---

### **JÁ DISPONÍVEL (via faster-whisper)** ✅

#### **Distil-Whisper Large-v3**

**Status**: ✅ **FUNCIONA EM PT** (testado e confirmado)

**CORREÇÃO IMPORTANTE**:
- ✅ Distil-Whisper **FUNCIONA em PT** mesmo treinado só em inglês
- ✅ Herda pesos do Whisper Large-v3 (multilingual)
- ✅ WER ~10-12% PT (um pouco pior que Large-v3)
- ✅ **6x mais rápido** que Large-v3
- ✅ Já disponível via faster-whisper (`model: distil-large-v3`)

**Recomendação**:
```yaml
# Usar quando velocidade > accuracy absoluta
whisper:
  backend: faster-whisper
  model: distil-large-v3  # ⭐ Recomendado para streaming
  language: pt
```

**Trade-off**:
- Accuracy: ⭐⭐⭐ (WER ~2-3% pior que Large-v3)
- Velocidade: ⭐⭐⭐⭐⭐ (6x mais rápido)
- **Decisão**: Vale a pena para streaming real-time!

---

#### **Deepgram Nova-3, AssemblyAI Slam-1**

**Motivo**: ❌ **Closed-source**

- Usar apenas como **benchmarks**
- Não implementar (não é open-source)

---

## 🎯 Matriz de Decisão por Caso de Uso

### **Caso de Uso 1: Streaming PT/EN (prioridade #1)**

| Modelo | Score | Motivo |
|--------|-------|--------|
| **Voxtral 24B** | ⭐⭐⭐⭐⭐ | Melhor WER, streaming nativo, PT nativo |
| Faster-Whisper | ⭐⭐⭐⭐ | Funciona, maduro, fallback confiável |
| Voxtral 3B | ⭐⭐⭐ | Mais rápido, mas WER um pouco pior |
| MLX Whisper | ⭐⭐⭐ | Excelente para M1 |

**Recomendação**: **Voxtral 24B** (CUDA) ou **Voxtral 3B** (M1/CPU)

---

### **Caso de Uso 2: Diarization (speaker identification)**

| Modelo | Score | Motivo |
|--------|-------|--------|
| **WhisperX** | ⭐⭐⭐⭐ | Única opção open-source, precisa validar PT |
| Hybrid Diarization | ⭐⭐⭐ | Experimental, para streaming |
| Sortformer | 🔬 | Pesquisar (mencionado mas não open ainda) |

**Recomendação**: **WhisperX** (batch) + **Hybrid approach** (streaming experimental)

---

### **Caso de Uso 3: Translation (PT→EN, PT→ES)**

| Modelo | Score | Motivo |
|--------|-------|--------|
| **IBM Granite** | ⭐⭐⭐⭐⭐ | Translation + Apache 2.0 (comercial OK) |
| SeamlessM4T | ⭐⭐⭐⭐ | Melhor translation, mas NC (não comercial) |
| Canary | ⭐⭐⭐ | Batch-only, NC |

**Recomendação**: **IBM Granite** (comercial) ou **SeamlessM4T** (não comercial)

---

### **Caso de Uso 4: Edge Devices / CPU**

| Modelo | Score | Motivo |
|--------|-------|--------|
| **Parakeet 0.6B** | ⭐⭐⭐⭐⭐ | Muito rápido, modelo pequeno |
| Voxtral 3B | ⭐⭐⭐⭐ | Pequeno, bom WER |
| Faster-Whisper tiny | ⭐⭐⭐ | Funciona, mas WER alto |

**Recomendação**: **Parakeet 0.6B** (velocidade) ou **Voxtral 3B** (qualidade)

---

### **Caso de Uso 5: Batch Processing (arquivos grandes)**

| Modelo | Score | Motivo |
|--------|-------|--------|
| **Canary 1B** | ⭐⭐⭐⭐⭐ | Melhor WER (5.63%), mas NC |
| IBM Granite | ⭐⭐⭐⭐⭐ | WER 5.85%, Apache 2.0 |
| Voxtral 24B | ⭐⭐⭐⭐ | WER ~5-6%, streaming também |
| WhisperX | ⭐⭐⭐⭐ | Se precisar diarization |

**Recomendação**: **IBM Granite** (comercial) ou **Canary** (não comercial)

---

## 📊 Score Total por Modelo

| Ranking | Modelo | Score | Comentário |
|---------|--------|-------|------------|
| 🥇 | **Voxtral 24B** | 95/100 | Melhor all-around, IMPLEMENTAR |
| 🥈 | **IBM Granite** | 90/100 | Translation + comercial |
| 🥉 | **WhisperX** | 85/100 | Diarization única, VALIDAR |
| 4 | Faster-Whisper | 80/100 | Maduro, confiável, MANTER |
| 5 | Voxtral 3B | 80/100 | Edge/M1, IMPLEMENTAR |
| 6 | SeamlessM4T | 75/100 | Translation melhor, mas NC |
| 7 | Parakeet | 75/100 | Velocidade incrível |
| 8 | MLX Whisper | 70/100 | Excelente para M1, MANTER |
| 9 | Canary | 70/100 | Batch-only, NC |
| 10 | Kyutai STT | 50/100 | Pesquisa, sem PT |
| 11 | Moshi | 40/100 | Caso de uso diferente |

---

## 🚀 Plano de Ação Final

### **Implementar Imediatamente** (FASE 2)

1. ✅ **Voxtral 24B + 3B** (3-5 dias)
2. ✅ **SeamlessM4T** (3-5 dias)
3. ✅ **IBM Granite** (2-3 dias)
4. ✅ **Parakeet** (2-3 dias)

### **Validar Existentes** (FASE 1)

1. ⚠️ **WhisperX** - testar diarization PT (RunPod)
2. ✅ **Faster-Whisper** - benchmark WER/latência
3. ✅ **MLX** - benchmark M1

### **Pesquisar** (FASE 4)

1. 🔬 **Kyutai STT** - aprender arquitetura DSM
2. 🔬 **Canary** - avaliar vs. Granite/Voxtral
3. 🔬 **Moshi** - avaliar viabilidade

### **Descartar**

1. ❌ Distil-Whisper standalone (usar via faster-whisper)
2. ❌ Deepgram/AssemblyAI (benchmark apenas)

---

**Status**: Roadmap de modelos completo
**Próximo**: Implementar FASE 0 (streaming), depois FASE 2 (Voxtral primeiro)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
