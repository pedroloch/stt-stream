# 🧪 Guia de Testes - Validação das Correções

**Data**: 2025-10-18
**Objetivo**: Validar correções do LocalAgreement e StreamingBuffer

---

## 📋 Pré-requisitos

1. **Servidor configurado**:
```bash
# Verificar server-config.example.yaml
buffer_trimming_sec: 15.0  # ✅ Deve estar presente
vad_threshold: 0.5         # ✅ Aumentado de 0.25
```

2. **Dependências instaladas**:
```bash
# Backend Python
uv pip install -e .

# Cliente sandbox
cd sandbox && bun install
```

3. **Modelo baixado**:
```bash
# O modelo será baixado automaticamente na primeira execução
# Recomendado para testes: small ou base (mais rápido)
```

---

## 🧪 Teste 1: Unit Tests (HypothesisBuffer)

**Objetivo**: Validar implementação do HypothesisBuffer

```bash
# Rodar testes unitários
uv run pytest tests/unit/test_hypothesis_buffer.py -v

# Output esperado:
test_init PASSED
test_insert_filters_old_words PASSED
test_insert_with_offset PASSED
test_flush_no_agreement PASSED
test_flush_with_full_agreement PASSED
test_flush_with_partial_agreement PASSED
test_flush_with_divergence PASSED
test_pop_commited PASSED
test_remove_duplicates_1gram PASSED
test_remove_duplicates_2gram PASSED
test_reset PASSED
test_complete PASSED
test_get_stats PASSED
test_realistic_streaming_scenario PASSED

======================== 14 passed in 0.5s ========================
```

**✅ Critério de sucesso**: Todos os testes passando

---

## 🧪 Teste 2: Servidor Básico

**Objetivo**: Validar que servidor inicia sem erros

```bash
# Terminal 1: Iniciar servidor com logs detalhados
python -m server.main --config server-config.example.yaml --log-level debug

# Output esperado:
==============================================================
🎤 WHISPER STREAM SERVER
==============================================================
✅ Modelo faster-whisper carregado
HypothesisBuffer inicializado
StreamingBuffer inicializado: min_chunk=1.0s, buffer_trimming=segment, trim_threshold=15.0s
✅ Servidor rodando em ws://0.0.0.0:9090
Health check: http://0.0.0.0:9090/health
WebSocket: ws://0.0.0.0:9090/ws
==============================================================
Modelo: small
Idioma: pt
Backend: auto
==============================================================
Servidor pronto para receber conexões!
```

**✅ Critérios de sucesso**:
- [x] Sem erros de import
- [x] HypothesisBuffer inicializado
- [x] `trim_threshold=15.0s` presente nos logs
- [x] Servidor rodando na porta 9090

---

## 🧪 Teste 3: Frase Original ("Olá, tudo bem...")

**Objetivo**: Validar que a frase original NÃO produz alucinações

**Frase de teste**:
```
"Olá, tudo bem por aí? Você gosta de xadrez? eu gosto bastante"
```

### Passos:

```bash
# Terminal 1: Servidor com debug
python -m server.main --config server-config.example.yaml --log-level debug

# Terminal 2: Cliente com debug
cd sandbox && DEBUG=1 bun start
```

### Output Esperado:

**Cliente (sandbox):**
```
🎤 WHISPER STREAM CLIENT
Server: ws://localhost:9090/ws | Debug: ON
✅ Conectado | 🎙️ Áudio Ativo
═══════════════════════════════════════════════════════════

[HH:MM:SS] 🎤 Olá, tudo bem por aí?                    (85%)

[HH:MM:SS] Você gosta de xadrez?                       (92%)

[HH:MM:SS] eu gosto bastante                           (88%)
```

**Servidor (logs):**
```
[DEBUG] Chunk #1 adicionado: 1.00s, buffer total: 1.00s, buffer_offset: 0.00s
[DEBUG] Transcrevendo buffer: 1.20s (offset: 0.00s)
[DEBUG] Extraídas 5 palavras com timestamps
[DEBUG] LocalAgreement: 0/2 histórico
[DEBUG] ⏳ Parcial: 'Olá, tudo bem'

[DEBUG] Chunk #2 adicionado: 1.00s, buffer total: 2.00s, buffer_offset: 0.00s
[DEBUG] Transcrevendo buffer: 2.10s (offset: 0.00s)
[DEBUG] Extraídas 8 palavras com timestamps
[INFO]  ✅ LocalAgreement confirmou 3 palavras: 'Olá, tudo bem'

[DEBUG] Chunk #3 adicionado: 1.00s, buffer total: 3.00s, buffer_offset: 0.00s
[DEBUG] Transcrevendo buffer: 3.20s (offset: 0.00s)
[DEBUG] Extraídas 12 palavras com timestamps
[INFO]  ✅ LocalAgreement confirmou 4 palavras: 'por aí? Você gosta'

# ... continua ...
```

### ❌ BUGS A EVITAR:

**1. "Muito obrigado" repetindo:**
```
# ❌ NÃO deve aparecer:
[HH:MM:SS] Muito obrigado                              (19%)
[HH:MM:SS] Muito obrigado                              (19%)
[HH:MM:SS] Muito obrigado                              (19%)
# (loop infinito)
```

**2. Timestamps resetando:**
```
# ❌ NÃO deve acontecer:
{"segments": [{"start": 0.0, "end": 0.18, ...}]}
{"segments": [{"start": 0.0, "end": 0.18, ...}]}
# (sempre 0.0 após cada chunk)
```

**3. Texto cortado:**
```
# ❌ NÃO deve aparecer:
"to bastante"  # Ao invés de "gosto bastante"
```

---

## 🧪 Teste 4: Silêncio (Anti-Alucinação)

**Objetivo**: Validar que silêncio NÃO produz transcrições

### Passos:

1. Iniciar servidor + cliente
2. **NÃO falar** por 10 segundos
3. Observar logs

### Output Esperado:

**Cliente:**
```
🎤 WHISPER STREAM CLIENT
═══════════════════════════════════════════════════════════
Aguardando transcrições... Fale no microfone para começar.


# ✅ NADA deve aparecer durante silêncio!
```

**Servidor (logs):**
```
[DEBUG] Chunk #1 adicionado: 1.00s, buffer total: 1.00s, buffer_offset: 0.00s
[DEBUG] Transcrevendo buffer: 1.00s (offset: 0.00s)
[DEBUG] Pulando segmento de silêncio (no_speech_prob=0.98): ''
[DEBUG] Transcrição vazia

# ✅ Sem transcrições durante silêncio!
```

### ❌ BUGS A EVITAR:

```
# ❌ NÃO deve aparecer durante silêncio:
[HH:MM:SS] Muito obrigado                              (19%)
[HH:MM:SS] Obrigado                                    (15%)
# (alucinações)
```

---

## 🧪 Teste 5: Buffer Trimming Conservador

**Objetivo**: Validar que buffer NÃO é trimmed agressivamente

### Passos:

1. Iniciar servidor com `--log-level debug`
2. Falar frase longa (> 10 palavras)
3. Observar logs de trimming

### Output Esperado:

```
[DEBUG] Buffer pequeno (3.50s < 15.0s), não trimming ainda
[DEBUG] Buffer pequeno (7.20s < 15.0s), não trimming ainda
[DEBUG] Buffer pequeno (12.80s < 15.0s), não trimming ainda

# Apenas quando buffer > 15s:
[DEBUG] Buffer grande (16.20s > 15.0s), fazendo trim
[INFO]  Buffer trimmed: removido 3.50s, restante 12.70s, buffer_offset agora: 3.50s
```

### ❌ BUGS A EVITAR:

```
# ❌ NÃO deve aparecer após CADA confirmação:
[INFO]  Buffer completamente trimmed
# (trim agressivo - causa alucinações!)
```

---

## 🧪 Teste 6: buffer_time_offset Incrementando

**Objetivo**: Validar que timestamps são absolutos

### Passos:

1. Servidor com `--log-level debug`
2. Falar frase longa continuamente
3. Observar `buffer_offset` nos logs

### Output Esperado:

```
[DEBUG] buffer total: 2.50s, buffer_offset: 0.00s
[DEBUG] buffer total: 5.20s, buffer_offset: 0.00s
# ... buffer cresce ...

# Após trim:
[INFO]  buffer_offset agora: 3.50s  # ✅ Incrementou!

# Próximas iterações:
[DEBUG] buffer total: 12.00s, buffer_offset: 3.50s
[DEBUG] buffer total: 14.50s, buffer_offset: 3.50s

# Após novo trim:
[INFO]  buffer_offset agora: 7.20s  # ✅ Incrementou novamente!
```

### ❌ BUGS A EVITAR:

```
# ❌ NÃO deve ficar sempre 0.0:
buffer_offset: 0.00s
buffer_offset: 0.00s
buffer_offset: 0.00s
# (timestamps relativos - bug!)
```

---

## 🧪 Teste 7: Context Window (Scrolled Away)

**Objetivo**: Validar que context usa apenas texto scrolled away

### Passos:

1. Servidor com `--log-level debug`
2. Falar frase longa (> 20 palavras) continuamente
3. Observar logs de "Prompt"

### Output Esperado:

```
# Antes do trim (sem texto scrolled away):
[DEBUG] Prompt (0 chars): '' (0 palavras scrolled away)

# Após trim (texto scrolled away):
[DEBUG] Prompt (87 chars): 'Olá, tudo bem por aí? Você gosta...' (12 palavras scrolled away)

# Após mais trim:
[DEBUG] Prompt (142 chars): '...gosta de xadrez? eu gosto bastante...' (23 palavras scrolled away)
```

### ❌ BUGS A EVITAR:

```
# ❌ NÃO deve ter muitas palavras SEM trim:
[DEBUG] Prompt (200 chars): '... (100 palavras scrolled away)'
# Mas buffer_offset ainda é 0.0s
# (context desalinhado - causa WER alto!)
```

---

## 📊 Resumo dos Critérios de Sucesso

### ✅ Checklist Final

- [ ] Unit tests passando (14/14)
- [ ] Servidor inicia sem erros
- [ ] Frase original transcrita corretamente
- [ ] SEM "Muito obrigado" ou loops
- [ ] SEM timestamps resetando para 0.0
- [ ] SEM texto cortado
- [ ] Silêncio NÃO produz transcrições
- [ ] Buffer trimming > 15s (conservador)
- [ ] `buffer_time_offset` incrementa após trim
- [ ] Context usa apenas scrolled away words
- [ ] Logs mostram "✅ Confirmado (N palavras)"

### 📈 Métricas Esperadas

| Métrica | Antes (Bugado) | Depois (Corrigido) |
|---------|---------------|-------------------|
| **Alucinações** | Frequentes | Nenhuma |
| **Loops infinitos** | Sim ("Muito obrigado") | Não |
| **Timestamps corretos** | Não (sempre 0.0) | Sim (absolutos) |
| **Texto completo** | Não (cortado) | Sim |
| **WER** | Alto | Normal |
| **Latência** | < 2s | < 2s (igual) |

---

## 🐛 Troubleshooting

Se encontrar problemas, consulte:
- `docs/STREAMING.md` - Seção "Troubleshooting - Bugs Comuns"
- `server/streaming/hypothesis_buffer.py` - Implementação de referência
- `tests/unit/test_hypothesis_buffer.py` - Casos de teste

---

**Última atualização**: 2025-10-18
**Versão**: 1.0

🤖 Generated with [Claude Code](https://claude.com/claude-code)
