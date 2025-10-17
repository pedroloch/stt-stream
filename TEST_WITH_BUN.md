# 🎙️ Como Testar Distil-Whisper com Cliente Bun

Guia rápido para testar o novo sistema com word timestamps!

---

## 🚀 Opção 1: Teste Rápido (Standalone)

```bash
# Testar backend diretamente
python test_distil_whisper.py
```

**O que faz:**
- ✅ Lista backends disponíveis
- ✅ Cria backend faster-whisper
- ✅ Inicializa modelo
- ✅ Testa com áudio dummy
- ✅ Mostra word timestamps (se houver fala)

---

## 🎙️ Opção 2: Servidor WebSocket + Cliente Bun

### Passo 1: Iniciar Servidor

```bash
# Terminal 1
python server_distil.py
```

**Você verá:**
```
============================================================
🎙️  Iniciando servidor com Faster-Whisper
============================================================
Criando backend faster-whisper...
Inicializando backend (pode demorar na primeira vez)...
✅ Backend pronto!
   Modelo: base
   Capabilities: word_timestamps, vad, streaming
============================================================
✅ Servidor rodando em ws://localhost:9090/ws
   Health check: http://localhost:9090/health
============================================================
```

### Passo 2: Iniciar Cliente Bun

```bash
# Terminal 2
bun start
```

**O cliente vai:**
- Conectar ao servidor
- Capturar áudio do microfone
- Enviar para o servidor
- Receber transcrições com word timestamps!

---

## ⚡ Upgrade para Distil-Large-v3 (6x mais rápido!)

### Editar `server_distil.py`

Linha 34, mudar:
```python
"model": "base",  # ← Mudar isso
```

Para:
```python
"model": "distil-large-v3",  # ⚡ 6x mais rápido!
```

**Benefícios:**
- 🚀 6x mais rápido que large-v3
- ✅ Mesma qualidade
- ✅ Menos alucinações
- ✅ Funciona no Mac!

---

## 📊 Ver Word Timestamps no Cliente

### Atual (texto simples)
```
Olá mundo
```

### Novo (com word timestamps) ⭐
```json
{
  "type": "transcription",
  "text": "Olá mundo",
  "is_final": true,
  "confidence": 0.95,
  "segments": [
    {
      "start": 0.0,
      "end": 1.5,
      "text": "Olá mundo",
      "words": [
        {"word": "Olá", "start": 0.0, "end": 0.5, "probability": 0.99},
        {"word": "mundo", "start": 0.6, "end": 1.5, "probability": 0.98}
      ]
    }
  ]
}
```

### Para renderizar no cliente Bun

**Editar `src/index.ts`:**

```typescript
// Adicionar após receber mensagem
interface Word {
  word: string;
  start: number;
  end: number;
  probability: number;
}

function renderTranscription(msg: any) {
  // Se tem words, mostrar palavra por palavra
  if (msg.segments?.[0]?.words) {
    console.log('\n📝 Transcrição com word timestamps:');
    for (const word of msg.segments[0].words) {
      const color = word.probability > 0.9 ? chalk.green : chalk.yellow;
      const time = `[${word.start.toFixed(2)}s-${word.end.toFixed(2)}s]`;
      console.log(`  ${color(word.word)} ${chalk.gray(time)}`);
    }
  } else {
    // Fallback: texto normal
    console.log(chalk.green(msg.text));
  }
}
```

---

## 🔍 Testar Health Check

```bash
curl http://localhost:9090/health
```

**Resposta:**
```json
{
  "status": "healthy",
  "backend": "faster-whisper",
  "model": "base",
  "initialized": true
}
```

---

## 🐛 Troubleshooting

### "Backend não inicializado"
- Aguardar alguns segundos (download do modelo na primeira vez)
- Verificar logs do servidor

### "Nenhuma transcrição"
- Verificar se microfone está funcionando
- Tentar falar mais alto/claro
- VAD threshold pode estar muito alto (ajustar `vad_threshold`)

### "Muito lento"
- ✅ Trocar para `distil-large-v3`
- ✅ Verificar que está usando `model: "base"` no mínimo
- ❌ Evitar `large-v3` no Mac (muito pesado)

---

## 📈 Comparação de Performance

| Modelo | Velocidade | Qualidade | Recomendado |
|--------|-----------|-----------|-------------|
| tiny | 🚀🚀🚀🚀🚀 | ⭐⭐ | Dev rápido |
| base | 🚀🚀🚀🚀 | ⭐⭐⭐ | ✅ Inicial |
| small | 🚀🚀🚀 | ⭐⭐⭐⭐ | Boa qualidade |
| medium | 🚀🚀 | ⭐⭐⭐⭐⭐ | Alta qualidade |
| **distil-large-v3** | 🚀🚀🚀🚀 | ⭐⭐⭐⭐⭐ | ⭐ **MELHOR** |
| large-v3 | 🚀 | ⭐⭐⭐⭐⭐ | Muito lento |

---

## ✅ Checklist

- [ ] Servidor iniciado (`python server_distil.py`)
- [ ] Health check OK (`curl localhost:9090/health`)
- [ ] Cliente Bun conectado (`bun start`)
- [ ] Microfone capturando
- [ ] Transcrições aparecendo
- [ ] (Opcional) Upgrade para distil-large-v3
- [ ] (Opcional) Atualizar cliente para mostrar word timestamps

---

## 🎉 Pronto!

Agora você tem:
- ✅ Servidor com faster-whisper
- ✅ Word timestamps funcionando!
- ✅ VAD (detecta silêncio)
- ✅ Pronto para upgrade para distil (6x mais rápido)

**Próximo passo:** Trocar para `distil-large-v3` e ver a diferença! ⚡
