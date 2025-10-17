#!/usr/bin/env python
"""
Script de teste para distil-whisper com word timestamps

Testa o backend faster-whisper com modelo distil-large-v3 (6x mais rapido!)
"""

import asyncio
import numpy as np
from server.backends import BackendRegistry
from server.models.capability import Capability

async def main():
    print("=" * 60)
    print("🎙️  TESTE: Distil-Whisper com Word Timestamps")
    print("=" * 60)

    # 1. Listar backends disponiveis
    print("\n1. Backends disponiveis no Mac:")
    available = BackendRegistry.list_available()
    for name, info in available.items():
        caps = [c.value for c in info.capabilities]
        print(f"   ✅ {name}: {', '.join(caps)}")

    # 2. Criar backend faster-whisper com distil-large-v3
    print("\n2. Criando backend faster-whisper...")
    print("   Modelo: distil-large-v3 (6x mais rapido!)")

    backend = BackendRegistry.create("faster-whisper", {
        "model": "base",  # Comece com base, depois mude para distil-large-v3
        "language": "pt",
        "enable_word_timestamps": True,
        "enable_vad": True,
        "vad_threshold": 0.75,
    })

    # 3. Verificar capabilities
    print("\n3. Capabilities do backend:")
    for cap in backend.info.capabilities:
        icon = "⭐" if cap == Capability.WORD_TIMESTAMPS else "✅"
        print(f"   {icon} {cap.value}")

    # 4. Inicializar (vai baixar o modelo se necessario)
    print("\n4. Inicializando backend...")
    print("   (Primeira vez pode demorar - download do modelo)")
    await backend.initialize()
    print("   ✅ Backend inicializado!")

    # 5. Testar com audio dummy (na vida real seria audio do microfone)
    print("\n5. Testando transcricao...")
    print("   Gerando audio dummy (1 segundo)...")

    # Audio dummy - 1 segundo de ruido branco
    audio = np.random.randn(16000).astype(np.float32)
    audio = audio / np.abs(audio).max()  # Normalizar

    print("   Transcrevendo...")
    result = await backend.transcribe_chunk(audio)

    # 6. Mostrar resultado
    print("\n6. RESULTADO:")
    print(f"   Texto: '{result.text}'")
    print(f"   Final: {result.is_final}")
    print(f"   Confianca: {result.confidence:.2f}")
    print(f"   Idioma: {result.language}")

    # 7. Word timestamps
    if result.segments:
        print(f"\n7. WORD TIMESTAMPS (⭐ Novidade!):")
        print(f"   Total de segmentos: {len(result.segments)}")

        for i, segment in enumerate(result.segments):
            print(f"\n   Segmento {i+1}:")
            print(f"     Tempo: {segment.start:.2f}s - {segment.end:.2f}s")
            print(f"     Texto: '{segment.text}'")

            if segment.words:
                print(f"     Palavras ({len(segment.words)}):")
                for word in segment.words[:5]:  # Mostrar primeiras 5
                    print(f"       - '{word.word}': {word.start:.2f}s - {word.end:.2f}s (conf: {word.probability:.2f})")
                if len(segment.words) > 5:
                    print(f"       ... e mais {len(segment.words) - 5} palavras")
            else:
                print("     (Sem word timestamps)")
    else:
        print("\n7. Nenhum segmento retornado (audio dummy nao tem fala)")

    # 8. Cleanup
    await backend.cleanup()

    print("\n" + "=" * 60)
    print("✅ TESTE COMPLETO!")
    print("=" * 60)
    print("\nPROXIMOS PASSOS:")
    print("1. Trocar 'base' por 'distil-large-v3' para 6x mais rapido")
    print("2. Conectar ao servidor WebSocket real")
    print("3. Usar audio real do microfone")
    print("4. Testar com o cliente Bun")

if __name__ == "__main__":
    asyncio.run(main())
