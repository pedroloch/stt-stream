"""
Batch API Handler - POST /v1/transcribe

Endpoint para transcrição batch de arquivos de áudio
Compatible com OpenAI Whisper-1 API
"""

import logging
import tempfile
from pathlib import Path

from aiohttp import web

from ..config import Config
from ..models.batch import ErrorResponse, TranscribeRequest, TranscribeResponse
from ..processors.batch_processor import (
    AudioTooLargeError,
    BatchProcessor,
    BatchProcessorError,
)

logger = logging.getLogger(__name__)


async def handle_batch_transcribe(request: web.Request) -> web.Response:
    """
    POST /v1/transcribe - Batch transcription endpoint

    Request: multipart/form-data
        - file: arquivo de áudio (obrigatório)
        - model: backend (default: "faster-whisper")
        - language: idioma (default: "pt")
        - task: "transcribe" | "translate" (default: "transcribe")
        - response_format: formato de saída (default: "verbose_json")
        - ... outros parâmetros (ver TranscribeRequest)

    Response: JSON
        - 200 OK: TranscribeResponse
        - 400 Bad Request: Validação falhou
        - 413 Payload Too Large: Arquivo muito grande
        - 500 Internal Server Error: Erro de processamento

    Example:
        curl -X POST http://localhost:9090/v1/transcribe \
          -F "file=@audio.mp3" \
          -F "model=faster-whisper" \
          -F "language=pt"
    """
    try:
        # 1. Parse multipart form
        logger.debug("Recebendo request batch...")

        reader = await request.multipart()

        file_data = None
        file_name = "audio"
        params = {}

        async for field in reader:
            if field.name == "file":
                # Ler arquivo
                file_name = field.filename or "audio"
                file_data = await field.read()
                logger.info(f"Arquivo recebido: {file_name} ({len(file_data)} bytes)")
            else:
                # Outros parâmetros
                value = await field.text()
                params[field.name] = value

        # 2. Validar que file foi enviado
        if not file_data:
            error = ErrorResponse(
                error="missing_file",
                message="Campo 'file' é obrigatório",
                details={"hint": "Use -F 'file=@audio.mp3' no curl"}
            )
            return web.json_response(error.model_dump(), status=400)

        # Validar tamanho
        file_size_mb = len(file_data) / (1024 * 1024)
        if file_size_mb > BatchProcessor.MAX_FILE_SIZE_MB:
            error = ErrorResponse(
                error="file_too_large",
                message=f"Arquivo muito grande: {file_size_mb:.1f}MB",
                details={"max_size_mb": BatchProcessor.MAX_FILE_SIZE_MB}
            )
            return web.json_response(error.model_dump(), status=413)

        # 3. Criar TranscribeRequest a partir de params
        # Converter strings para tipos corretos
        params_typed = _parse_params(params)
        transcribe_req = TranscribeRequest(**params_typed)

        logger.info(f"Parâmetros: model={transcribe_req.model}, language={transcribe_req.language}")

        # 4. Salvar arquivo temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_name).suffix) as tmp_file:
            tmp_file.write(file_data)
            tmp_path = Path(tmp_file.name)

        try:
            # 5. Processar com BatchProcessor
            config: Config = request.app["config"]
            processor = BatchProcessor(config)

            result: TranscribeResponse = await processor.process(
                audio_file=tmp_path,
                request=transcribe_req
            )

            # 6. Retornar resposta
            # Pydantic model serializa automaticamente para JSON
            # mode='json' converte datetime para string ISO
            return web.json_response(result.model_dump(mode='json'), status=200)

        finally:
            # Limpar arquivo temporário
            tmp_path.unlink(missing_ok=True)

    except AudioTooLargeError as e:
        logger.warning(f"Áudio muito longo: {e}")
        error = ErrorResponse(
            error="audio_too_long",
            message=str(e),
            details={"max_duration_sec": BatchProcessor.MAX_DURATION_SEC}
        )
        return web.json_response(error.model_dump(), status=413)

    except BatchProcessorError as e:
        logger.error(f"Erro de processamento: {e}")
        error = ErrorResponse(
            error="processing_error",
            message=str(e)
        )
        return web.json_response(error.model_dump(), status=500)

    except Exception as e:
        logger.error(f"Erro inesperado: {e}", exc_info=True)
        error = ErrorResponse(
            error="internal_error",
            message=f"Erro interno: {e}"
        )
        return web.json_response(error.model_dump(), status=500)


def _parse_params(params: dict[str, str]) -> dict:
    """
    Converte parâmetros string para tipos corretos

    Args:
        params: Parâmetros do form (todos strings)

    Returns:
        Dict com tipos corretos (bool, int, float)
    """
    result = {}

    # Mapeamento de conversões
    bool_fields = {
        "enable_diarization",
        "return_metrics",
    }

    int_fields = {
        "num_speakers",
    }

    float_fields = {
        "temperature",
        "no_speech_threshold",
        "compression_ratio_threshold",
    }

    for key, value in params.items():
        # Booleanos
        if key in bool_fields:
            result[key] = value.lower() in ("true", "1", "yes")
        # Integers
        elif key in int_fields:
            try:
                result[key] = int(value) if value else None
            except ValueError:
                result[key] = None
        # Floats
        elif key in float_fields:
            try:
                result[key] = float(value)
            except ValueError:
                result[key] = None
        # Strings (default)
        else:
            result[key] = value

    return result
