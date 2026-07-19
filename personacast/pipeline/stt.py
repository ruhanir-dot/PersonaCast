"""
speech to text transcriving the spoken reaction through the same gemini live api 
listener record reaction clip through streamlit `st.audio_input` mic, convert to 16kHz mono PCM, which is the live apis required input format 
send clip over live session, at end of audio read back model input audio trnascription

    - probably a better way to do this doing this for now 
"""
from __future__ import annotations

import asyncio
import audioop 
import wave
import io
from .. import config

def _to_pcm16k_mono(wav_bytes: bytes) -> bytes:
    """
    Convert the recorded wav  to 16kHz mono 16bit PCM
    """

    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        n_channels, sample_width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        frames = w.readframes(w.getnframes())
    
    if sample_width != 2: # check for 16 bit
        frames = audioop.lin2lin(frames, sample_width, 2)
        sample_width = 2
    if n_channels == 2: # check for mono
        frames = audioop.tomono(frames, 2, 0.5, 0.5)
    if rate != config.STT_INPUT_SAMPLE_RATE: # 16kHz 
        frames, _ = audioop.ratecv(frames, 2, 1, rate, config.STT_INPUT_SAMPLE_RATE, None)
    
    return frames

async def _transcribe_live(pcm:bytes) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=config.STT_API_KEY) # setup client
    ## live model oly supports audio output but we just read back the input audio trnascription of what user said 

    live_config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        input_audio_transcription=types.AudioTranscriptionConfig(),
    )

    parts: list[str] = []
    async with client.aio.live.connect(model=config.STT_MODEL, config=live_config) as session:
        await session.send_realtime_input(
            audio=types.Blob(data=pcm, mime_type=f"audio/pcm;rate={config.STT_INPUT_SAMPLE_RATE}")
        )
        await session.send_realtime_input(audio_stream_end=True)   # finalize the clip
        async for response in session.receive():
            server = response.server_content
            if server is None:
                continue
            if server.input_transcription and server.input_transcription.text:
                parts.append(server.input_transcription.text)
            if server.turn_complete:
                break
    return "".join(parts).strip()

def transcribe(wav_bytes: bytes) -> str:
    """
    transcribe recorded reaction WAV clip to text
    """
    if not wav_bytes:
        return ""
    pcm = _to_pcm16k_mono(wav_bytes)
    return asyncio.run(_transcribe_live(pcm))