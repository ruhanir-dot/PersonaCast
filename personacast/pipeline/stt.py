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
import numpy as np

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


_model = None

def _get_model(): 
   """
   load whisper model once per process and reuse it same cache in TTS as well
   """
   global _model

   if _model is None:
    from faster_whisper import WhisperModel
    _model = WhisperModel(config.STT_MODEL_SIZE, device="cpu", compute_type="int8")

   return _model 

def _transcribe_whisper(pcm: bytes) -> str: 
    """
    transcribe 16kHz mono int16 PCM locally
    """
    audio = np.frombuffer(pcm, dtype = np.int16).astype(np.float32) / 32768 # turn butes into 16bit signed integers --> 32 bit float --> normalize to -1 to 1 

    if audio.size == 0:
            return ""

    segments, _info = _get_model().transcribe(audio, language= 'en') # load in cached whispermodel and get text from wav

    return "".join(segment.text for segment in segments).strip()



def transcribe(wav_bytes: bytes) -> str:
    """
    transcribe recorded reaction WAV clip to text
    """
    if not wav_bytes:
        return ""
    return _transcribe_whisper(_to_pcm16k_mono(wav_bytes))