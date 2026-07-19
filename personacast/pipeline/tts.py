"""
text to speech using gemini live api through google-genai sdk 
replaces the old local kokora backend, model using: gemini-3.1-flash-live-preview

we synthesize as a .wav 

the live api is a streaming websocket session through async 
    - open connection 
    - send generated turn text
    - collect streamed audio chunks (16kHz 16-bit mono PCM)

- later we could stream text token by toke into persistent socket for realtime audio

"""

from __future__ import annotations

import asyncio
import re
import wave
from pathlib import Path

from .. import config

### prommpt for TTS model, so it reads as narrotor and not react to it or answer
_NARRATOR_SYSTEM = (
    "You are a text-to-speech narrator. Read the user's text aloud EXACTLY as written, "
    "verbatim, with natural spoken delivery. Do not greet, summarize, answer, add commentary, "
    "or say anything that is not in the provided text."
)


def _chunk(text, max_chars) -> list[str]:
    """
    split within less than max character chunks on paragraph, and then sentence    
    """
    chunks = [] # list of strings

    for paragraph in re.split(r"\n\s*\n", text.strip()): # split  document into indivdual paragraphs 
        paragraph = paragraph.strip() # remove trailing spaces
        if not paragraph: # if empty skip and move to next paragraph
            continue

        if len(paragraph) <= max_chars: # if paragraph under max character limit save to chunk list
            chunks.append(paragraph)
            continue

        buffer = "" # initializes a temperorary string to accumulate sentences

        for sentence in re.split(r"(?<=[.!?])\s+", paragraph): # split long paragraph
            
            if len(buffer) + len(sentence) + 1 > max_chars and buffer: #  if i add space and next sentence to buffer will it blow past max char limit
                chunks.append(buffer.strip()) # if adding sentence exceeds limit current buffer full 
                buffer = sentence # clear buffer start new chunk using sentence that didnt fit
            else:
                buffer = f"{buffer} {sentence}".strip() #if fits appends space and strip spaces
        
        if buffer:
            chunks.append(buffer.strip())# ensure text saved
    
    return chunks


async def _synthesize_live(chunks: list[str], out_path: Path):
    """
    open live session, send each text chunk, stream generated audio back into wav
    """

    from google import genai 
    from google.genai import types

    client = genai.Client(api_key= config.TTS_API_KEY)
    live_config = types.LiveConnectConfig( # set up config
        response_modalities= ['AUDIO'],
        system_instruction=types.Content(parts=[types.Part(text=_NARRATOR_SYSTEM)]),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=config.TTS_VOICE)
            )
        ),
    )

    with wave.open(str(out_path), "wb") as wav: 
        wav.setnchannels(1) # setting mono
        wav.setsampwidth(2) # 16 bit 2 bytes per sample 
        wav.setframerate(config.TTS_SAMPLE_RATE) # 16kHz

        async with client.aio.live.connect(model= config.TTS_MODEL, config=live_config) as session:
            for i, chunk in enumerate(chunks,1): 
                await session.send_realtime_input(text = chunk)
                got = 0 # byte counter 

                async for response in session.receive(): # read audio till turn complete 
                    if response.data: 
                        wav.writeframes(response.data)
                        got += len(response.data)
                    server = response.server_content
                    if server is not None and server.turn_complete:
                        break
                seconds = got / 2 / config.TTS_SAMPLE_RATE
                print(f"  tts chunk {i}/{len(chunks)} -> {seconds:.1f}s", flush=True)




def synthesize(script: str, out_path: str | Path) -> Path:
    """
    stream written audio chunk by chunk to output file, interrupted run has partial file
    """
    # config
    out_path = Path(out_path)
    chunks = _chunk(script, config.TTS_CHUNK_CHARS)
    asyncio.run(_synthesize_live(chunks, out_path))
    return out_path