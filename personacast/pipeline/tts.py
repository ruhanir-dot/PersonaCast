from __future__ import annotations

import re
from pathlib import Path

from .. import config

def _chunk(text, max_chars) -> list[str]:
    """
    split within less thank max character chunks on paragraph, and then sentence    
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


def synthesize(script: str, out_path: str | Path) -> Path:
    """
    stream wroitten audio chunk by chunk to output file, interrupted run has partial file
    """
    import soundfile as sf
    from kokoro_onnx import Kokoro

    # config
    out_path = Path(out_path)
    kokoro = Kokoro(config.KOKORO_MODEL_PATH, config.KOKORO_VOICES_PATH)
    chunks = _chunk(script, config.TTS_CHUNK_CHARS)

    # get the first chunk to learn the sample rate, then open the file and stream the rest
    first, sample_rate = kokoro.create(chunks[0], voice=config.TTS_VOICE, speed=1.0, lang="en-us")

    with sf.SoundFile(str(out_path), mode="w", samplerate=sample_rate, channels=1) as f:
        f.write(first)
        print(f"  tts chunk 1/{len(chunks)} -> {len(first)/sample_rate:.1f}s", flush=True)
        for i, chunk in enumerate(chunks[1:], 2):
            samples, _ = kokoro.create(chunk, voice=config.TTS_VOICE, speed=1.0, lang="en-us")
            f.write(samples)
            print(f"  tts chunk {i}/{len(chunks)} -> {len(samples)/sample_rate:.1f}s", flush=True)
   
    return out_path