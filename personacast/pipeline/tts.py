"""
text to speech, using local piper onnx
replaces old kokoro, and gemini live api
"""

from __future__ import annotations

import wave
from pathlib import Path

from .. import config

_voice = None # cache slot so outlives a single call

def _get_voice(): 
    """
    load piper voice once per process and reuse
    """

    global _voice # sets _voice as a module level variable instead of a function only variable 

    if _voice is None: # first run is the expensive of importing the voice and cache
        from piper import PiperVoice # imported in function so importing this module is cheap if onot needed 

        path = Path(config.PIPER_VOICE_PATH) # take config string into a path for here voice loves 
        if not path.exists(): # check if path exists if not raise error
            raise RuntimeError(
                    f"piper voice not found at {path}\n"
                    f"download it once with:\n"
                    f'  export SSL_CERT_FILE=$(python -c "import certifi;print(certifi.where())")\n'
                    f"  python -m piper.download_voices en_US-lessac-low\n"
                    f"then move the .onnx and .onnx.json into {path.parent}/"
            )

        _voice = PiperVoice.load(str(path)) # set voice variable to PiperVoice load at path

    return _voice 

def synthesize(script: str, out_path: str | Path) -> Path:
    """
    render script to a wav at a given out_path, no chunking
    """

    out_path = Path(out_path)
    voice = _get_voice() # get voice

    with wave.open(str(out_path), "wb") as wav: # initialize file object
        voice.synthesize_wav(script, wav) # given script write into wav

    return out_path


def wav_duration(path: str| Path) -> float: 
    """ 
    playback length of the wav file, use to map audio pause time to a certain sentence 
    """
    with wave.open(str(path), "rb") as w: # read wav file w/ open rb mode
        rate = w.getframerate() # get audio file framerate, how many audio frames recorded pers second 
        return w.getnframes() / float(rate) if rate else 0.0 # total number of frames in file/ by frames per second to get seconds of playback 

