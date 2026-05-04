"""
TTS (Text-To-Speech) generation -- gTTS → MP3 → WAV.

Caches generated files by MD5 hash of (text, lang) under static/voice_alerts/.
"""
import hashlib
import logging
import os
import subprocess
from typing import Optional

from utils import APP_ROOT

_LOG = logging.getLogger(__name__)

VOICE_DIR = os.path.join(APP_ROOT, "static", "voice_alerts")


def generate_mp3_to_static(text: str, lang: str = "zh-CN") -> Optional[str]:
    """Generate MP3 and return URL path for frontend playback."""
    mp3_path = _generate_mp3(text, lang)
    if mp3_path is None:
        return None
    return f"/static/voice_alerts/{os.path.basename(mp3_path)}"


def _generate_mp3(text: str, lang: str = "zh-CN") -> Optional[str]:
    """Generate MP3 file via gTTS and return local path."""
    try:
        from gtts import gTTS
    except ImportError:
        return None

    os.makedirs(VOICE_DIR, exist_ok=True)
    cache_key = hashlib.md5(f"{text}_{lang}".encode()).hexdigest()
    mp3_path = os.path.join(VOICE_DIR, f"alert_{cache_key}.mp3")
    if not os.path.exists(mp3_path):
        try:
            tts = gTTS(text=text, lang=lang[:2])
            tts.save(mp3_path)
        except Exception:
            return None
    return mp3_path


def generate_wav(text: str, lang: str = "zh-CN") -> Optional[str]:
    """Generate WAV file via gTTS + GStreamer and return local path.

    Returns None on failure.
    """
    mp3_path = _generate_mp3(text, lang)
    if mp3_path is None:
        return None

    wav_path = mp3_path.replace(".mp3", ".wav")
    if os.path.exists(wav_path):
        return wav_path

    try:
        subprocess.run(
            ["gst-launch-1.0", "-q",
             "filesrc", f"location={mp3_path}",
             "!", "decodebin",
             "!", "audioconvert",
             "!", "wavenc",
             "!", "filesink", f"location={wav_path}"],
            timeout=30,
        )
        if os.path.exists(wav_path):
            return wav_path
    except Exception:
        return None

    return None
