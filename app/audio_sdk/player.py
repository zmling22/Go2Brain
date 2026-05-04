"""
Voice alert playback -- Go2 speaker (megaphone) with local ALSA fallback.

Playback priority:
1. Go2 robot speaker (via audio_sdk.sdk / megaphone)
2. Local ALSA device (via aplay)

Never blocks the caller -- all playback runs in daemon threads.
"""
import logging
import os
import threading
import wave
from typing import Optional

from audio_sdk import sdk as audio_sdk
from audio_sdk.tts import generate_wav, generate_mp3_to_static  # noqa: F401  (re-exported for convenience)

_LOG = logging.getLogger(__name__)


def play_alert(text: str, lang: str = "zh-CN") -> None:
    """Play a TTS alert through the best available speaker.

    Tries Go2 speaker first, then local ALSA. Never blocks the caller.
    """
    thread = threading.Thread(
        target=_play_alert_sync, args=(text, lang), daemon=True,
    )
    thread.start()


def _play_alert_sync(text: str, lang: str = "zh-CN") -> None:
    """Synchronous alert playback (runs in daemon thread)."""
    wav = generate_wav(text, lang)
    if wav is None:
        return

    # 1. Try Go2 megaphone (ensure SDK init first)
    try:
        if not audio_sdk.is_available():
            audio_sdk.init_audio_client()

        if audio_sdk.is_available():
            with wave.open(wav, "r") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / rate if rate > 0 else 3.0

            ok = audio_sdk.play_wav_via_megaphone(wav, duration)
            if ok:
                return
    except Exception:
        _LOG.debug("Go2 playback unavailable, falling back to local")

    # 2. Fallback: local ALSA
    try:
        import subprocess
        subprocess.run(["aplay", wav], timeout=15)
    except Exception:
        _LOG.warning("Local audio playback failed")
