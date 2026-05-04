"""
Go2 robot audio SDK -- self-contained AudioHubClient initialisation.

Follows the pattern from go2_dashboard/app.py:
  DDSChannelFactoryInitialize → create_standard_sdk → create_robot → AudioHubClient
"""
import logging
import os
import threading
import time
from typing import Optional

_LOG = logging.getLogger(__name__)

_audio_client: Optional[object] = None
_init_lock = threading.Lock()
_init_done = False


def init_audio_client(serial: Optional[str] = None) -> bool:
    """Initialise the AudioHubClient (idempotent, thread-safe).

    Returns True if the client is available after the call.
    """
    global _audio_client, _init_done

    if _init_done:
        return _audio_client is not None

    with _init_lock:
        if _init_done:
            return _audio_client is not None

        try:
            from unitree_sdk2py.core.dds.channel import DDSChannelFactoryInitialize
            from unitree_sdk2py.sdk.sdk import create_standard_sdk
            from unitree_sdk2py.go2.audiohub.audiohub_client import AudioHubClient

            serial = serial or os.environ.get("GO2_SERIAL", "B42D2000XXXXXXXX")

            communicator = DDSChannelFactoryInitialize(domainId=0)
            sdk = create_standard_sdk("Go2DashboardAudio")
            robot = sdk.create_robot(communicator, serialNumber=serial)
            client = robot.ensure_client(AudioHubClient.default_service_name)
            client.SetTimeout(3.0)
            client.Init()

            _audio_client = client
            _LOG.info("AudioHubClient initialised (serial=%s)", serial)
        except Exception as exc:
            _LOG.warning("AudioHubClient init failed: %s", exc)
            _audio_client = None
        finally:
            _init_done = True

    return _audio_client is not None


def init_audio_client_async(serial: Optional[str] = None) -> None:
    """Initialise AudioHubClient in a daemon thread (non-blocking)."""
    thread = threading.Thread(
        target=init_audio_client, args=(serial,), daemon=True
    )
    thread.start()


def is_available() -> bool:
    """Check if AudioHubClient is available."""
    return _audio_client is not None


def get_client() -> Optional[object]:
    """Return the AudioHubClient instance, or None."""
    return _audio_client


def play_wav_via_megaphone(wav_path: str, duration_sec: float) -> bool:
    """Play a WAV file through Go2's speaker using megaphone mode.

    Returns True on success, False on failure.
    """
    if not is_available():
        _LOG.warning("AudioHubClient not available, cannot play on Go2")
        return False

    if not os.path.exists(wav_path):
        _LOG.warning("WAV file not found: %s", wav_path)
        return False

    try:
        _audio_client.MegaphoneEnter()
        _audio_client.MegaphoneUpload(wav_path)
        time.sleep(min(duration_sec, 30.0))
        _audio_client.MegaphoneExit()
        return True
    except Exception as exc:
        _LOG.error("Go2 megaphone playback failed: %s", exc)
        try:
            _audio_client.MegaphoneExit()
        except Exception:
            pass
        return False
