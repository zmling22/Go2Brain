"""Tests for audio_sdk module components."""
import os
import sys
import tempfile
import threading
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure app/ is on sys.path for imports
_APP_DIR = Path(__file__).resolve().parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))


# =============================================================================
#  Fixtures
# =============================================================================

@pytest.fixture
def fake_wav():
    """Create a real, minimal WAV file for tests that need wave.open to work."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000)
    yield path
    if os.path.exists(path):
        os.unlink(path)


# =============================================================================
#  Tests for audio_sdk.sdk
#  NOTE: DDSChannelFactoryInitialize, create_standard_sdk, and AudioHubClient
#  are imported *inside* init_audio_client(), so we patch at their source
#  (unitree_sdk2py.*) rather than on the audio_sdk.sdk module.
# =============================================================================

class TestSdk:

    def test_init_audio_client_success(self):
        """Successful init should set _audio_client and return True."""
        from audio_sdk import sdk
        sdk._init_done = False
        sdk._audio_client = None

        with patch("unitree_sdk2py.core.dds.channel.DDSChannelFactoryInitialize"):
            with patch("unitree_sdk2py.sdk.sdk.create_standard_sdk") as mock_sdk:
                mock_client = MagicMock()
                mock_robot = MagicMock()
                mock_robot.ensure_client.return_value = mock_client
                mock_sdk.return_value.create_robot.return_value = mock_robot

                result = sdk.init_audio_client(serial="TEST123")

                assert result is True
                assert sdk._audio_client is not None
                assert sdk._audio_client == mock_client
                mock_client.SetTimeout.assert_called_once_with(3.0)
                mock_client.Init.assert_called_once()

    def test_init_audio_client_failure_graceful(self):
        """Init failure should not raise; should return False."""
        from audio_sdk import sdk
        sdk._init_done = False
        sdk._audio_client = None

        with patch("unitree_sdk2py.core.dds.channel.DDSChannelFactoryInitialize",
                   side_effect=Exception("No robot")):
            result = sdk.init_audio_client()
            assert result is False
            assert sdk._audio_client is None
            assert sdk._init_done is True

    def test_init_audio_client_idempotent(self):
        """Second call should not re-initialise."""
        from audio_sdk import sdk
        sdk._init_done = True
        sdk._audio_client = "already_set"

        with patch("unitree_sdk2py.core.dds.channel.DDSChannelFactoryInitialize") as mock_dds:
            result = sdk.init_audio_client()
            mock_dds.assert_not_called()
            assert result is True

    def test_is_available(self):
        """is_available returns True only when client is set."""
        from audio_sdk import sdk
        sdk._audio_client = None
        assert sdk.is_available() is False
        sdk._audio_client = MagicMock()
        assert sdk.is_available() is True

    def test_get_client(self):
        """get_client returns the client or None."""
        from audio_sdk import sdk
        sdk._audio_client = None
        assert sdk.get_client() is None
        fake = MagicMock()
        sdk._audio_client = fake
        assert sdk.get_client() is fake

    def test_play_wav_via_megaphone_no_client(self):
        """Without client, play_wav_via_megaphone returns False."""
        from audio_sdk import sdk
        sdk._audio_client = None
        assert sdk.play_wav_via_megaphone("/fake.wav", 1.0) is False

    def test_play_wav_via_megaphone_missing_file(self):
        """With client but missing WAV, returns False."""
        from audio_sdk import sdk
        sdk._audio_client = MagicMock()
        assert sdk.play_wav_via_megaphone("/nonexistent/file.wav", 1.0) is False

    def test_play_wav_via_megaphone_success(self, fake_wav):
        """Successful megaphone playback calls enter/upload/sleep/exit."""
        from audio_sdk import sdk
        mock_client = MagicMock()
        sdk._audio_client = mock_client

        result = sdk.play_wav_via_megaphone(fake_wav, 2.0)

        assert result is True
        mock_client.MegaphoneEnter.assert_called_once()
        mock_client.MegaphoneUpload.assert_called_once_with(fake_wav)
        mock_client.MegaphoneExit.assert_called_once()

    def test_play_wav_via_megaphone_exception_exit(self, fake_wav):
        """On exception, MegaphoneExit is still called."""
        from audio_sdk import sdk
        mock_client = MagicMock()
        mock_client.MegaphoneUpload.side_effect = RuntimeError("upload fail")
        sdk._audio_client = mock_client

        result = sdk.play_wav_via_megaphone(fake_wav, 1.0)

        assert result is False
        mock_client.MegaphoneEnter.assert_called_once()
        mock_client.MegaphoneExit.assert_called_once()

    def test_init_audio_client_async_spawns_thread(self):
        """init_audio_client_async starts a daemon thread."""
        from audio_sdk import sdk
        original = threading.Thread
        captured_kwargs = {}

        class MockThread:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)
                self._inner = original(**kwargs)
            def start(self):
                self._inner.start()

        with patch("audio_sdk.sdk.threading.Thread", MockThread):
            sdk._init_done = False
            sdk._audio_client = None
            sdk.init_audio_client_async()

        assert captured_kwargs.get("daemon") is True
        assert captured_kwargs.get("target") == sdk.init_audio_client


# =============================================================================
#  Tests for audio_sdk.tts
#  NOTE: gTTS is imported inside function body, so we patch gtts.gTTS.
# =============================================================================

class TestTts:

    def test_generate_mp3_success(self):
        """_generate_mp3 returns a path when gTTS succeeds."""
        from audio_sdk import tts

        with patch("hashlib.md5") as mock_md5:
            mock_md5.return_value.hexdigest.return_value = "abc123"
            with patch("os.path.exists", return_value=False):
                with patch("gtts.gTTS") as mock_gtts:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        tts.VOICE_DIR = tmpdir
                        mp3_path = tts._generate_mp3("你好", "zh-CN")

                        assert mp3_path is not None
                        assert mp3_path.endswith(".mp3")
                        mock_gtts.assert_called_once_with(text="你好", lang="zh")

    def test_generate_mp3_cached(self):
        """_generate_mp3 returns cached path without calling gTTS."""
        from audio_sdk import tts

        with patch("os.path.exists", return_value=True):
            mp3_path = tts._generate_mp3("你好", "zh-CN")
            assert mp3_path is not None

    def test_generate_mp3_no_gtts(self):
        """_generate_mp3 returns None when gTTS is not installed."""
        from audio_sdk import tts

        with patch.dict("sys.modules", {"gtts": None}):
            result = tts._generate_mp3("hi", "en")
            assert result is None

    def test_generate_wav_gstreamer_success(self):
        """generate_wav converts MP3 to WAV via GStreamer."""
        from audio_sdk import tts

        with patch("audio_sdk.tts._generate_mp3",
                   return_value="/tmp/alert_abc123.mp3"):
            with patch("subprocess.run") as mock_run:
                # os.path.exists: first False (wav missing), then True (converted)
                exists_calls = iter([False, True])
                with patch("os.path.exists", side_effect=lambda p: next(exists_calls)):
                    wav = tts.generate_wav("你好", "zh-CN")
                    assert wav == "/tmp/alert_abc123.wav"
                    mock_run.assert_called_once()

    def test_generate_wav_no_mp3(self):
        """generate_wav returns None if MP3 generation fails."""
        from audio_sdk import tts

        with patch("audio_sdk.tts._generate_mp3", return_value=None):
            assert tts.generate_wav("你好") is None

    def test_generate_mp3_to_static(self):
        """generate_mp3_to_static returns URL path on success."""
        from audio_sdk import tts

        with patch("audio_sdk.tts._generate_mp3",
                   return_value="/base/static/voice_alerts/alert_abc.mp3"):
            url = tts.generate_mp3_to_static("hello", "en")
            assert url == "/static/voice_alerts/alert_abc.mp3"

    def test_generate_mp3_to_static_none(self):
        """generate_mp3_to_static returns None on failure."""
        from audio_sdk import tts

        with patch("audio_sdk.tts._generate_mp3", return_value=None):
            assert tts.generate_mp3_to_static("hello") is None


# =============================================================================
#  Tests for audio_sdk.player
#  NOTE: subprocess is imported *inside* _play_alert_sync, so we patch
#  top-level subprocess.run rather than audio_sdk.player.subprocess.run.
# =============================================================================

class TestPlayer:

    def test_play_alert_spawns_daemon_thread(self):
        """play_alert starts a daemon thread without blocking."""
        from audio_sdk import player
        original = threading.Thread
        captured_kwargs = {}

        class MockThread:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)
                self._inner = original(**kwargs)
            def start(self):
                self._inner.start()

        with patch("audio_sdk.player.threading.Thread", MockThread):
            with patch("audio_sdk.player._play_alert_sync"):
                player.play_alert("test")
                import time
                time.sleep(0.1)

        assert captured_kwargs.get("daemon") is True
        assert captured_kwargs.get("args") == ("test", "zh-CN")

    def test_play_alert_no_wav(self):
        """When generate_wav returns None, _play_alert_sync returns early."""
        from audio_sdk import player

        with patch("audio_sdk.player.generate_wav", return_value=None):
            with patch("audio_sdk.player.audio_sdk.is_available") as mock_avail:
                player._play_alert_sync("test")
                mock_avail.assert_not_called()

    def test_play_alert_tries_go2_first(self, fake_wav):
        """_play_alert_sync tries Go2 megaphone before falling back to aplay."""
        from audio_sdk import player

        with patch("audio_sdk.player.generate_wav", return_value=fake_wav):
            with patch("audio_sdk.player.audio_sdk.is_available",
                       return_value=True):
                with patch("audio_sdk.player.audio_sdk.init_audio_client"):
                    with patch(
                        "audio_sdk.player.audio_sdk.play_wav_via_megaphone",
                            return_value=True) as mock_mega:
                        with patch("subprocess.run") as mock_aplay:
                            player._play_alert_sync("test")

                            mock_mega.assert_called_once()
                            mock_aplay.assert_not_called()

    def test_play_alert_falls_back_to_aplay(self, fake_wav):
        """When Go2 fails, fall back to aplay."""
        from audio_sdk import player

        with patch("audio_sdk.player.generate_wav", return_value=fake_wav):
            with patch("audio_sdk.player.audio_sdk.is_available",
                       return_value=True):
                with patch("audio_sdk.player.audio_sdk.init_audio_client"):
                    with patch(
                        "audio_sdk.player.audio_sdk.play_wav_via_megaphone",
                            return_value=False) as mock_mega:
                        with patch("subprocess.run") as mock_aplay:
                            player._play_alert_sync("test")

                            mock_mega.assert_called_once()
                            mock_aplay.assert_called_once_with(
                                ["aplay", fake_wav], timeout=15)

    def test_play_alert_go2_exception_falls_back(self, fake_wav):
        """When Go2 raises, fall back to aplay."""
        from audio_sdk import player

        with patch("audio_sdk.player.generate_wav", return_value=fake_wav):
            with patch("audio_sdk.player.audio_sdk.is_available",
                       return_value=True):
                with patch("audio_sdk.player.audio_sdk.init_audio_client"):
                    with patch(
                        "audio_sdk.player.audio_sdk.play_wav_via_megaphone",
                            side_effect=RuntimeError("fail")):
                        with patch("subprocess.run") as mock_aplay:
                            player._play_alert_sync("test")
                            mock_aplay.assert_called_once_with(
                                ["aplay", fake_wav], timeout=15)

    def test_play_alert_init_if_not_available(self, fake_wav):
        """When Go2 not available, init_audio_client is called."""
        from audio_sdk import player

        with patch("audio_sdk.player.generate_wav", return_value=fake_wav):
            with patch("audio_sdk.player.audio_sdk.is_available",
                       side_effect=[False, True]):
                with patch("audio_sdk.player.audio_sdk.init_audio_client") as mock_init:
                    with patch(
                        "audio_sdk.player.audio_sdk.play_wav_via_megaphone",
                            return_value=True):
                        player._play_alert_sync("test")
                        mock_init.assert_called_once()

    def test_generate_mp3_to_static_re_exported(self):
        """player module re-exports generate_mp3_to_static for convenience."""
        from audio_sdk import player
        assert callable(player.generate_mp3_to_static)
