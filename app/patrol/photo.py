"""
Photo archival -- save detection photos for patrol sessions.
"""
import os
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


from utils import APP_ROOT

PHOTOS_BASE = os.path.join(APP_ROOT, "static", "patrol_photos")


def ensure_patrol_dirs(patrol_id: str) -> str:
    """Create ``patrol_photos/<patrol_id>/`` and return its path."""
    d = os.path.join(PHOTOS_BASE, patrol_id)
    os.makedirs(d, exist_ok=True)
    return d


def save_patrol_photo(
    bgr_frame: np.ndarray,
    bbox: List[float],
    label: str,
    confidence: float,
    photos_dir: str,
    seq: int,
) -> Optional[str]:
    """Draw detection overlay on *bgr_frame* and save as JPEG.

    Returns the relative URL path or *None* on failure.
    """
    try:
        annotated = bgr_frame.copy()
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            annotated, f"{label} {confidence:.2f}", (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
        )
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            annotated, ts, (8, annotated.shape[0] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

        fname = f"detection_{seq:04d}_{label}_{confidence:.2f}.jpg"
        path = os.path.join(photos_dir, fname)
        cv2.imwrite(path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return f"/patrol/photos/{os.path.basename(photos_dir)}/{fname}"
    except Exception:
        return None


def cleanup_old_patrol_sessions(max_keep: int = 10) -> None:
    """Remove oldest patrol photo sessions beyond *max_keep*."""
    if not os.path.isdir(PHOTOS_BASE):
        return
    sessions = sorted(
        d for d in os.listdir(PHOTOS_BASE)
        if os.path.isdir(os.path.join(PHOTOS_BASE, d))
    )
    while len(sessions) > max_keep:
        old = sessions.pop(0)
        import shutil
        shutil.rmtree(os.path.join(PHOTOS_BASE, old), ignore_errors=True)
