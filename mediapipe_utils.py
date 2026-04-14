"""
MediaPipe compatibility utilities for ASL Alphabet Recognition.

Provides a unified API for hand landmark extraction that works across:
- MediaPipe 0.10.30+  : Tasks API  (requires hand_landmarker.task model download)
- MediaPipe older     : Solutions API (mp.solutions.hands)
- Fallback            : returns None — CNN approaches still work; Landmark MLP is skipped

Usage
-----
    from mediapipe_utils import HandDetector, extract_landmarks, HAND_CONNECTIONS

    with HandDetector() as det:
        lms = det.process(image_bgr)   # returns list of (x,y,z) tuples or None
"""

import cv2
import numpy as np
import os
import logging
import warnings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hand skeleton connection pairs (identical to mp.solutions.hands.HAND_CONNECTIONS)
# ---------------------------------------------------------------------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),      # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),    # Ring finger
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
]

NUM_LANDMARKS = 21   # MediaPipe hand landmarker always returns 21 landmarks
FEATURE_DIM   = NUM_LANDMARKS * 3  # 63 (x, y, z per landmark)

# Module-level cache so detection only runs once per session
_mediapipe_mode: str | None = None   # 'tasks' | 'solutions' | 'unavailable'
_task_model_path: str | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _download_task_model(cache_dir: str = "/tmp/mediapipe_models") -> str | None:
    """Download the hand-landmarker `.task` model file if not already cached."""
    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(cache_dir, "hand_landmarker.task")
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        return model_path

    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )
    try:
        import urllib.request
        logger.info("Downloading hand-landmarker model to %s …", model_path)
        urllib.request.urlretrieve(url, model_path)
        if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
            logger.info("Model downloaded successfully.")
            return model_path
    except Exception as exc:
        logger.warning("Could not download hand-landmarker model: %s", exc)

    # Clean up an incomplete/empty file
    if os.path.exists(model_path):
        os.remove(model_path)
    return None


def _detect_mediapipe_mode() -> str:
    """
    Detect which MediaPipe API is available and cache the result.

    Detection order:
    1. Tasks API   (mediapipe 0.10.30+, requires model file download)
    2. Solutions API (older mediapipe builds that still expose mp.solutions)
    3. 'unavailable' — landmark-based approaches will be skipped gracefully
    """
    global _mediapipe_mode, _task_model_path
    if _mediapipe_mode is not None:
        return _mediapipe_mode

    # --- Attempt 1: Tasks API ---
    try:
        import mediapipe as mp  # noqa: F401
        from mediapipe.tasks import python as _mp_python  # noqa: F401
        from mediapipe.tasks.python import vision as _mp_vision  # noqa: F401

        model_path = _download_task_model()
        if model_path is not None:
            _task_model_path = model_path
            _mediapipe_mode = "tasks"
            logger.info("MediaPipe mode: Tasks API (0.10.30+)")
            return _mediapipe_mode
        else:
            logger.warning(
                "MediaPipe Tasks API is present but the hand-landmarker model "
                "could not be downloaded. Falling back to Solutions API."
            )
    except Exception as exc:
        logger.debug("Tasks API unavailable: %s", exc)

    # --- Attempt 2: Solutions API ---
    try:
        import mediapipe as mp
        _solutions = mp.solutions.hands  # will raise AttributeError if missing
        _mediapipe_mode = "solutions"
        logger.info("MediaPipe mode: Solutions API")
        return _mediapipe_mode
    except Exception as exc:
        logger.debug("Solutions API unavailable: %s", exc)

    # --- Fallback ---
    warnings.warn(
        "MediaPipe is not fully functional in this environment.\n"
        "Hand landmark extraction (Approach 1 – Landmark MLP) will be skipped.\n"
        "CNN approaches (Approaches 2–4) will still work using raw images as fallback.",
        RuntimeWarning,
        stacklevel=2,
    )
    _mediapipe_mode = "unavailable"
    return _mediapipe_mode


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_mediapipe_mode() -> str:
    """
    Return the detected MediaPipe API mode.

    Returns one of: ``'tasks'``, ``'solutions'``, ``'unavailable'``.
    """
    return _detect_mediapipe_mode()


class HandDetector:
    """
    Context manager that wraps a MediaPipe hand-landmark detector.

    Example::

        with HandDetector() as det:
            landmarks = det.process(image_bgr)
            if landmarks is not None:
                feats = np.array(landmarks, dtype=np.float32).flatten()  # shape (63,)
    """

    def __init__(self):
        self.mode = _detect_mediapipe_mode()
        self._detector = None

    def __enter__(self):
        if self.mode == "solutions":
            try:
                import mediapipe as mp
                self._detector = mp.solutions.hands.Hands(
                    static_image_mode=True,
                    max_num_hands=1,
                    min_detection_confidence=0.7,
                )
            except OSError as exc:
                # Missing system library (e.g. libEGL.so.1 on headless Linux)
                logger.warning("MediaPipe Solutions failed to load C library: %s — falling back to unavailable.", exc)
                self.mode = "unavailable"
        elif self.mode == "tasks":
            try:
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python import vision as mp_vision
                base_options = mp_python.BaseOptions(model_asset_path=_task_model_path)
                options = mp_vision.HandLandmarkerOptions(
                    base_options=base_options,
                    num_hands=1,
                    min_hand_detection_confidence=0.7,
                )
                self._detector = mp_vision.HandLandmarker.create_from_options(options)
            except OSError as exc:
                # Missing system library (e.g. libEGL.so.1 on headless Linux)
                logger.warning("MediaPipe Tasks failed to load C library: %s — falling back to unavailable.", exc)
                self.mode = "unavailable"
        # mode == 'unavailable' — _detector stays None
        return self

    def __exit__(self, *_args):
        if self._detector is not None:
            self._detector.close()
            self._detector = None

    def process(self, image_bgr: np.ndarray):
        """
        Detect and return hand landmarks for the given BGR image.

        Returns
        -------
        list[tuple[float, float, float]] | None
            21 ``(x, y, z)`` normalised-coordinate tuples if a hand is detected,
            otherwise ``None``.
        """
        if self._detector is None or self.mode == "unavailable":
            return None

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if self.mode == "solutions":
            result = self._detector.process(rgb)
            if not result.multi_hand_landmarks:
                return None
            lms = result.multi_hand_landmarks[0].landmark
            return [(lm.x, lm.y, lm.z) for lm in lms]

        if self.mode == "tasks":
            import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._detector.detect(mp_image)
            if not result.hand_landmarks:
                return None
            return [(lm.x, lm.y, lm.z) for lm in result.hand_landmarks[0]]

        return None  # should not reach here


def extract_landmarks(image_bgr: np.ndarray, detector: "HandDetector | None" = None) -> np.ndarray | None:
    """
    Extract 63 hand-landmark features from a BGR image.

    Parameters
    ----------
    image_bgr : np.ndarray
        OpenCV BGR image.
    detector : HandDetector, optional
        An already-opened ``HandDetector`` context. If omitted a temporary
        one is created (less efficient for batch processing).

    Returns
    -------
    np.ndarray | None
        Float32 array of shape ``(63,)`` or ``None`` if no hand was detected
        or MediaPipe is unavailable.
    """
    if detector is not None:
        lms = detector.process(image_bgr)
    else:
        with HandDetector() as det:
            lms = det.process(image_bgr)

    if lms is None:
        return None
    return np.array(lms, dtype=np.float32).flatten()


def get_hand_bbox(
    image_bgr: np.ndarray,
    padding: float = 0.20,
    detector: "HandDetector | None" = None,
) -> tuple[int, int, int, int] | None:
    """
    Return the bounding box of the detected hand in pixel coordinates.

    Parameters
    ----------
    image_bgr : np.ndarray
        OpenCV BGR image.
    padding : float
        Fractional padding around the tight bounding box (default 0.20 = 20 %).
    detector : HandDetector, optional
        An already-opened ``HandDetector`` context.

    Returns
    -------
    tuple[int, int, int, int] | None
        ``(x1, y1, x2, y2)`` pixel coordinates, or ``None`` if no hand found.
    """
    h, w = image_bgr.shape[:2]

    if detector is not None:
        lms = detector.process(image_bgr)
    else:
        with HandDetector() as det:
            lms = det.process(image_bgr)

    if lms is None:
        return None

    xs = [lm[0] for lm in lms]
    ys = [lm[1] for lm in lms]
    pad_x = padding * (max(xs) - min(xs))
    pad_y = padding * (max(ys) - min(ys))
    x1 = max(0, int((min(xs) - pad_x) * w))
    y1 = max(0, int((min(ys) - pad_y) * h))
    x2 = min(w, int((max(xs) + pad_x) * w))
    y2 = min(h, int((max(ys) + pad_y) * h))
    return (x1, y1, x2, y2)


def draw_skeleton(
    landmarks: list[tuple[float, float, float]] | None,
    output_size: tuple[int, int] = (224, 224),
    bg_color: tuple[int, int, int] = (255, 255, 255),
    conn_color: tuple[int, int, int] = (0, 0, 0),
    dot_color: tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    """
    Render hand-skeleton connections and joint dots on a blank canvas.

    Parameters
    ----------
    landmarks : list[tuple[float, float, float]] | None
        21 ``(x, y, z)`` normalised-coordinate tuples, or ``None``.
    output_size : tuple[int, int]
        ``(height, width)`` of the output image.
    bg_color, conn_color, dot_color : tuple[int, int, int]
        BGR colours for background, connections, and joints.

    Returns
    -------
    np.ndarray
        BGR image of shape ``(height, width, 3)``.  If *landmarks* is
        ``None`` the canvas is returned blank (all ``bg_color``).
    """
    h_out, w_out = output_size
    canvas = np.full((h_out, w_out, 3), bg_color, dtype=np.uint8)

    if landmarks is None:
        return canvas

    for s, e in HAND_CONNECTIONS:
        p1 = (int(landmarks[s][0] * w_out), int(landmarks[s][1] * h_out))
        p2 = (int(landmarks[e][0] * w_out), int(landmarks[e][1] * h_out))
        cv2.line(canvas, p1, p2, conn_color, 2)

    for lm in landmarks:
        cv2.circle(canvas, (int(lm[0] * w_out), int(lm[1] * h_out)), 4, dot_color, -1)

    return canvas


def print_mediapipe_info() -> None:
    """Print a summary of the detected MediaPipe environment."""
    mode = get_mediapipe_mode()
    try:
        import mediapipe as mp
        version = mp.__version__
    except ImportError:
        version = "not installed"

    print("=" * 60)
    print("MediaPipe Environment Info")
    print("=" * 60)
    print(f"  Version       : {version}")
    print(f"  Mode          : {mode}")
    if mode == "tasks":
        print(f"  Model file    : {_task_model_path}")
    elif mode == "unavailable":
        print("  ⚠️  Landmark MLP (Approach 1) will be skipped.")
        print("     CNN approaches (2–4) will use raw images as fallback.")
    print("=" * 60)
