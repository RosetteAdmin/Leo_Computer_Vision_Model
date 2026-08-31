"""Offline, CPU-only full-body movement and exercise monitoring system.

Module map (pipeline order):

* :mod:`src.landmarks`        - BlazePose landmark indices and skeleton topology
* :mod:`src.pose_extraction`  - MediaPipe wrapper -> all 33 landmarks per frame
* :mod:`src.angle_engine`     - universal whole-body angle/metric engine
* :mod:`src.features`         - per-frame and windowed classifier features
* :mod:`src.exercise_config`  - YAML exercise definitions (loader + validation)
* :mod:`src.classifier`       - exercise classifier + "unknown" rejection
* :mod:`src.rep_counter`      - config-driven state machine + generic fallback
* :mod:`src.mistake_detector` - angle / repetition / generic anomaly checks
* :mod:`src.feedback`         - error codes -> coaching text
* :mod:`src.overlay`          - OpenCV rendering
* :mod:`src.pipeline`         - the whole thing wired together, frame in -> state out
* :mod:`src.main`             - real-time application entry point
* :mod:`src.session`          - recording / loading landmark sessions
* :mod:`src.synthetic`        - synthetic test-bench data generator
* :mod:`src.train`            - classifier training
* :mod:`src.evaluate`         - metrics report generation
"""

__version__ = "1.0.0"
