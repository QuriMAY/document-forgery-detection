from pathlib import Path


class DocumentDetector:
    """YOLOv8-based stamp and signature detector.

    Gracefully degrades to empty detections when the model isn't trained yet.
    """

    def __init__(self, model_path: str = "models/best.pt"):
        self._model_path = model_path
        self._model = None
        self._available = False

        if Path(model_path).exists():
            self._load()

    def _load(self) -> None:
        from ultralytics import YOLO
        self._model = YOLO(self._model_path)
        self._available = True

    def detect(self, image_path: str, conf: float = 0.25) -> dict:
        if not self._available:
            return {
                "signatures": [],
                "stamps": [],
                "warning": "YOLO model not found — run train_yolo.py first.",
            }

        results = self._model(image_path, conf=conf, verbose=False)
        detections: dict = {"signatures": [], "stamps": []}

        for result in results:
            for box in result.boxes:
                label = result.names[int(box.cls)]
                entry = {
                    "bbox": [round(v, 1) for v in box.xyxy[0].tolist()],
                    "confidence": round(float(box.conf), 4),
                }
                if label == "signature":
                    detections["signatures"].append(entry)
                elif label == "stamp":
                    detections["stamps"].append(entry)

        return detections

    @property
    def is_available(self) -> bool:
        return self._available
