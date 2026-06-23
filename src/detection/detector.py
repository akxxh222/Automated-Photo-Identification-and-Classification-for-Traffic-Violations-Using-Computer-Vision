import numpy as np
import logging
from typing import List, Dict, Any

from src.tracking.tracker import UnifiedTracker

logger = logging.getLogger(__name__)

class VehicleDetector:
    def __init__(self) -> None:
        self.tracker = UnifiedTracker()

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        return self.tracker.process_frame(frame)
