from dataclasses import dataclass
from typing import Dict

@dataclass
class RecognitionResult:
    detected: bool
    text: str
    confidence: float
    metadata: Dict
