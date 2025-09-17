from dataclasses import dataclass

@dataclass
class RecognitionResult:
    """Data class for recognition results"""
    detected: bool
    text: str
    confidence: float
    metadata: dict
