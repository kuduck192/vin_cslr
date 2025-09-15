# Vietnamese Continuous Sign Language Recognition - VinBigdata Computer Vision Project

Real-time sign language recognition system with Text-to-Speech output.

## 📁 Project Structure

```
project/
├── main.py           # Main application with webcam UI
├── slr/
│   └── __init__.py   # Exports APIs
│   └── slr.py        # Sign Language Recognition module
├── tts/
│   └── __init__.py   # Exports APIs
│   └── slr.py        # Text-to-Speech module
├── requirements.txt  # Dependencies
└── README.md         # This file
```

## 🚀 Quick Start

### 1. Create virtual env
```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows
pip install --upgrade pip
```

### 2. Install Dependencies

```bash
# Install core dependencies
pip install opencv-python numpy

# Or install all from requirements
pip install -r requirements.txt
```

### 3. Run the Demo

```bash
python main.py
```

## 🎮 Controls

- **Q/ESC**: Quit application
- **I**: Toggle information panel
- **Space**: Pause/Resume video
- **R**: Reset recognition history

## 🔧 Features

### Current (Dummy Implementation)
- ✅ Real-time webcam capture
- ✅ Simulated sign language recognition
- ✅ Text-to-speech output (with pyttsx3)
- ✅ Recognition history display
- ✅ FPS counter
- ✅ Confidence scores
- ✅ Mirror mode for natural interaction

### Pipeline Flow
```
Video Input (Webcam)
    ↓
Frame Processing
    ↓
Sign Language Recognition (SLR)
    ↓
Text Output
    ↓
Text-to-Speech (TTS)
    ↓
Audio Output (Speaker)
```

## 🔨 Development

### Implementing Real SLR

Replace the dummy `sign_language_recognition()` function in `slr/slr.py` with your actual model:

```python
def sign_language_recognition(video: np.ndarray) -> dict:
    # Your model implementation here
    # Example with MediaPipe:
    # 1. Extract hand landmarks
    # 2. Process sequence of landmarks
    # 3. Run through LSTM/Transformer model
    # 4. Return predicted text
    
    return {
        'text': predicted_text,
        'confidence': confidence_score,
        'detected': True,
        # ... other metadata
    }
```

### Implementing Real TTS
Replace the dummy `sign_language_recognition()` function in `tts/tts.py` with your actual implementation:

```python
def text_to_speech(txt: str) -> np.ndarray:
    # Your TTS implementation
    
    return audio_array
```

## 📊 Configuration

Modify these parameters in `main.py`:

```python
# Recognition settings
self.recognition_interval = 0.5  # Process frequency (seconds)
self.confidence_threshold = 0.6  # Min confidence for valid detection
self.speak_cooldown = 2.0       # Avoid repeating same phrase

# Video settings
self.frame_width = 640
self.frame_height = 480
```