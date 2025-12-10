# Whisper (Speech-to-Text)

OpenAI's robust speech recognition model that handles multiple languages.

## 📋 Overview

**Task:** Audio → Text transcription
**Languages:** 99 languages supported
**Model Sizes:** Tiny, Base, Small, Medium, Large
**Accuracy:** 95%+ on English, varies by language
**Training Data:** 680k hours multilingual audio

## 🎯 Key Features

```
Robust to:
✓ Background noise
✓ Technical language
✓ Accents and dialects
✓ Multiple languages in same audio
✓ Music, laughter, etc.

Handles:
✓ Timestamps
✓ Language identification
✓ Task selection (transcribe vs translate)
✓ No need for fine-tuning
```

## 🚀 Quick Start

```python
import whisper

# Load model
model = whisper.load_model("base")  # or "tiny", "small", "medium", "large"

# Transcribe
result = model.transcribe("audio.mp3")

# Results
print(result["text"])  # Full transcription
print(result["language"])  # Detected language

# Detailed output
for segment in result["segments"]:
    print(f"{segment['start']:.1f}s - {segment['end']:.1f}s: {segment['text']}")
```

## 📊 Model Sizes and Performance

```
Model       Size    VRAM    Speed    Accuracy
────────────────────────────────────────────
tiny        39M     ~1GB    50×      55%
base        74M     ~1GB    32×      85%
small       244M    ~2GB    16×      92%
medium      769M    ~5GB    8×       96%
large       1550M   ~10GB   4×       99%

Speed = relative to real-time (16× = 16× faster than audio)
Accuracy = on test set (English)
```

## 🎯 Model Selection

```
Choose model based on:

Available compute?
├─ GPU with 1GB → "tiny" or "base"
├─ GPU with 5GB → "small" or "medium"
└─ GPU with 10GB+ → "large"

Need speed?
├─ Real-time/low-latency → "tiny" or "base"
├─ Batch processing → "medium" or "large"
└─ Quality critical → "large"

Language diversity?
├─ Mostly English → "base"
├─ Multiple languages → "medium" or "large"
└─ Rare languages → "large"

Default recommendation:
"base" or "small" (good balance)
```

## 💻 Running Locally

```python
import whisper
import torch

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base", device=device)

# Transcribe
result = model.transcribe("audio.mp3")
```

## 🌍 Multilingual Support

### Language Identification
```python
result = model.transcribe("audio.mp3")
print(result["language"])  # Detected language (e.g., "fr" for French)
```

### Translate to English
```python
# If audio is in French, translate to English
result = model.transcribe("audio.mp3", task="translate")
# Returns English translation instead of French transcription
```

### Supported Languages
```
99 languages including:
- Major: English, Spanish, French, German, Chinese, Japanese, Korean
- Many others: Arabic, Hindi, Portuguese, Russian, Thai, Vietnamese, etc.

Full list: See Whisper documentation
```

## 📊 Audio Preprocessing

```python
import librosa
import numpy as np

# Load audio
audio, sr = librosa.load("audio.mp3", sr=16000)

# Normalization (Whisper expects certain audio characteristics)
if np.max(np.abs(audio)) > 1.0:
    audio = audio / np.max(np.abs(audio))

# Whisper handles most preprocessing automatically
# But good preprocessing can improve results
```

## 🎯 Advanced Options

```python
# Verbose output
result = model.transcribe("audio.mp3", verbose=True)

# Word-level timestamps
result = model.transcribe("audio.mp3", language="en")
for segment in result["segments"]:
    for word_info in segment.get("words", []):
        print(f"{word_info['word']}: {word_info['start']:.2f}-{word_info['end']:.2f}s")

# Temperature (controls uncertainty)
# Higher temperature → more random
# Lower temperature → more confident
result = model.transcribe("audio.mp3", temperature=0.0)  # Most confident
```

## 📈 Performance Evaluation

```
Word Error Rate (WER):
WER = (S + D + I) / N
where:
S = substitutions
D = deletions
I = insertions
N = reference words

Example:
Reference: "hello world today"
Hypothesis: "hello world"
WER = 1/3 = 33%

Whisper WER:
- English: ~5-10%
- Spanish: ~10-15%
- Japanese: ~15-20%
- Rare languages: 20-50%+
```

## ⚠️ Common Issues

1. **Audio quality matters**
   ```
   Bad: Heavy background noise, poor microphone
   → Higher error rate

   Solution: Pre-process audio, remove noise
   ```

2. **Very long audio**
   ```
   Problem: Might lose context halfway through

   Solution:
   - Use large model for long audio
   - Break into chunks and transcribe separately
   ```

3. **Domain-specific language**
   ```
   Problem: Medical, legal, technical terms
   → Lower accuracy

   Solution:
   - No fine-tuning available
   - Post-process with spell checker for domain
   - Use human review for important content
   ```

4. **Cost for large-scale**
   ```
   Problem: Running large model is slow

   Solutions:
   - Use smaller model and accept lower accuracy
   - Batch processing (more efficient)
   - Use quantization (see below)
   ```

## 🚀 Optimization Techniques

### Quantization
```python
import whisper
from transformers import AutoModelForSpeechSeq2Seq
import torch

# Load with lower precision
model = whisper.load_model("base")
model = model.half()  # Float16 instead of Float32
# Half memory, similar accuracy
```

### Batch Processing
```python
audio_files = ["audio1.mp3", "audio2.mp3", ..., "audio100.mp3"]

# More efficient than one-by-one
results = []
for audio_file in audio_files:
    result = model.transcribe(audio_file)
    results.append(result)
```

### Streaming (with Faster-Whisper)
```python
from faster_whisper import WhisperModel

# Faster inference than official Whisper
model = WhisperModel("base", device="cuda", compute_type="float16")

# Can do streaming transcription
segments, info = model.transcribe("audio.mp3")
for segment in segments:
    print(f"[{segment.start:.2f}s] {segment.text}")
```

## 📈 Applications

| Domain | Use Case |
|--------|----------|
| **Transcription** | Podcast transcripts, meetings |
| **Accessibility** | Captions for videos |
| **Search** | Audio search (speech to text) |
| **Customer service** | Call transcription |
| **Accessibility** | Speech-to-text for deaf |
| **Translation** | Audio translation to English |
| **Data preparation** | Label audio data for training |

## 💡 Comparison with Alternatives

| Method | Accuracy | Speed | Cost | Requires Model |
|--------|----------|-------|------|---|
| Whisper (large) | 99% | Slow | Free | 1.5GB |
| Whisper (base) | 85% | Fast | Free | 74MB |
| Google Speech API | 98% | Fast | $ | Cloud |
| AWS Transcribe | 96% | Fast | $ | Cloud |
| OpenAI API | 99% | Fast | $ | Cloud |

## 🎓 Learning Outcomes

- [x] Whisper model architecture
- [x] Language identification and translation
- [x] Model selection for your use case
- [x] Audio preprocessing
- [x] Performance evaluation
- [x] Optimization techniques

## 📚 Resources

- **Whisper**: https://github.com/openai/whisper
- **Faster-Whisper**: https://github.com/guillaumekln/faster-whisper
- **OpenAI Documentation**: https://platform.openai.com/docs/guides/speech-to-text

## 💡 Production Checklist

```
✓ Choose model size (base or small recommended)
✓ Test on your audio domain
✓ Set up error handling (network, audio errors)
✓ Implement caching (avoid re-transcribing)
✓ Set up monitoring (accuracy, latency)
✓ Plan for updates (new Whisper versions)
✓ Consider API vs local (cost vs latency)
✓ Pre-process audio if needed (denoise, trim)
✓ Post-process output (fix common errors)
✓ Test with various audio qualities
```

---

**Last Updated:** December 2024
**Status:** ✅ Complete
