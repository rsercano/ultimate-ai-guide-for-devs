---
layout: default
title: Multi-modal AI
parent: "08 - Advanced Topics"
nav_order: 3
---

# Multi-modal AI

## TL;DR

Multi-modal AI processes multiple input types: text, images, audio, video. Vision-language models (GPT-4V, Claude Vision) enable image understanding. Speech models (Whisper) enable transcription. Combining these creates powerful applications.

## Multi-modal Capabilities

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTI-MODAL LANDSCAPE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   INPUT TYPES              MODELS                  OUTPUTS       │
│   ──────────              ──────                  ───────       │
│                                                                  │
│   [Image] ─────┐                                                │
│                ├───→ Vision-LLM ───→ [Text Description]         │
│   [Text]  ─────┘      (GPT-4V)       [Analysis]                 │
│                                      [Answers]                   │
│                                                                  │
│   [Audio] ─────────→ Whisper ──────→ [Transcription]            │
│                                                                  │
│   [Text]  ─────────→ TTS ──────────→ [Audio]                    │
│                      (ElevenLabs)                               │
│                                                                  │
│   [Text]  ─────────→ Image Gen ────→ [Image]                    │
│                      (DALL-E, SD)                               │
│                                                                  │
│   [Video] ─────────→ Video LLM ────→ [Analysis]                 │
│                      (Gemini 1.5)    [Summary]                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Vision-Language Models

### Available Models

| Model | Provider | Strengths |
|-------|----------|-----------|
| GPT-4V | OpenAI | Best overall, OCR |
| Claude 3 Vision | Anthropic | Charts, documents |
| Gemini Pro Vision | Google | Video understanding |
| LLaVA | Open Source | Self-hostable |
| Qwen-VL | Open Source | Multilingual |

### Basic Image Analysis

```python
from openai import OpenAI
import base64

client = OpenAI()

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def analyze_image(image_path: str, question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
                        }
                    }
                ]
            }
        ],
        max_tokens=500
    )
    return response.choices[0].message.content

# Usage
result = analyze_image("chart.png", "Describe the trends in this chart")
```

### Use Cases

| Use Case | Example |
|----------|---------|
| Document understanding | Extract data from invoices, receipts |
| Chart analysis | Interpret graphs and visualizations |
| UI testing | Verify screenshots match expected |
| Accessibility | Describe images for screen readers |
| Content moderation | Detect inappropriate images |
| Product catalogs | Auto-generate descriptions |

### Multiple Images

```python
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare these two product images"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img1}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img2}"}}
            ]
        }
    ]
)
```

## Speech-to-Text (Whisper)

### OpenAI API

```python
from openai import OpenAI

client = OpenAI()

def transcribe(audio_path: str) -> str:
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
    return transcript

# With timestamps
def transcribe_with_timestamps(audio_path: str):
    with open(audio_path, "rb") as audio_file:
        return client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["word"]
        )
```

### Local Whisper

```python
import whisper

model = whisper.load_model("base")  # tiny, base, small, medium, large
result = model.transcribe("audio.mp3")
print(result["text"])

# With word-level timestamps
result = model.transcribe("audio.mp3", word_timestamps=True)
```

### Whisper Model Sizes

| Model | VRAM | Speed | Quality |
|-------|------|-------|---------|
| tiny | 1GB | Very fast | Basic |
| base | 1GB | Fast | Good |
| small | 2GB | Medium | Better |
| medium | 5GB | Slow | Great |
| large-v3 | 10GB | Very slow | Best |

## Text-to-Speech

### OpenAI TTS

```python
from openai import OpenAI
from pathlib import Path

client = OpenAI()

response = client.audio.speech.create(
    model="tts-1",  # or tts-1-hd for higher quality
    voice="alloy",  # alloy, echo, fable, onyx, nova, shimmer
    input="Hello! This is a test of text to speech."
)

response.stream_to_file(Path("output.mp3"))
```

### ElevenLabs (Higher Quality)

```python
from elevenlabs import generate, save

audio = generate(
    text="Hello! This sounds more natural.",
    voice="Rachel",
    model="eleven_multilingual_v2"
)

save(audio, "output.mp3")
```

## Image Generation

### DALL-E 3

```python
from openai import OpenAI

client = OpenAI()

response = client.images.generate(
    model="dall-e-3",
    prompt="A futuristic cityscape at sunset, cyberpunk style",
    size="1024x1024",
    quality="hd",
    n=1
)

image_url = response.data[0].url
```

### Stable Diffusion (Local)

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

image = pipe("A futuristic cityscape at sunset").images[0]
image.save("output.png")
```

## Video Understanding

### Gemini 1.5 Pro

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_KEY")

video_file = genai.upload_file("video.mp4")

model = genai.GenerativeModel("gemini-1.5-pro")
response = model.generate_content([
    video_file,
    "Summarize the key events in this video"
])
print(response.text)
```

## Building Multi-modal Pipelines

### Example: Meeting Assistant

```python
class MeetingAssistant:
    def __init__(self):
        self.whisper = whisper.load_model("medium")
        self.llm = OpenAI()
    
    def process_meeting(self, video_path: str) -> dict:
        # 1. Extract audio
        audio_path = self._extract_audio(video_path)
        
        # 2. Transcribe
        transcript = self.whisper.transcribe(audio_path)
        
        # 3. Extract screenshots at key moments
        screenshots = self._extract_keyframes(video_path)
        
        # 4. Analyze screenshots (slides, whiteboard)
        slide_content = [
            self._analyze_image(img) for img in screenshots
        ]
        
        # 5. Generate summary
        summary = self._generate_summary(
            transcript["text"],
            slide_content
        )
        
        # 6. Extract action items
        action_items = self._extract_actions(transcript["text"])
        
        return {
            "transcript": transcript,
            "slides": slide_content,
            "summary": summary,
            "action_items": action_items
        }
```

### Example: Document Processor

```python
class DocumentProcessor:
    def process_document(self, pdf_path: str) -> dict:
        # Convert PDF to images
        images = pdf_to_images(pdf_path)
        
        results = []
        for i, img in enumerate(images):
            # Analyze each page
            analysis = self.analyze_image(img, 
                "Extract all text and describe any charts, tables, or figures"
            )
            results.append({
                "page": i + 1,
                "content": analysis
            })
        
        # Synthesize into structured output
        return self.synthesize(results)
```

## Costs & Latency

| Modality | Model | Cost | Latency |
|----------|-------|------|---------|
| Vision | GPT-4V | $0.01/image | 2-5s |
| Speech→Text | Whisper API | $0.006/min | Real-time |
| Text→Speech | TTS-1 | $0.015/1K chars | 1-2s |
| Image Gen | DALL-E 3 | $0.04-0.12/image | 10-30s |

## Key Takeaways

- Vision-language models enable powerful document and image understanding
- Whisper provides excellent transcription (local or API)
- Multi-modal pipelines combine capabilities for complex workflows
- Consider cost and latency when designing production systems
- Open source alternatives exist for all modalities

## References

- [OpenAI Vision Guide](https://platform.openai.com/docs/guides/vision)
- [Whisper Paper](https://openai.com/research/whisper)
- [LLaVA](https://llava-vl.github.io/)
- [Gemini Multi-modal](https://ai.google.dev/tutorials/python_quickstart)
