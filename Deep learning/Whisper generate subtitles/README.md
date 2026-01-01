## Notebook quick view - [click here](https://note-book-share.vercel.app/#/https://github.com/A-k-a-sh/ML-DL-Projects/blob/main/Deep%20learning/Whisper%20generate%20subtitles/whisper-generate-subtitles.ipynb)

# Whisper - Automatic Subtitle Generation

Automatic subtitle and transcription generation for videos using OpenAI's Whisper speech recognition model. This project demonstrates state-of-the-art automatic speech recognition (ASR) for creating accurate subtitles with timestamps.

## Overview

This notebook implements an end-to-end pipeline for converting spoken audio in videos to text subtitles using OpenAI's Whisper model. Whisper is a powerful transformer-based ASR system trained on 680,000 hours of multilingual data, capable of transcription, translation, and language identification.

### What is Whisper?

Whisper is a general-purpose speech recognition model from OpenAI that:
- Transcribes speech to text in 99+ languages
- Handles noisy audio, accents, and technical language
- Provides word-level timestamps for synchronization
- Works with various audio qualities
- Requires no fine-tuning for most use cases

## Key Features

**Platform Flexibility:**

### Running on Google Colab:
- **Google Drive Integration**: 
  - Mount drive to access video files: `drive.mount('/content/drive')`
  - Store videos in Google Drive folders
  - Output subtitles directly to Drive
  - Persistent storage across sessions
  
- **Free GPU Access**: Utilize Colab's free GPU for faster processing
- **No Local Installation**: Everything runs in the cloud
- **Easy Sharing**: Share notebooks with collaborators

**Setup Steps for Colab:**
1. Mount Google Drive
2. Install Whisper: `!pip install git+https://github.com/openai/whisper.git`
3. Install FFmpeg: `!sudo apt update && sudo apt install ffmpeg`
4. Specify file paths in Drive
5. Run Whisper with chosen model

### Running on Kaggle:
- **Public File URLs**: Use direct links to video files
- **Kaggle Datasets**: Upload videos as Kaggle datasets
- **Built-in GPU**: Leverage Kaggle's GPU quota
- **Notebook Publishing**: Share results publicly

**Whisper Model Sizes:**

| Model | Parameters | Size | Speed | VRAM Required | Use Case |
|-------|-----------|------|-------|---------------|----------|
| **tiny** | 39M | ~150 MB | ~32x real-time | 1 GB | Quick drafts, low-resource |
| **base** | 74M | ~290 MB | ~16x real-time | 1 GB | Fast transcription |
| **small** | 244M | ~966 MB | ~6x real-time | 2 GB | Balanced speed/accuracy |
| **medium** | 769M | ~3 GB | ~2x real-time | 5 GB | High accuracy |
| **large** | 1550M | ~6 GB | ~1x real-time | 10 GB | Best accuracy |
| **turbo** | ~809M | ~3 GB | ~8x real-time | 6 GB | Optimized large model |

**Model Selection Guide:**
- **tiny/base**: Quick transcripts, low accuracy requirements, limited hardware
- **small**: Good balance for most YouTube videos
- **medium**: Professional transcription, important content
- **large**: Maximum accuracy, multilingual, technical content
- **turbo**: Best of both worlds (speed + accuracy)

**Command-Line Interface:**

Basic usage:
```bash
whisper "path/to/video.mp4" --model turbo --output_format srt --output_dir "path/to/output/"
```

Advanced options:
```bash
whisper video.mp4 \
  --model medium \
  --output_format srt \
  --language en \
  --task transcribe \
  --output_dir ./subtitles/ \
  --fp16 True \
  --device cuda
```

**Output Formats:**
- **SRT** (SubRip): Most common subtitle format with timestamps
  ```
  1
  00:00:00,000 --> 00:00:02,500
  Welcome to this tutorial.
  
  2
  00:00:02,500 --> 00:00:05,000
  Today we'll learn about AI.
  ```
- **VTT** (WebVTT): Web-native subtitle format
- **TXT**: Plain text without timestamps
- **TSV**: Tab-separated values with timestamps
- **JSON**: Structured data with word-level timestamps

**Supported Languages:**
- English, Spanish, French, German, Chinese, Japanese, Korean, Arabic, Hindi, Russian, Portuguese, Italian, Dutch, Polish, Turkish, Vietnamese, Indonesian, Hebrew, Greek, Swedish, Danish, Norwegian, Finnish, Czech, Romanian, Thai, and 70+ more

**Key Parameters:**

- `--model`: Choose model size (tiny, base, small, medium, large, turbo)
- `--output_format`: Subtitle format (srt, vtt, txt, tsv, json)
- `--output_dir`: Where to save subtitle files
- `--language`: Force specific language (auto-detect if not specified)
- `--task`: 'transcribe' (same language) or 'translate' (to English)
- `--fp16`: Use half-precision for faster inference (default: True on GPU)
- `--device`: 'cuda' for GPU, 'cpu' for CPU
- `--verbose`: Show detailed progress

**Workflow:**

1. **Video Upload**:
   - Colab: Upload to Google Drive or direct upload
   - Kaggle: Use dataset or public URL
   
2. **Model Selection**: Choose based on accuracy needs and hardware
   
3. **Processing**:
   - FFmpeg extracts audio from video
   - Whisper processes audio in 30-second chunks
   - Generates transcription with timestamps
   
4. **Output**:
   - SRT file saved to specified directory
   - Can be used with any video player
   - Compatible with YouTube, VLC, streaming platforms

**Advanced Features:**

- **Automatic Language Detection**: Whisper identifies the spoken language
- **Punctuation & Capitalization**: Automatically added to transcripts
- **Diarization Support**: (with additional tools) Identify different speakers
- **Timestamps**: Word-level and segment-level timing information
- **Translation**: Transcribe and translate to English simultaneously
- **Noise Robustness**: Handles background noise, music, multiple speakers

## Technologies Used

- **OpenAI Whisper**: Core ASR model
  - Transformer-based encoder-decoder architecture
  - Trained on 680k hours of multilingual audio
  - Multitask learning (transcription + translation)
  
- **FFmpeg**: Audio/video processing
  - Extract audio streams from video
  - Format conversion
  - Audio resampling to 16kHz (Whisper requirement)
  
- **PyTorch**: Deep learning backend for Whisper
- **NumPy**: Audio array processing
- **Google Drive API** (Colab): File system integration
- **Kaggle API** (Kaggle): Dataset management

## Use Cases

1. **Content Creation**:
   - YouTube video subtitles
   - Podcast transcriptions
   - Course material accessibility
   
2. **Accessibility**:
   - Deaf/hard-of-hearing support
   - Language learning
   - Multi-language support
   
3. **Documentation**:
   - Meeting transcripts
   - Interview recordings
   - Lecture notes
   
4. **Media Production**:
   - Video editing with text
   - Content search and indexing
   - Automated closed captions
   
5. **Research & Analysis**:
   - Analyze spoken content at scale
   - Sentiment analysis on audio
   - Content moderation

## Performance Considerations

**Processing Time:**
- **tiny**: ~2 minutes for 1-hour video
- **base**: ~4 minutes for 1-hour video
- **small**: ~10 minutes for 1-hour video
- **medium**: ~30 minutes for 1-hour video
- **large**: ~60 minutes for 1-hour video
- **turbo**: ~8 minutes for 1-hour video

**Accuracy:**
- English: 95-98% WER (Word Error Rate) with large models
- Other languages: 85-95% depending on language and audio quality
- Technical content: Use larger models for better accuracy

**Hardware Requirements:**
- **GPU Recommended**: 5-32x faster than CPU
- **VRAM**: See model table above
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: ~1-10 GB per model

## Setup Instructions

### For Google Colab:

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Install Whisper
!pip install git+https://github.com/openai/whisper.git

# 3. Install FFmpeg
!sudo apt update && sudo apt install ffmpeg

# 4. Run Whisper
file_path = '/content/drive/MyDrive/videos/my_video.mp4'
output_dir = '/content/drive/MyDrive/subtitles/'

!whisper "{file_path}" --model turbo --output_format srt --output_dir "{output_dir}"
```

### For Kaggle:

```python
# 1. Install Whisper
!pip install git+https://github.com/openai/whisper.git

# 2. Download video from URL
!wget -O video.mp4 "https://example.com/video.mp4"

# 3. Run Whisper
!whisper video.mp4 --model small --output_format srt
```

## Learning Outcomes

1. Working with state-of-the-art ASR models
2. Audio processing with FFmpeg
3. Google Colab and cloud computing
4. Subtitle file formats (SRT, VTT)
5. Model selection for accuracy/speed tradeoff
6. Handling multilingual content
7. Timestamp synchronization
8. Large model deployment considerations

## Limitations & Considerations

- **Hallucinations**: May generate plausible but incorrect text for silence/noise
- **Language Mixing**: Can struggle with code-switching
- **Background Noise**: Extreme noise degrades performance
- **Processing Time**: Large models are slow without GPU
- **File Size**: Large videos require more processing time and storage

## Tips for Best Results

1. **Choose Appropriate Model**: Balance speed and accuracy
2. **Clean Audio**: Remove excessive background noise if possible
3. **Specify Language**: Improves accuracy if language is known
4. **Use GPU**: 10-30x faster than CPU processing
5. **Segment Long Videos**: Process in chunks for very long content
6. **Review Output**: Manual review recommended for critical applications
7. **Fine-tune Timestamps**: Adjust if synchronization is off

## Future Enhancements

- Real-time streaming transcription
- Speaker diarization integration
- Custom vocabulary for domain-specific terms
- Automatic subtitle formatting and styling
- Batch processing for multiple videos
- Web interface for easier usage
- Integration with video editing software
- Multi-language subtitle generation
