# Whisper - Automatic Subtitle Generation

Automatic subtitle generation for videos using OpenAI's Whisper speech recognition model.

## Overview
- Generates subtitles (SRT format) from video files automatically
- Uses OpenAI's Whisper model for audio transcription
- Supports multiple Whisper model sizes (tiny, base, small, medium, large, turbo)
- Can be run on Google Colab or Kaggle

## Features
- Automatic speech-to-text transcription
- SRT subtitle file generation
- Multiple model size options for accuracy vs. speed tradeoff
- Support for various languages
- Direct integration with Google Drive for file access

## Requirements
- OpenAI Whisper package
- FFmpeg for audio processing
- Google Drive (for Colab) or file upload (for Kaggle)

## Model Sizes
Models range from tiny to turbo, with different performance characteristics:
- Tiny/Base: Faster, less accurate
- Small/Medium: Balanced
- Large/Turbo: Most accurate, slower

## Usage Platforms
- **Google Colab**: Upload videos to Google Drive and process
- **Kaggle**: Use public file links to download and process

## Output Format
- SRT (SubRip Subtitle) format
- Contains timestamps and transcribed text
- Can be used with any video player that supports subtitles
