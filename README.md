# Google Gemini AI - Video, Audio & Image Processing

**Mr Fugu Data Science** (◕‿◕✿)

A comprehensive Google Colab notebook demonstrating how to use Google Gemini AI API for multimedia processing including video analysis, audio extraction, and image processing using Large Language Models.

**Video Tutorial**: [YouTube | How to use Gemini API to Extract Text from Images](https://www.youtube.com/watch?v=ddhkSxGC6Rs&list=PLsqV2CNcWMnTIvAejMN7cB-lHLWrsc5e4&index=4)

## About

This project demonstrates the capabilities of Google Gemini 1.5 Flash & Pro models for processing multimedia content including:

- 📹 **Video Analysis**: Extract insights, summaries, and specific information from video content
- 🎵 **Audio Processing**: Extract audio from videos and analyze audio content
- 🖼️ **Image Processing**: Extract text and analyze images using computer vision
- 🤖 **LLM Integration**: Combine multiple AI capabilities for comprehensive content analysis

## Google Gemini 1.5 Flash & Pro

- **Documentation**: [Gemini Flash Documentation](https://deepmind.google/technologies/gemini/flash/)

## Gemini Flash 1.5 Pricing

- **Pricing Information**: [https://ai.google.dev/pricing](https://ai.google.dev/pricing)
- **Available Countries**: [Regional Availability](https://ai.google.dev/gemini-api/docs/available-regions)

## Setup and Installation

### Required Dependencies

```python
# Install required packages
pip install -q -U google-generativeai  # Google Gemini AI API
pip install pytubefix                   # YouTube video downloading
pip install requests                    # HTTP requests
pip install python-dotenv              # Environment variables
```

### Environment Setup

```python
import google.generativeai as genai
from dotenv import load_dotenv
import os
from pytubefix import YouTube
import requests
from IPython.display import Image

# Load environment variables
load_dotenv()
api_key = os.getenv("Google_Gemini_API")

# For Google Colab users
try:
    from google.colab import userdata
    api_key = userdata.get('Google_Gemini_API') or api_key
except ImportError:
    pass  # Not running in Colab

# Configure Gemini AI
genai.configure(api_key=api_key)

# Initialize Gemini model
model = genai.GenerativeModel('models/gemini-1.5-pro-001')
```

## Features

### 1. YouTube Video Processing

Download and process YouTube videos for analysis:

```python
def sanitize_filename(filename):
    return "".join(c if c.isalnum() or c in " ._-" else "_" for c in filename)

def audio(thelink, path):
    """Extract audio from YouTube video"""
    try:
        yt = YouTube(thelink)
        print('Title:', yt.title)
        print('Views:', yt.views)
        yd = yt.streams.get_audio_only()
        yt_title = sanitize_filename(yt.title)
        yd.download(output_path=path, filename=f'{yt_title}.mp3')
        print('Finished downloading audio')
    except Exception as e:
        print(f"Error: {e}")

def high(thelink, path):
    """Download high quality video"""
    try:
        yt = YouTube(thelink)
        print('Title:', yt.title)
        print('Views:', yt.views)
        yt_title = sanitize_filename(yt.title)
        
        # Download highest resolution video
        video_stream = yt.streams.filter().order_by("resolution").last()
        audio_stream = yt.streams.get_audio_only()
        
        video_filename = f'{yt_title}.mp4'
        audio_filename = f'{yt_title}.mp3'
        
        video_stream.download(output_path=path, filename=video_filename)
        audio_stream.download(output_path=path, filename=audio_filename)
        
        print('Finished downloading video and audio')
    except Exception as e:
        print(f"Error: {e}")
```

### 2. Image Analysis with Gemini

Process and analyze images using Gemini's vision capabilities:

```python
# Load and analyze images
image_path = "your_image.jpg"
image = Image.open(image_path)

# Generate content from image
response = model.generate_content([
    "Analyze this image and describe what you see:",
    image
])

print(response.text)
```

### 3. Video Content Analysis

Analyze video content for insights and summaries:

```python
# Video analysis example
# You can summarize a video or find specific images within a video
# and extract information from it
```

### 4. Audio Transcription

Transcribe audio files with timecodes:

```python
import pathlib

# Create the prompt
prompt = """Can you transcribe this audio, in the format of timecode,caption and separate each section by a new line"""

# Load audio file and pass to Gemini
response = model.generate_content([
    prompt,
    {
        "mime_type": "audio/mp3",
        "data": pathlib.Path('path/to/your/audio.mp3').read_bytes()
    }
])

# Output transcription
print(response.text)
```

## Available Gemini Models

The notebook supports various Gemini models:

- `models/gemini-1.0-pro-vision-latest`
- `models/gemini-pro-vision`
- `models/gemini-1.5-pro-latest`
- `models/gemini-1.5-pro-001`
- `models/gemini-1.5-flash-latest`

## Use Cases

1. **Content Creation**: Analyze videos for content summaries and insights
2. **Educational Tools**: Extract educational content from multimedia sources
3. **Data Extraction**: Pull structured data from images and videos
4. **Accessibility**: Generate descriptions and transcripts for multimedia content
5. **Research**: Analyze large amounts of video/audio content efficiently

## File Structure

```
📁 Project Directory
├── 📄 README.md                                    # This file
├── 📓 GeminiAIP_Colab_VideoAudio_Img.ipynb        # Main Jupyter notebook
├── 📓 GeminiAIP_Colab_VideoAudio_Img copy.ipynb   # Backup notebook
├── 📄 GeminiAIP_Colab_VideoAudio_Img.html         # HTML export
└── 🖼️ IMG_7850.png                                # Sample image
```

## Getting Started

1. **Clone or download** this repository
2. **Set up your API key** in environment variables or Google Colab secrets
3. **Open the notebook** in Google Colab or Jupyter
4. **Run the cells** to start experimenting with Gemini AI
5. **Modify the examples** for your specific use case

## API Key Setup

### For Local Development:
Create a `.env` file:
```
Google_Gemini_API=your_api_key_here
```

### For Google Colab:
1. Go to the "Secrets" tab in Colab
2. Add a new secret named `Google_Gemini_API`
3. Paste your API key as the value

## Examples and Tutorials

This project is based on and inspired by various tutorials and examples. Check the citations section for detailed resources.

## Contributing

Feel free to contribute by:
- Adding new examples
- Improving documentation
- Reporting issues
- Suggesting new features

## Citations & References

### Core Resources
- [Gemini Flash Documentation](https://deepmind.google/technologies/gemini/flash/)
- [Building RAG Applications with LLM](https://medium.com/rahasak/build-rag-application-using-a-llm-running-on-local-computer-with-ollama-and-langchain-e6513853fda0)
- [Google Cloud AI/ML Notebooks](https://cloud.google.com/vertex-ai/docs/tutorials/jupyter-notebooks)
- [Gemini Getting Started Guide](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/getting-started/intro_gemini_1_5_pro.ipynb)

### LangChain & Prompt Engineering
- [Understanding Prompt Templates in LangChain](https://medium.com/@punya8147_26846/understanding-prompt-templates-in-langchain-f714cd7ab380)
- [Prompt Chaining with LLM](https://www.datacamp.com/tutorial/prompt-chaining-llm)
- [Image Extraction with LangChain and Gemini](https://medium.com/vectrix-ai/image-extraction-with-langchain-and-gemini-a-step-by-step-guide-02c79abcd679)

### Video & Audio Processing
- [Building a Video Insights Generator with Gemini Flash](https://medium.com/pythoneers/building-a-video-insights-generator-using-gemini-flash-e4ee4fefd3ab)
- [Extracting Audio from Video using MoviePy](https://medium.com/featurepreneur/extracting-audio-from-video-using-pythons-moviepy-library-e351cd652ab8)
- [Live Transcription with Python](https://dev.to/deepgram/live-transcription-with-python-and-django-4aj2)
- [Real-time Transcription](https://www.assemblyai.com/blog/real-time-transcription-in-python/)

### Specialized Applications
- [AI for Drug Discovery with Python](https://medium.com/@ibrahimmukherjee/ai-for-drug-discovery-with-python-code-47e1fe3a8233)
- [Building Asynchronous LLM Applications](https://diverger.medium.com/building-asynchronous-llm-applications-in-python-f775da7b15d1)
- [Voice Chat with Gemini](https://rohitraj-iit.medium.com/part-10-voice-chat-with-gemini-484514580033)
- [Video Translation with LLM](https://blogs.jollytoday.com/how-to-use-llm-such-as-gemini-and-chatgpt-for-video-translation-e22ff076e885)

### YouTube API & PyTube
- [YouTube Video Uploading Guide](https://developers.google.com/youtube/v3/guides/uploading_a_video)
- [PyTube Troubleshooting](https://www.reddit.com/r/learnpython/comments/156p9ju/pytube_errors_help_fix_or_recommend_other_library/)

### Google Cloud Examples
- [Gemini Python Tutorial](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/getting-started/intro_gemini_python.ipynb)
- [Multimodal Use Cases](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/intro_multimodal_use_cases.ipynb)

## License

This project is for educational purposes. Please ensure you comply with Google's API terms of service and YouTube's terms of service when using this code.

## Support

If you find this project helpful, please ⭐ star the repository and consider sharing it with others who might benefit from it.

---

**Created with ❤️ by Mr Fugu Data Science** (◔̯◔)
