# Prism - Trình tạo hoạt ảnh STEM tự động

**Prism** là một hệ thống AI tự động chuyển đổi các khái niệm STEM thành video hoạt ảnh giáo dục có lời tường thuật. Hệ thống sử dụng LLM để phân tích khái niệm, tạo mã Manim, render hoạt ảnh, và tổng hợp giọng nói đồng bộ.

## Tính năng chính

- 🎬 **Tự động tạo hoạt ảnh**: Chuyển đổi khái niệm STEM thành hoạt ảnh Manim
- 🗣️ **Lời tường thuật đa ngôn ngữ**: Hỗ trợ tiếng Việt và tiếng Anh
- 🤖 **Powered by AI**: Sử dụng Gemini 2.0 Flash cho reasoning và multimodal analysis
- 🎙️ **Text-to-Speech chất lượng cao**: Tích hợp ElevenLabs và OpenAI TTS
- 📊 **Giao diện web thân thiện**: Gradio UI đơn giản và dễ sử dụng

---

## Installation

### Prerequisites

**System Requirements:**
- Python 3.10+
- FFmpeg (for video processing)
- LaTeX (for mathematical notation in animations)

**API Keys Required:**
- [Google AI API Key](https://aistudio.google.com/app/apikey) (for LLM reasoning and multimodal video analysis)
- [ElevenLabs API Key](https://elevenlabs.io/) or [OpenAI API Key](https://platform.openai.com/) (for text-to-speech)

---

### Step 1: Install System Dependencies

#### Windows
```powershell
# Install Chocolatey if not already installed (run PowerShell as Administrator)
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install FFmpeg
choco install ffmpeg

# Install MiKTeX (LaTeX distribution for Windows)
choco install miktex

# After installation, restart your terminal and update PATH if needed
```

**Verify installations:**
```bash
ffmpeg -version
latex --version
```

---

### Step 2: Clone Repository

```bash
git clone https://github.com/qnguyen3/Prism.git
cd Prism
```

---

### Step 3: Install Python Environment (Using UV - Recommended)

We recommend using [UV](https://github.com/astral-sh/uv) for fast, reliable Python package management.

#### Install UV

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Create Virtual Environment and Install Dependencies

```bash
# Create virtual environment
uv venv

# Activate virtual environment
# macOS/Linux:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

**Alternative (using pip):**
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

### Step 4: Configure API Keys

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your API keys:
   ```bash
   # Required
   GOOGLE_API_KEY=your_google_ai_key_here
   
   # TTS Provider (choose one)
   TTS_PROVIDER=openai  # or "elevenlabs"
   OPENAI_API_KEY=your_openai_key_here
   # ELEVENLABS_API_KEY=your_elevenlabs_key_here
   ```

**Where to get API keys:**
- **Google AI**: Get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
- **OpenAI**: Sign up at [platform.openai.com](https://platform.openai.com/) and create an API key
- **ElevenLabs** (optional): Sign up at [elevenlabs.io](https://elevenlabs.io/) and get your API key from the profile page

---

### Step 5: Verify Manim Installation

Test that Manim is properly installed:

```bash
manim --version
```

If Manim is not found, ensure your virtual environment is activated and reinstall:
```bash
uv pip install --force-reinstall manim
```

---

## Usage

### Launch Gradio Web Interface

```bash
python app.py
```

The Gradio interface will open in your browser at `http://127.0.0.1:7860`

### Using the Interface

1. **Nhập khái niệm STEM bằng tiếng Việt** (e.g., "Giải thích thuật toán QuickSort", "Giải thích gradient descent", "Giải thích định lý Bayes")
2. **Chọn ngôn ngữ giọng đọc (TTS)**: Vietnamese hoặc English
   - Input luôn là tiếng Việt
   - Giọng đọc có thể chọn tiếng Việt hoặc tiếng Anh tùy người dùng
3. **Click "Tạo hoạt ảnh"**
4. **Chờ hệ thống xử lý** (thường mất 3-5 phút tùy độ phức tạp)
5. **Xem video đã tạo** với lời tường thuật đồng bộ

### Example Prompts

**Ví dụ prompt (luôn nhập bằng tiếng Việt):**
```
- Giải thích thuật toán bubble sort
- Giải thích gradient descent
- Giải thích định lý Bayes với ví dụ chẩn đoán y tế
- Giải thích backpropagation trong mạng neural
- Trực quan hóa biến đổi Fourier
- Giải thích định lý giới hạn trung tâm
- Giải thích cấu trúc dữ liệu cây nhị phân
- Giải thích thuật toán QuickSort
- Giải thích định lý Pythagoras
- Giải thích chuỗi Fibonacci
```

**Lưu ý**: 
- ✅ Input luôn bằng tiếng Việt
- ✅ Có thể chọn giọng đọc tiếng Việt hoặc tiếng Anh
- ✅ Hệ thống tự động hiểu và tạo hoạt ảnh phù hợp
```

---

## Architecture

```
User Input (STEM concept via Gradio)
  ↓
Concept Interpreter Agent (structured analysis)
  ↓
Manim Agent (scene planning → parallel code generation → rendering)
  ↓
Concatenated Silent Animation
  ↓
Script Generator (multimodal LLM analyzes video → timestamped narration)
  ↓
Audio Synthesizer (TTS with timing sync)
  ↓
Video Compositor (final MP4 with audio + subtitles)
  ↓
Display in Gradio
```

### Technology Stack

- **UI**: Gradio
- **Animation**: Manim Community Edition
- **LLMs**: 
  - Reasoning & Multimodal: Gemini 2.0 Flash
- **TTS**: OpenAI TTS hoặc ElevenLabs
- **Media Processing**: FFmpeg

---

## Configuration

Edit `config.py` to customize:

- **Animation quality**: `manim_quality` (l, m, h, p, k - từ thấp đến 4K)
- **LLM models**: `reasoning_model`, `multimodal_model` (Gemini models)
- **TTS provider**: `tts_provider` (openai hoặc elevenlabs)
- **TTS settings**: Voice ID, stability, similarity boost (cho ElevenLabs)
- **Video settings**: `video_codec`, `video_crf`, `audio_bitrate`
- **Language**: `target_language` (Vietnamese hoặc English)
- **Timeouts and retries**: Various `*_timeout` and `*_max_retries` settings

---

## Output Structure

```
output/
├── analyses/       # Concept analysis JSON files
├── scene_codes/    # Generated Manim code (cleaned up after success)
├── scenes/         # Individual scene videos (cleaned up after success)
├── animations/     # Concatenated silent animations (cleaned up after success)
├── scripts/        # Timestamped SRT narration scripts
├── audio/          # Generated speech audio (cleaned up after success)
│   └── segments/   # Individual audio segments (cleaned up after success)
└── final/          # Final videos with narration ✅ (KEPT)
```
