## Installation

### Prerequisites

**System Requirements:**
- Python 3.10+
- FFmpeg (for video processing)
- LaTeX (for mathematical notation in animations)

**API Keys Required:**
- [OpenRouter API Key](https://openrouter.ai/) (for LLM reasoning)
- [Google AI API Key](https://aistudio.google.com/app/apikey) (for multimodal video analysis)
- [ElevenLabs API Key](https://elevenlabs.io/) (for text-to-speech)

---

### Step 1: Install System Dependencies

#### macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install FFmpeg
brew install ffmpeg

# Install LaTeX (required for math rendering in Manim)
brew install --cask mactex

# After LaTeX installation, update PATH (add to ~/.zshrc or ~/.bash_profile)
export PATH="/Library/TeX/texbin:$PATH"
source ~/.zshrc  # or source ~/.bash_profile
```

#### Linux (Ubuntu/Debian)
```bash
# Update package list
sudo apt update

# Install FFmpeg
sudo apt install ffmpeg

# Install LaTeX (full TeX Live distribution)
sudo apt install texlive-full

# Alternative: minimal LaTeX install (faster, but may miss some packages)
# sudo apt install texlive texlive-latex-extra texlive-fonts-extra texlive-science
```

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
git clone https://github.com/qnguyen3/STEMViz.git
cd STEMViz
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
   OPENROUTER_API_KEY=your_openrouter_key_here
   GOOGLE_API_KEY=your_google_ai_key_here
   ELEVENLABS_API_KEY=your_elevenlabs_key_here
   ```

**Where to get API keys:**
- **OpenRouter**: Sign up at [openrouter.ai](https://openrouter.ai/) and create an API key
- **Google AI**: Get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
- **ElevenLabs**: Sign up at [elevenlabs.io](https://elevenlabs.io/) and get your API key from the profile page

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

1. **Enter a STEM concept** in the text box (e.g., "Explain QuickSort algorithm", "Demonstrate gradient descent", "Show Bayes' theorem")
2. **Click "Generate Animation"**
3. **Wait for the pipeline** to complete (typically 3-5 minutes depending on complexity)
4. **Watch the generated video** with synchronized narration

### Example Prompts

```
- Explain bubble sort algorithm
- Demonstrate gradient descent optimization
- Show Bayes' theorem with a medical diagnosis example
- Explain how backpropagation works in neural networks
- Visualize the Fourier transform
- Demonstrate the central limit theorem
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
  - Reasoning: Claude Sonnet 4.5 via OpenRouter
  - Multimodal: Gemini 2.5 Flash
- **TTS**: ElevenLabs
- **Media Processing**: FFmpeg

---

## Configuration

Edit `config.py` to customize:

- **Animation quality**: `manim_quality` (480p15, 720p30, 1080p60, 1440p60)
- **LLM models**: `reasoning_model`, `multimodal_model`
- **TTS settings**: `tts_voice_id`, `tts_stability`, `tts_similarity_boost`
- **Video settings**: `video_codec`, `video_crf`, `audio_bitrate`
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

**Note:** Temporary files are automatically cleaned up after successful video generation. Only final videos and scripts are preserved.

---

## Troubleshooting

### Common Issues

**1. "LaTeX not found" error**
```bash
# Verify LaTeX installation
latex --version

# macOS: Ensure PATH includes LaTeX
export PATH="/Library/TeX/texbin:$PATH"

# Linux: Reinstall TeX Live
sudo apt install texlive-full
```

**2. "FFmpeg not found" error**
```bash
# Verify FFmpeg installation
ffmpeg -version

# Reinstall if needed (macOS)
brew reinstall ffmpeg

# Linux
sudo apt reinstall ffmpeg
```

**3. "Manim command not found"**
```bash
# Ensure virtual environment is activated
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Reinstall Manim
uv pip install --force-reinstall manim
```

**4. API Key errors**
- Verify `.env` file exists and contains valid API keys
- Check API key quotas/limits on respective platforms
- Ensure no extra spaces or quotes around API keys in `.env`

**5. Out of memory errors**
- Reduce animation quality in `config.py`: `manim_quality = "720p30"`
- Reduce `manim_max_scene_duration` to simplify scenes
- Close other applications to free up RAM

**6. Slow generation**
- First run is slower due to LaTeX package downloads
- Subsequent runs are faster as packages are cached
- Complex concepts naturally take longer (3-5 minutes average)

---

## Development

### Project Structure

```
STEMViz/
├── agents/                  # AI agents
│   ├── concept_interpreter.py   # Analyzes and decomposes STEM concepts
│   ├── manim_agent.py            # Generates and renders Manim animations
│   └── manim_models.py           # Data models for animation pipeline
├── generation/              # Content generation
│   ├── script_generator.py       # Multimodal narration generation
│   ├── audio_synthesizer.py      # TTS audio synthesis
│   └── video_compositor.py       # Final video composition
├── rendering/               # Animation rendering
│   └── manim_renderer.py         # Manim code execution
├── utils/                   # Utilities
│   └── validators.py             # Input validation
├── config.py                # Centralized configuration
├── pipeline.py              # Main orchestration pipeline
├── app.py                   # Gradio web interface
└── requirements.txt         # Python dependencies
```

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under a non-commercial License - see the LICENSE file for details.

---

## Acknowledgments

- **Manim Community**: For the amazing mathematical animation engine
- **3Blue1Brown**: For inspiring educational math visualizations
- **Anthropic, Google, ElevenLabs**: For powerful AI APIs

---

## Citation

If you use STEMViz in your research or project, please cite:

```bibtex
@software{stemviz2025,
  author = {Nguyen, Quan},
  title = {STEMViz: AI-Powered STEM Concept Visualizer},
  year = {2025},
  url = {https://github.com/qnguyen3/STEMViz}
}
```

---

## Contact

**Quan Nguyen** - [@qnguyen3](https://github.com/qnguyen3)

Project Link: [https://github.com/qnguyen3/STEMViz](https://github.com/qnguyen3/STEMViz)

---

**⭐ Star this repo if you find it useful!**
