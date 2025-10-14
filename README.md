## Prism — Tạo video minh hoạ STEM bằng nhiều tác tử AI

Prism là hệ thống tạo video giáo dục: bạn nhập chủ đề STEM, các tác tử AI sẽ phân tích khái niệm, lập kế hoạch cảnh, sinh mã Manim để dựng hoạt hình 2D, tạo phụ đề SRT đồng bộ, tổng hợp giọng nói (TTS), rồi ghép thành video MP4 có thuyết minh và phụ đề (burn‑in) sẵn sàng chia sẻ.

### Tính năng chính
- Tác tử Concept Interpreter → phân tích chủ đề thành các tiểu mục
- Tác tử Manim → lập kế hoạch scene, sinh mã, render từng cảnh và nối video câm
- Script Generator → xem video câm và sinh phụ đề SRT theo ngôn ngữ mục tiêu
- TTS đa nhà cung cấp (ElevenLabs, OpenAI) → giọng Việt tự nhiên
- Ghép video + audio + phụ đề bằng FFmpeg, hỗ trợ font tiếng Việt khi burn‑in
- Giao diện Gradio tiếng Việt, mặc định ngôn ngữ thuyết minh là “Vietnamese”

---

## Cài đặt

### Yêu cầu hệ thống
- Python 3.10+
- FFmpeg
- LaTeX (để hiển thị công thức trong Manim)

### API keys cần có
- OpenRouter API Key (LLM reasoning)
- Google AI API Key (multimodal sinh phụ đề SRT)
- ElevenLabs hoặc OpenAI API Key (TTS)

### Cài hệ thống phụ thuộc
macOS:
```bash
brew install ffmpeg
brew install --cask mactex
export PATH="/Library/TeX/texbin:$PATH"
```
Ubuntu/Debian:
```bash
sudo apt update && sudo apt install -y ffmpeg texlive-full
```
Windows (PowerShell Admin):
```powershell
choco install ffmpeg
choco install miktex
```

### Tải mã nguồn và cài thư viện Python
```bash
git clone https://github.com/ngththuyen/Prism.git
cd Prism

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Cấu hình .env
Tạo file `.env` và điền:
   ```bash
OPENROUTER_API_KEY=...
GOOGLE_API_KEY=...
ELEVENLABS_API_KEY=...      # hoặc OPENAI_API_KEY=...
TTS_PROVIDER=elevenlabs     # hoặc openai
```

---

## Sử dụng

Chạy giao diện Gradio:
```bash
python app.py
```
Mặc định mở tại `http://127.0.0.1:7860`. Nhập chủ đề bằng tiếng Việt, chọn “Ngôn ngữ thuyết minh” (mặc định: Vietnamese), bấm “Tạo video”.

Ví dụ prompt:
```
Giải thích thuật toán Sắp xếp nổi bọt (Bubble Sort)
Trình bày Định lý Bayes trong chẩn đoán y khoa
Giải thích trực quan Gradient Descent
```

### Tuỳ chọn thời lượng video
- Chọn "Thời lượng video":
  - Ngắn (~30s): 2–3 cảnh ngắn, render nhanh.
  - Trung bình (~60s): 3–5 cảnh, cân bằng tốc độ/chất lượng.
  - Dài (~120s): 4–8 cảnh, nội dung chi tiết hơn.
Prism sẽ phân phối độ dài cảnh theo tổng thời lượng mục tiêu để giảm thời gian chờ khi chọn video ngắn.

---

## Cấu hình quan trọng (`config.py`)
- `target_language = "Vietnamese"` (mặc định)
- TTS:
  - `tts_provider = "elevenlabs" | "openai"`
  - ElevenLabs: `elevenlabs_voice_id`, `elevenlabs_model_id`, …
  - OpenAI: `openai_voice`, `openai_model`, …
- Phụ đề (burn‑in):
  - `subtitle_burn_in = True`
  - `subtitle_font_path` → ĐƯỜNG DẪN FONT có hỗ trợ tiếng Việt (ví dụ: NotoSans-Regular.ttf). Prism sẽ truyền `fontsdir` cho ffmpeg để đảm bảo dấu tiếng Việt hiển thị đúng.
- Manim: `manim_quality`, `manim_frame_rate`, `manim_max_scene_duration`, …

---

## Kiến trúc
```
Người dùng (Gradio)
  ↓
Concept Interpreter Agent
  ↓
Manim Agent (lập kế hoạch cảnh → sinh mã → render → nối video câm)
  ↓
Script Generator (LLM đa phương thức → phụ đề SRT có timestamp)
  ↓
TTS (đồng bộ thời gian)
  ↓
Video Compositor (FFmpeg: ghép + burn‑in phụ đề)
  ↓
Hiển thị trong Gradio
```

### Công nghệ
- UI: Gradio
- Animation: Manim Community Edition
- LLMs: OpenRouter (Reasoning), Google AI (Multimodal)
- TTS: ElevenLabs, OpenAI
- Media: FFmpeg

---

## Cấu trúc thư mục
```
Prism/
├── agents/
│   ├── concept_interpreter.py
│   ├── manim_agent.py
│   └── manim_models.py
├── generation/
│   ├── script_generator.py
│   ├── tts/
│   │   ├── elevenlabs_provider.py
│   │   └── openai_provider.py
│   └── video_compositor.py
├── rendering/
│   └── manim_renderer.py
├── utils/
├── app.py
├── pipeline.py
├── config.py
└── requirements.txt
```

### Thư mục đầu ra
```
output/
├── analyses/
├── scene_codes/
├── scenes/
├── animations/
├── scripts/      # *.srt
├── audio/
└── final/        # *.mp4 cuối cùng
```

---

## Mẹo & xử lý sự cố
- Manim không chạy: đảm bảo đã kích hoạt virtualenv, `manim --version` ok.
- FFmpeg/LaTeX thiếu: cài lại theo phần Cài đặt; kiểm tra `ffmpeg -version`, `latex --version`.
- Phụ đề lỗi dấu: đặt `subtitle_font_path` tới font có hỗ trợ tiếng Việt, ví dụ Noto Sans.
- Render chậm/lỗi bộ nhớ: giảm `manim_quality` hoặc `manim_max_scene_duration`.

---

## Giấy phép
Dự án sử dụng giấy phép phi thương mại. Xem chi tiết trong `LICENSE`.

---

## Góp ý & liên hệ
- Vui lòng mở issue hoặc PR để đóng góp tính năng/sửa lỗi.
- Chạy xem trước UI: `python app.py`.

Nếu Prism hữu ích, hãy gắn ⭐ cho repo nhé!
