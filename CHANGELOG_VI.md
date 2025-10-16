# Nhật ký thay đổi - Prism

## Phiên bản mới nhất - Cập nhật ngày [Hôm nay]

### 🎯 Cập nhật quan trọng: Input tiếng Việt, TTS tùy chọn

**Thay đổi chính:**
- ✅ **Input luôn là tiếng Việt**: Người dùng nhập khái niệm STEM bằng tiếng Việt
- ✅ **TTS linh hoạt**: Người dùng chọn giọng đọc tiếng Việt hoặc tiếng Anh
- ✅ **Hệ thống thông minh**: AI hiểu input tiếng Việt và tạo video phù hợp

### Thay đổi chính

#### 1. Đổi tên dự án từ STEMViz sang Prism
- ✅ Cập nhật tất cả tài liệu và file cấu hình
- ✅ Thay đổi branding trong giao diện người dùng
- ✅ Cập nhật README.md với thông tin về Prism

#### 2. Giao diện tiếng Việt
- ✅ Chuyển đổi toàn bộ UI trong `app.py` sang tiếng Việt
- ✅ Các label, placeholder, và button đều được Việt hóa
- ✅ Ví dụ mẫu được cung cấp bằng tiếng Việt

#### 3. Hỗ trợ ngôn ngữ - CẬP NHẬT MỚI
- ✅ **Input**: Luôn nhập bằng tiếng Việt
- ✅ **TTS/Giọng đọc**: Người dùng chọn tiếng Việt hoặc tiếng Anh
- ✅ Loại bỏ hỗ trợ tiếng Trung (Chinese) và tiếng Tây Ban Nha (Spanish)
- ✅ Ngôn ngữ mặc định cho TTS: **Tiếng Việt**
- ✅ Label trong UI: "Nhập khái niệm STEM (bằng tiếng Việt)" và "Ngôn ngữ giọng đọc (TTS)"

#### 4. Cập nhật cấu hình
- ✅ `config.py`: Đổi `target_language` mặc định thành "Vietnamese"
- ✅ `app.py`: Dropdown chỉ hiển thị Vietnamese và English
- ✅ `pipeline.py`: Cập nhật default language thành Vietnamese
- ✅ `script_generator.py`: Cập nhật docstring và default parameters

#### 5. Cập nhật tài liệu
- ✅ README.md: Thêm phần giới thiệu về Prism
- ✅ README.md: Cập nhật hướng dẫn sử dụng với ví dụ tiếng Việt
- ✅ README.md: Cập nhật Technology Stack
- ✅ README.md: Cập nhật phần cấu hình API keys
- ✅ TTS_USAGE.md: Đổi tên STEMViz thành Prism

### Chi tiết các file đã thay đổi

1. **app.py** - CẬP NHẬT MỚI
   - Đổi title thành "Prism - Trình tạo hoạt ảnh STEM"
   - Việt hóa tất cả text trong UI
   - Label: "Nhập khái niệm STEM (bằng tiếng Việt)"
   - Label: "Ngôn ngữ giọng đọc (TTS)"
   - Dropdown ngôn ngữ chỉ còn ["Vietnamese", "English"]
   - Default language: Vietnamese

2. **config.py**
   - `target_language: str = "Vietnamese"`
   - Comment cập nhật: "Supported: Vietnamese, English"

3. **pipeline.py**
   - Default parameter `target_language: str = "Vietnamese"`
   - Docstring cập nhật: "Target language for narration (Vietnamese, English)"

4. **generation/script_generator.py** - CẬP NHẬT MỚI
   - Default parameter `target_language: str = "Vietnamese"`
   - Docstring cập nhật
   - Prompt mới: Hướng dẫn AI hiểu video có thể về khái niệm tiếng Việt
   - Prompt mới: Tạo narration theo ngôn ngữ được chọn (Vietnamese/English)

5. **agents/concept_interpreter.py** - CẬP NHẬT MỚI
   - System prompt mới: Hỗ trợ nhận input tiếng Việt
   - Hướng dẫn: "You can receive input in Vietnamese or English"
   - Hướng dẫn: "Always analyze and understand the concept regardless of input language"
   - Output JSON vẫn dùng English cho tính tương thích

6. **README.md** - CẬP NHẬT MỚI
   - Thêm header và giới thiệu về Prism
   - Thêm phần "Tính năng chính"
   - Cập nhật hướng dẫn cài đặt
   - Thêm ví dụ tiếng Việt
   - Cập nhật Technology Stack
   - Cập nhật phần Configuration

6. **TTS_USAGE.md**
   - Đổi "STEMViz TTS system" thành "Prism TTS system"

### Hướng dẫn sử dụng sau khi cập nhật

1. **Khởi động ứng dụng:**
   ```bash
   python app.py
   ```

2. **Sử dụng giao diện:** - CẬP NHẬT MỚI
   - ✅ **Nhập khái niệm STEM bằng tiếng Việt** (luôn luôn tiếng Việt)
   - ✅ **Chọn ngôn ngữ giọng đọc (TTS)**: Vietnamese hoặc English
   - ✅ Nhấn "Tạo hoạt ảnh"
   - ✅ Chờ video được tạo (3-5 phút)

3. **Ví dụ prompt (luôn bằng tiếng Việt):**
   - "Giải thích thuật toán Bubble Sort"
   - "Giải thích Định lý Bayes"
   - "Giải thích Gradient Descent"
   - "Giải thích cấu trúc dữ liệu cây nhị phân"
   - "Giải thích thuật toán QuickSort"
   - "Trực quan hóa biến đổi Fourier"

4. **Lựa chọn TTS:**
   - Chọn "Vietnamese" → Video có giọng đọc tiếng Việt
   - Chọn "English" → Video có giọng đọc tiếng Anh
   - Input vẫn luôn là tiếng Việt trong cả hai trường hợp

### Lưu ý quan trọng

- ⚠️ Các video đã tạo trước đây vẫn hoạt động bình thường
- ⚠️ Không cần thay đổi cấu hình API keys
- ⚠️ Hệ thống vẫn hỗ trợ cả ElevenLabs và OpenAI TTS
- ✅ Giao diện mới thân thiện hơn với người dùng Việt Nam
- ✅ Tất cả tính năng cũ vẫn hoạt động như trước

### Các tính năng không thay đổi

- ✅ Khả năng tạo hoạt ảnh Manim
- ✅ Tích hợp LLM (Gemini 2.0 Flash)
- ✅ Text-to-Speech (ElevenLabs/OpenAI)
- ✅ Video composition và subtitle
- ✅ Chất lượng output video
- ✅ Hiệu suất xử lý

---

## Hỗ trợ

Nếu bạn gặp vấn đề sau khi cập nhật, vui lòng:
1. Kiểm tra file `.env` có đầy đủ API keys
2. Đảm bảo đã cài đặt đầy đủ dependencies
3. Xóa cache và thử lại: `rm -rf __pycache__`
4. Kiểm tra log trong `output/pipeline.log`
