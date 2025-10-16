# Hướng dẫn sử dụng Prism

## 🎯 Cách sử dụng đơn giản

### Bước 1: Khởi động ứng dụng
```bash
python app.py
```

### Bước 2: Mở trình duyệt
- Tự động mở tại: `http://127.0.0.1:7860`
- Hoặc click vào link trong terminal

### Bước 3: Nhập khái niệm STEM

**⚠️ QUAN TRỌNG: Luôn nhập bằng tiếng Việt**

Ví dụ:
```
✅ Giải thích thuật toán Bubble Sort
✅ Giải thích Định lý Bayes
✅ Giải thích Gradient Descent
✅ Trực quan hóa biến đổi Fourier
✅ Giải thích cấu trúc dữ liệu cây nhị phân
```

❌ **KHÔNG nhập tiếng Anh**:
```
❌ Explain Bubble Sort
❌ Explain Bayes Theorem
```

### Bước 4: Chọn ngôn ngữ giọng đọc (TTS)

Bạn có 2 lựa chọn:

1. **Vietnamese** (Mặc định)
   - Giọng đọc bằng tiếng Việt
   - Phù hợp cho học sinh, sinh viên Việt Nam

2. **English**
   - Giọng đọc bằng tiếng Anh
   - Phù hợp cho người học tiếng Anh hoặc môi trường quốc tế

### Bước 5: Tạo hoạt ảnh

1. Click nút **"Tạo hoạt ảnh"**
2. Chờ hệ thống xử lý (3-5 phút)
3. Xem video được tạo tự động

---

## 📋 Ví dụ cụ thể

### Ví dụ 1: Video tiếng Việt
- **Input**: "Giải thích thuật toán QuickSort"
- **Chọn TTS**: Vietnamese
- **Kết quả**: Video hoạt ảnh với giọng đọc tiếng Việt

### Ví dụ 2: Video tiếng Anh
- **Input**: "Giải thích thuật toán QuickSort" (vẫn tiếng Việt)
- **Chọn TTS**: English
- **Kết quả**: Video hoạt ảnh với giọng đọc tiếng Anh

---

## 🎓 Các khái niệm STEM phổ biến

### Thuật toán
```
- Giải thích thuật toán Bubble Sort
- Giải thích thuật toán QuickSort
- Giải thích thuật toán Merge Sort
- Giải thích thuật toán Binary Search
- Giải thích thuật toán Dijkstra
```

### Toán học
```
- Giải thích Định lý Pythagoras
- Giải thích Định lý Bayes
- Giải thích chuỗi Fibonacci
- Giải thích số phức
- Giải thích ma trận và định thức
```

### Machine Learning
```
- Giải thích Gradient Descent
- Giải thích Backpropagation
- Giải thích Neural Network
- Giải thích Decision Tree
- Giải thích K-means clustering
```

### Cấu trúc dữ liệu
```
- Giải thích cấu trúc dữ liệu Stack
- Giải thích cấu trúc dữ liệu Queue
- Giải thích cấu trúc dữ liệu cây nhị phân
- Giải thích Hash Table
- Giải thích Linked List
```

### Vật lý
```
- Giải thích định luật Newton
- Giải thích chuyển động parabol
- Giải thích sóng điện từ
- Giải thích hiệu ứng Doppler
```

---

## ⚙️ Cấu hình nâng cao

### Thay đổi chất lượng video
Chỉnh sửa file `config.py`:
```python
manim_quality: str = "p"  # p = 1080p60 (Production)
# Các tùy chọn: l, m, h, p, k
```

### Thay đổi TTS provider
Chỉnh sửa file `.env`:
```bash
# Dùng OpenAI TTS (mặc định)
TTS_PROVIDER=openai
OPENAI_API_KEY=your_key_here

# Hoặc dùng ElevenLabs
TTS_PROVIDER=elevenlabs
ELEVENLABS_API_KEY=your_key_here
```

---

## 🔧 Xử lý sự cố

### Lỗi: "No API key found"
**Giải pháp**: Kiểm tra file `.env` có đầy đủ API keys
```bash
GOOGLE_API_KEY=your_google_key
OPENAI_API_KEY=your_openai_key
```

### Lỗi: "Video generation failed"
**Giải pháp**:
1. Kiểm tra kết nối internet
2. Kiểm tra API keys còn credit
3. Xem log trong `output/pipeline.log`

### Video không có giọng đọc
**Giải pháp**:
1. Kiểm tra TTS provider trong `.env`
2. Kiểm tra API key của TTS provider
3. Thử đổi sang provider khác

---

## 📊 Hiệu suất

- **Thời gian tạo video**: 3-5 phút
- **Chất lượng video**: 1080p60 (mặc định)
- **Độ dài video**: 1-2 phút
- **Kích thước file**: 10-30 MB

---

## 💡 Mẹo sử dụng

1. **Khái niệm cụ thể**: Càng cụ thể càng tốt
   - ✅ "Giải thích thuật toán QuickSort với ví dụ mảng số"
   - ❌ "Giải thích thuật toán sắp xếp"

2. **Độ phức tạp vừa phải**: Không quá đơn giản, không quá phức tạp
   - ✅ "Giải thích Gradient Descent trong Machine Learning"
   - ❌ "Giải thích toàn bộ Machine Learning"

3. **Sử dụng ví dụ**: Yêu cầu ví dụ cụ thể
   - ✅ "Giải thích Định lý Bayes với ví dụ chẩn đoán y tế"

4. **Kiểm tra output**: Xem video trong thư mục `output/final/`

---

## 📞 Hỗ trợ

Nếu gặp vấn đề:
1. Đọc file `CHANGELOG_VI.md` để xem cập nhật mới nhất
2. Kiểm tra log trong `output/pipeline.log`
3. Đảm bảo đã cài đặt đầy đủ dependencies: `pip install -r requirements.txt`

---

## 🎉 Tính năng nổi bật

- ✅ **Tự động 100%**: Chỉ cần nhập khái niệm, hệ thống làm tất cả
- ✅ **Chất lượng cao**: Video 1080p60 với hoạt ảnh mượt mà
- ✅ **Đa ngôn ngữ TTS**: Chọn giọng đọc Việt hoặc Anh
- ✅ **AI thông minh**: Hiểu khái niệm và tạo hoạt ảnh phù hợp
- ✅ **Miễn phí**: Chỉ cần API keys (có gói miễn phí)
