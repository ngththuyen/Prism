#!/usr/bin/env python3
"""
Ứng dụng web Gradio cho Prism - Trình tạo hoạt ảnh STEM
Giao diện đơn giản: Nhập khái niệm -> Thanh tiến trình -> Trình phát video
"""

import gradio as gr
from pipeline import Pipeline
from pathlib import Path

pipeline = Pipeline()

def generate_animation(concept: str, language: str = "Vietnamese", progress=gr.Progress()):
    """
    Hàm tạo hoạt ảnh chính được gọi bởi Gradio

    Args:
        concept: Khái niệm STEM từ người dùng
        language: Ngôn ngữ mục tiêu cho lời tường thuật (Vietnamese, English)
        progress: Bộ theo dõi tiến trình Gradio

    Returns:
        Đường dẫn file video hoặc thông báo lỗi
    """
    if not concept or concept.strip() == "":
        return None
    
    def update_progress(message: str, percentage: float):
        progress(percentage, desc=message)
    
    result = pipeline.run(concept, progress_callback=update_progress, target_language=language)
    
    if result["status"] == "success" and result.get("video_result"):
        video_path = result["video_result"]["output_path"]
        if Path(video_path).exists():
            return video_path
        else:
            return None
    else:
        return None

with gr.Blocks(title="Prism - Trình tạo hoạt ảnh STEM") as demo:
    gr.Markdown("# Prism")
    gr.Markdown("Chuyển đổi các khái niệm STEM thành hoạt ảnh giáo dục có lời tường thuật")
    
    with gr.Row():
        with gr.Column():
            concept_input = gr.Textbox(
                label="Nhập khái niệm STEM (bằng tiếng Việt)",
                placeholder="Ví dụ: Giải thích thuật toán Bubble Sort, Định lý Bayes, Gradient Descent...",
                lines=2
            )
            language_dropdown = gr.Dropdown(
                choices=["Vietnamese", "English"],
                value="Vietnamese",
                label="Ngôn ngữ giọng đọc (TTS)"
            )
            generate_btn = gr.Button("Tạo hoạt ảnh", variant="primary")
        
    with gr.Row():
        video_output = gr.Video(
            label="Hoạt ảnh đã tạo",
            autoplay=True
        )
    
    gr.Examples(
        examples=[
            ["Giải thích thuật toán Bubble Sort"],
            ["Giải thích Định lý Bayes"],
            ["Giải thích Gradient Descent"]
        ],
        inputs=concept_input
    )
    
    generate_btn.click(
        fn=generate_animation,
        inputs=[concept_input, language_dropdown],
        outputs=video_output
    )

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)
