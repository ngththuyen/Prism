#!/usr/bin/env python3
"""
Gradio web app for Prism
Simple UI: Nhập chủ đề -> Tiến trình -> Xem video
"""

import gradio as gr
from pipeline import Pipeline
from pathlib import Path

pipeline = Pipeline()

def generate_animation(concept: str, language: str = "Vietnamese", duration_preset: str = "Ngắn (~30s)", progress=gr.Progress()):
    """
    Main generation function called by Gradio

    Args:
        concept: User input STEM concept
        language: Target language for narration (English, Chinese, Spanish, Vietnamese)
        progress: Gradio progress tracker

    Returns:
        Video file path or error message
    """
    if not concept or concept.strip() == "":
        return None
    
    def update_progress(message: str, percentage: float):
        progress(percentage, desc=message)
    
    result = pipeline.run(
        concept,
        progress_callback=update_progress,
        target_language=language,
        duration_preset=duration_preset
    )
    
    if result["status"] == "success" and result.get("video_result"):
        video_path = result["video_result"]["output_path"]
        if Path(video_path).exists():
            return video_path
        else:
            return None
    else:
        return None

with gr.Blocks(title="Prism") as demo:
    gr.Markdown("# Prism")
    gr.Markdown("Biến các khái niệm STEM thành video minh hoạ có thuyết minh")
    
    with gr.Row():
        with gr.Column():
            concept_input = gr.Textbox(
                label="Nhập chủ đề STEM",
                placeholder="Ví dụ: Giải thích Thuật toán Sắp xếp nổi bọt, Định lý Bayes, Gradient Descent...",
                lines=2
            )
            language_dropdown = gr.Dropdown(
                choices=["English", "Chinese", "Spanish", "Vietnamese"],
                value="Vietnamese",
                label="Ngôn ngữ thuyết minh"
            )
            duration_dropdown = gr.Dropdown(
                choices=["Ngắn (~30s)", "Trung bình (~60s)", "Dài (~120s)"],
                value="Ngắn (~30s)",
                label="Thời lượng video"
            )
            generate_btn = gr.Button("Tạo video", variant="primary")
        
    with gr.Row():
        video_output = gr.Video(
            label="Video đã tạo",
            autoplay=True
        )
    
    gr.Examples(
        examples=[
            ["Giải thích thuật toán Sắp xếp nổi bọt (Bubble Sort)"],
            ["Trình bày Định lý Bayes trong chẩn đoán y khoa"],
            ["Giải thích trực quan Gradient Descent"]
        ],
        inputs=concept_input
    )
    
    generate_btn.click(
        fn=generate_animation,
        inputs=[concept_input, language_dropdown, duration_dropdown],
        outputs=video_output
    )

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)
