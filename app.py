#!/usr/bin/env python3
"""
Gradio web app for STEM Animation Generator
Simple UI: Input concept -> Progress bar -> Video player
Supports Vietnamese UI and logging
"""

import gradio as gr
import logging
import sys
from pipeline import Pipeline
from pathlib import Path

# Configure logging with UTF-8 support
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("App")
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

pipeline = Pipeline()

# Translation dictionary for UI elements
UI_TEXT = {
    "English": {
        "title": "Prism",
        "description": "Transform STEM concepts into narrated educational animations",
        "concept_label": "Enter STEM Concept",
        "concept_placeholder": "e.g., Explain Bubble Sort, Bayes' Theorem, Gradient Descent...",
        "language_label": "Narration Language",
        "generate_button": "Generate Animation",
        "video_label": "Generated Animation",
        "examples": [
            ["Explain Bubble Sort"],
            ["Explain Bayes' Theorem"],
            ["Explain Gradient Descent"]
        ]
    },
    "Vietnamese": {
        "title": "Prism",
        "description": "Chuyển đổi các khái niệm STEM thành các hoạt hình giáo dục có lời dẫn",
        "concept_label": "Nhập Khái Niệm STEM",
        "concept_placeholder": "ví dụ: Giải thích thuật toán Sắp xếp Nổi, Định lý Bayes, Gradient Descent...",
        "language_label": "Ngôn Ngữ Lời Dẫn",
        "generate_button": "Tạo Hoạt Hình",
        "video_label": "Hoạt Hình Đã Tạo",
        "examples": [
            ["Giải thích thuật toán Sắp xếp Nổi"],
            ["Giải thích Định lý Bayes"],
            ["Giải thích Gradient Descent"]
        ]
    }
}

def generate_animation(concept: str, language: str = "English", progress=gr.Progress()):
    """
    Main generation function called by Gradio

    Args:
        concept: User input STEM concept (supports Vietnamese)
        language: Target language for narration (English, Chinese, Spanish, Vietnamese)
        progress: Gradio progress tracker

    Returns:
        Video file path or error message
    """
    logger.info(f"Generating animation for concept: {concept} in language: {language}")
    
    if not concept or concept.strip() == "":
        logger.warning("Empty concept provided")
        return None
    
    def update_progress(message: str, percentage: float):
        progress(percentage, desc=message)
    
    result = pipeline.run(concept, progress_callback=update_progress, target_language=language)
    
    if result["status"] == "success" and result.get("video_result"):
        video_path = result["video_result"]["output_path"]
        if Path(video_path).exists():
            logger.info(f"Video generated successfully: {video_path}")
            return video_path
        else:
            logger.error(f"Video file not found: {video_path}")
            return None
    else:
        logger.error(f"Pipeline failed: {result.get('error', 'Unknown error')}")
        return None

def create_interface(language: str = "English"):
    """Create Gradio interface with dynamic language support"""
    texts = UI_TEXT.get(language, UI_TEXT["English"])
    
    with gr.Blocks(title=texts["title"]) as demo:
        gr.Markdown(f"# {texts['title']}")
        gr.Markdown(texts["description"])
        
        with gr.Row():
            with gr.Column():
                concept_input = gr.Textbox(
                    label=texts["concept_label"],
                    placeholder=texts["concept_placeholder"],
                    lines=2
                )
                language_dropdown = gr.Dropdown(
                    choices=["English", "Chinese", "Spanish", "Vietnamese"],
                    value=language,
                    label=texts["language_label"]
                )
                generate_btn = gr.Button(texts["generate_button"], variant="primary")
        
        with gr.Row():
            video_output = gr.Video(
                label=texts["video_label"],
                autoplay=True
            )
        
        gr.Examples(
            examples=texts["examples"],
            inputs=concept_input
        )
        
        generate_btn.click(
            fn=generate_animation,
            inputs=[concept_input, language_dropdown],
            outputs=video_output
        )
        
        # Update interface when language changes
        language_dropdown.change(
            fn=create_interface,
            inputs=language_dropdown,
            outputs=demo
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False, inbrowser=True)