import torch
import sys
import os

print(f"Python version: {sys.version}")
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Flash Attention available: {torch.backends.cuda.flash_sdp_enabled()}")

try:
    import qwen_tts
    print("qwen_tts imported successfully.")
except ImportError as e:
    print(f"Failed to import qwen_tts: {e}")

try:
    import gradio as gr
    print("gradio imported successfully.")
except ImportError as e:
    print(f"Failed to import gradio: {e}")
