@echo off
set PYTHONPATH=temp_qwen_tts;%PYTHONPATH%
echo Running Qwen3-TTS App...
python src/app.py
pause
