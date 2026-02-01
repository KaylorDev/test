import gradio as gr
import os
from tts_engine import TTSEngine
from file_parser import parse_txt, parse_epub, parse_fb2
import tempfile
import soundfile as sf

# Initialize Engine (lazy loading will happen on first use)
engine = TTSEngine()

# Global variable to store the current voice prompt
current_voice_prompt = None
current_voice_name = "По умолчанию (Нет)"

def switch_model_ui(model_name):
    status_msg = f"Переключение на модель {model_name}..."
    print(status_msg)
    try:
        engine.switch_model(model_name)
        return f"Модель успешно переключена на {model_name}"
    except Exception as e:
        return f"Ошибка при смене модели: {e}"

def process_file_or_text(text_input, file_input):
    content = text_input
    if file_input is not None:
        file_path = file_input.name
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == '.txt':
                content = parse_txt(file_path)
            elif ext == '.epub':
                content = parse_epub(file_path)
            elif ext == '.fb2':
                content = parse_fb2(file_path)
            else:
                return "Неподдерживаемый формат файла."
        except Exception as e:
            return f"Ошибка чтения файла: {e}"
            
    return content

def generate_speech(text, file, voice_ref_audio, voice_ref_text):
    global current_voice_prompt
    
    # 1. Get content
    content = process_file_or_text(text, file)
    if not content or len(content.strip()) == 0:
        return None, "Ошибка: Нет текста для озвучки."
        
    # 2. Determine voice strategy
    # If user provided a reference audio in this call, use One-Shot
    # If global prompt is set, use that.
    # Otherwise, fail (Base model needs a clone reference) or allow if model supports it (Base usually doesn't without ref).
    
    status_msg = f"Генерация для {len(content)} символов..."
    print(status_msg)
    
    try:
        audio = None
        sr = None
        
        if voice_ref_audio is not None and voice_ref_text and len(voice_ref_text.strip()) > 0:
             # One-shot cloning
             print("Используется загруженный образец для клонирования.")
             audio, sr = engine.generate_with_audio_ref(content, voice_ref_audio, voice_ref_text)
        elif current_voice_prompt is not None:
             # Use stored prompt
             print("Используется сохраненный профиль голоса.")
             audio, sr = engine.generate(content, voice_prompt=current_voice_prompt)
        else:
             return None, "Ошибка: Вы должны предоставить образец голоса (аудио + текст) или создать профиль."
             
        if audio is not None:
            # Save to temporary file for Gradio to play
            # Gradio generic audio output wants a tuple (sr, data) or path
            return (sr, audio), "Готово!"
        else:
            return None, "Ошибка генерации."
            
    except Exception as e:
        return None, f"Ошибка: {e}"

def create_profile(ref_audio, ref_text):
    global current_voice_prompt, current_voice_name
    
    if ref_audio is None or not ref_text:
        return "Ошибка: Отсутствует аудио или текст."
        
    print(f"Создание профиля из {ref_audio}...")
    prompt = engine.create_voice_prompt(ref_audio, ref_text)
    
    if prompt:
        current_voice_prompt = prompt
        current_voice_name = "Свой голос (Готов)"
        return "Профиль голоса успешно создан!"
    else:
        return "Не удалось создать профиль голоса."

# --- UI Layout ---

with gr.Blocks(title="Qwen3-TTS Читалка") as demo:
    gr.Markdown("# Qwen3-TTS Озвучка и Клонирование")
    
    with gr.Tabs():
        # --- Tab 1: Text to Speech ---
        with gr.TabItem("Чтение / Озвучка"):
            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.TextArea(label="Введите текст", placeholder="Вставьте текст сюда...", lines=10)
                    file_upload = gr.File(label="Или загрузите файл (.txt, .epub, .fb2)", file_types=[".txt", ".epub", ".fb2"])
                
                with gr.Column(scale=1):
                    gr.Markdown("### Настройки системы")
                    model_dropdown = gr.Dropdown(
                        label="Выберите модель",
                        choices=["Qwen/Qwen3-TTS-12Hz-1.7B-Base", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"],
                        value="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
                    )
                    model_status = gr.Textbox(label="Статус модели", value="Модель 1.7B (Base) готова к использованию", interactive=False)
                    
                    gr.Markdown("### Настройки голоса")
                    
                    # Quick Clone inputs right here for convenience
                    ref_audio_input = gr.Audio(label="Образец голоса (аудио файл)", type="filepath")
                    ref_text_input = gr.Textbox(label="Текст образца (что сказано в аудио)", placeholder="Напишите в точности то, что говорится в аудио...")
                    
                    target_voice_status = gr.Markdown(f"Текущий профиль голоса: **{current_voice_name}**")
                    
                    generate_btn = gr.Button("Озвучить", variant="primary")
            
            output_audio = gr.Audio(label="Результат", autoplay=False)
            status_output = gr.Textbox(label="Статус", interactive=False)
            
            generate_btn.click(
                fn=generate_speech,
                inputs=[text_input, file_upload, ref_audio_input, ref_text_input],
                outputs=[output_audio, status_output]
            )

        # --- Tab 2: Voice Cloning Studio ---
        with gr.TabItem("Студия Клонирования Голоса"):
            gr.Markdown("Создайте профиль голоса здесь, чтобы не загружать образец каждый раз.")
            
            with gr.Row():
                with gr.Column():
                    clone_audio = gr.Audio(label="Загрузить образец голоса", type="filepath")
                    clone_text = gr.Textbox(label="Текст образца", placeholder="Введите точный текст из образца...")
                    create_btn = gr.Button("Создать профиль голоса")
                
                with gr.Column():
                    profile_status = gr.Textbox(label="Результат")
            
            create_btn.click(
                fn=create_profile,
                inputs=[clone_audio, clone_text],
                outputs=[profile_status]
            )
            # Update the status label in the other tab when profile changes
            create_btn.click(
                fn=lambda: f"Текущий профиль голоса: **Свой голос (Готов)**",
                inputs=[],
                outputs=[target_voice_status]
            )

        model_dropdown.change(
            fn=switch_model_ui,
            inputs=[model_dropdown],
            outputs=[model_status]
        )

if __name__ == "__main__":
    demo.launch()
