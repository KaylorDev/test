import gradio as gr
import os
from tts_engine import TTSEngine
from file_parser import parse_txt, parse_epub, parse_fb2
import tempfile
import soundfile as sf
import glob
import time
import psutil

# Initialize Engine (lazy loading will happen on first use)
engine = TTSEngine()

# State will be handled by Gradi State components inside the Blocks
# Removing globals initialization here as they are now local to the session


VOICES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "voices")
os.makedirs(VOICES_DIR, exist_ok=True)

def get_saved_voices():
    files = glob.glob(os.path.join(VOICES_DIR, "*.pt"))
    names = [os.path.splitext(os.path.basename(f))[0] for f in files]
    return ["Ничего"] + sorted(names)

def load_voice_ui(voice_name):
    if voice_name == "Ничего" or not voice_name:
        return None, "По умолчанию (Нет)", "Голос сброшен на стандартный."
    
    path = os.path.join(VOICES_DIR, f"{voice_name}.pt")
    if os.path.exists(path):
        prompt = engine.load_voice_prompt(path)
        if prompt is not None:
             return prompt, voice_name, f"Загружен голос: {voice_name}"
        else:
             return None, "По умолчанию (Нет)", "Ошибка загрузки голоса."
    else:
        return None, "По умолчанию (Нет)", "Файл голоса не найден."

def switch_model_ui(model_name):
    status_msg = f"Переключение на модель {model_name}..."
    print(status_msg)
    try:
        engine.switch_model(model_name)
        device_name = engine.device.upper()
        return f"Модель переключена на {model_name}. Устройство: {device_name}"
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

def generate_speech(text, file, voice_ref_audio, voice_ref_text, current_voice_prompt, progress=gr.Progress()):
    # 1. Get content
    content = process_file_or_text(text, file)
    if not content or len(content.strip()) == 0:
        return None, "Ошибка: Нет текста для озвучки."
        
    # 2. Determine voice strategy
    
    status_msg = f"Генерация для {len(content)} символов..."
    print(status_msg)
    progress(0, desc="Подготовка...")
    
    try:
        audio = None
        sr = None
        
        # Define a callback wrapper for Gradio progress
        def progress_wrapper(pct, msg):
            progress(pct, desc=msg)
        
        if voice_ref_audio is not None and voice_ref_text and len(voice_ref_text.strip()) > 0:
             # One-shot cloning
             print("Используется загруженный образец для клонирования.")
             audio, sr = engine.generate_with_audio_ref(content, voice_ref_audio, voice_ref_text, progress_callback=progress_wrapper)
        elif current_voice_prompt is not None:
             # Use stored prompt
             print("Используется сохраненный профиль голоса.")
             audio, sr = engine.generate(content, voice_prompt=current_voice_prompt, progress_callback=progress_wrapper)
        else:
             return None, "Ошибка: Вы должны предоставить образец голоса (аудио + текст) или создать профиль."
             
        if audio is not None:
            return (sr, audio), "Готово!"
        else:
            return None, "Ошибка генерации."
            
    except Exception as e:
        return None, f"Ошибка: {e}"

def create_profile(ref_audio, ref_text, voice_name_save):
    if ref_audio is None or not ref_text:
        return None, "По умолчанию (Нет)", "Ошибка: Отсутствует аудио или текст."
        
    print(f"Создание профиля из {ref_audio}...")
    prompt = engine.create_voice_prompt(ref_audio, ref_text)
    
    if prompt:
        new_prompt = prompt
        new_name = "Свой (unsaved)"
        msg = "Профиль создан (в памяти)."
        
        if voice_name_save and len(voice_name_save.strip()) > 0:
            safe_name = "".join([c for c in voice_name_save if c.isalpha() or c.isdigit() or c in (' ', '_', '-')]).strip()
            save_path = os.path.join(VOICES_DIR, f"{safe_name}.pt")
            saved = engine.save_voice_prompt(prompt, save_path)
            if saved:
                new_name = safe_name
                msg = f"Профиль создан и сохранен как '{safe_name}'!"
            else:
                msg = "Профиль создан, но ошибка при сохранении."
        
        return new_prompt, new_name, msg
    else:
        return None, "По умолчанию (Нет)", "Не удалось создать профиль голоса."
        
def refresh_voices_list():
    return gr.Dropdown(choices=get_saved_voices())

def update_monitor():
    info = engine.get_device_status()
    if isinstance(info, dict):
        ram = f"RAM: {info['ram_used_gb']} GB ({info['ram_percent']}%)"
        vram = ""
        if info['device_name'] != "CPU":
            vram = f" | VRAM: {info['vram_allocated_gb']} GB (Alloc) / {info['vram_reserved_gb']} GB (Rsrv)"
        return f"{ram}{vram} [{info['device_name']}]"
    return str(info)

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
                    model_status = gr.Textbox(label="Статус модели", value=f"Модель 1.7B (Base). Устройство: {engine.device.upper()}", interactive=False)
                    
                    gr.Markdown("### Настройки голоса")
                    
                    voice_dropdown = gr.Dropdown(
                        label="Выберите сохраненный голос",
                        choices=get_saved_voices(),
                        value="Ничего",
                        interactive=True
                    )
                    refresh_voices_btn = gr.Button("Обновить список голосов", size="sm")
                    
                    # Quick Clone inputs right here for convenience
                    ref_audio_input = gr.Audio(label="Образец голоса (аудио файл)", type="filepath")
                    ref_text_input = gr.Textbox(label="Текст образца (что сказано в аудио)", placeholder="Напишите в точности то, что говорится в аудио...")
                    
                    # Component States
                    voice_prompt_state = gr.State(None)
                    voice_name_state = gr.State("По умолчанию (Нет)")

                    target_voice_status = gr.Markdown(value="Текущий профиль голоса: **По умолчанию (Нет)**")
                    
                    generate_btn = gr.Button("Озвучить", variant="primary")
            
            output_audio = gr.Audio(label="Результат", autoplay=False)
            status_output = gr.Textbox(label="Статус", interactive=False)
            
            generate_btn.click(
                fn=generate_speech,
                inputs=[text_input, file_upload, ref_audio_input, ref_text_input, voice_prompt_state],
                outputs=[output_audio, status_output]
            )

        # --- Tab 2: Voice Cloning Studio ---
        with gr.TabItem("Студия Клонирования Голоса"):
            gr.Markdown("Создайте профиль голоса здесь, чтобы не загружать образец каждый раз.")
            
            with gr.Row():
                with gr.Column():
                    clone_audio = gr.Audio(label="Загрузить образец голоса", type="filepath")
                    clone_text = gr.Textbox(label="Текст образца", placeholder="Введите точный текст из образца...")
                    save_name_input = gr.Textbox(label="Имя для сохранения (опционально)", placeholder="Например: MyVoice1")
                    create_btn = gr.Button("Создать и, если имя задано, сохранить профиль")
                
                with gr.Column():
                    profile_status = gr.Textbox(label="Результат")
            
            create_btn.click(
                fn=create_profile,
                inputs=[clone_audio, clone_text, save_name_input],
                outputs=[voice_prompt_state, voice_name_state, profile_status]
            ).success( # Update status text after creation
                fn=lambda name: f"Текущий профиль голоса: **{name}**",
                inputs=[voice_name_state],
                outputs=[target_voice_status]
            ).then( # Auto refresh list just in case
                fn=refresh_voices_list,
                outputs=[voice_dropdown]
            )
            
            voice_dropdown.change(
                 fn=load_voice_ui,
                 inputs=[voice_dropdown],
                 outputs=[voice_prompt_state, voice_name_state, status_output]
            ).then(
                 fn=lambda name: f"Текущий профиль голоса: **{name}**",
                 inputs=[voice_name_state],
                 outputs=[target_voice_status]
            )
            
            refresh_voices_btn.click(
                fn=refresh_voices_list,
                outputs=[voice_dropdown]
            )

        model_dropdown.change(
            fn=switch_model_ui,
            inputs=[model_dropdown],
            outputs=[model_status]
        )
        
        # Monitor
        gr.Markdown("---")
        with gr.Row():
            monitor_toggle = gr.Checkbox(label="Включить мониторинг ресурсов", value=False)
            monitor_display = gr.Textbox(label="Статус системы", value="Мониторинг отключен", interactive=False)
            
        timer = gr.Timer(1.0, active=False)
        
        monitor_toggle.change(
            fn=lambda x: gr.Timer(active=x),
            inputs=[monitor_toggle],
            outputs=[timer]
        )
        
        timer.tick(
            fn=update_monitor,
            outputs=[monitor_display]
        )

if __name__ == "__main__":
    demo.launch()
