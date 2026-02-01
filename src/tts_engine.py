import torch
import soundfile as sf
import os
import uuid
import uuid
import numpy as np
from file_parser import split_text_into_chunks
# from qwen_tts import Qwen3TTSModel # Import will happen inside class to allow lazy loading or handle import errors gracefully

class TTSEngine:
    def __init__(self, model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base", device="cuda"):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        print(f"Initializing TTS Engine with device: {self.device}")

    def load_model(self):
        if self.model is None:
            print("Loading Qwen3-TTS model...")
            try:
                from qwen_tts import Qwen3TTSModel
                # Using float16 or bfloat16 is recommended for GPU
                dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
                
                self.model = Qwen3TTSModel.from_pretrained(
                    self.model_name,
                    device_map=self.device,
                    dtype=dtype,
                    attn_implementation="sdpa"
                )
                print(f"Model {self.model_name} loaded successfully with SDPA.")
            except Exception as e:
                print(f"Error loading model: {e}")
                # Fallback to standard attention if sdpa fails (unlikely, but good to have)
                print("Retrying with default attention...")
                from qwen_tts import Qwen3TTSModel
                dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
                self.model = Qwen3TTSModel.from_pretrained(
                    self.model_name,
                    device_map=self.device,
                    dtype=dtype,
                    attn_implementation="eager"
                )
                print(f"Model {self.model_name} loaded with standard (eager) attention.")

    def switch_model(self, new_model_name):
        if self.model_name == new_model_name and self.model is not None:
            print(f"Model {new_model_name} is already loaded.")
            return
        
        print(f"Switching model from {self.model_name} to {new_model_name}...")
        self.model_name = new_model_name
        
        # Clear memory
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
        
        # The model will be loaded on the next generate() or create_voice_prompt() call via load_model()
        self.load_model()

    def generate(self, text, voice_prompt=None, language="auto", progress_callback=None):
        """
        Generates audio from text.
        voice_prompt: Result of create_voice_prompt or None
        progress_callback: Optional function to call with (progress_0_to_1, status_message)
        """
        self.load_model()
        
        chunks = split_text_into_chunks(text)
        all_audio = []
        sr = 24000 # Default Qwen used usually, serves as fallback
        
        total_chunks = len(chunks)
        print(f"Total chunks to process: {total_chunks}")
        
        try:
            for i, chunk in enumerate(chunks):
                if progress_callback:
                    progress_callback((i / total_chunks), f"Генерация части {i+1} из {total_chunks}...")
                
                print(f"Processing chunk {i+1}/{total_chunks}: {chunk[:50]}...")
                
                if voice_prompt:
                     wavs, current_sr = self.model.generate_voice_clone(
                        text=chunk,
                        language=language,
                        voice_clone_prompt=voice_prompt
                    )
                else:
                    # Base model behavior fallback or error
                     return None, None
                
                sr = current_sr
                if len(wavs) > 0:
                    all_audio.append(wavs[0])
            
            if not all_audio:
                return None, None
                
            # Stitch audio
            final_audio = np.concatenate(all_audio)
            return final_audio, sr
            
        except Exception as e:
            print(f"Generation error: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def create_voice_prompt(self, audio_path, transcript):
        """
        Creates a voice profile from a reference audio.
        """
        self.load_model()
        try:
            # audio_path can be file path
            # transcript is the text content of that audio
            
            # If transcript is missing, we could try x_vector_only_mode=True?
            # But the user requirements mentioned "copy voice by fragment".
            
            prompt = self.model.create_voice_clone_prompt(
                ref_audio=audio_path,
                ref_text=transcript
            )
            return prompt
        except Exception as e:
            print(f"Error creating voice prompt: {e}")
            return None

    def generate_with_audio_ref(self, text, ref_audio_path, ref_text, language="auto", progress_callback=None):
        """
        One-shot generation with direct reference audio (no pre-computed prompt).
        """
        self.load_model()
        
        chunks = split_text_into_chunks(text)
        all_audio = []
        sr = 24000
        total_chunks = len(chunks)
        print(f"Total chunks to process (One-Shot): {total_chunks}")

        try:
            for i, chunk in enumerate(chunks):
                if progress_callback:
                    progress_callback((i / total_chunks), f"Генерация части {i+1} из {total_chunks}...")
                
                print(f"Processing chunk {i+1}/{total_chunks}...")

                wavs, current_sr = self.model.generate_voice_clone(
                    text=chunk,
                    language=language,
                    ref_audio=ref_audio_path,
                    ref_text=ref_text
                )
                sr = current_sr
                if len(wavs) > 0:
                    all_audio.append(wavs[0])
            
            if not all_audio:
                return None, None
                
            final_audio = np.concatenate(all_audio)
            return final_audio, sr
            
        except Exception as e:
            print(f"One-shot generation error: {e}")
            return None, None
