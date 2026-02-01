import torch
import soundfile as sf
import os
import uuid
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
                    attn_implementation="flash_attention_2" if self.device == "cuda" else None 
                )
                print(f"Model {self.model_name} loaded successfully.")
            except Exception as e:
                print(f"Error loading model: {e}")
                # Fallback to standard attention if flash_attention_2 fails
                if "flash_attention_2" in str(e):
                    print("Retrying without Flash Attention 2...")
                    from qwen_tts import Qwen3TTSModel
                    dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
                    self.model = Qwen3TTSModel.from_pretrained(
                        self.model_name,
                        device_map=self.device,
                        dtype=dtype
                    )
                    print(f"Model {self.model_name} loaded with standard attention.")
                else:
                    raise e

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

    def generate(self, text, voice_prompt=None, language="auto"):
        """
        Generates audio from text.
        voice_prompt: Result of create_voice_prompt or None (for default handling, though Base model needs a prompt or it might fail/be random)
        """
        self.load_model()
        
        # If no voice prompt is provided for Base model, we might need a default one or use a different generation method.
        # But Base model requires reference audio for cloning usually. 
        # Actually Qwen3-TTS Base acts as a voice cloner.
        
        try:
            if voice_prompt:
                 wavs, sr = self.model.generate_voice_clone(
                    text=text,
                    language=language,
                    voice_clone_prompt=voice_prompt
                )
            else:
                # If no prompt, we can't really "clone".
                # Maybe we should use a default reference if user didn't provide one?
                # Or just error out.
                # Let's assume the UI enforces having a voice selected.
                return None, None
            
            return wavs[0], sr
        except Exception as e:
            print(f"Generation error: {e}")
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

    def generate_with_audio_ref(self, text, ref_audio_path, ref_text, language="auto"):
        """
        One-shot generation with direct reference audio (no pre-computed prompt).
        """
        self.load_model()
        try:
            wavs, sr = self.model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio_path,
                ref_text=ref_text
            )
            return wavs[0], sr
        except Exception as e:
            print(f"One-shot generation error: {e}")
            return None, None
