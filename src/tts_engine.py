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

        # Try to optimize with torch.compile (PyTorch 2.0+)
        # Removed as per user request to speed up startup / visibility
        # try:
        #     if hasattr(torch, "compile") and self.device == "cuda":
        #         print("Compiling model with torch.compile...")
        #         # Reduce overhead mode is good for small batches/real-time
        #         self.model.model = torch.compile(self.model.model, mode="reduce-overhead")
        # except Exception as e:
        #     print(f"Compilation warning: {e}")

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
        import time
        t_start_load = time.time()
        self.load_model()
        print(f"Model consistency check: {getattr(self.model, 'device', 'unknown')} | {getattr(self.model, 'dtype', getattr(self.model.model, 'dtype', 'unknown'))}")
        print(f"Time to ensure model loaded: {time.time() - t_start_load:.4f}s")
        
        chunks = split_text_into_chunks(text)
        all_audio = []
        sr = 24000 # Default Qwen used usually, serves as fallback
        
        total_chunks = len(chunks)
        print(f"Total chunks to process: {total_chunks}")
        
        try:
            with torch.inference_mode():
                for i, chunk in enumerate(chunks):
                    t_chunk_start = time.time()
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
                    
                    t_chunk_end = time.time()
                    chunk_dur = len(wavs[0]) / current_sr if len(wavs) > 0 else 0
                    gen_dur = t_chunk_end - t_chunk_start
                    print(f"Chunk {i+1} generated {chunk_dur:.2f}s audio in {gen_dur:.2f}s (RTF: {gen_dur/chunk_dur if chunk_dur > 0 else 0:.2f})")

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
            
            with torch.inference_mode():
                prompt = self.model.create_voice_clone_prompt(
                    ref_audio=audio_path,
                    ref_text=transcript
                )
            return prompt
        except Exception as e:
            print(f"Error creating voice prompt: {e}")
            return None

    def save_voice_prompt(self, prompt, path):
        try:
            # prompt is a list of VoiceClonePromptItem
            # We serialize it to dicts to avoid pickle dependency issues with custom classes
            from dataclasses import asdict
            payload = {
                "items": [asdict(it) for it in prompt],
                "version": "1.0"
            }
            torch.save(payload, path)
            print(f"Saved voice prompt to {path}")
            return True
        except Exception as e:
            print(f"Error saving voice prompt: {e}")
            return False

    def load_voice_prompt(self, path):
        try:
            # We need the class definition to reconstruct
            from qwen_tts import VoiceClonePromptItem
            
            payload = torch.load(path, map_location=self.device) # weights_only=False by default usually, but explicit is safer if needed
            
            # Handle legacy (pickle) or new (dict) format
            if isinstance(payload, list): 
                # Old format (if any exist, though user just started) or direct pickle
                # This might fail if class not found, but if it loaded, great.
                print(f"Loaded legacy/list voice prompt from {path}")
                return payload
            
            if isinstance(payload, dict) and "items" in payload:
                items_raw = payload["items"]
                items = []
                for d in items_raw:
                     # Reconstruct items
                    ref_code = d.get("ref_code")
                    if ref_code is not None and not torch.is_tensor(ref_code):
                        ref_code = torch.tensor(ref_code)
                    
                    ref_spk = d.get("ref_spk_embedding")
                    if ref_spk is None:
                        continue # Should not happen
                    if not torch.is_tensor(ref_spk):
                        ref_spk = torch.tensor(ref_spk)
                        
                    # Move to device
                    if ref_code is not None:
                        ref_code = ref_code.to(self.device)
                    ref_spk = ref_spk.to(self.device)

                    item = VoiceClonePromptItem(
                        ref_code=ref_code,
                        ref_spk_embedding=ref_spk,
                        x_vector_only_mode=bool(d.get("x_vector_only_mode", False)),
                        icl_mode=bool(d.get("icl_mode", not bool(d.get("x_vector_only_mode", False)))),
                        ref_text=d.get("ref_text", None),
                    )
                    items.append(item)
                
                print(f"Loaded voice prompt from {path} to {self.device}")
                return items
            
            print("Unknown voice prompt format.")
            return None

        except Exception as e:
            print(f"Error loading voice prompt: {e}")
            # If pickle error, it might be because Qwen3TTSModel uses local class definitions?
            # Trying to import them might help.
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
            with torch.inference_mode():
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

    def get_device_status(self):
        status = {}
        try:
            # RAM
            import psutil
            vm = psutil.virtual_memory()
            status["ram_percent"] = vm.percent
            status["ram_used_gb"] = round(vm.used / (1024**3), 2)
            
            # VRAM (if CUDA)
            if torch.cuda.is_available():
                status["vram_reserved_gb"] = round(torch.cuda.memory_reserved() / (1024**3), 2)
                status["vram_allocated_gb"] = round(torch.cuda.memory_allocated() / (1024**3), 2)
                status["device_name"] = torch.cuda.get_device_name(0)
            else:
                status["vram_reserved_gb"] = 0
                status["vram_allocated_gb"] = 0
                status["device_name"] = "CPU"
                
            return status
        except Exception as e:
            return f"Error getting status: {e}"
