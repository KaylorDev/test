import torch
import sys
import os

# Add the current directory to sys.path so we can import modules
sys.path.append(os.getcwd())

from qwen_tts.core.tokenizer_25hz.vq.whisper_encoder import WhisperEncoder

def test_whisper_encoder_sdpa():
    print("Testing WhisperEncoder with SDPA...")
    
    # Dummy parameters for initialization
    n_mels = 80
    n_ctx = 1500
    n_state = 128 # reduced for speed
    n_head = 4
    n_layer = 2
    
    model = WhisperEncoder(n_mels, n_ctx, n_state, n_head, n_layer)
    model.eval()
    
    # Dummy input
    # x_list: List[Tensor], audio_mellens:List[int], audio_aftercnnlens:List[int], audio_seqlens:List[int]
    batch_size = 2
    seq_len = 100
    
    # Mock input tensors (mel spectrograms)
    # Shape: (n_mels, n_ctx) ? No, forward doc says (n_mels, n_ctx)
    # Typically (batch, n_mels, time) but the code does split/etc.
    # Let's look at forward signature: x_list: List[Tensor] where each is (n_mels, n_ctx)
    
    x1 = torch.randn(n_mels, seq_len * 2) # *2 for conv reduction estimate
    x2 = torch.randn(n_mels, seq_len * 2)
    x_list = [x1, x2]
    
    # aftercnn_x_list logic in forward:
    # conv1 (k=3, p=1) -> same length?
    # conv2 (stride=2) -> length / 2
    # So if input length is L, output is L/2.
    
    audio_aftercnnlens = [x.shape[1] // 2 for x in x_list]
    audio_mellens = [x.shape[1] for x in x_list]
    audio_seqlens = [l // 2 for l in audio_aftercnnlens] # roughly
    
    # We need to ensure audio_aftercnnlens are accurate for the code to work? 
    # Actually let's just approximate.
    
    try:
        with torch.no_grad():
            output = model(x_list, audio_mellens, audio_aftercnnlens, audio_seqlens)
        print("Forward pass successful!")
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_whisper_encoder_sdpa()
