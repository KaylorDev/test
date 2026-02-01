import torch
import torch.nn.functional as F

def check_sdpa_backend():
    print(f"PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA Available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
        
        # Test SDPA
        q = torch.randn(1, 8, 128, 64, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 8, 128, 64, device="cuda", dtype=torch.float16)
        v = torch.randn(1, 8, 128, 64, device="cuda", dtype=torch.float16)
        
        # Check backends
        backends = {
            "flash": torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False),
            "mem_efficient": torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True),
            "math": torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
        }
        
        print("\nChecking SDPA Backends:")
        for name, ctx in backends.items():
            try:
                with ctx:
                    F.scaled_dot_product_attention(q, k, v)
                    print(f"  [{name}]: AVAILABLE")
            except RuntimeError as e:
                print(f"  [{name}]: NOT AVAILABLE ({e})")
                
    else:
        print("CUDA not available.")

if __name__ == "__main__":
    check_sdpa_backend()
