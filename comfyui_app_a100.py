import modal

# Tentukan GPU yang kompatibel dengan FlashAttention-2
GPU_CONFIG = modal.gpu.L40S()  # atau modal.gpu.A10G untuk RTX 3090

# Buat environment Modal dengan dependensi yang diperlukan
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential", "libssl-dev", "curl", "wget", "git")
    .pip_install(
        "torch==2.8.0",
        "packaging",
        "ninja",
        "pytest"
    )
    # Instal CUDA toolkit 12.4 (dukungan terbaik untuk A100)
    .run_commands(
        "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get install -y cuda-toolkit-12-9",
        "rm cuda-keyring_1.1-1_all.deb"
    )
    # Instal FlashAttention-2 dari source
    .run_commands(
        "git clone https://github.com/Dao-AILab/flash-attention.git --recursive",
        "cd flash-attention && pip install . --no-build-isolation"
    )
)

app = modal.App("flash-attention", image=image)

@app.function(gpu=GPU_CONFIG, timeout=600)
def test_flash_attention():
    import torch
    from flash_attn import flash_attn_func
    
    # Konfigurasi test
    batch_size = 2
    seqlen_q = 1024
    seqlen_k = 1024
    nheads = 12
    headdim = 128
    
    # Generate data dummy di GPU
    q = torch.randn(batch_size, seqlen_q, nheads, headdim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch_size, seqlen_k, nheads, headdim, device="cuda", dtype=torch.float16)
    v = torch.randn(batch_size, seqlen_k, nheads, headdim, device="cuda", dtype=torch.float16)
    
    # Jalankan FlashAttention
    with torch.no_grad():
        out = flash_attn_func(q, k, v, causal=True)
    
    print(f"✅ FlashAttention berhasil dijalankan!")
    print(f"Output shape: {out.shape}")
    print(f"Device: {out.device}")
    print(f"Dtype: {out.dtype}")
    
    return {
        "success": True,
        "output_shape": list(out.shape),
        "device": str(out.device),
        "dtype": str(out.dtype)
    }

@app.local_entrypoint()
def main():
    result = test_flash_attention.remote()
    print("Hasil test FlashAttention:", result)
