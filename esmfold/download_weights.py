import os, time, sys
print("HF_HOME:", os.environ.get("HF_HOME"))
print("TORCH_HOME:", os.environ.get("TORCH_HOME"))
print("XDG_CACHE_HOME:", os.environ.get("XDG_CACHE_HOME"))
sys.stdout.flush()

from transformers import EsmForProteinFolding, AutoTokenizer

t0 = time.time()
print("Downloading/loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
print(f"Tokenizer ready in {time.time()-t0:.1f}s", flush=True)

t1 = time.time()
print("Downloading/loading model (this is the big one)...", flush=True)
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
print(f"Model ready in {time.time()-t1:.1f}s", flush=True)

n_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {n_params:,}", flush=True)
print("DONE", flush=True)
