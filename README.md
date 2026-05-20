# HY-World 2.0 — Windows + Blackwell + Single-GPU Port

> **Community port — not affiliated with Tencent.** This repo contains patches and helper scripts to run Tencent's [HY-World 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) worldgen pipeline (image → explorable 3D Gaussian Splat world) on **Windows 11 with a single NVIDIA Blackwell GPU**.
>
> Upstream is designed for Linux + 8× H20 GPUs. Getting it running on Windows + single GPU required ~18 specific patches and workarounds, all documented below.

**Original project:** [Tencent-Hunyuan/HY-World-2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) (Apache 2.0).
**This port:** also Apache 2.0.

**Tested configuration:**
- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB, sm_120)
- Driver: 596.36 (CUDA 13.2)
- OS: Windows 11 Pro
- Conda: miniforge3
- End-to-end runtime: ~60 minutes per scene

## Outputs you get

From a single input panorama:
- `point_cloud_7999.ply` (~115–180 MB) — full 3D Gaussian Splat scene, 2–3M Gaussians
- `fuse_post.ply` (~12 MB) — extracted mesh
- `fuse_simplified.ply` (~1 MB) — simplified mesh
- `point_cloud_7999_oriented.ply` — same 3DGS rotated to SuperSplat's convention

---

## Prerequisites

| Component | Required | Notes |
|---|---|---|
| GPU | NVIDIA Blackwell (RTX PRO 6000 / 5090 / 5080) | sm_120 architecture |
| VRAM | ≥48 GB | 24 GB might work with offloading; not tested |
| RAM | ≥64 GB | We had 128 GB |
| Driver | 596+ | Required for CUDA 13 (vLLM Blackwell wheel) |
| CUDA Toolkit | 12.9 installed system-wide | At `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9` |
| MSVC | Visual Studio 2022 Build Tools, "Desktop development with C++" | For pytorch3d, gsplat, recast builds |
| Conda | miniforge3 25+ | At `C:\Users\<user>\miniforge3` |
| Disk | ~300 GB free | Weights + intermediate outputs |
| HuggingFace account | With approved access to `facebook/sam3` | Gated repo, manual approval ~1-24 hr |

## Architecture overview

Two separate conda environments, communicating over HTTP:

```
  ┌─────────────────────────────┐        ┌────────────────────────────────┐
  │ vllm-serve env (Python 3.12)│        │ hyworld2 env (Python 3.11)     │
  │ torch 2.11+cu130            │  HTTP  │ torch 2.7.1+cu128              │
  │ vLLM 0.20 (devnen Windows)  │◀──────▶│ Worldgen pipeline (5 stages)   │
  │ Qwen3-VL-8B-Instruct        │  :18000│ pytorch3d, gsplat, recast etc. │
  └─────────────────────────────┘        └────────────────────────────────┘
                  │                                       │
                  └─────────── shares one GPU ────────────┘
                              (~46 GB + ~50 GB)
```

Why two envs? vLLM Blackwell requires CUDA 13 / PyTorch 2.11; the worldgen pipeline pins PyTorch 2.7. They can't coexist in one env.

---

## Installation

### Step 1: Driver and CUDA toolkit

1. Install NVIDIA driver 596+ (RTX Enterprise or Data Center). Verify: `nvidia-smi` shows "CUDA Version: 13.x".
2. Install CUDA Toolkit 12.9 from NVIDIA's downloads page (the `nvcc.exe` you'll need for source builds is in there).
3. Install Visual Studio 2022 Build Tools with "Desktop development with C++" workload.

### Step 2: Clone the repo and apply patches

```powershell
cd C:\path\to\workspace   # anywhere with 300+ GB free, e.g. D:\models\
git clone --recursive https://github.com/Tencent-Hunyuan/HY-World-2.0.git
cd HY-World-2.0
```

Clone this port repo alongside (so you have the patches and helpers):

```powershell
cd ..
git clone https://github.com/mikecaronna/hunyuanworld2.0-windows-blackwell.git
```

Apply all patches to the HY-World-2.0 clone:

```powershell
cd HY-World-2.0
git apply ..\hunyuanworld2.0-windows-blackwell\patches\*.patch
```

### Step 3: hyworld2 env (worldgen pipeline)

```powershell
& "C:\Users\<user>\miniforge3\Scripts\conda.exe" create -n hyworld2 python=3.11.15 -y
$py = "C:\Users\<user>\miniforge3\envs\hyworld2\python.exe"

# PyTorch
& $py -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

# Patched main requirements (cupy → cupy-cuda12x)
& $py -m pip install -r requirements.txt

# gsplat (NOT in upstream requirements but required by worldrecon)
& $py -m pip install gsplat

# CMake (needed for spz build attempt, even though we skip spz)
& $py -m pip install cmake

# scikit-learn (needed for stage 3)
& $py -m pip install scikit-learn

# huggingface_hub CLI
& $py -m pip install "huggingface_hub[cli]"
```

**Install FlashAttention 2 (Blackwell Windows prebuilt wheel):**

Download `flash_attn-2.7.4.post1-cp311-cp311-win_amd64.whl` from
[marcorez8/flash-attn-windows-blackwell](https://huggingface.co/marcorez8/flash-attn-windows-blackwell)
(use the `torch2.7.0-cu128` variant; it works on 2.7.1 too).

```powershell
& $py -m pip install path\to\flash_attn-2.7.4.post1-cp311-cp311-win_amd64.whl
```

**Install requirements_git.txt (excluding spz which fails on Windows):**

```powershell
& $py -m pip install --no-build-isolation `
    "git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787" `
    "git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5" `
    "git+https://github.com/facebookresearch/pytorch3d.git" `
    "git+https://github.com/microsoft/MoGe.git@0286b495230a074aadf1c76cc5c679e943e5d1c6"
```

pytorch3d will build from source (~15 min with nvcc + cl.exe). PyTorch 2.7.1+cu128 is outside pytorch3d's official support matrix but builds successfully on Blackwell.

**Custom CUDA submodules:**

```powershell
$env:TORCH_CUDA_ARCH_LIST = "12.0"

# gsplat_maskgaussian — clone glm into expected location first
cd hyworld2\worldgen\third_party\gsplat_maskgaussian\gsplat\cuda\csrc
if (-not (Test-Path third_party)) { mkdir third_party }
cd third_party
git clone --depth 1 --branch 1.0.1 https://github.com/g-truc/glm.git
cd ..\..\..\..

& $py -m pip install -e . --no-build-isolation
cd ..\..\..\..

# navmesh / recast
git submodule update --init --recursive
cd hyworld2\worldgen\third_party\navmesh
& $py -m pip install . --no-build-isolation
cd ..\..\..\..
```

**Glob normalization (sitecustomize.py):**

This is critical — without it, the worldgen pipeline can't find its own intermediate files (Windows mixed path separators).

Copy `env/sitecustomize.py` from this port to the env's site-packages:

```powershell
Copy-Item .\env\sitecustomize.py "C:\Users\<user>\miniforge3\envs\hyworld2\Lib\site-packages\sitecustomize.py"
```

### Step 4: vllm-serve env (VLM for trajectory planning)

```powershell
& "C:\Users\<user>\miniforge3\Scripts\conda.exe" create -n vllm-serve python=3.12 -y
$pyv = "C:\Users\<user>\miniforge3\envs\vllm-serve\python.exe"

# torch 2.11 + cu130
& $pyv -m pip install torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

Download `vllm-0.20.0+cu132.devnen.2-cp312-cp312-win_amd64.whl` from
[devnen/vllm-windows releases](https://github.com/devnen/vllm-windows/releases) (Blackwell variant).

```powershell
& $pyv -m pip install path\to\vllm-0.20.0+cu132.devnen.2-cp312-cp312-win_amd64.whl
```

**CUDA junction for flashinfer:** vLLM's flashinfer needs `<CUDA_LIB_PATH>/bin/cudart64_13.dll` but the system CUDA toolkit is 12.9. Torch 2.11+cu130 bundles cu130 DLLs — create a junction so flashinfer finds them:

```powershell
$fakeRoot = "C:\Users\<user>\cuda13_fake"
mkdir $fakeRoot -Force | Out-Null
cmd /c mklink /J "$fakeRoot\bin" "C:\Users\<user>\miniforge3\envs\vllm-serve\Lib\site-packages\torch\lib"
```

### Step 5: Download model weights

```powershell
& "C:\Users\<user>\miniforge3\envs\hyworld2\Scripts\hf.exe" auth login
# Paste your HF token (must have access to facebook/sam3)

# WorldMirror (used by Phase 1 + Phase 2b stage 3)
hf download tencent/HY-World-2.0 --include "HY-WorldMirror-2.0/*" --local-dir D:\path\to\HY-World-2.0\ckpts

# Qwen-Image-Edit-2509 (Phase 2a lighter HY-Pano backend; not used in Phase 2b but useful)
hf download Qwen/Qwen-Image-Edit-2509 --local-dir D:\path\to\HY-World-2.0\ckpts\Qwen-Image-Edit-2509

# HY-Pano LoRA (Phase 2a)
hf download tencent/HY-World-2.0 --include "HY-Pano-2.0/pytorch_lora_weights.safetensors" --local-dir D:\path\to\HY-World-2.0\ckpts

# WorldStereo 2.0 (Phase 2b stage 3)
hf download hanshanxue/WorldStereo --local-dir D:\path\to\HY-World-2.0\ckpts\WorldStereo

# Qwen3-VL-8B-Instruct (Phase 2b stages 1-2, served via vLLM)
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir D:\path\to\HY-World-2.0\ckpts\Qwen3-VL-8B-Instruct
```

Total weight footprint: ~140 GB. Plus optional Phase 2c HunyuanImage-3.0 (157 GB more, deferred — see Known Limitations).

### Step 6: Request SAM3 access

Visit https://huggingface.co/facebook/sam3 and click "Agree and access repository". Manual approval ~1-24 hours.

---

## Patches

All patches are to the upstream `Tencent-Hunyuan/HY-World-2.0` codebase. Listed in the order they're hit by the pipeline.

### P1: `requirements.txt`
```diff
- cupy==13.6.0
+ cupy-cuda12x==13.6.0
```
**Why:** cupy from PyPI requires CUDA toolkit headers at build time. cupy-cuda12x is prebuilt for Windows.

### P2: `hyworld2/worldgen/traj_render.py:71`
```diff
-        backend="cpu:gloo,cuda:nccl",
+        backend="gloo",
```
**Why:** NCCL is Linux-only. PyTorch Windows wheels don't include it.

### P3: `hyworld2/worldgen/traj_render.py:101`
```diff
-            view_id, traj_id = traj_path.split('/')[-2], traj_path.split('/')[-1]
+            view_id, traj_id = os.path.basename(os.path.dirname(traj_path)), os.path.basename(traj_path)
```
**Why:** Windows paths use backslashes. `split('/')` returns the whole string instead of the path components.

### P4: `hyworld2/worldgen/gen_gs_data.py:91` and `video_gen.py:54`
Same as P2: `backend="cpu:gloo,cuda:nccl"` → `"gloo"`.

### P5: `hyworld2/worldgen/src/pointcloud.py` — single-GPU bypass in `multi_gpu_point_rendering`
The upstream uses `dist.all_gather` on CUDA tensors via gloo, which silently produces zero outputs on Windows. Added a single-GPU fast path that skips the gather entirely. See `patches/pointcloud.patch` for the full diff.

### P6: `hyworld2/worldgen/src/retrieval_wm.py` — bypass internal torchrun call
The `apply_worldmirror` method internally spawns `torchrun --nproc_per_node=N -m worldrecon.pipeline`. `torchrun` on Windows fails on libuv even with `USE_LIBUV=0`. Replace with direct python subprocess + manual env vars + drop `--use_fsdp`:

```python
# Add at top: import sys
wm_cmd = [
    sys.executable, "-m", "worldrecon.pipeline",
    # ... existing args minus --use_fsdp ...
]
wm_env = os.environ.copy()
wm_env.update({"RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0",
               "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29509",
               "USE_LIBUV": "0"})
result = subprocess.run(wm_cmd, cwd="..", env=wm_env)
```

### P7: `hyworld2/worldgen/src/data_utils.py:17`
```diff
- from glob import glob
+ from glob import glob as _orig_glob
+ def glob(*args, **kwargs):
+     return [p.replace('\\', '/') for p in _orig_glob(*args, **kwargs)]
```
**Why:** Local override in case sitecustomize doesn't get loaded by every subprocess.

### P8: `hyworld2/panogen/hunyuan_image_3/modeling_hunyuan_image_3.py:933` (only needed if testing HY-Pano heavy backend / Phase 2c)
```diff
-            self.layers[layer_idx].lazy_initialization(key_states)
+            self.layers[layer_idx].lazy_initialization(key_states, value_states)
```
**Why:** `transformers==5.x` changed `StaticLayer.lazy_initialization` signature. Hunyuan code was written for 4.57.x.

---

## Running the pipeline

### Start vLLM server (terminal 1)

```powershell
$env:Path = "C:\Users\<user>\miniforge3\envs\vllm-serve\Scripts;C:\Users\<user>\miniforge3\envs\vllm-serve;" + $env:Path
$env:CUDA_VISIBLE_DEVICES = "0"
$env:CUDA_LIB_PATH = "C:\Users\<user>\cuda13_fake"
$env:CUDA_HOME    = "C:\Users\<user>\cuda13_fake"

vllm.exe serve "D:\path\to\HY-World-2.0\ckpts\Qwen3-VL-8B-Instruct" `
    --served-model-name "Qwen/Qwen3-VL-8B-Instruct" `
    --host 127.0.0.1 --port 18000 `
    --max-model-len 32768 --trust-remote-code `
    --gpu-memory-utilization 0.45
```

Wait for "Application startup complete." Verify: `Invoke-WebRequest http://127.0.0.1:18000/v1/models` returns HTTP 200.

**Note:** Port **18000**, not 8000. Windows reserves port 8000 (HyperV / WSL).

### Run worldgen pipeline (terminal 2)

Use the `run_full_pipeline.ps1` helper script in this port. It sets every env var correctly and chains all 6 stages.

```powershell
$env:HF_TOKEN = "hf_..."   # your token with SAM3 access
powershell -ExecutionPolicy Bypass -File .\run_full_pipeline.ps1 `
    -PanoramaPath "D:\path\to\panorama.png" `
    -OutputName "scene_name" `
    -RepoRoot "D:\path\to\HY-World-2.0"
```

Optional flags: `-HyworldEnv`, `-CudaPath`, `-VllmHost`, `-VllmPort` (defaults to `127.0.0.1:18000`).

Required env vars (the script sets these automatically):
- `CUDA_PATH` = system CUDA toolkit (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9`)
- `CUDA_HOME` = same
- `USE_LIBUV=0`
- `RANK=0`, `WORLD_SIZE=1`, `LOCAL_RANK=0`, `MASTER_ADDR=127.0.0.1`, `MASTER_PORT=29520`
- `TORCHDYNAMO_DISABLE=1`, `TORCH_COMPILE_DISABLE=1` (Triton not available on Windows)
- `HF_TOKEN` = your HuggingFace token
- `TARGET` path uses **forward slashes**: `D:/path/...` not `D:\path\...`

### Stages

| Stage | Script | Duration | What |
|---|---|---|---|
| 1 | `traj_generate.py` | ~80s | SAM3 segmentation + Qwen3-VL labeling + trajectory planning |
| 2 | `traj_render.py` | ~40s | Point-cloud preview rendering for 9 trajectories |
| 3 | `video_gen.py` | ~35 min | WorldStereo diffusion + WorldMirror reconstruction |
| 4 | `gen_gs_data.py` | ~50s | Prep depths/normals/cameras for training |
| 5 | `world_gs_trainer.py` | ~22 min | 3DGS training (8000 steps single-GPU) + mesh extraction |
| 6 | `Tools/rotate_3dgs.py` | ~30s | Reorient PLY to SuperSplat convention |

Total: ~60 minutes per scene.

---

## Known issues and limitations

### Cleanup-only `dist.barrier` crash at end of stage 5

After training completes and PLYs are saved, the script crashes on a `dist.barrier()` call where `dist` isn't imported in scope. This is a Python scoping bug in `world_gs_trainer.py:1713`. **All outputs are saved before the crash** — pipeline is effectively successful, the error appears in the exit code only.

### PLY orientation

HY-World 2.0 outputs are barrel-rolled when loaded in SuperSplat. Fix: rotate **Z axis -90°**. The included `run_full_pipeline.ps1` auto-applies this via `Tools/rotate_3dgs.py`, producing both raw and oriented PLYs.

### vLLM dies if you reboot

vLLM runs as a foreground process in its terminal. If the terminal closes or the machine reboots, you need to restart it manually before running the pipeline again.

### Triton / torch.compile disabled

Several diffusion components in the pipeline try to compile via Triton, which isn't available on Windows. `TORCHDYNAMO_DISABLE=1` is set globally to fall back to eager execution. Inference works, just doesn't get the compile speedup.

### spz format skipped

Niantic's `.spz` (compressed Gaussian Splat) export is optional in HY-World 2.0. Its build fails on Windows (CMake zlib FetchContent error). We skip it and use the standard `.ply` output. Unaffected.

### Phase 2c (HunyuanImage-3.0 heavy HY-Pano backend) deferred

The heavy panorama generator (80B MoE) requires `flashinfer-python` for usable inference speed on a single GPU. Without it, CPU offloading via `accelerate` makes a single panorama take many hours. Mike's Phase 2b uses the lighter Qwen-Image-Edit backend, which works fine.

---

## File inventory

Helper files added by this port:

| Path | Purpose |
|---|---|
| `run_full_pipeline.ps1` | Single-command end-to-end pipeline runner |
| `Tools/rotate_3dgs.py` | 3DGS PLY rotation utility (positions + per-Gaussian quaternions) |
| `WINDOWS_BLACKWELL_PORT.md` | This document |

Files modified in the upstream codebase:

- `requirements.txt`
- `hyworld2/worldgen/traj_render.py`
- `hyworld2/worldgen/video_gen.py`
- `hyworld2/worldgen/gen_gs_data.py`
- `hyworld2/worldgen/src/pointcloud.py`
- `hyworld2/worldgen/src/retrieval_wm.py`
- `hyworld2/worldgen/src/data_utils.py`
- `hyworld2/panogen/hunyuan_image_3/modeling_hunyuan_image_3.py` (Phase 2c only)

Environment-level:

- `C:\Users\<user>\miniforge3\envs\hyworld2\Lib\site-packages\sitecustomize.py`
- `C:\Users\<user>\cuda13_fake\` (CUDA junction for vLLM/flashinfer)

---

## Credits

- Original project: [Tencent-Hunyuan HY-World 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0)
- Windows vLLM port: [devnen/vllm-windows](https://github.com/devnen/vllm-windows)
- FlashAttention Blackwell Windows wheel: [marcorez8/flash-attn-windows-blackwell](https://huggingface.co/marcorez8/flash-attn-windows-blackwell)
- Windows + Blackwell + single-GPU adaptations: documented here.
