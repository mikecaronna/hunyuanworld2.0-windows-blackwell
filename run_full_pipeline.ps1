# Run the full HY-World 2.0 worldgen pipeline end-to-end on a panorama.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File run_full_pipeline.ps1 `
#       -PanoramaPath <path-to-input-panorama.png> `
#       -OutputName <scene_name> `
#       -RepoRoot <path-to-HY-World-2.0-clone> `
#       [-HyworldEnv <path-to-hyworld2-conda-env>] `
#       [-CudaPath <path-to-system-CUDA-toolkit>] `
#       [-VllmHost 127.0.0.1] [-VllmPort 18000]
#
# Requires:
#   - vLLM server already running with Qwen3-VL-8B-Instruct at -VllmHost:-VllmPort
#   - HF_TOKEN env var set in your shell (or `hf auth login` already done in the hyworld2 env)

param(
    [Parameter(Mandatory=$true)][string]$PanoramaPath,
    [Parameter(Mandatory=$true)][string]$OutputName,
    [Parameter(Mandatory=$true)][string]$RepoRoot,
    [string]$HyworldEnv = "$env:USERPROFILE\miniforge3\envs\hyworld2",
    [string]$CudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9",
    [string]$VllmHost = "127.0.0.1",
    [int]$VllmPort = 18000
)

$ErrorActionPreference = "Continue"

# Sanity checks
if (-not (Test-Path $PanoramaPath)) { Write-Error "Panorama not found: $PanoramaPath"; exit 1 }
if (-not (Test-Path $HyworldEnv)) { Write-Error "Conda env not found: $HyworldEnv"; exit 1 }
if (-not (Test-Path "$RepoRoot\hyworld2\worldgen\traj_generate.py")) { Write-Error "RepoRoot does not look like an HY-World 2.0 clone: $RepoRoot"; exit 1 }
if (-not (Test-Path $CudaPath)) { Write-Error "CUDA toolkit not found at: $CudaPath"; exit 1 }

$env:Path = "$HyworldEnv;$HyworldEnv\Library\bin;$HyworldEnv\Lib\site-packages\torch\lib;$HyworldEnv\Scripts;" + $env:Path
$env:PYTHONIOENCODING = "utf-8"
$env:CUDA_PATH = $CudaPath
$env:CUDA_HOME = $CudaPath
$env:CUDA_VISIBLE_DEVICES = "0"
$env:USE_LIBUV = "0"
$env:RANK = "0"
$env:WORLD_SIZE = "1"
$env:LOCAL_RANK = "0"
$env:MASTER_ADDR = "127.0.0.1"
$env:MASTER_PORT = "29520"
$env:TORCHDYNAMO_DISABLE = "1"
$env:TORCH_COMPILE_DISABLE = "1"
$env:HYWORLD_CKPTS_DIR = "$RepoRoot\ckpts"

if (-not $env:HF_TOKEN) {
    Write-Output "Note: HF_TOKEN not set. Pipeline will rely on `hf auth login` cached credentials."
}

$py = "$HyworldEnv\python.exe"
$ToolsDir = $PSScriptRoot
$TARGET_WIN = "$RepoRoot\outputs\phase2b_test\$OutputName"
$TARGET = $TARGET_WIN.Replace("\", "/")
$RESULT = "$TARGET/result"
$VllmModel = "Qwen/Qwen3-VL-8B-Instruct"

Write-Output "=== Setup ==="
Write-Output "Target dir:  $TARGET_WIN"
Write-Output "vLLM:        $VllmHost`:$VllmPort"
Write-Output ""

if (-not (Test-Path $TARGET_WIN)) { New-Item -ItemType Directory -Path $TARGET_WIN -Force | Out-Null }
Copy-Item $PanoramaPath "$TARGET_WIN\panorama.png" -Force
Write-Output "Staged panorama: $TARGET_WIN\panorama.png"

cd "$RepoRoot\hyworld2\worldgen"

Write-Output ""
Write-Output "=== Stage 1: Trajectory planning (Qwen3-VL + SAM3) ~80s ==="
& $py traj_generate.py --target_path $TARGET --llm_addr $VllmHost --llm_port $VllmPort --llm_name $VllmModel --apply_nav_traj --apply_up_route --apply_recon_iteration --force_vlm
if ($LASTEXITCODE -ne 0) { Write-Output "STAGE 1 FAILED ($LASTEXITCODE)"; exit 1 }

Write-Output ""
Write-Output "=== Stage 2: Trajectory rendering ~40s ==="
& $py traj_render.py --target_path $TARGET --llm_addr $VllmHost --llm_port $VllmPort --llm_name $VllmModel
if ($LASTEXITCODE -ne 0) { Write-Output "STAGE 2 FAILED ($LASTEXITCODE)"; exit 1 }

Write-Output ""
Write-Output "=== Stage 3: WorldStereo + WorldMirror (longest stage, ~35 min) ==="
& $py video_gen.py --target_path $TARGET
if ($LASTEXITCODE -ne 0) { Write-Output "STAGE 3 FAILED ($LASTEXITCODE)"; exit 1 }

Write-Output ""
Write-Output "=== Stage 4: GS data preparation ~50s ==="
& $py gen_gs_data.py --root_path $TARGET --save_normal --split_sky
if ($LASTEXITCODE -ne 0) { Write-Output "STAGE 4 FAILED ($LASTEXITCODE)"; exit 1 }

Write-Output ""
Write-Output "=== Stage 5: 3DGS training (~22 min, ends with cleanup error - non-fatal) ==="
& $py -m world_gs_trainer default --data_dir "$TARGET/gs_data" --result_dir $RESULT --max_steps 8000 --save_steps 8000 --eval_steps 8000 --ply_steps 8000 --save_ply --disable_video --use_scale_regularization --antialiased --depth_loss --normal_loss --sky_depth_from_pcd --use_mask_gaussian --mask_export_stochastic --no-mask-export-anchor-protection --use_anchor_protection --export_mesh

Write-Output ""
Write-Output "=== Stage 6: Reorient PLY to SuperSplat convention (Z -90deg) ==="
$origPly = "$RESULT/ply/point_cloud_7999.ply"
$orientedPly = "$RESULT/ply/point_cloud_7999_oriented.ply"
if (Test-Path $origPly.Replace("/", "\")) {
    & $py "$ToolsDir\Tools\rotate_3dgs.py" $origPly $orientedPly --axis Z --angle -90
} else {
    Write-Output "Skip reorient: $origPly not found"
}

Write-Output ""
Write-Output "=== ALL STAGES COMPLETE ==="
Write-Output "Output (SuperSplat-ready): $orientedPly"
Write-Output "Output (original orientation): $origPly"
