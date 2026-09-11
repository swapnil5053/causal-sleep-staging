# Finish the seed 42 GRU experiment unattended.
#
# Waits for the training already in progress to exit, trains whatever folds are still
# missing, scores both arms under both protocols, checks its own output, and computes the
# decisive statistic. Safe to re-run: anything already finished is skipped.
#
#     powershell -ExecutionPolicy Bypass -File C:\CAP\run_gru_s42.ps1
#
# Everything printed is also written to gru_s42_overnight.log.

$ErrorActionPreference = "Continue"
Set-Location C:\CAP
Start-Transcript -Path "gru_s42_overnight.log" -Append | Out-Null

function Say($text) { Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $text" }

$started = Get-Date

# ---------------------------------------------------------------- wait for the GPU
if (Get-Process python -ErrorAction SilentlyContinue) {
    Say "python is running, waiting for it to finish before touching the GPU"
    while (Get-Process python -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 30 }
    Say "that run has exited"
}

# ---------------------------------------------------------------------- training
$arms = @(
    @{ name = "gru";           config = "configs\sleep78_streaming_gru.yaml";           ckpt = "checkpoints_78streaming_gru_s42" },
    @{ name = "gru_noncausal"; config = "configs\sleep78_streaming_gru_noncausal.yaml"; ckpt = "checkpoints_78streaming_gru_noncausal_s42" }
)

foreach ($arm in $arms) {
    foreach ($fold in 0..4) {
        $checkpoint = Join-Path $arm.ckpt "best_model_fold_$fold.pth"
        if (Test-Path $checkpoint) {
            Say "skip training $($arm.name) fold $fold, checkpoint already exists"
            continue
        }
        Say "training $($arm.name) fold $fold"
        python -m src.train.train --config $arm.config --fold $fold
    }
}

# ------------------------------------------------------------- checkpoint check
$missing = @()
foreach ($arm in $arms) {
    foreach ($fold in 0..4) {
        $checkpoint = Join-Path $arm.ckpt "best_model_fold_$fold.pth"
        if (-not (Test-Path $checkpoint)) { $missing += $checkpoint }
    }
}
if ($missing.Count -gt 0) {
    Say "STOPPING. These checkpoints were never written, so those folds failed:"
    $missing | ForEach-Object { Write-Host "    $_" }
    Stop-Transcript | Out-Null
    exit 1
}
Say "all 10 checkpoints present"

# -------------------------------------------------------------------- evaluation
foreach ($arm in $arms) {
    $logDir = "logs_78streaming_$($arm.name)_s42"
    Remove-Item (Join-Path $logDir "test_metrics_summary.csv") -ErrorAction SilentlyContinue
    Remove-Item (Join-Path $logDir "test_metrics_summary_streaming30.csv") -ErrorAction SilentlyContinue
    foreach ($fold in 0..4) {
        Say "scoring $($arm.name) fold $fold, tiled"
        python -m src.eval.evaluate --config $arm.config --fold $fold
        Say "scoring $($arm.name) fold $fold, streaming stride 30"
        python -m src.eval.evaluate --config $arm.config --fold $fold --stream_stride 30
    }
}

# ------------------------------------------------------------------- row check
$incomplete = @()
foreach ($arm in $arms) {
    $logDir = "logs_78streaming_$($arm.name)_s42"
    foreach ($name in @("test_metrics_summary.csv", "test_metrics_summary_streaming30.csv")) {
        $path = Join-Path $logDir $name
        $rows = if (Test-Path $path) { (Get-Content $path).Count - 1 } else { -1 }
        Write-Host ("    {0,-58} {1} folds" -f $path, $rows)
        if ($rows -ne 5) { $incomplete += "$path has $rows folds" }
    }
}
if ($incomplete.Count -gt 0) {
    Say "STOPPING. Not computing the excess on an incomplete run:"
    $incomplete | ForEach-Object { Write-Host "    $_" }
    Stop-Transcript | Out-Null
    exit 1
}
Say "all four summary files hold five folds"

# ------------------------------------------------------------------- statistics
Say "computing the within-arm protocol excess"
python scripts\protocol_excess.py --label "Sleep-EDF-78, GRU, seed 42" --seed 42=logs_78streaming_gru_s42,logs_78streaming_gru_noncausal_s42 --out results\generated\protocol_excess_gru_s42.md

Say "computing the between-arm causality cost under the tiled protocol, for reference"
python scripts\pool_seeds.py --seed 42=logs_78streaming_gru_s42,logs_78streaming_gru_noncausal_s42 --out results\generated\statistics_gru_tiled_s42.md

Say "finished in $((Get-Date) - $started)"
Say "the number that matters is the pooled excess in results\generated\protocol_excess_gru_s42.md"
Stop-Transcript | Out-Null
