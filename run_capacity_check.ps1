# Capacity check: does the tiling penalty shrink as the causal model gets stronger?
#
# Causal arm only, seed 42, five folds, scored under both protocols. No control arm is
# needed because the protocol gain is a within-arm quantity.
#
# Optional. Run it only if the three-seed GRU sweep has finished and there is still time
# before the deadline. Takes the wide variant by default; pass -IncludeXWide to add the
# largest one, which roughly doubles the runtime.
#
#     powershell -ExecutionPolicy Bypass -File C:\CAP\run_capacity_check.ps1
#     powershell -ExecutionPolicy Bypass -File C:\CAP\run_capacity_check.ps1 -IncludeXWide

param([switch]$IncludeXWide)

$ErrorActionPreference = "Continue"
Set-Location C:\CAP
Start-Transcript -Path "capacity_check.log" -Append | Out-Null

function Say($text) { Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $text" }

$started = Get-Date
Say "run_capacity_check starting"

$waitStarted = Get-Date
if (Get-Process python -ErrorAction SilentlyContinue) {
    Say "waiting for the running sweep to finish before touching the GPU"
    while (Get-Process python -ErrorAction SilentlyContinue) {
        if (((Get-Date) - $waitStarted).TotalHours -ge 8) {
            Say "STOPPING. Still waiting after 8 hours."
            Stop-Transcript | Out-Null
            exit 1
        }
        Start-Sleep -Seconds 60
    }
}

Say "generating the capacity configs"
python scripts\make_capacity_configs.py

$variants = @(@{ suffix = "wide"; config = "configs\sleep78_streaming_causal_wide.yaml" })
if ($IncludeXWide) {
    $variants += @{ suffix = "xwide"; config = "configs\sleep78_streaming_causal_xwide.yaml" }
}

foreach ($v in $variants) {
    $ckpt = "checkpoints_78streaming_causal_$($v.suffix)_s42"
    $log = "logs_78streaming_causal_$($v.suffix)_s42"

    foreach ($fold in 0..4) {
        $checkpoint = Join-Path $ckpt "best_model_fold_$fold.pth"
        if (Test-Path $checkpoint) { Say "skip $($v.suffix) fold $fold, already trained"; continue }
        Say "training $($v.suffix) fold $fold"
        python -m src.train.train --config $v.config --fold $fold
    }

    $missing = 0..4 | Where-Object { -not (Test-Path (Join-Path $ckpt "best_model_fold_$_.pth")) }
    if ($missing) {
        Say "STOPPING. $($v.suffix) is missing folds: $($missing -join ', ')"
        Stop-Transcript | Out-Null
        exit 1
    }

    Remove-Item (Join-Path $log "test_metrics_summary.csv") -ErrorAction SilentlyContinue
    Remove-Item (Join-Path $log "test_metrics_summary_streaming30.csv") -ErrorAction SilentlyContinue
    foreach ($fold in 0..4) {
        Say "scoring $($v.suffix) fold $fold, tiled"
        python -m src.eval.evaluate --config $v.config --fold $fold --no_save_predictions
        Say "scoring $($v.suffix) fold $fold, streaming stride 30"
        python -m src.eval.evaluate --config $v.config --fold $fold --stream_stride 30 --no_save_predictions
    }
}

# --------------------------------------------------------- gain against strength
$arms = @(
    '--arm', 'GRU causal, Sleep-EDF-78=logs_78streaming_gru_s42',
    '--arm', 'GRU control, Sleep-EDF-78=logs_78streaming_gru_noncausal_s42',
    '--arm', 'TCN causal, Sleep-EDF-78=logs_78streaming_causal_s42',
    '--arm', 'TCN control, Sleep-EDF-78=logs_78streaming_noncausal_s42',
    '--arm', 'TCN causal, DOD-H=results\dodh_causal_s42',
    '--arm', 'TCN control, DOD-H=results\dodh_noncausal_s42'
)
foreach ($v in $variants) {
    $arms += '--arm'
    $arms += "TCN causal $($v.suffix), Sleep-EDF-78=logs_78streaming_causal_$($v.suffix)_s42"
}

Say "protocol gain against absolute score, across every arm on disk"
python scripts\gain_vs_strength.py @arms --out results\generated\gain_vs_strength.md

Say "finished in $((Get-Date) - $started)"
Stop-Transcript | Out-Null
