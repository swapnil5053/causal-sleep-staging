# Seeds 43 and 44 for the GRU experiment, then the three-seed pooled statistics.
#
# Same structure as run_gru_s42.ps1: waits for the GPU, trains only what is missing,
# scores both arms under both protocols, checks its own output, then pools all three
# seeds so the design matches the published TCN and DOD-H tables exactly.
#
#     powershell -ExecutionPolicy Bypass -File runs\run_gru_s4344.ps1
#
# Everything printed is also written to gru_s4344_overnight.log.

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $root ".git"))) { $root = $PSScriptRoot }
Set-Location $root
Start-Transcript -Path "gru_s4344_overnight.log" -Append | Out-Null

function Say($text) { Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $text" }

$started = Get-Date
Say "run_gru_s4344 starting"

# ------------------------------------------------------------------ disk headroom
$freeGb = [math]::Round((Get-PSDrive -Name C).Free / 1GB, 1)
Say "free space on C: $freeGb GB"
if ($freeGb -lt 5) {
    Say "STOPPING. Under 5 GB free and the 40 evaluation runs write about 1 GB of prediction files."
    Stop-Transcript | Out-Null
    exit 1
}

# ---------------------------------------------------------------- wait for the GPU
$waitStarted = Get-Date
$running = Get-Process python -ErrorAction SilentlyContinue
if ($running) {
    foreach ($p in $running) { Say "waiting on python pid $($p.Id), started $($p.StartTime.ToString('HH:mm:ss'))" }
    while (Get-Process python -ErrorAction SilentlyContinue) {
        if (((Get-Date) - $waitStarted).TotalHours -ge 4) {
            Say "STOPPING. Still waiting on a python process after 4 hours."
            Get-Process python -ErrorAction SilentlyContinue | ForEach-Object { Say "  pid $($_.Id) started $($_.StartTime)" }
            Stop-Transcript | Out-Null
            exit 1
        }
        Start-Sleep -Seconds 30
    }
    Say "that run has exited"
} else {
    Say "no python running, starting immediately"
}

# ------------------------------------------------------------------------- runs
$runs = @()
foreach ($seed in @("43", "44")) {
    $runs += @{ seed = $seed; name = "gru_s$seed";           config = "configs\sleep78_streaming_gru_s$seed.yaml";           ckpt = "checkpoints_78streaming_gru_s$seed";           log = "logs_78streaming_gru_s$seed" }
    $runs += @{ seed = $seed; name = "gru_noncausal_s$seed"; config = "configs\sleep78_streaming_gru_noncausal_s$seed.yaml"; ckpt = "checkpoints_78streaming_gru_noncausal_s$seed"; log = "logs_78streaming_gru_noncausal_s$seed" }
}

foreach ($run in $runs) {
    if (-not (Test-Path $run.config)) {
        Say "STOPPING. Missing config $($run.config). Re-run scripts\make_gru_configs.py."
        Stop-Transcript | Out-Null
        exit 1
    }
}

# --------------------------------------------------------------------- training
foreach ($run in $runs) {
    foreach ($fold in 0..4) {
        $checkpoint = Join-Path $run.ckpt "best_model_fold_$fold.pth"
        if (Test-Path $checkpoint) {
            Say "skip training $($run.name) fold $fold, checkpoint already exists"
            continue
        }
        Say "training $($run.name) fold $fold"
        python -m src.train.train --config $run.config --fold $fold
    }
}

$missing = @()
foreach ($run in $runs) {
    foreach ($fold in 0..4) {
        $checkpoint = Join-Path $run.ckpt "best_model_fold_$fold.pth"
        if (-not (Test-Path $checkpoint)) { $missing += $checkpoint }
    }
}
if ($missing.Count -gt 0) {
    Say "STOPPING. These checkpoints were never written, so those folds failed:"
    $missing | ForEach-Object { Write-Host "    $_" }
    Stop-Transcript | Out-Null
    exit 1
}
Say "all 20 checkpoints present"

# ------------------------------------------------------------------- evaluation
foreach ($run in $runs) {
    Remove-Item (Join-Path $run.log "test_metrics_summary.csv") -ErrorAction SilentlyContinue
    Remove-Item (Join-Path $run.log "test_metrics_summary_streaming30.csv") -ErrorAction SilentlyContinue
    foreach ($fold in 0..4) {
        Say "scoring $($run.name) fold $fold, tiled"
        python -m src.eval.evaluate --config $run.config --fold $fold
        Say "scoring $($run.name) fold $fold, streaming stride 30"
        python -m src.eval.evaluate --config $run.config --fold $fold --stream_stride 30
    }
}

$incomplete = @()
foreach ($run in $runs) {
    foreach ($name in @("test_metrics_summary.csv", "test_metrics_summary_streaming30.csv")) {
        $path = Join-Path $run.log $name
        $rows = if (Test-Path $path) { @(Get-Content $path).Count - 1 } else { -1 }
        Write-Host ("    {0,-62} {1} folds" -f $path, $rows)
        if ($rows -ne 5) { $incomplete += "$path has $rows folds" }
    }
}
if ($incomplete.Count -gt 0) {
    Say "STOPPING. Not pooling an incomplete run:"
    $incomplete | ForEach-Object { Write-Host "    $_" }
    Stop-Transcript | Out-Null
    exit 1
}
Say "all eight summary files hold five folds"

# ------------------------------------------------------- three-seed statistics
$report = "results\generated\protocol_excess_gru_pooled.md"
Say "pooling all three seeds, 15 paired measurements, same design as the published tables"
python scripts\protocol_excess.py --label "Sleep-EDF-78, GRU, three seeds" --seed 42=logs_78streaming_gru_s42,logs_78streaming_gru_noncausal_s42 --seed 43=logs_78streaming_gru_s43,logs_78streaming_gru_noncausal_s43 --seed 44=logs_78streaming_gru_s44,logs_78streaming_gru_noncausal_s44 --out $report

Say "between-arm causality cost, tiled protocol, three seeds"
python scripts\pool_seeds.py --seed 42=logs_78streaming_gru_s42,logs_78streaming_gru_noncausal_s42 --seed 43=logs_78streaming_gru_s43,logs_78streaming_gru_noncausal_s43 --seed 44=logs_78streaming_gru_s44,logs_78streaming_gru_noncausal_s44 --out results\generated\statistics_gru_tiled_pooled.md

# ----------------------------------------------------------------- buffer curve
# No retraining. One sweep per arm yields kappa at every buffer position, which is the
# GRU equivalent of the paper's per-quartile table. This is what shows whether the
# bidirectional control pays its own cold-start cost at the far edge of the buffer.
Say "buffer-position sweep, causal arm"
python scripts\latency_sweep.py --config configs\sleep78_streaming_gru.yaml --stride 5 --out results\generated\latency_gru_causal_s5.md

Say "buffer-position sweep, control arm"
python scripts\latency_sweep.py --config configs\sleep78_streaming_gru_noncausal.yaml --stride 5 --out results\generated\latency_gru_noncausal_s5.md

# ---------------------------------------------------------------------- verdict
Write-Host ""
Write-Host "================================================================"
if (Test-Path $report) {
    Get-Content $report | ForEach-Object { Write-Host $_ }
    Write-Host "================================================================"
    try {
        $pooled = (Select-String -Path $report -Pattern '\*\*Pooled\*\*' | Select-Object -First 1).Line
        $cells = $pooled.Split('|')
        $causalGain = [double](($cells[2] -replace '[^0-9\.\-\+]', ''))
        $excess = [double](($cells[4] -replace '[^0-9\.\-\+]', ''))
        $folds = ($cells[5] -replace '\*', '').Trim()
        Say "pooled causal-arm gain $causalGain, excess $excess, folds positive $folds"
        Say "TCN reference on this dataset: causal-arm gain +0.0350, excess +0.0314, 15/15"
    } catch {
        Say "could not parse the pooled row automatically, read the table above"
    }
} else {
    Say "the pooled report was not written, something failed in the last step"
}
Write-Host "================================================================"
Say "finished in $((Get-Date) - $started)"
Stop-Transcript | Out-Null
