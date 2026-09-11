# Finish the seed 42 GRU experiment unattended.
#
# Waits for the training already in progress to exit, trains whatever folds are still
# missing, scores both arms under both protocols, checks its own output, and computes the
# decisive statistic. Safe to re-run: anything already finished is skipped.
#
#     powershell -ExecutionPolicy Bypass -File runs\run_gru_s42.ps1
#
# Everything printed is also written to gru_s42_overnight.log.

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $root ".git"))) { $root = $PSScriptRoot }
Set-Location $root
Start-Transcript -Path "gru_s42_overnight.log" -Append | Out-Null

function Say($text) { Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $text" }

$started = Get-Date
Say "run_gru_s42 starting"

# ------------------------------------------------------------------ disk headroom
$drive = Get-PSDrive -Name C
$freeGb = [math]::Round($drive.Free / 1GB, 1)
Say "free space on C: $freeGb GB"
if ($freeGb -lt 5) {
    Say "STOPPING. Under 5 GB free. The 20 evaluation runs write about 0.5 GB of prediction files"
    Say "and a disk-full failure part way through would leave a half-scored run."
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
            Say "If that is an editor or a notebook rather than the training run, close it and re-run this script."
            Get-Process python -ErrorAction SilentlyContinue | ForEach-Object { Say "  pid $($_.Id) started $($_.StartTime)" }
            Stop-Transcript | Out-Null
            exit 1
        }
        Start-Sleep -Seconds 30
    }
    Say "that run has exited after $([math]::Round(((Get-Date) - $waitStarted).TotalMinutes, 1)) minutes of waiting"
} else {
    Say "no python running, starting immediately"
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
        $rows = if (Test-Path $path) { @(Get-Content $path).Count - 1 } else { -1 }
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
$report = "results\generated\protocol_excess_gru_s42.md"
Say "computing the within-arm protocol excess"
python scripts\protocol_excess.py --label "Sleep-EDF-78, GRU, seed 42" --seed 42=logs_78streaming_gru_s42,logs_78streaming_gru_noncausal_s42 --out $report

Say "computing the between-arm causality cost under the tiled protocol (informational, may not support a single seed)"
python scripts\pool_seeds.py --seed 42=logs_78streaming_gru_s42,logs_78streaming_gru_noncausal_s42 --out results\generated\statistics_gru_tiled_s42.md

# ---------------------------------------------------------------------- verdict
Write-Host ""
Write-Host "================================================================"
if (Test-Path $report) {
    Get-Content $report | ForEach-Object { Write-Host $_ }
    Write-Host "================================================================"
    try {
        $pooled = (Select-String -Path $report -Pattern '\*\*Pooled\*\*' | Select-Object -First 1).Line
        $cells = $pooled.Split('|')
        $excess = [double](($cells[4] -replace '[^0-9\.\-\+]', ''))
        $folds = ($cells[5] -replace '\*', '').Trim()
        Say "pooled excess $excess, folds positive $folds"
        if ($excess -ge 0.02 -and $folds -eq "5/5") {
            Say "GATE PASSED. The protocol effect reproduces on the recurrent encoder."
            Say "Next: seeds 43 and 44, then the paper edits."
        } else {
            Say "GATE NOT MET. Do not run more seeds. This is a finding and gets reported as one."
        }
    } catch {
        Say "could not parse the pooled row automatically, read the table above"
    }
} else {
    Say "the excess report was not written, something failed in the last step"
}
Write-Host "================================================================"
Say "finished in $((Get-Date) - $started)"
Stop-Transcript | Out-Null
