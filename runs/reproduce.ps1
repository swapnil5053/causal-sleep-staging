# Regenerate every number the papers report, from committed per-fold CSVs. No retraining.
#
#     powershell -ExecutionPolicy Bypass -File runs\reproduce.ps1
#
# Reports are written under results\generated\. Compare them against the committed copies in
# results\ before submitting anything; any disagreement is a bug in one of the two.

$root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $root ".git"))) { $root = $PSScriptRoot }
Set-Location $root

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path "results\generated" | Out-Null

function Section($name) { Write-Host ""; Write-Host "=== $name" }

$sleep78 = @(
    '--seed', '42=logs_78streaming_causal_s42,logs_78streaming_noncausal_s42',
    '--seed', '43=logs_78streaming_causal_s43,logs_78streaming_noncausal_s43',
    '--seed', '44=logs_78streaming_causal_s44,logs_78streaming_noncausal_s44'
)
$dodh = @(
    '--seed', '42=results\dodh_causal_s42,results\dodh_noncausal_s42',
    '--seed', '43=results\dodh_causal_s43,results\dodh_noncausal_s43',
    '--seed', '44=results\dodh_causal_s44,results\dodh_noncausal_s44'
)
$gru = @(
    '--seed', '42=logs_78streaming_gru_s42,logs_78streaming_gru_noncausal_s42',
    '--seed', '43=logs_78streaming_gru_s43,logs_78streaming_gru_noncausal_s43',
    '--seed', '44=logs_78streaming_gru_s44,logs_78streaming_gru_noncausal_s44'
)

Section "protocol excess and the cost of causality under each protocol"
python scripts\protocol_excess.py --label "Sleep-EDF-78, TCN" @sleep78 --out results\generated\protocol_excess_sleep78.md
python scripts\protocol_excess.py --label "DOD-H, TCN" @dodh --out results\generated\protocol_excess_dodh.md
python scripts\protocol_excess.py --label "Sleep-EDF-78, GRU" @gru --out results\generated\protocol_excess_gru_pooled.md

Section "between-arm causality cost, tiled protocol"
python scripts\pool_seeds.py @sleep78 --out results\generated\statistics_streaming_pooled.md
python scripts\pool_seeds.py @dodh --out results\generated\statistics_dodh.md
python scripts\pool_seeds.py @gru --out results\generated\statistics_gru_tiled_pooled.md

Section "protocol gain against absolute score"
python scripts\gain_vs_strength.py `
    --arm "TCN causal, Sleep-EDF-78=logs_78streaming_causal_s42" `
    --arm "TCN control, Sleep-EDF-78=logs_78streaming_noncausal_s42" `
    --arm "TCN causal, DOD-H=results\dodh_causal_s42" `
    --arm "TCN control, DOD-H=results\dodh_noncausal_s42" `
    --arm "GRU causal, Sleep-EDF-78=logs_78streaming_gru_s42" `
    --arm "GRU control, Sleep-EDF-78=logs_78streaming_gru_noncausal_s42" `
    --out results\generated\gain_vs_strength.md

Section "structural check on the archived result CSVs"
python scripts\validate_results.py

Write-Host ""
Write-Host "Reports are in results\generated\. Diff them against the committed copies in results\."
Write-Host "Anything that disagrees is a bug, not a rounding difference: every one of these is"
Write-Host "deterministic given the same CSVs."
