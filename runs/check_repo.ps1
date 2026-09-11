# Pre-tag hygiene gate for the public repository.
#
# Checks only files git actually tracks, so anything under .gitignore is out of scope. Run it
# before tagging the commit the paper cites, and again after any edit to the tracked tree.
#
#     powershell -ExecutionPolicy Bypass -File runs\check_repo.ps1
#
# Exits non-zero if anything fails, so it can gate a release.

$root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $root ".git"))) { $root = $PSScriptRoot }
Set-Location $root

$failures = @()

function Section($name) { Write-Host ""; Write-Host "--- $name" }

# This script contains the patterns it searches for, so it would match itself. Exclude it from
# the content scans; it is still covered by every structural check below.
$self = "runs/check_repo.ps1"

$tracked = git ls-files
$textPattern = '\.(py|md|ps1|yaml|yml|cff|txt|tex|bib|cfg|toml|json)$'

Section "tracked files"
$absent = $tracked | Where-Object { -not (Test-Path $_) }
$present = $tracked | Where-Object { Test-Path $_ }
$text = $present | Where-Object { $_ -match $textPattern -and $_ -ne $self }
Write-Host "$($tracked.Count) tracked, $($text.Count) of them scanned as text"
if ($absent) {
    Write-Host "  tracked but missing from the working tree:"
    $absent | ForEach-Object { Write-Host "    $_" }
    $failures += "files are tracked but absent from the working tree"
}

# ------------------------------------------------------------------ assistant strings
Section "assistant strings in tracked files"
$tells = 'claude|anthropic|openai|chatgpt|copilot|co-authored-by|as an AI|language model|I apologize'
$hits = $text | ForEach-Object { Select-String -Path $_ -Pattern $tells -AllMatches }
if ($hits) {
    $hits | ForEach-Object { Write-Host "  $($_.Path):$($_.LineNumber): $($_.Line.Trim())" }
    $failures += "assistant strings found in tracked files"
} else { Write-Host "  none" }

# ---------------------------------------------------------------------- em dashes
Section "em dashes"
$dash = $text | ForEach-Object { Select-String -Path $_ -Pattern ([char]0x2014) -AllMatches }
if ($dash) {
    $dash | ForEach-Object { Write-Host "  $($_.Path):$($_.LineNumber): $($_.Line.Trim())" }
    $failures += "em dashes found"
} else { Write-Host "  none" }

# ------------------------------------------------------------- machine-specific paths
Section "absolute and machine-specific paths"
$paths = $text | ForEach-Object { Select-String -Path $_ -Pattern '[A-Z]:\\(CAP|Users)|/home/[a-z]|/Users/' -AllMatches }
if ($paths) {
    $paths | ForEach-Object { Write-Host "  $($_.Path):$($_.LineNumber): $($_.Line.Trim())" }
    $failures += "machine-specific paths found"
} else { Write-Host "  none" }

# ------------------------------------------------------------------ commit messages
Section "commit messages on this branch"
$messages = git log --format="%H %s%n%b" -n 200
$msgHits = $messages | Select-String -Pattern $tells
if ($msgHits) {
    $msgHits | ForEach-Object { Write-Host "  $($_.Line.Trim())" }
    $failures += "assistant strings in commit messages (needs an interactive rebase, do not do this casually)"
} else { Write-Host "  none in the last 200 commits" }

# --------------------------------------------------------------- accidentally tracked
Section "run artifacts that should not be tracked"
# results/ is curated by hand and is meant to carry per-fold CSVs and a small number of archived
# checkpoints, so it is exempt. Live run directories and the data trees are not.
$leaked = $tracked |
    Where-Object { $_ -notmatch '(^|/)\.gitkeep$' } |
    Where-Object { $_ -notmatch '^results/' } |
    Where-Object { $_ -match '^(logs_|checkpoints_|data/raw/|data/processed/)' -or $_ -match '\.(edf|npz|h5)$' }
if ($leaked) {
    $leaked | ForEach-Object { Write-Host "  $_" }
    $failures += "run artifacts are tracked"
} else { Write-Host "  none" }

Section "tracked files inside an ignored directory"
$shadowed = $tracked | Where-Object { $_ -match '^results/generated/' }
if ($shadowed) {
    $shadowed | ForEach-Object { Write-Host "  $_" }
    $failures += "files under results/generated/ are tracked although .gitignore excludes it"
} else { Write-Host "  none" }

Section "large tracked files (over 5 MB)"
$big = $present | Get-Item | Where-Object { $_.Length -gt 5MB }
if ($big) {
    $big | ForEach-Object { Write-Host ("  {0}  {1:N1} MB" -f $_.Name, ($_.Length / 1MB)) }
    $failures += "large files are tracked"
} else { Write-Host "  none" }

# ------------------------------------------------------------------------- tests
Section "unit tests"
python -m unittest discover -s tests -q
if ($LASTEXITCODE -ne 0) { $failures += "unit tests failed" }

# -------------------------------------------------------------------------- lint
Section "lint"
$ruff = Get-Command ruff -ErrorAction SilentlyContinue
if ($ruff) {
    ruff check --select=F,E9 --output-format=concise .
    if ($LASTEXITCODE -ne 0) { $failures += "ruff reported undefined names or syntax errors" }
} else {
    Write-Host "  ruff not installed, skipping (pip install ruff)"
}

# ------------------------------------------------------------------------ verdict
Write-Host ""
Write-Host "================================================================"
if ($failures.Count -eq 0) {
    Write-Host "PASS. The tracked tree is clean."
    exit 0
}
Write-Host "FAIL:"
$failures | ForEach-Object { Write-Host "  - $_" }
exit 1
