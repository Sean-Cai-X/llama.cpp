param(
	[string]$NodeExe = "node",
	[string]$Label = "",
	[int]$Keep = 20,
	[switch]$SkipCompare,
	[switch]$SkipSnapshot
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$runIndexScript = Join-Path $scriptDir "run-bundle-index.ps1"
$compareScript = Join-Path $scriptDir "compare-fingerprint-snapshots.ps1"
$snapshotScript = Join-Path $scriptDir "save-fingerprint-snapshot.ps1"
$serverDir = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $scriptDir))
$indexRoot = Join-Path $serverDir "public\\bundle.parts.index"
$currentFingerprintFile = Join-Path $indexRoot "fingerprints\\fingerprints.records.json"
$latestSnapshotFile = Join-Path $indexRoot "fingerprints\\snapshots\\latest.records.json"

function Get-EffectiveLabel {
	param(
		[string]$InputLabel
	)

	if ($InputLabel) {
		return $InputLabel
	}

	return (Get-Date).ToString("yyyyMMdd-HHmmss")
}

$effectiveLabel = Get-EffectiveLabel -InputLabel $Label

Write-Host "==> Running full bundle index pipeline" -ForegroundColor Cyan
powershell -ExecutionPolicy Bypass -File $runIndexScript -NodeExe $NodeExe
if ($LASTEXITCODE -ne 0) {
	throw "Bundle index pipeline failed (exit code $LASTEXITCODE)"
}

if (-not $SkipCompare) {
	if (Test-Path $latestSnapshotFile) {
		Write-Host "==> Comparing against latest fingerprint snapshot" -ForegroundColor Cyan
		powershell -ExecutionPolicy Bypass -File $compareScript `
			-Previous $latestSnapshotFile `
			-Current $currentFingerprintFile `
			-Label $effectiveLabel `
			-NodeExe $NodeExe
		if ($LASTEXITCODE -ne 0) {
			throw "Fingerprint comparison failed (exit code $LASTEXITCODE)"
		}
	} else {
		Write-Host "==> No latest fingerprint snapshot found, skipping compare" -ForegroundColor Yellow
	}
}

if (-not $SkipSnapshot) {
	Write-Host "==> Saving current fingerprint snapshot" -ForegroundColor Cyan
	powershell -ExecutionPolicy Bypass -File $snapshotScript `
		-Label $effectiveLabel `
		-Keep $Keep `
		-NodeExe $NodeExe
	if ($LASTEXITCODE -ne 0) {
		throw "Fingerprint snapshot save failed (exit code $LASTEXITCODE)"
	}
}

Write-Host "Bundle incremental pipeline completed." -ForegroundColor Green
