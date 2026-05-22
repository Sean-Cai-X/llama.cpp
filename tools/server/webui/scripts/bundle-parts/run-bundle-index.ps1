param(
	[string]$NodeExe = "node"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$steps = @(
	"01-rebuild-bundle.mjs",
	"02-scan-ast.mjs",
	"03-slice-by-structure.mjs",
	"04-refine-oversized-chunks.mjs",
	"05-build-lookup-indexes.mjs",
	"06-export-rag-corpus.mjs",
	"07-build-fingerprint-indexes.mjs"
)

function Resolve-NodeCommand {
	param(
		[string]$CommandName
	)

	$command = Get-Command $CommandName -ErrorAction SilentlyContinue
	if ($command) {
		return $command.Source
	}

	throw "Node executable not found: $CommandName"
}

$nodeCommand = Resolve-NodeCommand -CommandName $NodeExe

foreach ($step in $steps) {
	$scriptPath = Join-Path $scriptDir $step
	if (-not (Test-Path $scriptPath)) {
		throw "Missing script: $scriptPath"
	}

	Write-Host "==> Running $step" -ForegroundColor Cyan
	& $nodeCommand $scriptPath
	if ($LASTEXITCODE -ne 0) {
		throw "Step failed: $step (exit code $LASTEXITCODE)"
	}
}

Write-Host "Bundle index pipeline completed." -ForegroundColor Green
