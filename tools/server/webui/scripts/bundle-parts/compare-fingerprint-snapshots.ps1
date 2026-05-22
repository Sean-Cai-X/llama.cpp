param(
	[Parameter(Mandatory = $true)]
	[string]$Previous,

	[string]$Current = "",
	[string]$Label = "default",
	[string]$NodeExe = "node"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$scriptPath = Join-Path $scriptDir "08-compare-fingerprint-snapshots.mjs"

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
$args = @($scriptPath, "--previous", $Previous, "--label", $Label)

if ($Current) {
	$args += @("--current", $Current)
}

& $nodeCommand @args
if ($LASTEXITCODE -ne 0) {
	throw "Fingerprint comparison failed (exit code $LASTEXITCODE)"
}
