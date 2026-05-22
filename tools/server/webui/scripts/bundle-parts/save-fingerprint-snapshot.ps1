param(
	[string]$Label = "",
	[int]$Keep = 20,
	[string]$NodeExe = "node"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$scriptPath = Join-Path $scriptDir "09-save-fingerprint-snapshot.mjs"

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
$args = @($scriptPath, "--keep", "$Keep")

if ($Label) {
	$args += @("--label", $Label)
}

& $nodeCommand @args
if ($LASTEXITCODE -ne 0) {
	throw "Fingerprint snapshot save failed (exit code $LASTEXITCODE)"
}
