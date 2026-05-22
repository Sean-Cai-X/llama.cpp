param(
	[Parameter(Mandatory = $true)]
	[string]$Input,

	[string]$Label = "",
	[string]$NodeExe = "node"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$scriptPath = Join-Path $scriptDir "11-summarize-fingerprint-diff.mjs"

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
$args = @($scriptPath, "--input", $Input)

if ($Label) {
	$args += @("--label", $Label)
}

& $nodeCommand @args
if ($LASTEXITCODE -ne 0) {
	throw "Fingerprint diff summary failed (exit code $LASTEXITCODE)"
}
