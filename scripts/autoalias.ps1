param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]] $Args
)

$root = Split-Path -Parent $PSScriptRoot
$defaultPython = "F:\ComfyUI\.venv\Scripts\python.exe"

if ($env:AUTOALIAS_PYTHON) {
  $python = $env:AUTOALIAS_PYTHON
} elseif (Test-Path $defaultPython) {
  $python = $defaultPython
} else {
  $python = "python"
}

$env:PYTHONPATH = "$root\src"
& $python -m autoalias.cli @Args
exit $LASTEXITCODE

