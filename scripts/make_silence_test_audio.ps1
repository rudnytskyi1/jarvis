$ErrorActionPreference = 'Stop'
$rowanRoot = Split-Path -Parent $PSScriptRoot
$rowanClips = Join-Path $rowanRoot 'data\silence-smoke'
New-Item -ItemType Directory -Path $rowanClips -Force | Out-Null
Add-Type -AssemblyName System.Speech
$rowanSynth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$rowanFormat = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo (16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$rowanCases = @(
    @{text='Shut up'; stop=$true},
    @{text='Stop talking'; stop=$true},
    @{text='Be quiet'; stop=$true},
    @{text='Do not stop talking'; stop=$false},
    @{text='My friend said shut up'; stop=$false},
    @{text='Open the latest video about Mars'; stop=$false}
)
try {
    for ($rowanIndex=0; $rowanIndex -lt $rowanCases.Count; $rowanIndex++) {
        $rowanCase = $rowanCases[$rowanIndex]
        $rowanCase.file = "case-$rowanIndex.wav"
        $rowanSynth.SetOutputToWaveFile((Join-Path $rowanClips $rowanCase.file), $rowanFormat)
        $rowanSynth.Speak($rowanCase.text)
        $rowanSynth.SetOutputToNull()
    }
    $rowanCases | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $rowanClips 'cases.json') -Encoding UTF8
} finally { $rowanSynth.Dispose() }
