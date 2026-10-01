Add-Type -AssemblyName System.Speech

$output = Join-Path $PSScriptRoot "english_test_audio.wav"
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = -1
$synth.Volume = 100

$voice = $synth.GetInstalledVoices() | Where-Object {
    $_.VoiceInfo.Culture.Name -like "en-*"
} | Select-Object -First 1
if ($null -ne $voice) {
    $synth.SelectVoice($voice.VoiceInfo.Name)
}

$ssml = @"
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
  <prosody rate="90%">
    The night is falling on the window<break time="700ms"/>
    I am still here waiting for you<break time="900ms"/>
    You said the story is not over<break time="700ms"/>
    We can leave the past behind
  </prosody>
</speak>
"@

$synth.SetOutputToWaveFile($output)
$synth.SpeakSsml($ssml)
$synth.Dispose()
Write-Output "generated: $output"
