param(
    [ValidateSet('CheckOnly', 'Execute')]
    [string]$Mode = 'CheckOnly'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ($PSCommandPath) {
    throw 'Gen6r4 stage1 rejects direct path invocation; use the authenticated EncodedCommand stage0.'
}
foreach ($name in @(
    'P1_R4_STAGE0_SOURCE_SHA256',
    'P1_R4_STAGE0_ENCODED_SHA256',
    'P1_R4_STAGE1_PATH',
    'P1_R4_STAGE1_STREAM',
    'P1_R4_STAGE1_BYTES',
    'P1_R4_STAGE1_SHA256'
)) {
    if (-not (Get-Variable -Scope Global -Name $name -ErrorAction SilentlyContinue)) {
        throw "Authenticated stage0 launch binding is absent: $name"
    }
}
if (-not $env:P1_WORKSPACE_ROOT -or -not $env:P1_POWERSHELL_HOST) {
    throw 'P1_WORKSPACE_ROOT and P1_POWERSHELL_HOST are required.'
}

$workspace = [System.IO.Path]::GetFullPath($env:P1_WORKSPACE_ROOT)
$actualHost = [System.IO.Path]::GetFullPath([Environment]::ProcessPath)
$expectedHost = [System.IO.Path]::GetFullPath($env:P1_POWERSHELL_HOST)
if (-not $actualHost.Equals($expectedHost, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'The running PowerShell host is not the externally selected held host.'
}

$script:Held = [ordered]@{}
$script:NativeImages = [ordered]@{}

function Get-StreamSha256([System.IO.Stream]$Stream) {
    $position = $Stream.Position
    try {
        $Stream.Position = 0
        $sha = [System.Security.Cryptography.SHA256]::Create()
        try {
            return [Convert]::ToHexString($sha.ComputeHash($Stream)).ToLowerInvariant()
        }
        finally {
            $sha.Dispose()
        }
    }
    finally {
        $Stream.Position = $position
    }
}

function Open-HeldFile(
    [string]$Label,
    [string]$Path,
    [long]$ExpectedBytes,
    [string]$ExpectedSha256
) {
    $full = [System.IO.Path]::GetFullPath($Path)
    if ([System.IO.File]::GetAttributes($full).HasFlag([System.IO.FileAttributes]::ReparsePoint)) {
        throw "Reparse-point file rejected: $Label"
    }
    $stream = [System.IO.FileStream]::new(
        $full,
        [System.IO.FileMode]::Open,
        [System.IO.FileAccess]::Read,
        [System.IO.FileShare]::Read,
        131072,
        [System.IO.FileOptions]::SequentialScan
    )
    try {
        $sha256 = Get-StreamSha256 $stream
        if ($stream.Length -ne $ExpectedBytes -or $sha256 -ne $ExpectedSha256) {
            throw "Pinned bytes differ: $Label"
        }
        $record = [pscustomobject]@{
            Label = $Label
            Path = $full
            Stream = $stream
            Bytes = $stream.Length
            Sha256 = $sha256
        }
        $script:Held[$Label] = $record
        return $record
    }
    catch {
        $stream.Dispose()
        throw
    }
}

function Open-HeldPowerShellDistribution([string]$HostPath) {
    $root = Split-Path -Parent $HostPath
    $rows = [System.Collections.Generic.List[string]]::new()
    $totalBytes = [long]0
    $files = Get-ChildItem -LiteralPath $root -Recurse -File -Force |
        Sort-Object { $_.FullName.Substring($root.Length).Replace('\', '/') } -CaseSensitive
    foreach ($item in $files) {
        if ($item.Attributes.HasFlag([System.IO.FileAttributes]::ReparsePoint)) {
            throw "PowerShell distribution reparse point rejected: $($item.FullName)"
        }
        $relative = $item.FullName.Substring($root.Length).TrimStart('\').Replace('\', '/')
        $stream = [System.IO.FileStream]::new(
            $item.FullName,
            [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::Read,
            [System.IO.FileShare]::Read,
            131072,
            [System.IO.FileOptions]::SequentialScan
        )
        $sha256 = Get-StreamSha256 $stream
        $script:Held["POWERSHELL_DISTRIBUTION::$relative"] = [pscustomobject]@{
            Label = "POWERSHELL_DISTRIBUTION::$relative"
            Path = $item.FullName
            Stream = $stream
            Bytes = $stream.Length
            Sha256 = $sha256
        }
        $rows.Add("$relative`0$($stream.Length)`0$sha256`n")
        $totalBytes += $stream.Length
    }
    $canonical = [System.Text.Encoding]::UTF8.GetBytes([string]::Concat($rows))
    $inventorySha = [Convert]::ToHexString(
        [System.Security.Cryptography.SHA256]::HashData($canonical)
    ).ToLowerInvariant()
    if ($files.Count -ne 983 -or $totalBytes -ne 296034085 -or
        $inventorySha -ne 'fcbbc18499e682ca08a0860dcb3b5353099a2a846e9eedc50afbb0c28ed728dc') {
        throw 'External PowerShell distribution inventory differs.'
    }
}

function Assert-HeldFilesUnchanged {
    foreach ($record in $script:Held.Values) {
        if ($record.Stream.Length -ne $record.Bytes -or
            (Get-StreamSha256 $record.Stream) -ne $record.Sha256) {
            throw "Held-file same-handle exit rehash failed: $($record.Label)"
        }
    }
    foreach ($record in $script:NativeImages.Values) {
        if ($record.Stream.Length -ne $record.Bytes -or
            (Get-StreamSha256 $record.Stream) -ne $record.Sha256) {
            throw "Native private-image same-handle exit rehash failed: $($record.Path)"
        }
    }
}

function Write-Frame([System.IO.Stream]$Stream, [hashtable]$Value) {
    $json = $Value | ConvertTo-Json -Depth 100 -Compress
    $payload = [System.Text.Encoding]::UTF8.GetBytes($json)
    $header = [BitConverter]::GetBytes([uint64]$payload.Length)
    $Stream.Write($header, 0, $header.Length)
    $Stream.Write($payload, 0, $payload.Length)
    $Stream.Flush()
}

function Read-Frame([System.IO.Stream]$Stream) {
    $header = [byte[]]::new(8)
    $read = $Stream.Read($header, 0, 8)
    if ($read -ne 8) { throw 'Child protocol frame header is absent.' }
    $length = [BitConverter]::ToUInt64($header, 0)
    if ($length -lt 1 -or $length -gt 67108864) { throw 'Child protocol frame size differs.' }
    $payload = [byte[]]::new([int]$length)
    $offset = 0
    while ($offset -lt $payload.Length) {
        $count = $Stream.Read($payload, $offset, $payload.Length - $offset)
        if ($count -le 0) { throw 'Child protocol frame is truncated.' }
        $offset += $count
    }
    return [System.Text.Encoding]::UTF8.GetString($payload) | ConvertFrom-Json -AsHashtable
}

function Stage-AuthenticatedNativeImage([string]$SourcePath) {
    $full = [System.IO.Path]::GetFullPath($SourcePath)
    if (-not $script:Held.Contains("NATIVE_SOURCE::$full")) {
        $sourceStream = [System.IO.FileStream]::new(
            $full,
            [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::Read,
            [System.IO.FileShare]::Read,
            131072,
            [System.IO.FileOptions]::SequentialScan
        )
        $sourceSha = Get-StreamSha256 $sourceStream
        $script:Held["NATIVE_SOURCE::$full"] = [pscustomobject]@{
            Label = "NATIVE_SOURCE::$full"
            Path = $full
            Stream = $sourceStream
            Bytes = $sourceStream.Length
            Sha256 = $sourceSha
        }
    }
    $source = $script:Held["NATIVE_SOURCE::$full"]
    $privateRoot = Join-Path ([System.IO.Path]::GetTempPath()) (
        'p1-v6r4-native-' + [Guid]::NewGuid().ToString('N')
    )
    [System.IO.Directory]::CreateDirectory($privateRoot) | Out-Null
    $privatePath = Join-Path $privateRoot ([System.IO.Path]::GetFileName($full))
    $imageWriter = [System.IO.FileStream]::new(
        $privatePath,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::Read,
        131072,
        [System.IO.FileOptions]::WriteThrough
    )
    $source.Stream.Position = 0
    $source.Stream.CopyTo($imageWriter)
    $imageWriter.Flush($true)
    $imageWriter.Position = 0
    $writerSha = Get-StreamSha256 $imageWriter
    if ($imageWriter.Length -ne $source.Bytes -or $writerSha -ne $source.Sha256) {
        throw 'Native private-image staging digest differs.'
    }
    # Open the final read-only share-deny-write/delete handle before closing the
    # sole writer, so there is no mutable pathname interval between staging and
    # authentication.  Only this read handle survives into the loader window.
    $image = [System.IO.FileStream]::new(
        $privatePath,
        [System.IO.FileMode]::Open,
        [System.IO.FileAccess]::Read,
        [System.IO.FileShare]::Read,
        131072,
        [System.IO.FileOptions]::SequentialScan
    )
    $imageSha = Get-StreamSha256 $image
    if ($image.Length -ne $source.Bytes -or $imageSha -ne $source.Sha256) {
        $image.Dispose()
        throw 'Native private-image held-read digest differs.'
    }
    $imageWriter.Dispose()
    $record = [pscustomobject]@{
        Path = $privatePath
        SourcePath = $full
        Stream = $image
        Bytes = $image.Length
        Sha256 = $imageSha
    }
    $script:NativeImages[$privatePath] = $record
    return $record
}

$stage1Path = [System.IO.Path]::GetFullPath([string]$global:P1_R4_STAGE1_PATH)
$expectedStage1 = Join-Path $workspace 'scripts/launch_p1_multiscale_cross_layer_offset_drift_unary_v6r4.ps1'
if (-not $stage1Path.Equals($expectedStage1, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Stage0-held stage1 path differs.'
}
if ($global:P1_R4_STAGE1_STREAM.Length -ne [long]$global:P1_R4_STAGE1_BYTES -or
    (Get-StreamSha256 $global:P1_R4_STAGE1_STREAM) -ne [string]$global:P1_R4_STAGE1_SHA256) {
    throw 'Stage0-held stage1 bytes changed before execution.'
}
$script:Held['EXTERNAL_LAUNCHER'] = [pscustomobject]@{
    Label = 'EXTERNAL_LAUNCHER'
    Path = $stage1Path
    Stream = $global:P1_R4_STAGE1_STREAM
    Bytes = [long]$global:P1_R4_STAGE1_BYTES
    Sha256 = [string]$global:P1_R4_STAGE1_SHA256
}

Open-HeldPowerShellDistribution $actualHost
$null = Open-HeldFile 'PYTHON_STUB' (Join-Path $workspace '.venv-p1/Scripts/python.exe') 274424 '0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14'
$bootstrap = Open-HeldFile 'BOOTSTRAP' (Join-Path $workspace 'scripts/bootstrap_p1_multiscale_cross_layer_offset_drift_unary_v6r4.py') 24944 '17f36b63e165a3843f2d815df792f0cf1a1938a1583138bf11bc204a1e17a6bb'
$trust = Open-HeldFile 'STARTUP_TRUST' (Join-Path $workspace 'configs/experiments/p1_multiscale_cross_layer_offset_drift_unary_v6r4_startup_trust.json') 6498 '9b8148f1aae2dde54adc026dfdde2cdabc3c2b930a6596f2275c9e8169e2c989'

$python = $script:Held['PYTHON_STUB'].Path
$cachePrefix = $trust.Path
$arguments = @('-I', '-S', '-B', '-X', "pycache_prefix=$cachePrefix", $bootstrap.Path)

try {
    if ($Mode -eq 'CheckOnly') {
        & $python @arguments '--check-only'
        if ($LASTEXITCODE -ne 0) { throw "Gen6r4 static check failed: $LASTEXITCODE" }
    }
    else {
        $info = [System.Diagnostics.ProcessStartInfo]::new()
        $info.FileName = $python
        $info.UseShellExecute = $false
        $info.CreateNoWindow = $true
        $info.RedirectStandardInput = $true
        $info.RedirectStandardOutput = $true
        $info.RedirectStandardError = $true
        foreach ($argument in @($arguments + '--execute')) { $null = $info.ArgumentList.Add($argument) }
        $process = [System.Diagnostics.Process]::new()
        $process.StartInfo = $info
        if (-not $process.Start()) { throw 'Private Python child did not start.' }
        $nonce = [Convert]::ToHexString([System.Security.Cryptography.RandomNumberGenerator]::GetBytes(32)).ToLowerInvariant()
        $envelope = @{
            schema_version = 'p1_v6r4_external_launch_envelope.v1'
            generation = 'p1_multiscale_cross_layer_offset_drift_unary_v6r4'
            role = 'parent'
            cell = $null
            nonce = $nonce
            encoded_command_sha256 = [string]$global:P1_R4_STAGE0_ENCODED_SHA256
            launcher_sha256 = [string]$global:P1_R4_STAGE1_SHA256
            public_cli_fields_absent = $true
        }
        Write-Frame $process.StandardInput.BaseStream $envelope
        $challenge = Read-Frame $process.StandardOutput.BaseStream
        if ($challenge.kind -ne 'child_challenge' -or $challenge.challenge.Length -ne 64) {
            throw 'Private child challenge differs.'
        }
        $responseBytes = [System.Text.Encoding]::ASCII.GetBytes(
            $nonce + $challenge.challenge + $envelope.encoded_command_sha256 + $envelope.launcher_sha256
        )
        $responseSha = [Convert]::ToHexString(
            [System.Security.Cryptography.SHA256]::HashData($responseBytes)
        ).ToLowerInvariant()
        Write-Frame $process.StandardInput.BaseStream @{
            kind = 'supervisor_challenge_response'
            response_sha256 = $responseSha
        }
        $request = Read-Frame $process.StandardOutput.BaseStream
        if ($request.kind -ne 'request_preexisting_control_pins') {
            throw 'Private child control request differs.'
        }
        Write-Frame $process.StandardInput.BaseStream @{
            kind = 'preexisting_control_pins'
            qa_receipt = $null
            authorization = $null
        }
        $process.StandardInput.Close()
        $process.WaitForExit()
        if ($process.ExitCode -eq 0) {
            throw 'Owner-static Execute unexpectedly passed without QA/authorization.'
        }
    }
}
finally {
    Assert-HeldFilesUnchanged
    foreach ($record in $script:NativeImages.Values) { $record.Stream.Dispose() }
    foreach ($record in $script:Held.Values) { $record.Stream.Dispose() }
}
