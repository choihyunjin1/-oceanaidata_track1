[CmdletBinding()]
param(
    [ValidateSet('CheckOnly', 'Execute')]
    [string]$Mode = 'CheckOnly'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# This script is not its own trust root.  Fresh independent QA must authenticate
# the exact absolute PowerShell host and these launcher bytes before either one
# executes.  The checks below preserve that externally established identity.
if ($PSVersionTable.PSVersion.ToString() -ne '7.6.4') {
    throw 'Canonical PowerShell 7.6.4 is required.'
}
if (
    -not $env:P1_POWERSHELL_HOST -or
    -not [System.IO.Path]::IsPathFullyQualified($env:P1_POWERSHELL_HOST)
) {
    throw 'P1_POWERSHELL_HOST must be an externally authenticated absolute path.'
}
$expectedHost = [System.IO.Path]::GetFullPath($env:P1_POWERSHELL_HOST)
$actualHost = [System.IO.Path]::GetFullPath([Environment]::ProcessPath)
if ($actualHost.ToLowerInvariant() -ne $expectedHost.ToLowerInvariant()) {
    throw 'Exact externally pinned PowerShell host path is required.'
}

Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

public static class P1Gen6r3Native {
    [StructLayout(LayoutKind.Sequential)]
    public struct BY_HANDLE_FILE_INFORMATION {
        public uint FileAttributes;
        public System.Runtime.InteropServices.ComTypes.FILETIME CreationTime;
        public System.Runtime.InteropServices.ComTypes.FILETIME LastAccessTime;
        public System.Runtime.InteropServices.ComTypes.FILETIME LastWriteTime;
        public uint VolumeSerialNumber;
        public uint FileSizeHigh;
        public uint FileSizeLow;
        public uint NumberOfLinks;
        public uint FileIndexHigh;
        public uint FileIndexLow;
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    public static extern SafeFileHandle CreateFileW(
        string fileName,
        uint desiredAccess,
        uint shareMode,
        IntPtr securityAttributes,
        uint creationDisposition,
        uint flagsAndAttributes,
        IntPtr templateFile
    );

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool GetFileInformationByHandle(
        SafeFileHandle handle,
        out BY_HANDLE_FILE_INFORMATION information
    );
}
'@

$genericRead = [Convert]::ToUInt32('80000000', 16)
$fileShareRead = [uint32]0x00000001
$openExisting = [uint32]3
$openReparsePoint = [uint32]0x00200000
$reparseAttribute = [uint32]0x00000400
$script:Held = [ordered]@{}

function Get-NormalizedPath([string]$Path) {
    return [System.IO.Path]::GetFullPath($Path).TrimEnd('\').ToLowerInvariant()
}

function Assert-PlainAncestorChain([string]$Path) {
    $item = Get-Item -LiteralPath $Path -Force
    while ($null -ne $item) {
        if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Reparse path is forbidden: $Path"
        }
        if ($null -ne $item.LinkType) {
            throw "Linked path is forbidden: $Path"
        }
        if ($item -is [System.IO.DirectoryInfo]) {
            $item = $item.Parent
        }
        else {
            $item = $item.Directory
        }
    }
}

function Get-HandleInformation([Microsoft.Win32.SafeHandles.SafeFileHandle]$Handle) {
    $information = New-Object P1Gen6r3Native+BY_HANDLE_FILE_INFORMATION
    if (-not [P1Gen6r3Native]::GetFileInformationByHandle($Handle, [ref]$information)) {
        throw [System.ComponentModel.Win32Exception]::new(
            [System.Runtime.InteropServices.Marshal]::GetLastWin32Error()
        )
    }
    return $information
}

function Get-HandleIdentity($Information) {
    return @(
        [uint64]$Information.VolumeSerialNumber,
        (([uint64]$Information.FileIndexHigh -shl 32) -bor [uint64]$Information.FileIndexLow),
        (([uint64]$Information.FileSizeHigh -shl 32) -bor [uint64]$Information.FileSizeLow),
        [uint64]$Information.NumberOfLinks,
        [uint64]$Information.FileAttributes
    ) -join ':'
}

function Get-StreamSha256([System.IO.FileStream]$Stream) {
    $Stream.Position = 0
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try {
        $digest = [Convert]::ToHexString($algorithm.ComputeHash($Stream)).ToLowerInvariant()
    }
    finally {
        $algorithm.Dispose()
    }
    $Stream.Position = 0
    return $digest
}

function Open-HeldFile(
    [string]$Label,
    [string]$Path,
    [long]$ExpectedBytes,
    [AllowNull()][string]$ExpectedSha256
) {
    $absolute = [System.IO.Path]::GetFullPath($Path)
    Assert-PlainAncestorChain $absolute
    $handle = [P1Gen6r3Native]::CreateFileW(
        $absolute,
        $genericRead,
        $fileShareRead,
        [IntPtr]::Zero,
        $openExisting,
        $openReparsePoint,
        [IntPtr]::Zero
    )
    if ($handle.IsInvalid) {
        throw [System.ComponentModel.Win32Exception]::new(
            [System.Runtime.InteropServices.Marshal]::GetLastWin32Error()
        )
    }
    try {
        $before = Get-HandleInformation $handle
        if (
            $before.NumberOfLinks -ne 1 -or
            ($before.FileAttributes -band $reparseAttribute) -ne 0
        ) {
            throw "Single-link non-reparse file required: $Label"
        }
        $stream = [System.IO.FileStream]::new(
            $handle,
            [System.IO.FileAccess]::Read,
            1048576,
            $false
        )
        $digest = Get-StreamSha256 $stream
        $length = $stream.Length
        if ($ExpectedBytes -ge 0 -and $length -ne $ExpectedBytes) {
            throw "Pinned byte count changed: $Label"
        }
        if (-not [string]::IsNullOrEmpty($ExpectedSha256) -and $digest -ne $ExpectedSha256) {
            throw "Pinned SHA-256 changed: $Label"
        }
        Assert-PlainAncestorChain $absolute
        $after = Get-HandleInformation $stream.SafeFileHandle
        if ((Get-HandleIdentity $before) -ne (Get-HandleIdentity $after)) {
            throw "Held identity changed during authentication: $Label"
        }
        $script:Held[$Label] = [pscustomobject]@{
            Path = $absolute
            Stream = $stream
            Identity = Get-HandleIdentity $before
            Bytes = $length
            Sha256 = $digest
        }
        return $script:Held[$Label]
    }
    catch {
        $handle.Dispose()
        throw
    }
}

function Assert-HeldFilesUnchanged {
    foreach ($entry in $script:Held.GetEnumerator()) {
        $value = $entry.Value
        Assert-PlainAncestorChain $value.Path
        $information = Get-HandleInformation $value.Stream.SafeFileHandle
        if (
            (Get-HandleIdentity $information) -ne $value.Identity -or
            $information.NumberOfLinks -ne 1 -or
            ($information.FileAttributes -band $reparseAttribute) -ne 0 -or
            $value.Stream.Length -ne $value.Bytes -or
            (Get-StreamSha256 $value.Stream) -ne $value.Sha256
        ) {
            throw "Held file changed before child completion: $($entry.Key)"
        }
    }
}

function Open-HeldPowerShellDistribution([string]$HostPath) {
    $distributionRoot = [System.IO.Path]::GetFullPath((Split-Path -Parent $HostPath))
    Assert-PlainAncestorChain $distributionRoot
    [string[]]$paths = @(
        Get-ChildItem -LiteralPath $distributionRoot -File -Recurse -Force |
            ForEach-Object { $_.FullName }
    )
    $rows = [System.Collections.Generic.List[string]]::new()
    [long]$totalBytes = 0
    foreach ($path in $paths) {
        $relative = [System.IO.Path]::GetRelativePath($distributionRoot, $path).Replace('\', '/')
        $held = Open-HeldFile "POWERSHELL_DISTRIBUTION::$relative" $path -1 $null
        $rows.Add("$relative`0$($held.Bytes)`0$($held.Sha256)`n")
        $totalBytes += $held.Bytes
    }
    [string[]]$canonicalRows = $rows.ToArray()
    [Array]::Sort($canonicalRows, [StringComparer]::Ordinal)
    $inventoryBytes = [System.Text.Encoding]::UTF8.GetBytes([string]::Concat($canonicalRows))
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try {
        $inventorySha256 = [Convert]::ToHexString(
            $algorithm.ComputeHash($inventoryBytes)
        ).ToLowerInvariant()
    }
    finally {
        $algorithm.Dispose()
    }
    if (
        $paths.Count -ne 983 -or
        $totalBytes -ne 296034085 -or
        $inventorySha256 -ne 'fcbbc18499e682ca08a0860dcb3b5353099a2a846e9eedc50afbb0c28ed728dc'
    ) {
        throw 'Externally QA-pinned PowerShell distribution inventory differs.'
    }
    return $distributionRoot
}

if (
    -not $env:P1_WORKSPACE_ROOT -or
    -not [System.IO.Path]::IsPathFullyQualified($env:P1_WORKSPACE_ROOT)
) {
    throw 'P1_WORKSPACE_ROOT must be an externally supplied absolute path.'
}
$workspace = [System.IO.Path]::GetFullPath($env:P1_WORKSPACE_ROOT)
Assert-PlainAncestorChain $workspace
if ((Get-NormalizedPath (Get-Location).Path) -ne (Get-NormalizedPath $workspace)) {
    throw 'Canonical launcher working directory must equal P1_WORKSPACE_ROOT.'
}
if ($Mode -eq 'Execute') {
    if (-not $env:P1_DATA_DIR -or -not [System.IO.Path]::IsPathFullyQualified($env:P1_DATA_DIR)) {
        throw 'Execute mode requires an externally supplied absolute P1_DATA_DIR.'
    }
    Assert-PlainAncestorChain ([System.IO.Path]::GetFullPath($env:P1_DATA_DIR))
}

$stage0PathVariable = Get-Variable -Scope Global -Name P1_STAGE0_LAUNCHER_PATH -ErrorAction Stop
$stage0StreamVariable = Get-Variable -Scope Global -Name P1_STAGE0_LAUNCHER_STREAM -ErrorAction Stop
$stage0IdentityVariable = Get-Variable -Scope Global -Name P1_STAGE0_LAUNCHER_IDENTITY -ErrorAction Stop
$stage0BytesVariable = Get-Variable -Scope Global -Name P1_STAGE0_LAUNCHER_BYTES -ErrorAction Stop
$stage0ShaVariable = Get-Variable -Scope Global -Name P1_STAGE0_LAUNCHER_SHA256 -ErrorAction Stop
$stage0Path = [System.IO.Path]::GetFullPath([string]$stage0PathVariable.Value)
$stage0Stream = $stage0StreamVariable.Value
$expectedLauncherPath = Join-Path $workspace 'scripts/launch_p1_multiscale_cross_layer_offset_drift_unary_v6r3.ps1'
if (
    $stage0Stream -isnot [System.IO.FileStream] -or
    (Get-NormalizedPath $stage0Path) -ne (Get-NormalizedPath $expectedLauncherPath) -or
    [long]$stage0BytesVariable.Value -ne $stage0Stream.Length
) {
    throw 'Pinned inline stage-0 launcher handoff differs.'
}
$stage0Information = Get-HandleInformation $stage0Stream.SafeFileHandle
if (
    (Get-HandleIdentity $stage0Information) -ne [string]$stage0IdentityVariable.Value -or
    $stage0Information.NumberOfLinks -ne 1 -or
    ($stage0Information.FileAttributes -band $reparseAttribute) -ne 0 -or
    (Get-StreamSha256 $stage0Stream) -ne [string]$stage0ShaVariable.Value
) {
    throw 'Pinned inline stage-0 launcher handle changed before stage-1.'
}
$script:Held['EXTERNAL_LAUNCHER'] = [pscustomobject]@{
    Path = $stage0Path
    Stream = $stage0Stream
    Identity = [string]$stage0IdentityVariable.Value
    Bytes = [long]$stage0BytesVariable.Value
    Sha256 = [string]$stage0ShaVariable.Value
    Stage0Owned = $true
}
$launcher = $script:Held['EXTERNAL_LAUNCHER']
$null = Open-HeldPowerShellDistribution $actualHost
$null = Open-HeldFile 'POWERSHELL_HOST' $actualHost 301368 'db6dd81183fe57d22e03b911ec9a30a2fd7c40542e97743615355a6fb44f458f'
$stub = Open-HeldFile 'VENV_STUB' (Join-Path $workspace '.venv-p1/Scripts/python.exe') 274424 '0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14'
$pyvenv = Open-HeldFile 'PYVENV_CFG' (Join-Path $workspace '.venv-p1/pyvenv.cfg') 339 'd1fb970854073922d49959ae01539088550613e316cb67f9fac858f586361174'
$startupTrust = Open-HeldFile 'STARTUP_TRUST' (Join-Path $workspace 'configs/experiments/p1_multiscale_cross_layer_offset_drift_unary_v6r3_startup_trust.json') 8249 '7dcc0c4a79eb3d1d67e22a1d9889e0160c082a925f460b884b1b7e50f5d75dfc'
$bootstrap = Open-HeldFile 'BOOTSTRAP' (Join-Path $workspace 'scripts/bootstrap_p1_multiscale_cross_layer_offset_drift_unary_v6r3.py') 74549 'e9e78d80559e234d6fde88a97e02d87325a2313435d32823956bd37cf65deecc'

$reader = [System.IO.BinaryReader]::new($pyvenv.Stream, [System.Text.Encoding]::UTF8, $true)
try {
    $pyvenv.Stream.Position = 0
    $pyvenvText = [System.Text.Encoding]::UTF8.GetString($reader.ReadBytes([int]$pyvenv.Bytes))
    $pyvenv.Stream.Position = 0
}
finally {
    $reader.Dispose()
}
$pyvenvValues = @{}
foreach ($line in ($pyvenvText -split "`r?`n")) {
    if (-not $line.Contains('=')) {
        continue
    }
    $pair = $line.Split('=', 2)
    $key = $pair[0].Trim().ToLowerInvariant()
    if ($pyvenvValues.ContainsKey($key)) {
        throw "Duplicate pyvenv.cfg key: $key"
    }
    $pyvenvValues[$key] = $pair[1].Trim()
}
foreach ($key in @('home', 'executable', 'version', 'include-system-site-packages')) {
    if (-not $pyvenvValues.ContainsKey($key)) {
        throw "Missing pyvenv.cfg key: $key"
    }
}
if (
    -not [System.IO.Path]::IsPathFullyQualified($pyvenvValues['home']) -or
    -not [System.IO.Path]::IsPathFullyQualified($pyvenvValues['executable']) -or
    $pyvenvValues['version'] -ne '3.12.10' -or
    $pyvenvValues['include-system-site-packages'].ToLowerInvariant() -ne 'false'
) {
    throw 'Pinned pyvenv.cfg values differ.'
}
$baseRoot = [System.IO.Path]::GetFullPath($pyvenvValues['home'])
$basePython = [System.IO.Path]::GetFullPath($pyvenvValues['executable'])
if (
    (Get-NormalizedPath $basePython) -ne (Get-NormalizedPath (Join-Path $baseRoot 'python.exe'))
) {
    throw 'pyvenv.cfg home/executable binding differs.'
}
Assert-PlainAncestorChain $baseRoot

$startupPins = @(
    @('BASE_PYTHON', 'python.exe', 104952, '4d6f5f81a4bca11191c4c7c6b43632694d0a4ce74e068619d8fdc161d469859a'),
    @('PYTHON312_DLL', 'python312.dll', 6945272, '9a0e3435aaa680d868150f87ab3e388ad2eebc22f87e036155c7b4eda8cd2120'),
    @('PYTHON3_DLL', 'python3.dll', 70376, 'fb975a606e7fbf74f64260e3f60c3490b4f74a183c0926fd6ed1ac4c52ac7b1c'),
    @('VCRUNTIME140_DLL', 'vcruntime140.dll', 120400, '052ad6a20d375957e82aa6a3c441ea548d89be0981516ca7eb306e063d5027f4'),
    @('ENCODINGS_INIT', 'Lib/encodings/__init__.py', 6058, '8b997e9f7beef09de01c34ac34191866d3ab25e17164e08f411940b070bc3e74'),
    @('ENCODINGS_ALIASES', 'Lib/encodings/aliases.py', 16228, '1893cfb597bc5eafd38ef03ac85d8874620112514eb42660408811929cc0d6f8'),
    @('ENCODINGS_UTF8', 'Lib/encodings/utf_8.py', 1047, '9c54c7db8ce0722ca4ddb5f45d4e170357e37991afb3fcdc091721bf6c09257e'),
    @('ENCODINGS_CP949', 'Lib/encodings/cp949.py', 1062, 'da13fd6f1bd7a1d3b48aed1fc75f7516d6a33814086cf971e030625590e9dda0')
)
foreach ($pin in $startupPins) {
    $null = Open-HeldFile $pin[0] (Join-Path $baseRoot $pin[1]) $pin[2] $pin[3]
}

$cachePrefix = Get-NormalizedPath $startupTrust.Path
$cachePrefixItem = Get-Item -LiteralPath $cachePrefix -Force
if ($cachePrefixItem -isnot [System.IO.FileInfo]) {
    throw 'The canonical pycache prefix must be the held regular-file sentinel.'
}
foreach ($name in @('PYTHONHOME', 'PYTHONPATH', 'PYTHONSTARTUP')) {
    [Environment]::SetEnvironmentVariable($name, $null, 'Process')
}
foreach ($name in @('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')) {
    [Environment]::SetEnvironmentVariable($name, '1', 'Process')
}
[Environment]::SetEnvironmentVariable('PYTHONHASHSEED', '0', 'Process')

$bootstrapMode = if ($Mode -eq 'CheckOnly') { '--check-only' } else { '--execute' }
$childExit = 1
try {
    & $stub.Path -I -S -B -X "pycache_prefix=$cachePrefix" $bootstrap.Path $bootstrapMode
    $childExit = $LASTEXITCODE
}
finally {
    try {
        if ((Get-Item -LiteralPath $cachePrefix -Force) -isnot [System.IO.FileInfo]) {
            throw 'The held regular-file pycache sentinel changed.'
        }
        Assert-HeldFilesUnchanged
    }
    finally {
        foreach ($entry in $script:Held.GetEnumerator()) {
            if ($null -eq $entry.Value.PSObject.Properties['Stage0Owned']) {
                $entry.Value.Stream.Dispose()
            }
        }
    }
}

$global:P1_STAGE1_CHILD_EXIT = $childExit
return
