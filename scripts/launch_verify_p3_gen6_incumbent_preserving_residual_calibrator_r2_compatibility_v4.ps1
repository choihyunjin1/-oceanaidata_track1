[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Root,
    [Parameter(Mandatory = $true)]
    [string]$AuthenticatedLauncherPath,
    [Parameter(Mandatory = $true)]
    [string]$AuthenticatedLauncherSha256,
    [ValidateSet('check-only')]
    [string]$Mode = 'check-only'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ($PSVersionTable.PSVersion.ToString() -ne '7.6.4') {
    throw 'Canonical PowerShell 7.6.4 is required.'
}
if (-not $env:P3_POWERSHELL_HOST -or -not [System.IO.Path]::IsPathFullyQualified($env:P3_POWERSHELL_HOST)) {
    throw 'P3_POWERSHELL_HOST must be an externally authenticated absolute path.'
}
$ExpectedPowerShellHost = $env:P3_POWERSHELL_HOST
if (
    [System.IO.Path]::GetFullPath([Environment]::ProcessPath).ToLowerInvariant() -ne
    [System.IO.Path]::GetFullPath($ExpectedPowerShellHost).ToLowerInvariant()
) {
    throw 'Exact externally pinned PowerShell host path is required.'
}

Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.IO;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

public static class P3CompatV4Native {
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

$GenericRead = [Convert]::ToUInt32('80000000', 16)
$FileShareRead = [uint32]0x00000001
$OpenExisting = [uint32]3
$OpenReparsePoint = [uint32]0x00200000
$ReparseAttribute = [uint32]0x00000400
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
    $information = New-Object P3CompatV4Native+BY_HANDLE_FILE_INFORMATION
    if (-not [P3CompatV4Native]::GetFileInformationByHandle($Handle, [ref]$information)) {
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
    $handle = [P3CompatV4Native]::CreateFileW(
        $absolute,
        $GenericRead,
        $FileShareRead,
        [IntPtr]::Zero,
        $OpenExisting,
        $OpenReparsePoint,
        [IntPtr]::Zero
    )
    if ($handle.IsInvalid) {
        throw [System.ComponentModel.Win32Exception]::new(
            [System.Runtime.InteropServices.Marshal]::GetLastWin32Error()
        )
    }
    try {
        $information = Get-HandleInformation $handle
        if (
            $information.NumberOfLinks -ne 1 -or
            ($information.FileAttributes -band $ReparseAttribute) -ne 0
        ) {
            throw "Single-link non-reparse file required: $Label"
        }
        $stream = [System.IO.FileStream]::new($handle, [System.IO.FileAccess]::Read, 1048576, $false)
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
        if ((Get-HandleIdentity $information) -ne (Get-HandleIdentity $after)) {
            throw "Held identity changed during authentication: $Label"
        }
        $script:Held[$Label] = [pscustomobject]@{
            Path = $absolute
            Stream = $stream
            Identity = Get-HandleIdentity $information
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
            ($information.FileAttributes -band $ReparseAttribute) -ne 0 -or
            $value.Stream.Length -ne $value.Bytes -or
            (Get-StreamSha256 $value.Stream) -ne $value.Sha256
        ) {
            throw "Held file changed before child completion: $($entry.Key)"
        }
    }
}

function Get-DistributionEntries([string]$RootPath) {
    $rootFull = [System.IO.Path]::GetFullPath($RootPath).TrimEnd('\')
    Assert-PlainAncestorChain $rootFull
    $directories = [System.Collections.Generic.List[string]]::new()
    $files = [System.Collections.Generic.Dictionary[string, string]]::new(
        [System.StringComparer]::Ordinal
    )
    $pending = [System.Collections.Generic.Stack[string]]::new()
    $pending.Push($rootFull)
    while ($pending.Count -gt 0) {
        $current = $pending.Pop()
        foreach ($child in [System.IO.Directory]::EnumerateFileSystemEntries($current)) {
            $item = Get-Item -LiteralPath $child -Force
            if (
                ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0 -or
                $null -ne $item.LinkType
            ) {
                throw "PowerShell distribution link/reparse is forbidden: $child"
            }
            $relative = $item.FullName.Substring($rootFull.Length + 1).Replace('\', '/')
            if ($item -is [System.IO.DirectoryInfo]) {
                $directories.Add($relative)
                $pending.Push($item.FullName)
            }
            elseif ($item -is [System.IO.FileInfo]) {
                $files.Add($relative, $item.FullName)
            }
            else {
                throw "Unsupported PowerShell distribution entry: $child"
            }
        }
    }
    $directoryArray = [string[]]$directories.ToArray()
    [Array]::Sort($directoryArray, [System.StringComparer]::Ordinal)
    $fileArray = [string[]]$files.Keys
    [Array]::Sort($fileArray, [System.StringComparer]::Ordinal)
    return [pscustomobject]@{
        Root = $rootFull
        Directories = $directoryArray
        Files = $fileArray
        FilePaths = $files
    }
}

function Open-HeldPowerShellDistribution([string]$RootPath) {
    $expectedDirectories = 53
    $expectedFiles = 983
    $expectedBytes = [long]296034085
    $expectedPayloadBytes = 112750
    $expectedSha256 = 'eef4626964532f664559724e8ce95b2a95b6cb4729d275ac6cd0da81e0115444'
    $entries = Get-DistributionEntries $RootPath
    if (
        $entries.Directories.Count -ne $expectedDirectories -or
        $entries.Files.Count -ne $expectedFiles
    ) {
        throw 'PowerShell distribution path inventory count changed.'
    }
    $builder = [System.Text.StringBuilder]::new()
    foreach ($relative in $entries.Directories) {
        [void]$builder.Append("D`0$relative`n")
    }
    $totalBytes = [long]0
    foreach ($relative in $entries.Files) {
        $held = Open-HeldFile "POWERSHELL_DIST::$relative" $entries.FilePaths[$relative] -1 $null
        $totalBytes += [long]$held.Bytes
        [void]$builder.Append("F`0$relative`0$($held.Bytes)`0$($held.Sha256)`n")
    }
    $payload = [System.Text.UTF8Encoding]::new($false).GetBytes($builder.ToString())
    $digest = [Convert]::ToHexString(
        [System.Security.Cryptography.SHA256]::HashData($payload)
    ).ToLowerInvariant()
    if (
        $totalBytes -ne $expectedBytes -or
        $payload.Length -ne $expectedPayloadBytes -or
        $digest -ne $expectedSha256
    ) {
        throw 'PowerShell distribution authenticated inventory changed.'
    }
    $script:PowerShellDistribution = [pscustomobject]@{
        Root = $entries.Root
        Directories = $entries.Directories
        Files = $entries.Files
        FileBytes = $totalBytes
        PayloadBytes = $payload.Length
        Sha256 = $digest
    }
}

function Assert-PowerShellDistributionNamesUnchanged {
    $current = Get-DistributionEntries $script:PowerShellDistribution.Root
    if (
        [string]::Join("`0", $current.Directories) -ne
            [string]::Join("`0", $script:PowerShellDistribution.Directories) -or
        [string]::Join("`0", $current.Files) -ne
            [string]::Join("`0", $script:PowerShellDistribution.Files)
    ) {
        throw 'PowerShell distribution names changed before child completion.'
    }
}

$workspace = [System.IO.Path]::GetFullPath($Root, (Get-Location).Path)
Assert-PlainAncestorChain $workspace
if ((Get-NormalizedPath (Get-Location).Path) -ne (Get-NormalizedPath $workspace)) {
    throw 'Canonical launcher working directory must equal -Root.'
}
if (-not $env:P3_WORKSPACE_ROOT -or (Get-NormalizedPath $env:P3_WORKSPACE_ROOT) -ne (Get-NormalizedPath $workspace)) {
    throw 'P3_WORKSPACE_ROOT must equal the canonical workspace.'
}
if (-not $env:P3_DATA_DIR) {
    throw 'P3_DATA_DIR is required.'
}
foreach ($name in 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS') {
    if ([Environment]::GetEnvironmentVariable($name) -ne '1') {
        throw "Canonical thread environment changed: $name"
    }
}
if ($env:PYTHONHASHSEED -ne '0') {
    throw 'PYTHONHASHSEED must equal 0.'
}

$expectedLauncherPath = Join-Path $workspace 'scripts/launch_verify_p3_gen6_incumbent_preserving_residual_calibrator_r2_compatibility_v4.ps1'
if (
    (Get-NormalizedPath $AuthenticatedLauncherPath) -ne (Get-NormalizedPath $expectedLauncherPath) -or
    $AuthenticatedLauncherSha256 -notmatch '^[0-9a-f]{64}$'
) {
    throw 'Authenticated stage-1 launcher attestation changed.'
}
$launcher = Open-HeldFile 'EXTERNAL_LAUNCHER' $AuthenticatedLauncherPath -1 $AuthenticatedLauncherSha256
$pwsh = Open-HeldFile 'POWERSHELL_HOST' ([Environment]::ProcessPath) 301368 'db6dd81183fe57d22e03b911ec9a30a2fd7c40542e97743615355a6fb44f458f'
Open-HeldPowerShellDistribution (Split-Path -Parent $pwsh.Path)
$stubPath = Join-Path $workspace '.venv-p1/Scripts/python.exe'
$cfgPath = Join-Path $workspace '.venv-p1/pyvenv.cfg'
$stub = Open-HeldFile 'VENV_STUB' $stubPath 274424 '0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14'
$cfg = Open-HeldFile 'PYVENV_CFG' $cfgPath 339 'd1fb970854073922d49959ae01539088550613e316cb67f9fac858f586361174'
$cfgText = [System.Text.Encoding]::UTF8.GetString((New-Object System.IO.BinaryReader($cfg.Stream, [System.Text.Encoding]::UTF8, $true)).ReadBytes([int]$cfg.Bytes))
$cfg.Stream.Position = 0
$homeLine = @($cfgText -split "`r?`n" | Where-Object { $_ -like 'home = *' })
if ($homeLine.Count -ne 1) {
    throw 'Pinned pyvenv.cfg home is malformed.'
}
$baseRoot = [System.IO.Path]::GetFullPath($homeLine[0].Substring(7))
Assert-PlainAncestorChain $baseRoot

$startupPins = @(
    @('BASE_PYTHON', 'python.exe', 104952, '4d6f5f81a4bca11191c4c7c6b43632694d0a4ce74e068619d8fdc161d469859a'),
    @('PYTHON3_DLL', 'python3.dll', 70376, 'fb975a606e7fbf74f64260e3f60c3490b4f74a183c0926fd6ed1ac4c52ac7b1c'),
    @('PYTHON312_DLL', 'python312.dll', 6945272, '9a0e3435aaa680d868150f87ab3e388ad2eebc22f87e036155c7b4eda8cd2120'),
    @('VCRUNTIME140_DLL', 'vcruntime140.dll', 120400, '052ad6a20d375957e82aa6a3c441ea548d89be0981516ca7eb306e063d5027f4'),
    @('VCRUNTIME140_1_DLL', 'vcruntime140_1.dll', 49776, '6a99bc0128e0c7d6cbbf615fcc26909565e17d4ca3451b97f8987f9c6acbc6c8'),
    @('ENCODINGS_INIT', 'Lib/encodings/__init__.py', 6058, '8b997e9f7beef09de01c34ac34191866d3ab25e17164e08f411940b070bc3e74'),
    @('ENCODINGS_ALIASES', 'Lib/encodings/aliases.py', 16228, '1893cfb597bc5eafd38ef03ac85d8874620112514eb42660408811929cc0d6f8'),
    @('ENCODINGS_UTF8', 'Lib/encodings/utf_8.py', 1047, '9c54c7db8ce0722ca4ddb5f45d4e170357e37991afb3fcdc091721bf6c09257e'),
    @('ENCODINGS_CP949', 'Lib/encodings/cp949.py', 1062, 'da13fd6f1bd7a1d3b48aed1fc75f7516d6a33814086cf971e030625590e9dda0')
)
foreach ($pin in $startupPins) {
    $null = Open-HeldFile $pin[0] (Join-Path $baseRoot $pin[1]) $pin[2] $pin[3]
}

$implementationPins = @(
    @('V4_BOOTSTRAP', 'scripts/bootstrap_verify_p3_gen6_incumbent_preserving_residual_calibrator_r2_compatibility_v4.py', 116592, '1114f2bc4e9d7cbe91c1beca31570c7146e617b4970f5583c9c02d58f3522018'),
    @('V4_CONFIG', 'configs/experiments/p3_gen6_incumbent_preserving_residual_calibrator_v1r2_compatibility_verifier_v4.json', 21782, '0dce5765d5ad7fa00b5cd76731d14754fa74e3142616a0be44ddc4ac6c2c67ad'),
    @('V4_HELPER', 'src/p3_wave/gen6_incumbent_preserving_residual_calibrator_r2_compatibility_verifier_v4.py', 14690, '3f9451cbae2f9b89b938435247d65555f39f5219f5e33b761025131dd3194473'),
    @('V4_CLI', 'scripts/verify_p3_gen6_incumbent_preserving_residual_calibrator_r2_compatibility_v4.py', 1007, '84cf5ff2e1060227cb0cc0e419b7e1e65d0a3c23b51a2df68e6ee6a8efe55331'),
    @('V4_TESTS', 'tests/test_p3_gen6_incumbent_preserving_residual_calibrator_r2_compatibility_verifier_v4.py', 25028, '5fe0cc39a467f89a6b87a31021810a4fd579bf81eb9ba231527c5d6b31727bcf'),
    @('PYCACHE_BLOCK_SENTINEL', 'scripts/p3_gen6_r2_compatibility_v4_pycache_block.sentinel', 74, 'ddb8423e21b551829ced83fb63c56d17df123ba9459fc44a3ddfbbb8735c55bd')
)
foreach ($pin in $implementationPins) {
    $null = Open-HeldFile $pin[0] (Join-Path $workspace $pin[1]) $pin[2] $pin[3]
}

$cacheSentinel = $script:Held['PYCACHE_BLOCK_SENTINEL'].Path

$childExit = 1
try {
    & $stub.Path -I -S -B -X "pycache_prefix=$cacheSentinel" $script:Held['V4_BOOTSTRAP'].Path `
        --root $workspace `
        --mode $Mode `
        --external-launcher-sha256 $launcher.Sha256 `
        --startup-pycache-sentinel $cacheSentinel
    $childExit = $LASTEXITCODE
    Assert-PowerShellDistributionNamesUnchanged
    Assert-HeldFilesUnchanged
}
finally {
    foreach ($entry in $script:Held.GetEnumerator()) {
        $entry.Value.Stream.Dispose()
    }
}

if ($childExit -ne 0) {
    throw "Authenticated Python check-only child failed with exit code $childExit."
}
