Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ExpectedHostBytes = [long]301368
$ExpectedHostSha256 = 'db6dd81183fe57d22e03b911ec9a30a2fd7c40542e97743615355a6fb44f458f'
$ExpectedLauncherBytes = [long]16551
$ExpectedLauncherSha256 = '08809e3913fb57d950e9e3cd34f5f992ec48dd8fd5796503ef1936f0bb4f3e08'
$LauncherRelative = 'scripts/launch_verify_p3_gen6_incumbent_preserving_residual_calibrator_r2_compatibility_v4.ps1'

if (
    -not $env:P3_WORKSPACE_ROOT -or
    -not [System.IO.Path]::IsPathFullyQualified($env:P3_WORKSPACE_ROOT) -or
    -not $env:P3_POWERSHELL_HOST -or
    -not [System.IO.Path]::IsPathFullyQualified($env:P3_POWERSHELL_HOST)
) {
    throw 'Externally authenticated absolute workspace and PowerShell host are required.'
}
$workspace = [System.IO.Path]::GetFullPath($env:P3_WORKSPACE_ROOT)
$hostPath = [System.IO.Path]::GetFullPath($env:P3_POWERSHELL_HOST)
$launcherPath = [System.IO.Path]::GetFullPath((Join-Path $workspace $LauncherRelative))
if (
    [System.IO.Path]::GetFullPath([Environment]::ProcessPath).ToLowerInvariant() -ne
        $hostPath.ToLowerInvariant() -or
    [System.IO.Path]::GetFullPath((Get-Location).Path).ToLowerInvariant() -ne
        $workspace.ToLowerInvariant()
) {
    throw 'Stage-0 host or working-directory identity changed.'
}

Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

public static class P3CompatV4Stage0Native {
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
$FileShareRead = [uint32]1
$OpenExisting = [uint32]3
$OpenReparsePoint = [uint32]0x00200000
$ReparseAttribute = [uint32]0x00000400

function Assert-Stage0PlainPath([string]$Path) {
    $item = Get-Item -LiteralPath $Path -Force
    while ($null -ne $item) {
        if (
            ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0 -or
            $null -ne $item.LinkType
        ) {
            throw "Stage-0 link/reparse is forbidden: $Path"
        }
        $item = if ($item -is [System.IO.DirectoryInfo]) { $item.Parent } else { $item.Directory }
    }
}

function Get-Stage0HandleInfo([Microsoft.Win32.SafeHandles.SafeFileHandle]$Handle) {
    $information = New-Object P3CompatV4Stage0Native+BY_HANDLE_FILE_INFORMATION
    if (-not [P3CompatV4Stage0Native]::GetFileInformationByHandle($Handle, [ref]$information)) {
        throw [System.ComponentModel.Win32Exception]::new(
            [System.Runtime.InteropServices.Marshal]::GetLastWin32Error()
        )
    }
    return $information
}

function Get-Stage0Identity($Information) {
    return @(
        [uint64]$Information.VolumeSerialNumber,
        (([uint64]$Information.FileIndexHigh -shl 32) -bor [uint64]$Information.FileIndexLow),
        (([uint64]$Information.FileSizeHigh -shl 32) -bor [uint64]$Information.FileSizeLow),
        [uint64]$Information.NumberOfLinks,
        [uint64]$Information.FileAttributes
    ) -join ':'
}

function Get-Stage0StreamSha([System.IO.FileStream]$Stream) {
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

function Open-Stage0HeldFile(
    [string]$Path,
    [long]$ExpectedBytes,
    [string]$ExpectedSha256
) {
    $absolute = [System.IO.Path]::GetFullPath($Path)
    Assert-Stage0PlainPath $absolute
    $handle = [P3CompatV4Stage0Native]::CreateFileW(
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
        $before = Get-Stage0HandleInfo $handle
        if (
            $before.NumberOfLinks -ne 1 -or
            ($before.FileAttributes -band $ReparseAttribute) -ne 0
        ) {
            throw 'Stage-0 requires a single-link non-reparse file.'
        }
        $stream = [System.IO.FileStream]::new($handle, [System.IO.FileAccess]::Read, 1048576, $false)
        $digest = Get-Stage0StreamSha $stream
        if ($stream.Length -ne $ExpectedBytes -or $digest -ne $ExpectedSha256) {
            throw 'Stage-0 held file pin changed.'
        }
        Assert-Stage0PlainPath $absolute
        $after = Get-Stage0HandleInfo $stream.SafeFileHandle
        if ((Get-Stage0Identity $before) -ne (Get-Stage0Identity $after)) {
            throw 'Stage-0 held file identity changed during authentication.'
        }
        return [pscustomobject]@{
            Path = $absolute
            Stream = $stream
            Identity = Get-Stage0Identity $before
            Bytes = $stream.Length
            Sha256 = $digest
        }
    }
    catch {
        $handle.Dispose()
        throw
    }
}

$heldHost = Open-Stage0HeldFile $hostPath $ExpectedHostBytes $ExpectedHostSha256
$launcher = Open-Stage0HeldFile $launcherPath $ExpectedLauncherBytes $ExpectedLauncherSha256
try {
    $reader = [System.IO.BinaryReader]::new(
        $launcher.Stream,
        [System.Text.UTF8Encoding]::new($false, $true),
        $true
    )
    try {
        $launcherRaw = $reader.ReadBytes([int]$launcher.Bytes)
    }
    finally {
        $reader.Dispose()
    }
    $launcher.Stream.Position = 0
    $launcherText = [System.Text.UTF8Encoding]::new($false, $true).GetString($launcherRaw)
    $launcherBlock = [System.Management.Automation.ScriptBlock]::Create($launcherText)
    & $launcherBlock `
        -Root $workspace `
        -AuthenticatedLauncherPath $launcher.Path `
        -AuthenticatedLauncherSha256 $launcher.Sha256 `
        -Mode 'check-only'
}
finally {
    foreach ($held in @($launcher, $heldHost)) {
        Assert-Stage0PlainPath $held.Path
        $final = Get-Stage0HandleInfo $held.Stream.SafeFileHandle
        if (
            (Get-Stage0Identity $final) -ne $held.Identity -or
            $final.NumberOfLinks -ne 1 -or
            ($final.FileAttributes -band $ReparseAttribute) -ne 0 -or
            $held.Stream.Length -ne $held.Bytes -or
            (Get-Stage0StreamSha $held.Stream) -ne $held.Sha256
        ) {
            throw 'Stage-0 held file changed during authenticated in-memory execution.'
        }
        $held.Stream.Dispose()
    }
}
