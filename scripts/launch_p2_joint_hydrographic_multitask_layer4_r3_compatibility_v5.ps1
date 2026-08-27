[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Root
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Add-Type -TypeDefinition @"
using System;
using System.ComponentModel;
using System.IO;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using Microsoft.Win32.SafeHandles;

public sealed class P2HeldFile : IDisposable
{
    public readonly string RequestedPath;
    public readonly string FinalPath;
    public readonly SafeFileHandle Handle;
    public readonly FileStream Stream;
    public readonly byte[] Raw;
    public readonly long Size;
    public readonly string Sha256;
    public readonly uint NumberOfLinks;
    private readonly uint volumeSerial;
    private readonly uint fileIndexHigh;
    private readonly uint fileIndexLow;
    private readonly long lastWrite;

    internal P2HeldFile(
        string requestedPath,
        string finalPath,
        SafeFileHandle handle,
        FileStream stream,
        byte[] raw,
        P2StableNative.BY_HANDLE_FILE_INFORMATION info)
    {
        RequestedPath = requestedPath;
        FinalPath = finalPath;
        Handle = handle;
        Stream = stream;
        Raw = raw;
        Size = raw.LongLength;
        Sha256 = P2StableNative.Hash(raw);
        NumberOfLinks = info.NumberOfLinks;
        volumeSerial = info.VolumeSerialNumber;
        fileIndexHigh = info.FileIndexHigh;
        fileIndexLow = info.FileIndexLow;
        lastWrite = ((long)info.LastWriteTime.dwHighDateTime << 32)
            | info.LastWriteTime.dwLowDateTime;
    }

    public void PostChildRehash(long expectedBytes, string expectedSha256)
    {
        P2StableNative.BY_HANDLE_FILE_INFORMATION info =
            P2StableNative.Information(Handle);
        long observedSize = ((long)info.FileSizeHigh << 32) | info.FileSizeLow;
        long observedWrite = ((long)info.LastWriteTime.dwHighDateTime << 32)
            | info.LastWriteTime.dwLowDateTime;
        if (
            info.VolumeSerialNumber != volumeSerial
            || info.FileIndexHigh != fileIndexHigh
            || info.FileIndexLow != fileIndexLow
            || info.NumberOfLinks != 1
            || observedSize != Size
            || observedWrite != lastWrite)
        {
            throw new InvalidOperationException(
                "P2 v5 held-file identity changed after child exit: " + RequestedPath);
        }
        string finalPath = P2StableNative.FinalPath(Handle);
        if (!String.Equals(
            Path.GetFullPath(finalPath),
            Path.GetFullPath(FinalPath),
            StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException(
                "P2 v5 held-file final path changed after child exit: " + RequestedPath);
        }
        Stream.Position = 0;
        string digest;
        using (SHA256 sha = SHA256.Create())
        {
            digest = Convert.ToHexString(sha.ComputeHash(Stream)).ToLowerInvariant();
        }
        if (Size != expectedBytes || !String.Equals(
            digest, expectedSha256, StringComparison.Ordinal))
        {
            throw new InvalidOperationException(
                "P2 v5 held-file bytes changed after child exit: " + RequestedPath);
        }
    }

    public void Dispose()
    {
        Stream.Dispose();
        Handle.Dispose();
    }
}

public static class P2StableNative
{
    public const uint GENERIC_READ = 0x80000000;
    public const uint FILE_SHARE_READ = 0x00000001;
    public const uint OPEN_EXISTING = 3;
    public const uint FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000;
    private const uint FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400;
    private const uint VOLUME_NAME_DOS = 0;

    [StructLayout(LayoutKind.Sequential)]
    public struct FILETIME
    {
        public uint dwLowDateTime;
        public uint dwHighDateTime;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct BY_HANDLE_FILE_INFORMATION
    {
        public uint FileAttributes;
        public FILETIME CreationTime;
        public FILETIME LastAccessTime;
        public FILETIME LastWriteTime;
        public uint VolumeSerialNumber;
        public uint FileSizeHigh;
        public uint FileSizeLow;
        public uint NumberOfLinks;
        public uint FileIndexHigh;
        public uint FileIndexLow;
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern SafeFileHandle CreateFileW(
        string lpFileName,
        uint dwDesiredAccess,
        uint dwShareMode,
        IntPtr lpSecurityAttributes,
        uint dwCreationDisposition,
        uint dwFlagsAndAttributes,
        IntPtr hTemplateFile);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool GetFileInformationByHandle(
        SafeFileHandle hFile,
        out BY_HANDLE_FILE_INFORMATION lpFileInformation);

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern uint GetFinalPathNameByHandleW(
        SafeFileHandle hFile,
        StringBuilder lpszFilePath,
        uint cchFilePath,
        uint dwFlags);

    public static string Hash(byte[] raw)
    {
        using (SHA256 sha = SHA256.Create())
        {
            return Convert.ToHexString(sha.ComputeHash(raw)).ToLowerInvariant();
        }
    }

    public static BY_HANDLE_FILE_INFORMATION Information(SafeFileHandle handle)
    {
        BY_HANDLE_FILE_INFORMATION info;
        if (!GetFileInformationByHandle(handle, out info))
        {
            throw new Win32Exception(Marshal.GetLastWin32Error());
        }
        return info;
    }

    public static string FinalPath(SafeFileHandle handle)
    {
        StringBuilder buffer = new StringBuilder(32768);
        uint length = GetFinalPathNameByHandleW(
            handle, buffer, (uint)buffer.Capacity, VOLUME_NAME_DOS);
        if (length == 0 || length >= buffer.Capacity)
        {
            throw new Win32Exception(Marshal.GetLastWin32Error());
        }
        string value = buffer.ToString();
        if (value.StartsWith(@"\\?\UNC\", StringComparison.OrdinalIgnoreCase))
        {
            return @"\\" + value.Substring(8);
        }
        if (value.StartsWith(@"\\?\", StringComparison.OrdinalIgnoreCase))
        {
            return value.Substring(4);
        }
        return value;
    }

    public static void AssertNoReparseChain(string path, bool leafMayBeMissing)
    {
        string full = Path.GetFullPath(path);
        string root = Path.GetPathRoot(full);
        string tail = full.Substring(root.Length);
        string current = root;
        bool missing = false;
        foreach (string part in tail.Split(
            new char[] { Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar },
            StringSplitOptions.RemoveEmptyEntries))
        {
            current = Path.Combine(current, part);
            bool exists = File.Exists(current) || Directory.Exists(current);
            if (!exists)
            {
                if (!leafMayBeMissing)
                {
                    throw new FileNotFoundException(
                        "P2 v5 stable path component is missing", current);
                }
                missing = true;
                continue;
            }
            if (missing)
            {
                throw new InvalidOperationException(
                    "P2 v5 path exists below a missing ancestor: " + current);
            }
            FileAttributes attributes = File.GetAttributes(current);
            if ((attributes & FileAttributes.ReparsePoint) != 0)
            {
                throw new InvalidOperationException(
                    "P2 v5 reparse ancestor is forbidden: " + current);
            }
        }
    }

    public static P2HeldFile OpenReadStable(string path)
    {
        string full = Path.GetFullPath(path);
        AssertNoReparseChain(full, false);
        SafeFileHandle handle = CreateFileW(
            full,
            GENERIC_READ,
            FILE_SHARE_READ,
            IntPtr.Zero,
            OPEN_EXISTING,
            FILE_FLAG_OPEN_REPARSE_POINT,
            IntPtr.Zero);
        if (handle.IsInvalid)
        {
            throw new Win32Exception(Marshal.GetLastWin32Error(), full);
        }
        try
        {
            BY_HANDLE_FILE_INFORMATION info = Information(handle);
            if ((info.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0)
            {
                throw new InvalidOperationException(
                    "P2 v5 opened target is a reparse point: " + full);
            }
            if (info.NumberOfLinks != 1)
            {
                throw new InvalidOperationException(
                    "P2 v5 regular file must have exactly one hard link: " + full);
            }
            string finalPath = FinalPath(handle);
            if (!String.Equals(
                Path.GetFullPath(finalPath),
                full,
                StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidOperationException(
                    "P2 v5 stable handle final path differs: " + full);
            }
            FileStream stream = new FileStream(handle, FileAccess.Read, 1024 * 1024, false);
            try
            {
                byte[] raw;
                using (MemoryStream memory = new MemoryStream())
                {
                    stream.CopyTo(memory);
                    raw = memory.ToArray();
                }
                long size = ((long)info.FileSizeHigh << 32) | info.FileSizeLow;
                if (raw.LongLength != size)
                {
                    throw new InvalidOperationException(
                        "P2 v5 stable read was truncated: " + full);
                }
                stream.Position = 0;
                return new P2HeldFile(full, finalPath, handle, stream, raw, info);
            }
            catch
            {
                stream.Dispose();
                throw;
            }
        }
        catch
        {
            handle.Dispose();
            throw;
        }
    }
}
"@

function Assert-Pin {
    param(
        [Parameter(Mandatory = $true)]
        [P2HeldFile]$Held,
        [Parameter(Mandatory = $true)]
        [long]$Bytes,
        [Parameter(Mandatory = $true)]
        [string]$Sha256,
        [Parameter(Mandatory = $true)]
        [string]$Label
    )
    if ($Held.Size -ne $Bytes -or $Held.Sha256 -cne $Sha256) {
        throw "P2 v5 external startup pin changed: $Label"
    }
}

$hostText = [Environment]::GetEnvironmentVariable("P2_POWERSHELL_HOST")
if ([string]::IsNullOrWhiteSpace($hostText) -or -not [IO.Path]::IsPathFullyQualified($hostText)) {
    throw "P2_POWERSHELL_HOST must name the externally pinned absolute PowerShell host"
}
$hostPath = [IO.Path]::GetFullPath($hostText)
$processPath = [Environment]::ProcessPath
if (
    [string]::IsNullOrWhiteSpace($processPath) -or
    -not [string]::Equals(
        [IO.Path]::GetFullPath($processPath),
        $hostPath,
        [StringComparison]::OrdinalIgnoreCase
    )
) {
    throw "P2 v5 launcher host differs from P2_POWERSHELL_HOST"
}

$workspace = [IO.Path]::GetFullPath($Root)
if (-not [IO.Directory]::Exists($workspace)) {
    throw "P2 v5 workspace directory is missing"
}
[P2StableNative]::AssertNoReparseChain($workspace, $false)

$canonicalPycache = [IO.Path]::Combine(
    $workspace,
    "configs",
    "experiments",
    "p2_joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v5.pycache_sentinel"
)
[P2StableNative]::AssertNoReparseChain($canonicalPycache, $false)
if (-not [IO.File]::Exists($canonicalPycache) -or [IO.Directory]::Exists($canonicalPycache)) {
    throw "P2 v5 canonical pycache sentinel must be a regular file before child start"
}

$startupPins = [ordered]@{
    PYCACHE_SENTINEL = [ordered]@{ scope = "workspace"; relative = "configs/experiments/p2_joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v5.pycache_sentinel"; bytes = 67L; sha256 = "932aa01b4026ed06cc675d1312fdf5f29f2247327319c3eaceec24fe7991f903" }
    VENV_PYTHON = [ordered]@{ scope = "workspace"; relative = ".venv-p1/Scripts/python.exe"; bytes = 274424L; sha256 = "0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14" }
    PYVENV_CFG = [ordered]@{ scope = "workspace"; relative = ".venv-p1/pyvenv.cfg"; bytes = 339L; sha256 = "d1fb970854073922d49959ae01539088550613e316cb67f9fac858f586361174" }
    BASE_PYTHON = [ordered]@{ scope = "base"; relative = "python.exe"; bytes = 104952L; sha256 = "4d6f5f81a4bca11191c4c7c6b43632694d0a4ce74e068619d8fdc161d469859a" }
    PYTHON3_DLL = [ordered]@{ scope = "base"; relative = "python3.dll"; bytes = 70376L; sha256 = "fb975a606e7fbf74f64260e3f60c3490b4f74a183c0926fd6ed1ac4c52ac7b1c" }
    PYTHON312_DLL = [ordered]@{ scope = "base"; relative = "python312.dll"; bytes = 6945272L; sha256 = "9a0e3435aaa680d868150f87ab3e388ad2eebc22f87e036155c7b4eda8cd2120" }
    VCRUNTIME140_DLL = [ordered]@{ scope = "base"; relative = "vcruntime140.dll"; bytes = 120400L; sha256 = "052ad6a20d375957e82aa6a3c441ea548d89be0981516ca7eb306e063d5027f4" }
    VCRUNTIME140_1_DLL = [ordered]@{ scope = "base"; relative = "vcruntime140_1.dll"; bytes = 49776L; sha256 = "6a99bc0128e0c7d6cbbf615fcc26909565e17d4ca3451b97f8987f9c6acbc6c8" }
    ENCODINGS_INIT = [ordered]@{ scope = "base"; relative = "Lib/encodings/__init__.py"; bytes = 6058L; sha256 = "8b997e9f7beef09de01c34ac34191866d3ab25e17164e08f411940b070bc3e74" }
    ENCODINGS_ALIASES = [ordered]@{ scope = "base"; relative = "Lib/encodings/aliases.py"; bytes = 16228L; sha256 = "1893cfb597bc5eafd38ef03ac85d8874620112514eb42660408811929cc0d6f8" }
    ENCODINGS_UTF8 = [ordered]@{ scope = "base"; relative = "Lib/encodings/utf_8.py"; bytes = 1047L; sha256 = "9c54c7db8ce0722ca4ddb5f45d4e170357e37991afb3fcdc091721bf6c09257e" }
    ENCODINGS_CP949 = [ordered]@{ scope = "base"; relative = "Lib/encodings/cp949.py"; bytes = 1062L; sha256 = "da13fd6f1bd7a1d3b48aed1fc75f7516d6a33814086cf971e030625590e9dda0" }
}

$heldAll = [System.Collections.Generic.List[P2HeldFile]]::new()
$startupHeld = [ordered]@{}
$envelope = $null
try {
    $stageZeroContext = $global:P2V5StageZeroContext
    if ($null -eq $stageZeroContext -or $stageZeroContext.Contract -cne "P2_V5_PATH_IMMUTABLE_INLINE_STAGE_ZERO_V1") {
        throw "P2 v5 stage one requires the authenticated inline stage-zero context"
    }
    if (-not [string]::IsNullOrEmpty($PSCommandPath)) {
        throw "P2 v5 stage one must be evaluated from authenticated memory, never by path"
    }
    $hostHeld = [P2StableNative]::OpenReadStable($hostPath)
    $heldAll.Add($hostHeld)
    Assert-Pin -Held $hostHeld -Bytes 301368L -Sha256 "db6dd81183fe57d22e03b911ec9a30a2fd7c40542e97743615355a6fb44f458f" -Label "externally injected PowerShell host"

    $launcherHeld = $stageZeroContext.StageOneHeld
    $ExternalLauncherAttestation = [string]$stageZeroContext.StageOneSha256
    if (
        $launcherHeld.Size -ne $stageZeroContext.StageOneBytes -or
        $launcherHeld.Sha256 -cne $ExternalLauncherAttestation
    ) {
        throw "P2 v5 authenticated in-memory stage-one context changed"
    }

    $bootstrapPath = [IO.Path]::Combine(
        $workspace,
        "scripts",
        "bootstrap_p2_joint_hydrographic_multitask_layer4_r3_compatibility_v5.py"
    )
    $bootstrapHeld = [P2StableNative]::OpenReadStable($bootstrapPath)
    $heldAll.Add($bootstrapHeld)
    Assert-Pin -Held $bootstrapHeld -Bytes 72079L -Sha256 "56b6f507105160550e62c98284181fab00c3a49f0cd720d6b1cea456924e8de3" -Label "v5 Python bootstrap"

    foreach ($role in @("PYCACHE_SENTINEL", "VENV_PYTHON", "PYVENV_CFG")) {
        $pin = $startupPins[$role]
        $path = [IO.Path]::GetFullPath(
            [IO.Path]::Combine($workspace, ($pin.relative -replace "/", "\"))
        )
        $held = [P2StableNative]::OpenReadStable($path)
        $heldAll.Add($held)
        Assert-Pin -Held $held -Bytes $pin.bytes -Sha256 $pin.sha256 -Label $role
        $startupHeld[$role] = $held
    }

    $pyvenvText = [Text.Encoding]::UTF8.GetString($startupHeld.PYVENV_CFG.Raw)
    $homes = @(
        $pyvenvText -split "\r?\n" |
            Where-Object { $_.StartsWith("home =") } |
            ForEach-Object { $_.Split("=", 2)[1].Trim() }
    )
    if ($homes.Count -ne 1 -or -not [IO.Path]::IsPathFullyQualified($homes[0])) {
        throw "P2 v5 pyvenv.cfg base-prefix resolution changed"
    }
    $base = [IO.Path]::GetFullPath($homes[0])
    [P2StableNative]::AssertNoReparseChain($base, $false)

    foreach ($role in $startupPins.Keys) {
        if ($startupHeld.Contains($role)) {
            continue
        }
        $pin = $startupPins[$role]
        if ($pin.scope -cne "base") {
            throw "P2 v5 startup scope changed: $role"
        }
        $path = [IO.Path]::GetFullPath(
            [IO.Path]::Combine($base, ($pin.relative -replace "/", "\"))
        )
        $held = [P2StableNative]::OpenReadStable($path)
        $heldAll.Add($held)
        Assert-Pin -Held $held -Bytes $pin.bytes -Sha256 $pin.sha256 -Label $role
        $startupHeld[$role] = $held
    }

    $venvPython = $startupHeld.VENV_PYTHON.FinalPath
    $start = [Diagnostics.ProcessStartInfo]::new()
    $start.FileName = $venvPython
    $start.WorkingDirectory = $workspace
    $start.UseShellExecute = $false
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    foreach ($argument in @(
        "-I",
        "-S",
        "-B",
        "-Xpycache_prefix=$canonicalPycache",
        $bootstrapPath,
        "--root",
        $workspace,
        "--mode",
        "check-only",
        "--external-launcher-attestation",
        $ExternalLauncherAttestation,
        "--powershell-runtime-attestation",
        $stageZeroContext.PowerShellRuntimeInventory.sha256,
        "--stage-zero-contract",
        $stageZeroContext.Contract,
        "--stage-zero-attestation",
        $stageZeroContext.StageZeroSha256
    )) {
        [void]$start.ArgumentList.Add($argument)
    }

    $child = [Diagnostics.Process]::new()
    $child.StartInfo = $start
    if (-not $child.Start()) {
        throw "P2 v5 Python child did not start"
    }
    $stdoutTask = $child.StandardOutput.ReadToEndAsync()
    $stderrTask = $child.StandardError.ReadToEndAsync()
    $child.WaitForExit()
    $stdout = $stdoutTask.GetAwaiter().GetResult()
    $stderr = $stderrTask.GetAwaiter().GetResult()
    if ($child.ExitCode -ne 0) {
        throw "P2 v5 Python child failed closed (exit $($child.ExitCode)): $stderr"
    }
    if (-not [string]::IsNullOrWhiteSpace($stderr)) {
        throw "P2 v5 Python child wrote unexpected stderr: $stderr"
    }
    $childReport = $stdout | ConvertFrom-Json -AsHashtable

    foreach ($role in $startupPins.Keys) {
        $pin = $startupPins[$role]
        $startupHeld[$role].PostChildRehash($pin.bytes, $pin.sha256)
    }
    $bootstrapHeld.PostChildRehash(72079L, "56b6f507105160550e62c98284181fab00c3a49f0cd720d6b1cea456924e8de3")
    $hostHeld.PostChildRehash(301368L, "db6dd81183fe57d22e03b911ec9a30a2fd7c40542e97743615355a6fb44f458f")
    $launcherHeld.PostRehash($launcherHeld.Size, $ExternalLauncherAttestation)

    if (-not [IO.File]::Exists($canonicalPycache) -or [IO.Directory]::Exists($canonicalPycache)) {
        throw "P2 v5 canonical pycache sentinel disappeared during child execution"
    }

    $envelope = [ordered]@{
        external_startup_trust = [ordered]@{
            model = "EXTERNAL_PREHOST_RUNTIME_HOLD_PLUS_PATH_IMMUTABLE_INLINE_STAGE_ZERO"
            host_path_source = "P2_POWERSHELL_HOST"
            host_observed_pin = [ordered]@{
                path = $hostHeld.FinalPath
                bytes = $hostHeld.Size
                sha256 = $hostHeld.Sha256
            }
            launcher_observed_pin = [ordered]@{
                path = $launcherHeld.FinalPath
                bytes = $launcherHeld.Size
                sha256 = $launcherHeld.Sha256
            }
            bootstrap_pin = [ordered]@{
                path = "scripts/bootstrap_p2_joint_hydrographic_multitask_layer4_r3_compatibility_v5.py"
                bytes = $bootstrapHeld.Size
                sha256 = $bootstrapHeld.Sha256
            }
            host_self_authentication_claimed = $false
            launcher_self_authentication_claimed = $false
            host_requires_independent_preexecution_pin = $true
            launcher_requires_independent_preexecution_pin = $true
            launcher_requires_independent_pin = $true
            stage_zero_contract = $stageZeroContext.Contract
            stage_zero_observed_pin = [ordered]@{
                path = "scripts/launch_p2_joint_hydrographic_multitask_layer4_r3_compatibility_v5_stage0.ps1"
                bytes = $stageZeroContext.StageZeroBytes
                sha256 = $stageZeroContext.StageZeroSha256
            }
            stage_zero_encoded_command = $true
            stage_zero_path_execution = $false
            stage_one_authenticated_before_evaluation = $true
            stage_one_evaluated_from_memory = $true
            powershell_runtime_inventory = $stageZeroContext.PowerShellRuntimeInventory
            startup_files = $startupPins.Count
            startup_files_prehashed = $startupHeld.Count
            startup_files_post_child_rehashed = $startupHeld.Count
            bootstrap_prehashed = 1
            bootstrap_post_child_rehashed = 1
            host_post_child_rehashed = 1
            launcher_post_child_rehashed = 1
            all_handles_held_until_child_exit = $true
            share_write_allowed = $false
            share_delete_allowed = $false
            open_reparse_point = $true
            all_regular_file_nlinks = 1
            canonical_pycache_sentinel_regular_held_before = $true
            canonical_pycache_sentinel_regular_held_after = $true
        }
        child_report = $childReport
    }
}
finally {
    foreach ($held in $heldAll) {
        $held.Dispose()
    }
}

$envelope | ConvertTo-Json -Depth 100 -Compress
