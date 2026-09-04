Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not [string]::IsNullOrEmpty($PSCommandPath)) {
    throw "P2 v5 stage zero must be supplied as path-immutable -EncodedCommand bytes"
}

Add-Type -TypeDefinition @"
using System;
using System.ComponentModel;
using System.IO;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using Microsoft.Win32.SafeHandles;

public sealed class P2V5StageZeroHeld : IDisposable
{
    public readonly string RequestedPath;
    public readonly string FinalPath;
    public readonly bool IsDirectory;
    public readonly SafeFileHandle Handle;
    public readonly FileStream Stream;
    public readonly long Size;
    public readonly string Sha256;
    public readonly uint NumberOfLinks;
    private readonly uint volumeSerial;
    private readonly uint fileIndexHigh;
    private readonly uint fileIndexLow;
    private readonly long lastWrite;

    internal P2V5StageZeroHeld(
        string requestedPath,
        string finalPath,
        bool isDirectory,
        SafeFileHandle handle,
        FileStream stream,
        long size,
        string sha256,
        P2V5StageZeroNative.BY_HANDLE_FILE_INFORMATION info)
    {
        RequestedPath = requestedPath;
        FinalPath = finalPath;
        IsDirectory = isDirectory;
        Handle = handle;
        Stream = stream;
        Size = size;
        Sha256 = sha256;
        NumberOfLinks = info.NumberOfLinks;
        volumeSerial = info.VolumeSerialNumber;
        fileIndexHigh = info.FileIndexHigh;
        fileIndexLow = info.FileIndexLow;
        lastWrite = ((long)info.LastWriteTime.dwHighDateTime << 32)
            | info.LastWriteTime.dwLowDateTime;
    }

    public byte[] ReadAllBytes()
    {
        if (IsDirectory || Stream == null)
        {
            throw new InvalidOperationException("P2 v5 directory has no byte buffer");
        }
        Stream.Position = 0;
        using (MemoryStream memory = new MemoryStream())
        {
            Stream.CopyTo(memory);
            Stream.Position = 0;
            return memory.ToArray();
        }
    }

    public void PostRehash(long expectedBytes, string expectedSha256)
    {
        P2V5StageZeroNative.BY_HANDLE_FILE_INFORMATION info =
            P2V5StageZeroNative.Information(Handle);
        long observedSize = ((long)info.FileSizeHigh << 32) | info.FileSizeLow;
        long observedWrite = ((long)info.LastWriteTime.dwHighDateTime << 32)
            | info.LastWriteTime.dwLowDateTime;
        if (
            info.VolumeSerialNumber != volumeSerial
            || info.FileIndexHigh != fileIndexHigh
            || info.FileIndexLow != fileIndexLow
            || observedWrite != lastWrite
            || (!IsDirectory && (info.NumberOfLinks != 1 || observedSize != Size)))
        {
            throw new InvalidOperationException(
                "P2 v5 stage-zero held identity changed: " + RequestedPath);
        }
        string finalPath = P2V5StageZeroNative.FinalPath(Handle);
        if (!String.Equals(
            Path.GetFullPath(finalPath),
            Path.GetFullPath(FinalPath),
            StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException(
                "P2 v5 stage-zero held final path changed: " + RequestedPath);
        }
        if (!IsDirectory)
        {
            Stream.Position = 0;
            string digest;
            using (SHA256 sha = SHA256.Create())
            {
                digest = Convert.ToHexString(sha.ComputeHash(Stream)).ToLowerInvariant();
            }
            Stream.Position = 0;
            if (Size != expectedBytes || !String.Equals(
                digest, expectedSha256, StringComparison.Ordinal))
            {
                throw new InvalidOperationException(
                    "P2 v5 stage-zero held bytes changed: " + RequestedPath);
            }
        }
    }

    public void Dispose()
    {
        if (Stream != null)
        {
            Stream.Dispose();
        }
        Handle.Dispose();
    }
}

public static class P2V5StageZeroNative
{
    public const uint GENERIC_READ = 0x80000000;
    public const uint GENERIC_WRITE = 0x40000000;
    public const uint FILE_SHARE_READ = 0x00000001;
    public const uint FILE_SHARE_WRITE = 0x00000002;
    public const uint FILE_SHARE_DELETE = 0x00000004;
    public const uint OPEN_EXISTING = 3;
    public const uint FILE_FLAG_BACKUP_SEMANTICS = 0x02000000;
    public const uint FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000;
    private const uint FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400;
    private const uint VOLUME_NAME_DOS = 0;
    private const int ERROR_SHARING_VIOLATION = 32;

    public static string[] SortOrdinal(string[] values)
    {
        Array.Sort(values, StringComparer.Ordinal);
        return values;
    }

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

    public static void AssertNoReparseChain(string path)
    {
        string full = Path.GetFullPath(path);
        string root = Path.GetPathRoot(full);
        string current = root;
        foreach (string part in full.Substring(root.Length).Split(
            new char[] { Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar },
            StringSplitOptions.RemoveEmptyEntries))
        {
            current = Path.Combine(current, part);
            if (!File.Exists(current) && !Directory.Exists(current))
            {
                throw new FileNotFoundException("P2 v5 path component is missing", current);
            }
            if ((File.GetAttributes(current) & FileAttributes.ReparsePoint) != 0)
            {
                throw new InvalidOperationException(
                    "P2 v5 reparse path component is forbidden: " + current);
            }
        }
    }

    public static void AssertExternallyHeld(string path, bool isDirectory)
    {
        uint flags = FILE_FLAG_OPEN_REPARSE_POINT;
        if (isDirectory)
        {
            flags |= FILE_FLAG_BACKUP_SEMANTICS;
        }
        using (SafeFileHandle probe = CreateFileW(
            Path.GetFullPath(path),
            GENERIC_READ,
            0,
            IntPtr.Zero,
            OPEN_EXISTING,
            flags,
            IntPtr.Zero))
        {
            if (!probe.IsInvalid)
            {
                throw new InvalidOperationException(
                    "P2 v5 runtime path was not externally share-deny held before host start: "
                    + path);
            }
            int error = Marshal.GetLastWin32Error();
            if (error != ERROR_SHARING_VIOLATION)
            {
                throw new Win32Exception(error,
                    "P2 v5 external pre-host hold was not proven: " + path);
            }
        }
    }

    public static P2V5StageZeroHeld OpenStable(string path, bool isDirectory)
    {
        string full = Path.GetFullPath(path);
        AssertNoReparseChain(full);
        uint flags = FILE_FLAG_OPEN_REPARSE_POINT;
        if (isDirectory)
        {
            flags |= FILE_FLAG_BACKUP_SEMANTICS;
        }
        SafeFileHandle handle = CreateFileW(
            full,
            GENERIC_READ,
            FILE_SHARE_READ,
            IntPtr.Zero,
            OPEN_EXISTING,
            flags,
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
                throw new InvalidOperationException("P2 v5 opened reparse target: " + full);
            }
            if (!isDirectory && info.NumberOfLinks != 1)
            {
                throw new InvalidOperationException(
                    "P2 v5 runtime file must have exactly one hard link: " + full);
            }
            string finalPath = FinalPath(handle);
            if (!String.Equals(
                Path.GetFullPath(finalPath), full, StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidOperationException(
                    "P2 v5 stable handle final path differs: " + full);
            }
            if (isDirectory)
            {
                return new P2V5StageZeroHeld(
                    full, finalPath, true, handle, null, 0, "", info);
            }
            FileStream stream = new FileStream(handle, FileAccess.Read, 1024 * 1024, false);
            string digest;
            using (SHA256 sha = SHA256.Create())
            {
                digest = Convert.ToHexString(sha.ComputeHash(stream)).ToLowerInvariant();
            }
            stream.Position = 0;
            long size = ((long)info.FileSizeHigh << 32) | info.FileSizeLow;
            return new P2V5StageZeroHeld(
                full, finalPath, false, handle, stream, size, digest, info);
        }
        catch
        {
            handle.Dispose();
            throw;
        }
    }
}
"@

$workspaceText = [Environment]::GetEnvironmentVariable("P2_V5_WORKSPACE_ROOT")
$hostText = [Environment]::GetEnvironmentVariable("P2_POWERSHELL_HOST")
$externalDigest = [Environment]::GetEnvironmentVariable(
    "P2_V5_EXTERNAL_RUNTIME_INVENTORY_SHA256"
)
$stageZeroBytesText = [Environment]::GetEnvironmentVariable("P2_V5_STAGE_ZERO_BYTES")
$stageZeroSha256 = [Environment]::GetEnvironmentVariable("P2_V5_STAGE_ZERO_SHA256")
$stageZeroBytes = 0L
if (
    [string]::IsNullOrWhiteSpace($workspaceText) -or
    [string]::IsNullOrWhiteSpace($hostText) -or
    -not [IO.Path]::IsPathFullyQualified($workspaceText) -or
    -not [IO.Path]::IsPathFullyQualified($hostText)
) {
    throw "P2 v5 stage zero requires exact absolute workspace and host environment values"
}
if (
    -not [long]::TryParse($stageZeroBytesText, [ref]$stageZeroBytes) -or
    $stageZeroBytes -le 0 -or
    $stageZeroSha256 -cnotmatch "\A[0-9a-f]{64}\z"
) {
    throw "P2 v5 stage-zero external source attestation changed"
}
$workspace = [IO.Path]::GetFullPath($workspaceText)
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
    throw "P2 v5 exact PowerShell host changed"
}
[P2V5StageZeroNative]::AssertNoReparseChain($workspace)
[P2V5StageZeroNative]::AssertNoReparseChain($hostPath)
$runtimeRoot = [IO.Path]::GetDirectoryName($hostPath)

$records = [System.Collections.Generic.List[object]]::new()
$pending = [System.Collections.Generic.Queue[IO.DirectoryInfo]]::new()
$pending.Enqueue([IO.DirectoryInfo]::new($runtimeRoot))
while ($pending.Count -gt 0) {
    $directory = $pending.Dequeue()
    foreach ($child in $directory.EnumerateFileSystemInfos()) {
        if (($child.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "P2 v5 PowerShell runtime contains a reparse entry: $($child.FullName)"
        }
        $relative = [IO.Path]::GetRelativePath($runtimeRoot, $child.FullName).Replace("\", "/")
        if ($child -is [IO.DirectoryInfo]) {
            $records.Add([pscustomobject]@{ Kind = "d"; Relative = $relative; Path = $child.FullName })
            $pending.Enqueue($child)
        }
        elseif ($child -is [IO.FileInfo]) {
            $records.Add([pscustomobject]@{ Kind = "f"; Relative = $relative; Path = $child.FullName })
        }
        else {
            throw "P2 v5 PowerShell runtime contains a special entry: $($child.FullName)"
        }
    }
}
$recordByRelative = @{}
foreach ($record in $records) {
    $recordByRelative[$record.Relative] = $record
}
$relativeNames = [P2V5StageZeroNative]::SortOrdinal(
    [string[]]@($records | ForEach-Object { $_.Relative })
)
$ordered = @($relativeNames | ForEach-Object { $recordByRelative[$_] })
if ($ordered.Count -ne 1036) {
    throw "P2 v5 PowerShell runtime entry count changed"
}

$heldAll = [System.Collections.Generic.List[P2V5StageZeroHeld]]::new()
$runtimeHeld = [System.Collections.Generic.List[P2V5StageZeroHeld]]::new()
$stageOneHeld = $null
try {
    [P2V5StageZeroNative]::AssertExternallyHeld($runtimeRoot, $true)
    $rootHeld = [P2V5StageZeroNative]::OpenStable($runtimeRoot, $true)
    $heldAll.Add($rootHeld)
    $runtimeHeld.Add($rootHeld)

    $digestText = [Text.StringBuilder]::new()
    $directoryCount = 0
    $fileCount = 0
    $fileBytes = 0L
    foreach ($record in $ordered) {
        $isDirectory = $record.Kind -ceq "d"
        [P2V5StageZeroNative]::AssertExternallyHeld($record.Path, $isDirectory)
        $held = [P2V5StageZeroNative]::OpenStable($record.Path, $isDirectory)
        $heldAll.Add($held)
        $runtimeHeld.Add($held)
        if ($isDirectory) {
            $directoryCount += 1
            [void]$digestText.Append("d`0$($record.Relative)`n")
        }
        else {
            $fileCount += 1
            $fileBytes += $held.Size
            [void]$digestText.Append(
                "f`0$($record.Relative)`0$($held.Size)`0$($held.Sha256)`n"
            )
        }
    }
    $digestBytes = [Text.UTF8Encoding]::new($false).GetBytes($digestText.ToString())
    $runtimeDigest = [Convert]::ToHexString(
        [Security.Cryptography.SHA256]::HashData($digestBytes)
    ).ToLowerInvariant()
    if (
        $directoryCount -ne 53 -or
        $fileCount -ne 983 -or
        $fileBytes -ne 296034085L -or
        $runtimeDigest -cne "9a197570fffc3399d9c8477ef0199e31ad950701de7b133df7c4669d42099be1" -or
        $externalDigest -cne $runtimeDigest
    ) {
        throw "P2 v5 complete PowerShell runtime inventory changed"
    }

    $stageOnePath = [IO.Path]::Combine(
        $workspace,
        "scripts",
        "launch_p2_joint_hydrographic_multitask_layer4_r3_compatibility_v5.ps1"
    )
    $stageOneHeld = [P2V5StageZeroNative]::OpenStable($stageOnePath, $false)
    $heldAll.Add($stageOneHeld)
    $stageOneExpectedBytes = 22133L
    $stageOneExpectedSha256 = "519a5eaf85ee1cf8735d9de0156470be5a7504de246f92142d26b591c592814f"
    if (
        $stageOneHeld.Size -ne $stageOneExpectedBytes -or
        $stageOneHeld.Sha256 -cne $stageOneExpectedSha256
    ) {
        throw "P2 v5 stage-one pin changed before evaluation"
    }
    $stageOneRaw = $stageOneHeld.ReadAllBytes()
    $strictUtf8 = [Text.UTF8Encoding]::new($false, $true)
    $stageOneText = $strictUtf8.GetString($stageOneRaw)
    if ($strictUtf8.GetByteCount($stageOneText) -ne $stageOneRaw.Length) {
        throw "P2 v5 stage-one UTF-8 round trip changed"
    }
    $global:P2V5StageZeroContext = [pscustomobject]@{
        Contract = "P2_V5_PATH_IMMUTABLE_INLINE_STAGE_ZERO_V1"
        StageZeroBytes = $stageZeroBytes
        StageZeroSha256 = $stageZeroSha256
        StageOneHeld = $stageOneHeld
        StageOneBytes = $stageOneExpectedBytes
        StageOneSha256 = $stageOneExpectedSha256
        PowerShellRuntimeInventory = [ordered]@{
            directories = $directoryCount
            files = $fileCount
            file_bytes = $fileBytes
            algorithm = "SHA256_SORTED_ORDINAL_TYPE_NUL_RELATIVE_NUL_BYTES_NUL_FILE_SHA256_LF"
            sha256 = $runtimeDigest
            externally_share_deny_held_before_host_start = $true
            external_hold_probes = 1 + $ordered.Count
            stage_zero_same_handle_holds = $runtimeHeld.Count
        }
    }
    $stageOne = [ScriptBlock]::Create($stageOneText)
    $result = & $stageOne -Root $workspace
    $stageOneHeld.PostRehash($stageOneExpectedBytes, $stageOneExpectedSha256)
    foreach ($held in $runtimeHeld) {
        if ($held.IsDirectory) {
            $held.PostRehash(0, "")
        }
        else {
            $held.PostRehash($held.Size, $held.Sha256)
        }
    }
    $result
}
finally {
    Remove-Variable -Name P2V5StageZeroContext -Scope Global -ErrorAction SilentlyContinue
    foreach ($held in $heldAll) {
        $held.Dispose()
    }
}
