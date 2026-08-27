"""Externally pinned trust root for the P3 Gen6r2 compatibility-v3 check.

Canonical invocation is the repository venv ``python.exe -I -S -B``.  The
audit hook below is deliberately installed before every import except ``sys``.
All executable Python sources are authenticated and executed from buffers;
all semantic artifacts are read once through Windows handles that deny write
and delete sharing and remain held until final verification.
"""

import sys

_PREBOOT_MODULES = frozenset(sys.modules)


class BootstrapV3Error(RuntimeError):
    """The isolated runtime, byte trust, or read-only contract failed closed."""


_AUDIT = {
    "phase": "EARLIEST",
    "attempts": {
        "write_or_create": 0,
        "remove_or_rename": 0,
        "link_or_metadata_mutation": 0,
        "process_launch": 0,
        "forbidden_import": 0,
        "bytecode": 0,
        "protected_reopen": 0,
    },
    "protected": set(),
    "normalizer": None,
    "registry_internal": False,
}


def _deny(category, message):
    _AUDIT["attempts"][category] += 1
    raise BootstrapV3Error(message)


def _audit_hook(event, args):
    if event == "open":
        path = args[0] if args else None
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        text = path if isinstance(path, str) else ""
        lowered = text.lower()
        if lowered.endswith((".pyc", ".pyo")):
            _deny("bytecode", "bytecode access is forbidden")
        if isinstance(mode, str) and any(character in mode for character in "wax+"):
            _deny("write_or_create", "write/create open is forbidden")
        if isinstance(flags, int) and flags & _AUDIT.get("write_flag_mask", 0):
            _deny("write_or_create", "write/create descriptor open is forbidden")
        normalizer = _AUDIT.get("normalizer")
        if (
            text
            and normalizer is not None
            and not _AUDIT["registry_internal"]
            and normalizer(text) in _AUDIT["protected"]
        ):
            _deny("protected_reopen", "protected semantic path reopen is forbidden")
    elif event == "_winapi.CreateFile":
        path = args[0] if args else None
        access = args[1] if len(args) > 1 else 0
        if isinstance(access, int) and access & 0x40000000:
            _deny("write_or_create", "Windows write handle is forbidden")
        normalizer = _AUDIT.get("normalizer")
        if (
            isinstance(path, str)
            and normalizer is not None
            and not _AUDIT["registry_internal"]
            and normalizer(path) in _AUDIT["protected"]
        ):
            _deny("protected_reopen", "protected Windows handle reopen is forbidden")
    elif event in {"os.mkdir"}:
        _deny("write_or_create", "directory creation is forbidden")
    elif event in {"os.remove", "os.rmdir", "os.rename", "os.replace"}:
        _deny("remove_or_rename", "remove/rename/replace is forbidden")
    elif event in {
        "os.link",
        "os.symlink",
        "os.chmod",
        "os.chown",
        "os.utime",
        "os.truncate",
    }:
        _deny("link_or_metadata_mutation", "link or metadata mutation is forbidden")
    elif (
        event in {"os.system", "os.startfile", "subprocess.Popen", "_winapi.CreateProcess"}
        or event.startswith("os.spawn")
        or event.startswith("os.exec")
    ):
        _deny("process_launch", "process launch is forbidden")
    elif event in {"socket.connect", "socket.bind"}:
        _deny("process_launch", "network endpoint use is forbidden")
    elif event == "import" and args:
        name = args[0]
        filename = args[1] if len(args) > 1 else None
        if isinstance(name, str):
            if name == (
                "p3_wave.gen6_incumbent_preserving_residual_calibrator_execution_r2"
            ) or name.startswith(
                "p3_wave.gen6_incumbent_preserving_residual_calibrator_execution_r2."
            ):
                _deny("forbidden_import", "r2 execution engine import is forbidden")
            root = name.partition(".")[0]
            if root in {"scipy", "sklearn", "torch"} or (
                _AUDIT["phase"] in {"EARLIEST", "STDLIB_AUTH", "SOURCE_AUTH"}
                and root in {"numpy", "pandas", "pyarrow"}
            ):
                _deny("forbidden_import", "forbidden or preauthentication numerical import")
        if (
            isinstance(filename, str)
            and "site-packages" in filename.lower()
            and _AUDIT["phase"] in {"EARLIEST", "STDLIB_AUTH", "SOURCE_AUTH"}
        ):
            _deny("forbidden_import", "site-packages import before RECORD authentication")


sys.addaudithook(_audit_hook)

# These modules are built-in or frozen in the pinned CPython runtime.  Their
# origins are checked immediately after import, before any disk-backed module.
import _winapi  # noqa: E402
import msvcrt  # noqa: E402
import os  # noqa: E402
import stat  # noqa: E402

IDENTITY = (
    "P3_GEN6_INCUMBENT_PRESERVING_RESIDUAL_CALIBRATOR_R2_COMPATIBILITY_V3_"
    "EARLIEST_AUDIT_TRUST_ROOT"
)
BOOTSTRAP_RELATIVE = (
    "scripts/bootstrap_verify_p3_gen6_incumbent_preserving_residual_calibrator_"
    "r2_compatibility_v3.py"
)
V3_CONFIG_RELATIVE = (
    "configs/experiments/p3_gen6_incumbent_preserving_residual_calibrator_"
    "v1r2_compatibility_verifier_v3.json"
)
V3_HELPER_MODULE = (
    "p3_wave.gen6_incumbent_preserving_residual_calibrator_"
    "r2_compatibility_verifier_v3"
)
V3_CLI_MODULE = V3_HELPER_MODULE + "_cli"
V2_HELPER_MODULE = (
    "p3_wave.gen6_incumbent_preserving_residual_calibrator_"
    "r2_compatibility_verifier_v2"
)
V2_CLI_MODULE_AUTH = V2_HELPER_MODULE + "_cli_authenticated"
V1_HELPER_MODULE = (
    "p3_wave.gen6_incumbent_preserving_residual_calibrator_"
    "r2_compatibility_verifier_v1"
)
R2_MODULE = "p3_wave.gen6_incumbent_preserving_residual_calibrator_contract_r2"
ENGINE_MODULE = "p3_wave.gen6_incumbent_preserving_residual_calibrator_execution_r2"
REPARSE_ATTRIBUTE = 0x400
FILE_SHARE_READ = 0x1
FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
GENERIC_READ = 0x80000000
OPEN_EXISTING = 3
O_BINARY = getattr(os, "O_BINARY", 0)
_AUDIT["write_flag_mask"] = (
    os.O_WRONLY
    | os.O_RDWR
    | os.O_CREAT
    | os.O_TRUNC
    | os.O_APPEND
    | getattr(os, "O_EXCL", 0)
)

IMPLEMENTATION_ROLES = {
    "CONFIG": V3_CONFIG_RELATIVE,
    "HELPER": (
        "src/p3_wave/gen6_incumbent_preserving_residual_calibrator_"
        "r2_compatibility_verifier_v3.py"
    ),
    "CLI": (
        "scripts/verify_p3_gen6_incumbent_preserving_residual_calibrator_"
        "r2_compatibility_v3.py"
    ),
    "TESTS": (
        "tests/test_p3_gen6_incumbent_preserving_residual_calibrator_"
        "r2_compatibility_verifier_v3.py"
    ),
}

# CONFIG/HELPER/CLI/TESTS are frozen in the final pinning pass.  This map is
# the sole noncyclic authority; CONFIG cannot authenticate itself.
PINNED_SOURCES = {
    "V3_CONFIG": {"path": V3_CONFIG_RELATIVE, "bytes": 11701, "sha256": "5f69a379a03dce661fd55464628aa5adedda9a9bdfa17bab1574501197ed4084"},
    "V3_HELPER": {"path": IMPLEMENTATION_ROLES["HELPER"], "bytes": 13069, "sha256": "4943d4db0310a68949d87f112ccca185c9e48920241b2fac93111dc851a1a39e"},
    "V3_CLI": {"path": IMPLEMENTATION_ROLES["CLI"], "bytes": 1007, "sha256": "2863c7b4e5fe5b16bb2df3b753ba8c935f5a8767fafb135782171dcc4d7349bb"},
    "V3_TESTS": {"path": IMPLEMENTATION_ROLES["TESTS"], "bytes": 13804, "sha256": "b337173af7b97ea4088bc41717cde2c8192378c72469aa0d4f2c4542086a032d"},
    "V2_BOOTSTRAP": {"path": "scripts/bootstrap_verify_p3_gen6_incumbent_preserving_residual_calibrator_r2_compatibility_v2.py", "bytes": 22071, "sha256": "d2dec0e2d05d53da3d0489f8af9762d7e57524326143b1e9afa91d2a47537733"},
    "V2_CONFIG": {"path": "configs/experiments/p3_gen6_incumbent_preserving_residual_calibrator_v1r2_compatibility_verifier_v2.json", "bytes": 11074, "sha256": "a80aedd91cc1ed73d638fcaa2827f73344220d49b3f2c1073458e7040c044cc1"},
    "V2_HELPER": {"path": "src/p3_wave/gen6_incumbent_preserving_residual_calibrator_r2_compatibility_verifier_v2.py", "bytes": 15462, "sha256": "054c271e2ba0d8aac8fc8f4436884b491ab7c03a397fd10aae0e0478ddcd681b"},
    "V2_CLI": {"path": "scripts/verify_p3_gen6_incumbent_preserving_residual_calibrator_r2_compatibility_v2.py", "bytes": 1417, "sha256": "39faf8b6f6d6a1acb043ed038cce396a85a4265be04678e8a9cdbc134980df13"},
    "V2_TESTS": {"path": "tests/test_p3_gen6_incumbent_preserving_residual_calibrator_r2_compatibility_verifier_v2.py", "bytes": 12901, "sha256": "a64f7e865422d2ace395e17a3717cca1ebec9fa35a57c06c477f0983f23abb67"},
    "R2_CONTRACT": {"path": "src/p3_wave/gen6_incumbent_preserving_residual_calibrator_contract_r2.py", "bytes": 73236, "sha256": "24245a5ed9e47c335607560bb02185259984ffae70b18edb7e8afd57a4aafe51"},
    "V1_CONFIG": {"path": "configs/experiments/p3_gen6_incumbent_preserving_residual_calibrator_v1r2_compatibility_verifier_v1.json", "bytes": 12799, "sha256": "6b92a6eb67adfb042958cb518633ead4e2c70ffb1e7de35eceafccd6c6e42d2a"},
    "V1_HELPER": {"path": "src/p3_wave/gen6_incumbent_preserving_residual_calibrator_r2_compatibility_verifier_v1.py", "bytes": 47746, "sha256": "9749feb754815915a50bab2dcf6a6ed687159047874fcd9ad0cda376e9ea0375"},
    "V1_CLI": {"path": "scripts/verify_p3_gen6_incumbent_preserving_residual_calibrator_r2_compatibility_v1.py", "bytes": 2546, "sha256": "25a24339319b43da3d9f97b422ac07dd574bb27b10532d7cf983a21726b86912"},
    "V1_TESTS": {"path": "tests/test_p3_gen6_incumbent_preserving_residual_calibrator_r2_compatibility_verifier_v1.py", "bytes": 12727, "sha256": "20b343905ad4670bfa9b970880577216560edb8bdcf048f5f30e97b1f7bbc255"},
    "V1_OWNER_NO_GO": {"path": "artifacts/p3_gen6_incumbent_preserving_residual_calibrator_20260823_v1r2_compatibility_verifier_v1_no_go/OWNER_STATIC_QA_NO_GO_20260823.json", "bytes": 5814, "sha256": "b721a6de42429754fb1b98062f54832848af92dcc58fbfb204bc5299c344b620"},
    "V1_TOMBSTONE": {"path": "artifacts/p3_gen6_incumbent_preserving_residual_calibrator_20260823_v1r2_compatibility_verifier_v1_no_go/EXECUTION_TOMBSTONE.json", "bytes": 3006, "sha256": "380830f54ff7d0c2d78e4ad11592bf8180d7fd3d2a59118d93e5944980661783"},
    "V2_OWNER_NO_GO": {"path": "artifacts/p3_gen6_incumbent_preserving_residual_calibrator_20260823_v1r2_compatibility_verifier_v2_no_go/OWNER_STATIC_QA_NO_GO_20260823.json", "bytes": 4794, "sha256": "f97a45f567c328b134618e2a902689470d86ab7a7a7b194e5f2d01c8b8b3ea43"},
    "V2_TOMBSTONE": {"path": "artifacts/p3_gen6_incumbent_preserving_residual_calibrator_20260823_v1r2_compatibility_verifier_v2_no_go/EXECUTION_TOMBSTONE.json", "bytes": 3419, "sha256": "91dae10e7e4c61e875da24999396afda90837aff4492bcd64671d11297e4ad13"},
}

STDLIB_EXPECTED = {
    "directories": 197,
    "files": 2443,
    "file_bytes": 66487423,
    "sha256": "2abb9059cf7d3cb2a9c74d296b528ddede30d51be764a0b357aeabe091b88afb",
}

DEPENDENCY_EXPECTED = {
    "numpy": ("numpy-2.3.5.dist-info", "2.3.5", 110571, "28e9a4fcb2fa550a51a5c6f6639c2a5d3aed11407aded2cb8981747eb9640ca9", 1332, 54719592, "be581587c7bfd68dc61d7dfbbf71c3c5e75d365c7741342fa52c4094bfd89b23", 21, 30400080, "1863ef07e2bfd7937678ba726deba524672587a4abb071eea5886a1dc9d972ed"),
    "pandas": ("pandas-3.0.1.dist-info", "3.0.1", 249268, "98e530a3a2f22b4865b342652b94846b0d75b4aa7590f7a53ba6aea322e0d0e2", 2989, 64018796, "a4730c50dcbebcd6582ed688b1cbb968bf91517a7c7b6af328b00b0efa59d6ab", 46, 13719632, "8168570b35c01f683ed2f53d16794e963511d47b62fedeaf46a12f01d81f3c9a"),
    "pyarrow": ("pyarrow-25.0.1.dist-info", "25.0.1", 78570, "1eddf4fb72b1b071868dc02d6fc8242125d98c6557ae6af8f783b1c84ef6a797", 867, 90535543, "4c5735c0678d1e127f1a67529fad6c43fd184a02639062efdf72b0dd2bf188f2", 33, 68921192, "25aa14a5a9c0299785d394083d0c7710f7329f17b256689ee21379b727b9cf4c"),
    "python_dateutil": ("python_dateutil-2.9.0.post0.dist-info", "2.9.0.post0", 3125, "0c26b4b1542dbd1ebd8d2babdd501aed583d6ada9595517f936f00fe4ff9d254", 44, 742900, "727a2cf7b75cf7e3d2f29b1f02f3d2185701962b5c64985d42aaed68422cf26a", 0, 0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    "six": ("six-1.17.0.dist-info", "1.17.0", 561, "d834e846ba51c0e7371968d0b5a0cdebdaa2f9ea2f0447a40b594fa96ca5d89f", 8, 79452, "5267dd37edca79c94ab79a02c53441496e81371d631018938cbcfc6dc58303a5", 0, 0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    "tzdata": ("tzdata-2026.3.dist-info", "2026.3", 56831, "4e88bcacd1b80a7aff99638129717f504c6225c887a28f4ba0e193783f03c30e", 656, 588722, "17540955217f069acd5978d3e6b92f7853f5090235dfc264c53a48f3b3b39807", 0, 0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
}


_SHA256_K = (
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
)

_FAST_HASH = None


def _ror(value, count):
    return ((value >> count) | (value << (32 - count))) & 0xFFFFFFFF


def _sha256_bytes(data):
    message = bytearray(data)
    bit_length = len(message) * 8
    message.append(0x80)
    while len(message) % 64 != 56:
        message.append(0)
    message.extend(bit_length.to_bytes(8, "big"))
    state = [
        0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
        0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
    ]
    for offset in range(0, len(message), 64):
        chunk = message[offset : offset + 64]
        words = [int.from_bytes(chunk[index : index + 4], "big") for index in range(0, 64, 4)]
        for index in range(16, 64):
            s0 = _ror(words[index - 15], 7) ^ _ror(words[index - 15], 18) ^ (words[index - 15] >> 3)
            s1 = _ror(words[index - 2], 17) ^ _ror(words[index - 2], 19) ^ (words[index - 2] >> 10)
            words.append((words[index - 16] + s0 + words[index - 7] + s1) & 0xFFFFFFFF)
        a, b, c, d, e, f, g, h = state
        for index in range(64):
            big1 = _ror(e, 6) ^ _ror(e, 11) ^ _ror(e, 25)
            choice = (e & f) ^ ((~e) & g)
            temp1 = (h + big1 + choice + _SHA256_K[index] + words[index]) & 0xFFFFFFFF
            big0 = _ror(a, 2) ^ _ror(a, 13) ^ _ror(a, 22)
            majority = (a & b) ^ (a & c) ^ (b & c)
            temp2 = (big0 + majority) & 0xFFFFFFFF
            h, g, f, e, d, c, b, a = g, f, e, (d + temp1) & 0xFFFFFFFF, c, b, a, (temp1 + temp2) & 0xFFFFFFFF
        state = [(left + right) & 0xFFFFFFFF for left, right in zip(state, (a, b, c, d, e, f, g, h), strict=True)]
    return b"".join(value.to_bytes(4, "big") for value in state)


def _sha256(data):
    if _FAST_HASH is not None:
        return _FAST_HASH(data).hexdigest()
    return _sha256_bytes(data).hex()


if _sha256(b"") != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" or _sha256(b"abc") != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad":
    raise BootstrapV3Error("pure-Python SHA-256 self-test failed")


def _origin(module_name, expected):
    module = sys.modules.get(module_name)
    spec = getattr(module, "__spec__", None)
    if module is None or spec is None or spec.origin != expected:
        raise BootstrapV3Error(f"bootstrap primitive origin changed: {module_name}")


for _name, _expected_origin in (
    ("sys", "built-in"),
    ("_winapi", "built-in"),
    ("msvcrt", "built-in"),
    ("os", "frozen"),
    ("stat", "frozen"),
):
    _origin(_name, _expected_origin)

sys.dont_write_bytecode = True


def _norm(path):
    return os.path.normcase(os.path.abspath(os.fspath(path)))


_AUDIT["normalizer"] = _norm


def _identity(info):
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_mode),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
        int(getattr(info, "st_file_attributes", 0)),
        int(getattr(info, "st_reparse_tag", 0)),
    )


def _core_identity(info):
    value = _identity(info)
    # Windows CRT fstat reports creation-time differently from path lstat.
    # Windows CRT fstat normalizes executable permission bits, too.  Device,
    # file index, length, and mtime are stable across both APIs; the full path
    # identity (including mode/ctime/attributes/tag) is still pinned via lstat.
    return (value[0], value[1], value[3], value[4])


def _is_reparse(path):
    info = os.lstat(path)
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & REPARSE_ATTRIBUTE
    )


def _plain_absolute(path, kind=None):
    absolute = os.path.abspath(os.fspath(path))
    drive, tail = os.path.splitdrive(absolute)
    current = drive + os.sep
    for part in [item for item in tail.split(os.sep) if item]:
        current = os.path.join(current, part)
        if not os.path.lexists(current):
            raise BootstrapV3Error(f"required path ancestor is missing: {absolute}")
        if _is_reparse(current):
            raise BootstrapV3Error(f"link/reparse path is forbidden: {absolute}")
    if os.path.realpath(absolute) != absolute:
        raise BootstrapV3Error(f"canonical path resolves through an alias: {absolute}")
    info = os.lstat(absolute)
    if kind == "file" and not stat.S_ISREG(info.st_mode):
        raise BootstrapV3Error(f"regular file required: {absolute}")
    if kind == "directory" and not stat.S_ISDIR(info.st_mode):
        raise BootstrapV3Error(f"plain directory required: {absolute}")
    return absolute


def _contained(root, relative, must_exist=True, kind=None):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise BootstrapV3Error("relative path is not canonical POSIX text")
    parts = relative.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise BootstrapV3Error("relative path is not strictly contained")
    candidate = os.path.abspath(os.path.join(root, *parts))
    prefix = os.path.normcase(root.rstrip(os.sep) + os.sep)
    if not os.path.normcase(candidate).startswith(prefix):
        raise BootstrapV3Error("path escapes its canonical root")
    if must_exist:
        return _plain_absolute(candidate, kind)
    probe = candidate
    while not os.path.lexists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            raise BootstrapV3Error("missing path escaped containment")
        probe = parent
    _plain_absolute(probe, "directory" if os.path.isdir(probe) else None)
    return candidate


def _closed_read(path):
    path = _plain_absolute(path, "file")
    before = os.lstat(path)
    descriptor = os.open(path, os.O_RDONLY | O_BINARY)
    try:
        opened = os.fstat(descriptor)
        if _core_identity(opened) != _core_identity(before):
            raise BootstrapV3Error(
                f"file identity changed while opening: {path}: "
                f"{_core_identity(before)} != {_core_identity(opened)}"
            )
        chunks = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after_open = os.fstat(descriptor)
        if _core_identity(after_open) != _core_identity(before):
            raise BootstrapV3Error("file identity changed during read")
    finally:
        os.close(descriptor)
    after = os.lstat(path)
    if _identity(after) != _identity(before) or _is_reparse(path):
        raise BootstrapV3Error("file identity changed after read")
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        raise BootstrapV3Error("file length changed during read")
    return raw, _identity(before)


def _aggregate_file_entries(entries):
    payload = bytearray()
    for relative, size, digest in sorted(entries):
        payload.extend(relative.encode("utf-8"))
        payload.extend(b"\0")
        payload.extend(str(size).encode("ascii"))
        payload.extend(b"\0")
        payload.extend(digest.encode("ascii"))
        payload.extend(b"\n")
    return _sha256(payload)


def _stdlib_inventory(base):
    entries = []
    directories = []
    source_buffers = {}
    native_paths = []
    for root_name in ("DLLs", "Lib"):
        root = _plain_absolute(os.path.join(base, root_name), "directory")
        for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
            dirnames[:] = sorted(
                name
                for name in dirnames
                if name not in {"__pycache__", "site-packages"}
            )
            for name in list(dirnames):
                child = os.path.join(current, name)
                if _is_reparse(child):
                    raise BootstrapV3Error("stdlib directory link/reparse is forbidden")
                directories.append(os.path.relpath(child, base).replace("\\", "/"))
            for name in sorted(filenames):
                if name.lower().endswith((".pyc", ".pyo")):
                    continue
                path = os.path.join(current, name)
                raw, unused_identity = _closed_read(path)
                del unused_identity
                relative = os.path.relpath(path, base).replace("\\", "/")
                entries.append((relative, len(raw), _sha256(raw)))
                if name.lower().endswith(".py"):
                    source_buffers[_norm(path)] = raw
                if name.lower().endswith((".pyd", ".dll")):
                    native_paths.append(path)
    for name in ("python3.dll", "python312.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
        path = _plain_absolute(os.path.join(base, name), "file")
        raw, unused_identity = _closed_read(path)
        del unused_identity
        entries.append((name, len(raw), _sha256(raw)))
        native_paths.append(path)
    payload = bytearray()
    file_map = {relative: (size, digest) for relative, size, digest in entries}
    for kind, relative in sorted(
        [("directory", path) for path in directories]
        + [("file", path) for path in file_map],
        key=lambda item: item[1],
    ):
        payload.extend(kind.encode("ascii") + b"\0" + relative.encode("utf-8") + b"\0")
        if kind == "file":
            size, digest = file_map[relative]
            payload.extend(str(size).encode("ascii") + b"\0" + digest.encode("ascii"))
        else:
            payload.extend(b"\0")
        payload.extend(b"\n")
    observed = {
        "directories": len(directories),
        "files": len(entries),
        "file_bytes": sum(item[1] for item in entries),
        "sha256": _sha256(payload),
    }
    if observed != STDLIB_EXPECTED:
        raise BootstrapV3Error(f"canonical stdlib inventory changed: {observed}")
    return observed, source_buffers, native_paths


class _MemorySourceLoader:
    def __init__(self, fullname, path, raw, is_package):
        self.fullname = fullname
        self.path = path
        self.raw = raw
        self.is_package_value = is_package

    def create_module(self, spec):
        del spec
        return None

    def exec_module(self, module):
        module.__file__ = self.path
        module.__loader__ = self
        if self.is_package_value:
            module.__path__ = [os.path.dirname(self.path)]
            module.__package__ = self.fullname
        else:
            module.__package__ = self.fullname.rpartition(".")[0]
        code = compile(self.raw, self.path, "exec", dont_inherit=True, optimize=0)
        exec(code, module.__dict__)

    def get_filename(self, fullname):
        if fullname != self.fullname:
            raise ImportError(fullname)
        return self.path

    def get_data(self, path):
        key = _norm(path)
        source = _SOURCE_PATH_BUFFERS.get(key)
        if source is None:
            raise OSError("unregistered authenticated source buffer")
        return source

    def is_package(self, fullname):
        if fullname != self.fullname:
            raise ImportError(fullname)
        return self.is_package_value


class _MemorySourceFinder:
    def find_spec(self, fullname, path=None, target=None):
        del path, target
        entry = _SOURCE_MODULES.get(fullname)
        if entry is not None:
            source_path, raw, is_package = entry
            loader = _MemorySourceLoader(fullname, source_path, raw, is_package)
            spec = sys.modules["_frozen_importlib"].ModuleSpec(
                fullname,
                loader,
                origin=source_path,
                is_package=is_package,
            )
            spec.has_location = True
            spec.cached = None
            if is_package:
                spec.submodule_search_locations = [os.path.dirname(source_path)]
            return spec
        native_path = _NATIVE_MODULES.get(fullname)
        if native_path is None:
            return None
        loader = sys.modules["_frozen_importlib_external"].ExtensionFileLoader(
            fullname, native_path
        )
        spec = sys.modules["_frozen_importlib"].ModuleSpec(
            fullname, loader, origin=native_path, is_package=False
        )
        spec.has_location = True
        return spec


def _source_module_map(root, path_buffers):
    result = {}
    prefix = _norm(root).rstrip(os.sep) + os.sep
    for normalized, raw in path_buffers.items():
        if not normalized.startswith(prefix) or not normalized.lower().endswith(".py"):
            continue
        relative = normalized[len(prefix) :].replace("\\", "/")
        parts = relative.split("/")
        filename = parts[-1]
        if filename == "__init__.py":
            module_parts = parts[:-1]
            is_package = True
        else:
            module_parts = parts[:-1] + [filename[:-3]]
            is_package = False
        if not module_parts or not all(part.isidentifier() for part in module_parts):
            continue
        name = ".".join(module_parts)
        path = os.path.join(root, *relative.split("/"))
        prior = result.get(name)
        entry = (path, raw, is_package)
        if prior is not None and prior != entry:
            raise BootstrapV3Error(f"duplicate authenticated source module: {name}")
        result[name] = entry
    return result


def _native_module_map(root, paths):
    result = {}
    prefix = _norm(root).rstrip(os.sep) + os.sep
    for path in paths:
        normalized = _norm(path)
        if not normalized.startswith(prefix) or not normalized.lower().endswith(".pyd"):
            continue
        relative = normalized[len(prefix) :].replace("\\", "/")
        parts = relative.split("/")
        module_parts = parts[:-1] + [parts[-1].split(".", 1)[0]]
        if not all(part.isidentifier() for part in module_parts):
            continue
        name = ".".join(module_parts)
        prior = result.get(name)
        if prior is not None and _norm(prior) != normalized:
            raise BootstrapV3Error(f"duplicate authenticated native module: {name}")
        result[name] = path
    return result


_INITIAL_SYS_PATH = tuple(sys.path)
_INITIAL_META_PATH = tuple(sys.meta_path)
_SOURCE_PATH_BUFFERS = {}
_SOURCE_MODULES = {}
_NATIVE_MODULES = {}


class _HeldRegistry:
    def __init__(self, workspace, data_root):
        self.workspace = workspace
        self.data_root = data_root
        self.entries = {}
        self.directory_handles = []
        self.native_handles = []
        self.read_count = 0
        self.final_verified = False

    def _hold_file(self, path):
        canonical = _plain_absolute(path, "file")
        before = os.lstat(canonical)
        _AUDIT["registry_internal"] = True
        try:
            handle = _winapi.CreateFile(
                canonical,
                GENERIC_READ,
                FILE_SHARE_READ,
                0,
                OPEN_EXISTING,
                FILE_FLAG_OPEN_REPARSE_POINT,
                0,
            )
            descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | O_BINARY)
        finally:
            _AUDIT["registry_internal"] = False
        try:
            opened = os.fstat(descriptor)
            if _core_identity(opened) != _core_identity(before):
                raise BootstrapV3Error("held file identity changed while opening")
            chunks = []
            while True:
                chunk = os.read(descriptor, 1 << 20)
                if not chunk:
                    break
                chunks.append(chunk)
            after_open = os.fstat(descriptor)
            after_path = os.lstat(canonical)
            if (
                _core_identity(after_open) != _core_identity(before)
                or _identity(after_path) != _identity(before)
                or _is_reparse(canonical)
            ):
                raise BootstrapV3Error("held file identity changed during single read")
            raw = b"".join(chunks)
            if len(raw) != before.st_size:
                raise BootstrapV3Error("held file length differs")
            return descriptor, raw, _identity(before)
        except BaseException:
            os.close(descriptor)
            raise

    def lock_absolute(self, path, expected=None, label="semantic file", protect=True):
        canonical = _plain_absolute(path, "file")
        key = _norm(canonical)
        prior = self.entries.get(key)
        if prior is not None:
            if expected is not None and (
                prior["pin"]["bytes"] != expected["bytes"]
                or prior["pin"]["sha256"] != expected["sha256"]
            ):
                raise BootstrapV3Error(f"conflicting repeated pin: {label}")
            prior["protected"] = prior["protected"] or protect
            prior["labels"].append(label)
            return prior["raw"]
        descriptor, raw, identity = self._hold_file(canonical)
        observed = {"bytes": len(raw), "sha256": _sha256(raw)}
        if expected is not None and observed != {
            "bytes": expected["bytes"],
            "sha256": expected["sha256"],
        }:
            os.close(descriptor)
            raise BootstrapV3Error(f"pinned bytes changed: {label}")
        pin = observed if expected is None else dict(expected)
        self.entries[key] = {
            "path": canonical,
            "fd": descriptor,
            "raw": raw,
            "identity": identity,
            "pin": pin,
            "labels": [label],
            "protected": protect,
        }
        self.read_count += 1
        return raw

    def lock_pin(self, pin, label="semantic pin", root_kind="workspace", protect=True):
        if set(pin) != {"path", "bytes", "sha256"}:
            raise BootstrapV3Error(f"pin schema changed: {label}")
        relative = pin["path"]
        size = pin["bytes"]
        digest = pin["sha256"]
        if (
            not isinstance(relative, str)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise BootstrapV3Error(f"pin value changed: {label}")
        root = self.data_root if root_kind == "data" else self.workspace
        path = _contained(root, relative, True, "file")
        expected = {"path": relative, "bytes": size, "sha256": digest}
        return self.lock_absolute(path, expected, label, protect)

    def bytes_for_path(self, path):
        entry = self.entries.get(_norm(path))
        if entry is None:
            raise BootstrapV3Error(f"semantic path is not in the held registry: {path}")
        return entry["raw"]

    def has_path(self, path):
        return _norm(path) in self.entries

    def protect_all(self):
        _AUDIT["protected"] = {
            key for key, entry in self.entries.items() if entry["protected"]
        }

    def hold_directory(self, path):
        canonical = _plain_absolute(path, "directory")
        _AUDIT["registry_internal"] = True
        try:
            handle = _winapi.CreateFile(
                canonical,
                GENERIC_READ,
                FILE_SHARE_READ,
                0,
                OPEN_EXISTING,
                FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS,
                0,
            )
            descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | O_BINARY)
        finally:
            _AUDIT["registry_internal"] = False
        identity = _identity(os.lstat(canonical))
        if _core_identity(os.fstat(descriptor)) != _core_identity(os.lstat(canonical)):
            os.close(descriptor)
            raise BootstrapV3Error("held directory handle identity differs")
        self.directory_handles.append((canonical, descriptor, identity))

    def hold_native(self, path):
        canonical = _plain_absolute(path, "file")
        descriptor, raw, identity = self._hold_file(canonical)
        self.native_handles.append((canonical, descriptor, identity, _sha256(raw)))

    def final_verify(self):
        rehashes = 0
        for entry in self.entries.values():
            current_fd = os.fstat(entry["fd"])
            current_path = os.lstat(entry["path"])
            if (
                _core_identity(current_fd) != _core_identity(current_path)
                or _identity(current_path) != entry["identity"]
                or _sha256(entry["raw"]) != entry["pin"]["sha256"]
                or len(entry["raw"]) != entry["pin"]["bytes"]
                or _is_reparse(entry["path"])
            ):
                raise BootstrapV3Error("held semantic file changed before final verification")
            os.lseek(entry["fd"], 0, os.SEEK_SET)
            digest = _FAST_HASH()
            size = 0
            while True:
                block = os.read(entry["fd"], 1 << 20)
                if not block:
                    break
                size += len(block)
                digest.update(block)
            if size != entry["pin"]["bytes"] or digest.hexdigest() != entry["pin"]["sha256"]:
                raise BootstrapV3Error("same-handle final semantic rehash changed")
            rehashes += 1
        for path, descriptor, identity, digest in self.native_handles:
            if (
                _identity(os.lstat(path)) != identity
                or _core_identity(os.fstat(descriptor)) != _core_identity(os.lstat(path))
                or not digest
            ):
                raise BootstrapV3Error("held native file changed")
        for path, descriptor, identity in self.directory_handles:
            if (
                _identity(os.lstat(path)) != identity
                or _core_identity(os.fstat(descriptor)) != _core_identity(os.lstat(path))
                or _is_reparse(path)
            ):
                raise BootstrapV3Error("held directory identity changed")
        self.final_verified = True
        self.final_rehashes = rehashes

    def close(self):
        for entry in self.entries.values():
            os.close(entry["fd"])
        for unused_path, descriptor, unused_identity, unused_digest in self.native_handles:
            del unused_path, unused_identity, unused_digest
            os.close(descriptor)
        for unused_path, descriptor, unused_identity in self.directory_handles:
            del unused_path, unused_identity
            os.close(descriptor)
        self.entries.clear()
        self.native_handles.clear()
        self.directory_handles.clear()

    def report(self):
        return {
            "files": len(self.entries),
            "bytes": sum(len(entry["raw"]) for entry in self.entries.values()),
            "single_read_count": self.read_count,
            "windows_share_read_only_handles_held": True,
            "protected_reopens": _AUDIT["attempts"]["protected_reopen"],
            "final_identity_and_hash_verified": self.final_verified,
            "same_handle_final_rehashes": getattr(self, "final_rehashes", 0),
        }


def _parse_early_arguments():
    values = list(sys.argv[1:])
    result = {"root": None, "config": None, "mode": "check-only"}
    index = 0
    seen = set()
    while index < len(values):
        option = values[index]
        if option not in {"--root", "--config", "--mode"} or option in seen:
            raise BootstrapV3Error("canonical CLI arguments changed")
        seen.add(option)
        if index + 1 >= len(values) or values[index + 1].startswith("--"):
            raise BootstrapV3Error(f"missing value for {option}")
        value = values[index + 1]
        result[option[2:]] = value
        index += 2
    if result["root"] is None:
        raise BootstrapV3Error("exactly one --root is required")
    if result["mode"] != "check-only":
        raise BootstrapV3Error("only check-only mode is allowed")
    return result


def _assert_initial_runtime(workspace):
    flags = sys.flags
    if (
        flags.isolated != 1
        or flags.no_site != 1
        or flags.safe_path is not True
        or flags.ignore_environment != 1
        or flags.no_user_site != 1
        or flags.dont_write_bytecode != 1
        or sys.dont_write_bytecode is not True
        or sys.version_info[:3] != (3, 12, 10)
    ):
        raise BootstrapV3Error("canonical python.exe -I -S -B runtime is required")
    expected_path = (
        sys.base_prefix + "\\python312.zip",
        sys.base_prefix + "\\DLLs",
        sys.base_prefix + "\\Lib",
        sys.base_prefix,
    )
    if _INITIAL_SYS_PATH != expected_path or tuple(sys.path) != expected_path:
        raise BootstrapV3Error("exact initial isolated sys.path changed")
    expected_meta = (
        ("_frozen_importlib", "BuiltinImporter"),
        ("_frozen_importlib", "FrozenImporter"),
        ("_frozen_importlib_external", "PathFinder"),
    )
    observed_meta = tuple(
        (getattr(item, "__module__", ""), getattr(item, "__qualname__", ""))
        for item in _INITIAL_META_PATH
    )
    if observed_meta != expected_meta or tuple(sys.meta_path) != _INITIAL_META_PATH:
        raise BootstrapV3Error("exact initial import finder chain changed")
    expected_modules = {
        "__main__", "_abc", "_codecs", "_codecs_kr", "_frozen_importlib",
        "_frozen_importlib_external", "_imp", "_io", "_multibytecodec", "_signal",
        "_thread", "_warnings", "_weakref", "abc", "builtins", "codecs",
        "encodings", "encodings.aliases", "encodings.cp949", "encodings.utf_8",
        "io", "marshal", "nt", "sys", "time", "winreg", "zipimport",
    }
    if set(_PREBOOT_MODULES) != expected_modules:
        raise BootstrapV3Error("pre-bootstrap module set changed")
    expected_executable = _contained(
        workspace, ".venv-p1/Scripts/python.exe", True, "file"
    )
    if _norm(sys.executable) != _norm(expected_executable):
        raise BootstrapV3Error("canonical venv python executable changed")
    original = list(sys.orig_argv)
    if len(original) < 5 or original[1:4] != ["-I", "-S", "-B"]:
        raise BootstrapV3Error("canonical interpreter option order changed")
    bootstrap = _contained(workspace, BOOTSTRAP_RELATIVE, True, "file")
    if _norm(original[4]) != _norm(bootstrap) or _norm(sys.argv[0]) != _norm(bootstrap):
        raise BootstrapV3Error("canonical bootstrap script entry changed")
    return {
        "initial_sys_path": list(expected_path),
        "initial_meta_path": [".".join(item) for item in expected_meta],
        "prebootstrap_modules": sorted(expected_modules),
        "orig_argv_flags": original[1:4],
    }


def _strict_json(raw, label):
    import json

    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise BootstrapV3Error(f"duplicate JSON key in {label}: {key}")
            value[key] = item
        return value

    def reject(constant):
        raise BootstrapV3Error(f"non-finite JSON value in {label}: {constant}")

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=unique,
            parse_constant=reject,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapV3Error(f"invalid strict JSON buffer: {label}") from exc
    if not isinstance(value, dict):
        raise BootstrapV3Error(f"JSON root is not an object: {label}")
    return value


def _pin(relative, size, digest):
    return {"path": relative, "bytes": size, "sha256": digest}


def _verify_source_pin_schema():
    for role, pin in PINNED_SOURCES.items():
        if set(pin) != {"path", "bytes", "sha256"}:
            raise BootstrapV3Error(f"source pin schema changed: {role}")
        if (
            not isinstance(pin["bytes"], int)
            or isinstance(pin["bytes"], bool)
            or pin["bytes"] <= 0
            or not isinstance(pin["sha256"], str)
            or len(pin["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in pin["sha256"])
        ):
            raise BootstrapV3Error(f"source pin is not frozen: {role}")


def _record_rows(raw, label):
    import csv
    import io

    try:
        rows = list(csv.reader(io.StringIO(raw.decode("utf-8", errors="strict"))))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise BootstrapV3Error(f"invalid dependency RECORD: {label}") from exc
    if not rows or any(len(row) != 3 for row in rows):
        raise BootstrapV3Error(f"dependency RECORD schema changed: {label}")
    return rows


def _record_digest(text):
    import base64

    if not text.startswith("sha256="):
        raise BootstrapV3Error("dependency RECORD hash algorithm changed")
    encoded = text.partition("=")[2]
    try:
        return base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).hex()
    except (ValueError, TypeError) as exc:
        raise BootstrapV3Error("dependency RECORD digest is malformed") from exc


def _authenticate_dependencies(registry, venv_root, site_root):
    reports = {}
    dependency_sources = {}
    registered = set()
    for distribution, expected in DEPENDENCY_EXPECTED.items():
        (
            dist_info,
            version,
            record_bytes,
            record_sha,
            expected_files,
            expected_bytes,
            expected_inventory,
            expected_native_files,
            expected_native_bytes,
            expected_native_inventory,
        ) = expected
        record_path = _contained(
            site_root, f"{dist_info}/RECORD", True, "file"
        )
        record_raw = registry.lock_absolute(
            record_path,
            _pin(f"{dist_info}/RECORD", record_bytes, record_sha),
            f"{distribution} RECORD",
            protect=False,
        )
        file_entries = []
        native_entries = []
        seen_paths = set()
        for row_path, row_hash, row_size in _record_rows(record_raw, distribution):
            candidate = os.path.abspath(
                os.path.join(site_root, *row_path.replace("\\", "/").split("/"))
            )
            prefix = os.path.normcase(venv_root.rstrip(os.sep) + os.sep)
            if not os.path.normcase(candidate).startswith(prefix):
                raise BootstrapV3Error("dependency RECORD escapes canonical venv")
            canonical = _plain_absolute(candidate, "file")
            key = _norm(canonical)
            if key in seen_paths:
                raise BootstrapV3Error("duplicate path in dependency RECORD")
            seen_paths.add(key)
            if registry.has_path(canonical):
                raw = registry.bytes_for_path(canonical)
            else:
                raw = registry.lock_absolute(
                    canonical,
                    None,
                    f"{distribution} RECORD member",
                    protect=False,
                )
            observed_digest = _sha256(raw)
            if row_hash and observed_digest != _record_digest(row_hash):
                raise BootstrapV3Error("dependency RECORD member hash changed")
            if row_size and (not row_size.isdecimal() or len(raw) != int(row_size)):
                raise BootstrapV3Error("dependency RECORD member size changed")
            relative_venv = os.path.relpath(canonical, venv_root).replace("\\", "/")
            item = (relative_venv, len(raw), observed_digest)
            file_entries.append(item)
            registered.add(key)
            lowered = canonical.lower()
            if lowered.endswith((".pyd", ".dll", ".so", ".dylib")):
                native_entries.append(item)
            if lowered.endswith(".py") and os.path.normcase(canonical).startswith(
                os.path.normcase(site_root.rstrip(os.sep) + os.sep)
            ):
                dependency_sources[key] = raw
        observed = {
            "version": version,
            "dist_info": dist_info,
            "record_bytes": len(record_raw),
            "record_sha256": _sha256(record_raw),
            "files": len(file_entries),
            "file_bytes": sum(item[1] for item in file_entries),
            "inventory_sha256": _aggregate_file_entries(file_entries),
            "native_files": len(native_entries),
            "native_bytes": sum(item[1] for item in native_entries),
            "native_sha256": _aggregate_file_entries(native_entries),
        }
        wanted = {
            "version": version,
            "dist_info": dist_info,
            "record_bytes": record_bytes,
            "record_sha256": record_sha,
            "files": expected_files,
            "file_bytes": expected_bytes,
            "inventory_sha256": expected_inventory,
            "native_files": expected_native_files,
            "native_bytes": expected_native_bytes,
            "native_sha256": expected_native_inventory,
        }
        if observed != wanted:
            raise BootstrapV3Error(
                f"authenticated dependency inventory changed: {distribution}: {observed}"
            )
        reports[distribution] = observed
    return reports, dependency_sources, registered


def _dependency_module_map(site_root, buffers):
    return _source_module_map(site_root, buffers)


def _verify_dependency_module_origins(site_root, registered_paths):
    allowed_roots = {"numpy", "pandas", "pyarrow", "dateutil", "six", "tzdata", "_cyutility"}
    modules = []
    for name, module in sorted(sys.modules.items()):
        root = name.partition(".")[0]
        if root not in allowed_roots:
            continue
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        if origin in {"built-in", "frozen"}:
            continue
        loader = getattr(module, "__loader__", None)
        loader_name = type(loader).__name__
        if (
            name == "six.moves"
            or name.startswith("six.moves.")
        ) and origin is None and loader_name == "_SixMetaPathImporter":
            modules.append(
                {"module": name, "origin": "<authenticated-six-virtual>", "loader": loader_name}
            )
            continue
        if not isinstance(origin, str) or _norm(origin) not in registered_paths:
            raise BootstrapV3Error(f"dependency module origin is not RECORD-authenticated: {name}")
        if origin.lower().endswith((".pyc", ".pyo")):
            raise BootstrapV3Error("dependency module loaded from bytecode")
        if loader_name not in {"_MemorySourceLoader", "ExtensionFileLoader"}:
            raise BootstrapV3Error(f"dependency loader is not authenticated: {name}")
        modules.append({"module": name, "origin": os.path.relpath(origin, site_root).replace("\\", "/"), "loader": loader_name})
    for required in ("numpy", "pandas", "pyarrow"):
        if required not in sys.modules:
            raise BootstrapV3Error(f"required numerical package was not imported: {required}")
    return modules


def _verify_all_loaded_origins(registry, registered_dependency_paths, bootstrap_path):
    reports = []
    initial_source_modules = {"encodings", "encodings.aliases", "encodings.cp949", "encodings.utf_8"}
    authenticated_native_paths = {_norm(path) for path in _NATIVE_MODULES.values()}
    for name, module in sorted(sys.modules.items()):
        spec = getattr(module, "__spec__", None)
        origin = getattr(spec, "origin", None)
        loader = getattr(module, "__loader__", None)
        loader_name = type(loader).__name__
        if origin in {"built-in", "frozen"}:
            continue
        if name == "__main__" and _norm(getattr(module, "__file__", "")) == _norm(
            bootstrap_path
        ):
            reports.append({"module": name, "origin": BOOTSTRAP_RELATIVE, "loader": loader_name})
            continue
        if (
            (name == "six.moves" or name.startswith("six.moves."))
            and origin is None
            and loader_name == "_SixMetaPathImporter"
        ):
            reports.append({"module": name, "origin": "<authenticated-six-virtual>", "loader": loader_name})
            continue
        if (
            name
            in {
                "_cython_3_2_1",
                "_cython_3_2_4",
                "_cython_3_2_5",
                "cython_runtime",
                "typing.io",
                "typing.re",
            }
            and origin is None
            and getattr(module, "__file__", None) is None
            and loader is None
        ):
            reports.append(
                {"module": name, "origin": "<authenticated-runtime-virtual>", "loader": loader_name}
            )
            continue
        file_path = origin if isinstance(origin, str) else getattr(module, "__file__", None)
        if not isinstance(file_path, str):
            raise BootstrapV3Error(f"loaded module lacks authenticated origin: {name}")
        key = _norm(file_path)
        if key in _SOURCE_PATH_BUFFERS:
            if name in initial_source_modules:
                if loader_name != "SourceFileLoader":
                    raise BootstrapV3Error("initial interpreter source loader changed")
            elif loader_name not in {"_MemorySourceLoader", "ExtensionFileLoader"}:
                raise BootstrapV3Error(f"authenticated module loader changed: {name}")
        elif key in registered_dependency_paths:
            if loader_name != "ExtensionFileLoader":
                raise BootstrapV3Error(f"dependency native loader changed: {name}")
        elif key in authenticated_native_paths:
            if loader_name != "ExtensionFileLoader":
                raise BootstrapV3Error(f"stdlib native loader changed: {name}")
        elif registry.has_path(file_path) and loader is None:
            pass
        else:
            raise BootstrapV3Error(f"loaded module origin is outside authenticated inventory: {name}")
        if file_path.lower().endswith((".pyc", ".pyo")):
            raise BootstrapV3Error("loaded module origin uses forbidden bytecode")
        reports.append({"module": name, "origin": file_path, "loader": loader_name})
    return reports


def _collect_pin_objects(value, output, label):
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"}.issubset(value):
            relative = value["path"]
            root_kind = "data" if relative in {"README.md", "train_wave.csv"} else "workspace"
            key = (root_kind, relative)
            exact = {
                "path": value["path"],
                "bytes": value["bytes"],
                "sha256": value["sha256"],
            }
            prior = output.get(key)
            if prior is not None and prior != exact:
                raise BootstrapV3Error(f"conflicting transitive pin: {label}")
            output[key] = exact
            return
        for key, child in value.items():
            _collect_pin_objects(child, output, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _collect_pin_objects(child, output, f"{label}[{index}]")


def _lock_collected_pins(registry, pins):
    for (root_kind, relative), pin in sorted(pins.items()):
        registry.lock_pin(
            pin,
            label=f"transitive semantic pin {root_kind}:{relative}",
            root_kind=root_kind,
            protect=True,
        )


def _build_semantic_registry(registry, source_buffers, v3_config):
    v2_config = _strict_json(source_buffers["V2_CONFIG"], "compatibility-v2 config")
    v1_config = _strict_json(source_buffers["V1_CONFIG"], "compatibility-v1 config")
    r2_pin = v1_config["r2_implementation_pins"]["CONFIG"]
    r2_raw = registry.lock_pin(r2_pin, label="frozen r2 config", protect=True)
    r2_config = _strict_json(r2_raw, "frozen r2 config")
    lock_pin = v1_config["r2_control_pins"]["ATTEMPT_LOCK"]
    lock_raw = registry.lock_pin(lock_pin, label="frozen r2 attempt lock", protect=True)
    attempt_lock = _strict_json(lock_raw, "frozen r2 attempt lock")
    collected = {}
    for label, value in (
        ("v3", v3_config),
        ("v2", v2_config),
        ("v1", v1_config),
        ("r2", r2_config),
        ("attempt_lock", attempt_lock),
    ):
        _collect_pin_objects(value, collected, label)
    for _role, pin in PINNED_SOURCES.items():
        collected[("workspace", pin["path"])] = dict(pin)
    extras = {
        ".git/config": _pin(
            ".git/config",
            645,
            "8630fd27025aabe4e750d901f3766d8da9ea2a379221e7dc31bc27a857ee10e7",
        ),
        "src/p3_wave/corrected_repeated_forward.py": _pin(
            "src/p3_wave/corrected_repeated_forward.py",
            17628,
            "4b614e2d0fd2259c77d462a69eff27245579d98671ec2a083ce4116278417474",
        ),
        "src/p3_wave/validation.py": _pin(
            "src/p3_wave/validation.py",
            4339,
            "2e76445dd69bcfa2a48efec571a3f9876b7ba104494a159c33338caa347f8e30",
        ),
    }
    for relative, pin in extras.items():
        collected[("workspace", relative)] = pin
    _lock_collected_pins(registry, collected)

    disposition = v3_config["v2_disposition"]
    owner = _strict_json(
        registry.bytes_for_path(
            _contained(registry.workspace, disposition["owner_no_go"]["path"], True, "file")
        ),
        "v2 owner NO-GO",
    )
    tombstone = _strict_json(
        registry.bytes_for_path(
            _contained(
                registry.workspace,
                disposition["execution_tombstone"]["path"],
                True,
                "file",
            )
        ),
        "v2 execution tombstone",
    )
    review = owner.get("review", {})
    finding_ids = [item.get("id", item.get("code")) for item in review.get("findings", [])]
    if (
        review.get("reviewer") != disposition["reviewer"]
        or review.get("verdict") != "NO-GO"
        or review.get("p0_count") != 0
        or review.get("p1_count") != 2
        or review.get("independent_receipt_file_exists") is not False
        or review.get("independent_receipt_hash_exists") is not False
        or finding_ids != disposition["finding_ids"]
        or tombstone.get("status")
        != "PERMANENTLY_TOMBSTONED_NEVER_USE_AS_CANONICAL_ATTESTATION"
        or tombstone.get("review", {}).get("finding_ids") != disposition["finding_ids"]
        or tombstone.get("owner_no_go_receipt") != disposition["owner_no_go"]
        or any(value is not True for value in tombstone.get("forbidden_actions", {}).values())
    ):
        raise BootstrapV3Error("v2 owner NO-GO/tombstone lineage changed")

    v9 = v1_config["v9_anchor"]
    if (
        v9.get("bytes") != 15812
        or v9.get("sha256")
        != "232b6ed3133de11ee05150ec439efe05baa315bbb64ea0f319ffcbddd421b965"
        or v9.get("head_sequence") != 5
        or v9.get("head_event_sha256")
        != "1b3e01be70c6f8ed2df04038deac3b3642804f70f9f17a238826c64d68090317"
        or v9.get("uploads") != 0
    ):
        raise BootstrapV3Error("central v9 seq5 anchor changed")
    return {
        "v2_config": v2_config,
        "v1_config": v1_config,
        "r2_config": r2_config,
        "attempt_lock": attempt_lock,
        "pin_count": len(collected),
        "v9": dict(v9),
    }


def _absence(workspace, relative):
    candidate = _contained(workspace, relative, False, None)
    return not os.path.lexists(candidate)


def _exec_buffer(name, path, raw, injected=None):
    import types

    if name in sys.modules:
        raise BootstrapV3Error(f"authenticated module already exists: {name}")
    module = types.ModuleType(name)
    module.__file__ = path
    module.__package__ = name.rpartition(".")[0]
    module.__loader__ = None
    module.__spec__ = None
    if injected:
        module.__dict__.update(injected)
    sys.modules[name] = module
    try:
        code = compile(raw, path, "exec", dont_inherit=True, optimize=0)
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _patch_semantic_io(registry, r2_module):
    import importlib.metadata
    import io
    from pathlib import Path

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    original = {
        "read_bytes": Path.read_bytes,
        "read_text": Path.read_text,
        "np_load": np.load,
        "pq_read_table": pq.read_table,
        "sha256_file": r2_module.sha256_file,
        "metadata_version": importlib.metadata.version,
    }

    def held_bytes(path):
        return registry.bytes_for_path(path)

    def read_bytes(path):
        return held_bytes(path)

    def read_text(path, encoding=None, errors=None):
        codec = "utf-8" if encoding is None else encoding
        policy = "strict" if errors is None else errors
        return held_bytes(path).decode(codec, errors=policy)

    def np_load(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)):
            return original["np_load"](io.BytesIO(held_bytes(file)), *args, **kwargs)
        return original["np_load"](file, *args, **kwargs)

    def pq_read_table(source, *args, **kwargs):
        if isinstance(source, (str, os.PathLike, Path)):
            return original["pq_read_table"](
                pa.BufferReader(held_bytes(source)), *args, **kwargs
            )
        return original["pq_read_table"](source, *args, **kwargs)

    def sha256_file(path):
        return _sha256(held_bytes(path))

    versions = {
        "numpy": "2.3.5",
        "pandas": "3.0.1",
        "pyarrow": "25.0.1",
        "scipy": "1.18.0",
        "scikit-learn": "1.9.0",
    }

    def metadata_version(name):
        if name not in versions:
            raise BootstrapV3Error(f"unregistered package version query: {name}")
        return versions[name]

    Path.read_bytes = read_bytes
    Path.read_text = read_text
    np.load = np_load
    pq.read_table = pq_read_table
    r2_module.sha256_file = sha256_file
    importlib.metadata.version = metadata_version

    def restore():
        Path.read_bytes = original["read_bytes"]
        Path.read_text = original["read_text"]
        np.load = original["np_load"]
        pq.read_table = original["pq_read_table"]
        r2_module.sha256_file = original["sha256_file"]
        importlib.metadata.version = original["metadata_version"]

    return restore, original


def _quick_reverify(registry, phase, phases):
    if not isinstance(phase, str) or not phase or phase in phases:
        raise BootstrapV3Error("reverification phase is malformed or replayed")
    for entry in registry.entries.values():
        if not entry["protected"]:
            continue
        current_path = os.lstat(entry["path"])
        if (
            _identity(current_path) != entry["identity"]
            or _core_identity(os.fstat(entry["fd"])) != _core_identity(current_path)
            or len(entry["raw"]) != entry["pin"]["bytes"]
            or _sha256(entry["raw"]) != entry["pin"]["sha256"]
        ):
            raise BootstrapV3Error(f"authenticated bytes changed at {phase}")
    phases.append(phase)


def _v2_context(registry, source_buffers, workspace, r2_module, v1_module, phases):
    from types import MappingProxyType

    v2_config = _strict_json(source_buffers["V2_CONFIG"], "v2 context config")
    v2_roles = {
        "CONFIG": PINNED_SOURCES["V2_CONFIG"],
        "HELPER": PINNED_SOURCES["V2_HELPER"],
        "CLI": PINNED_SOURCES["V2_CLI"],
        "TESTS": PINNED_SOURCES["V2_TESTS"],
        "R2_CONTRACT": PINNED_SOURCES["R2_CONTRACT"],
        "V1_CONFIG": PINNED_SOURCES["V1_CONFIG"],
        "V1_HELPER": PINNED_SOURCES["V1_HELPER"],
        "V1_CLI": PINNED_SOURCES["V1_CLI"],
        "V1_TESTS": PINNED_SOURCES["V1_TESTS"],
        "V1_OWNER_NO_GO": PINNED_SOURCES["V1_OWNER_NO_GO"],
        "V1_TOMBSTONE": PINNED_SOURCES["V1_TOMBSTONE"],
    }
    buffers = {
        "CONFIG": source_buffers["V2_CONFIG"],
        "HELPER": source_buffers["V2_HELPER"],
        "CLI": source_buffers["V2_CLI"],
        "TESTS": source_buffers["V2_TESTS"],
        "R2_CONTRACT": source_buffers["R2_CONTRACT"],
        "V1_CONFIG": source_buffers["V1_CONFIG"],
        "V1_HELPER": source_buffers["V1_HELPER"],
        "V1_CLI": source_buffers["V1_CLI"],
        "V1_TESTS": source_buffers["V1_TESTS"],
        "V1_OWNER_NO_GO": source_buffers["V1_OWNER_NO_GO"],
        "V1_TOMBSTONE": source_buffers["V1_TOMBSTONE"],
    }
    claimed = set()
    token = object()

    def checked_path(relative, must_exist=True, kind=None):
        from pathlib import Path

        return Path(_contained(workspace, relative, must_exist, kind))

    def reverify(phase):
        _quick_reverify(registry, "legacy_" + phase, phases)

    def claim_phase(phase):
        if phase != "CLI_VERIFY_ONCE" or phase in claimed:
            raise BootstrapV3Error("legacy v2 capability is invalid or replayed")
        claimed.add(phase)

    data = {
        "token": token,
        "root": __import__("pathlib").Path(workspace),
        "buffers": MappingProxyType(dict(buffers)),
        "pins": MappingProxyType(
            {role: MappingProxyType(dict(pin)) for role, pin in v2_roles.items()}
        ),
        "implementation_roles": MappingProxyType(dict(v2_config["implementation_roles"])),
        "reverify": reverify,
        "checked_path": checked_path,
        "claim_phase": claim_phase,
        "r2_module": r2_module,
        "v1_module": v1_module,
    }
    return MappingProxyType(data), data, token, claimed


def _run_authenticated_modules(
    registry,
    source_buffers,
    workspace,
    v3_config,
    runtime_report,
    dependency_report,
):
    from pathlib import Path
    from types import MappingProxyType

    phases = []
    loaded = []
    r2_path = _contained(workspace, PINNED_SOURCES["R2_CONTRACT"]["path"], True, "file")
    v1_path = _contained(workspace, PINNED_SOURCES["V1_HELPER"]["path"], True, "file")
    v2_path = _contained(workspace, PINNED_SOURCES["V2_HELPER"]["path"], True, "file")
    v3_path = _contained(workspace, PINNED_SOURCES["V3_HELPER"]["path"], True, "file")
    cli_path = _contained(workspace, PINNED_SOURCES["V3_CLI"]["path"], True, "file")

    _quick_reverify(registry, "pre_r2_contract_exec", phases)
    r2_module = _exec_buffer(R2_MODULE, r2_path, source_buffers["R2_CONTRACT"])
    loaded.append(R2_MODULE)
    _quick_reverify(registry, "pre_v1_helper_exec", phases)
    v1_module = _exec_buffer(V1_HELPER_MODULE, v1_path, source_buffers["V1_HELPER"])
    loaded.append(V1_HELPER_MODULE)
    v2_context, v2_context_data, v2_token, v2_claimed = _v2_context(
        registry, source_buffers, workspace, r2_module, v1_module, phases
    )
    _quick_reverify(registry, "pre_v2_helper_exec", phases)
    v2_module = _exec_buffer(
        V2_HELPER_MODULE,
        v2_path,
        source_buffers["V2_HELPER"],
        {
            "__trusted_bootstrap_context__": v2_context,
            "__trusted_bootstrap_token__": v2_token,
            "__trusted_v1_module__": v1_module,
            "__trusted_r2_module__": r2_module,
        },
    )
    loaded.append(V2_HELPER_MODULE)
    v2_context_data["v2_helper"] = v2_module
    _quick_reverify(registry, "pre_v2_cli_exec", phases)
    v2_cli_module = _exec_buffer(
        V2_CLI_MODULE_AUTH,
        _contained(workspace, PINNED_SOURCES["V2_CLI"]["path"], True, "file"),
        source_buffers["V2_CLI"],
        {
            "__trusted_bootstrap_context__": v2_context,
            "__trusted_bootstrap_token__": v2_token,
            "__trusted_v2_helper__": v2_module,
        },
    )
    loaded.append(V2_CLI_MODULE_AUTH)

    outer_token = object()
    claimed = set()
    legacy_used = False
    context_data = {}

    def claim_phase(phase, presented):
        if presented is not outer_token or phase not in {
            "V3_CLI_ONCE",
            "V3_CHECK_ONLY_ONCE",
        } or phase in claimed:
            raise BootstrapV3Error("v3 phase capability is invalid or replayed")
        if phase == "V3_CHECK_ONLY_ONCE" and "V3_CLI_ONCE" not in claimed:
            raise BootstrapV3Error("v3 helper entered before authenticated CLI")
        claimed.add(phase)

    def reverify(phase):
        _quick_reverify(registry, phase, phases)

    def canonical_path(relative, root_kind="workspace"):
        root = registry.data_root if root_kind == "data" else workspace
        return Path(_contained(root, relative, True, "file"))

    def same_path(left, right):
        return _norm(left) == _norm(right)

    def source_buffer(role):
        return source_buffers[role]

    def source_pin(role):
        return dict(PINNED_SOURCES[role])

    def absent(relative):
        return _absence(workspace, relative)

    def runtime_result():
        return dict(runtime_report)

    def semantic_result():
        return registry.report()

    def run_legacy_replay(presented):
        nonlocal legacy_used
        if presented is not outer_token or legacy_used:
            raise BootstrapV3Error("legacy replay capability is invalid or replayed")
        legacy_used = True
        restore, originals = _patch_semantic_io(registry, r2_module)
        original_prefixes = r2_module.PREFIX_FRACTIONS
        original_inventory = r2_module._control_inventory
        caught = None
        try:
            _quick_reverify(registry, "pre_legacy_v2_verify", phases)
            result = v2_cli_module.run(
                root=Path(workspace), requested_config=None, mode="check-only"
            )
            _quick_reverify(registry, "post_legacy_v2_verify", phases)
            return result
        except BaseException as exc:
            caught = exc
            raise
        finally:
            restore()
            if (
                r2_module.PREFIX_FRACTIONS is not original_prefixes
                or r2_module._control_inventory is not original_inventory
                or __import__("pathlib").Path.read_bytes is not originals["read_bytes"]
                or __import__("pathlib").Path.read_text is not originals["read_text"]
                or sys.modules["numpy"].load is not originals["np_load"]
                or sys.modules["pyarrow.parquet"].read_table is not originals["pq_read_table"]
                or r2_module.sha256_file is not originals["sha256_file"]
            ):
                raise BootstrapV3Error("semantic adapters or legacy globals were not restored") from caught

    subordinate = {
        role: dict(PINNED_SOURCES["V3_" + role])
        for role in ("HELPER", "CLI", "TESTS")
    }
    context_data.update(
        {
            "token": outer_token,
            "workspace": Path(workspace),
            "implementation_roles": dict(IMPLEMENTATION_ROLES),
            "subordinate_pins": subordinate,
            "runtime_contract": v3_config["canonical_runtime_contract"],
            "semantic_read_contract": v3_config["stable_semantic_read_contract"],
            "canonical_path": canonical_path,
            "same_path": same_path,
            "source_buffer": source_buffer,
            "source_pin": source_pin,
            "absent": absent,
            "claim_phase": claim_phase,
            "reverify": reverify,
            "run_legacy_replay": run_legacy_replay,
            "runtime_report": runtime_result,
            "semantic_report": semantic_result,
            "dependency_report": dict(dependency_report),
        }
    )
    context = MappingProxyType(context_data)
    _quick_reverify(registry, "pre_v3_helper_exec", phases)
    v3_module = _exec_buffer(
        V3_HELPER_MODULE,
        v3_path,
        source_buffers["V3_HELPER"],
        {"__trusted_v3_context__": context, "__trusted_v3_token__": outer_token},
    )
    loaded.append(V3_HELPER_MODULE)
    context_data["helper"] = v3_module
    _quick_reverify(registry, "pre_v3_cli_exec", phases)
    cli_module = _exec_buffer(
        V3_CLI_MODULE,
        cli_path,
        source_buffers["V3_CLI"],
        {
            "__trusted_v3_context__": context,
            "__trusted_v3_token__": outer_token,
            "__trusted_v3_helper__": v3_module,
        },
    )
    loaded.append(V3_CLI_MODULE)
    result = cli_module.run(root=Path(workspace), requested_config=None, mode="check-only")
    if (
        claimed != {"V3_CLI_ONCE", "V3_CHECK_ONLY_ONCE"}
        or not legacy_used
        or v2_claimed != {"CLI_VERIFY_ONCE"}
    ):
        raise BootstrapV3Error(
            "single-use phase capabilities were not consumed exactly once: "
            f"v3={claimed}, legacy_used={legacy_used}, v2={v2_claimed}"
        )
    return result, phases, loaded


def _validate_v3_config(config):
    if (
        config.get("schema_version")
        != "p3_gen6_incumbent_preserving_residual_calibrator.r2_compatibility_verifier.v3"
        or config.get("identity")
        != "P3_GEN6_INCUMBENT_PRESERVING_RESIDUAL_CALIBRATOR_R2_COMPATIBILITY_VERIFIER_V3"
        or config.get("verifier_only") is not True
        or config.get("check_only_default") is not True
        or config.get("append_only_successor_of_v2") is not True
        or config.get("implementation_roles") != IMPLEMENTATION_ROLES
    ):
        raise BootstrapV3Error("v3 config identity or role map changed")
    subordinate = {
        role: dict(PINNED_SOURCES["V3_" + role])
        for role in ("HELPER", "CLI", "TESTS")
    }
    if config.get("authenticated_subordinate_pins") != subordinate:
        raise BootstrapV3Error("v3 config does not bind subordinate implementation bytes")
    expected_v2 = {
        "BOOTSTRAP": PINNED_SOURCES["V2_BOOTSTRAP"],
        "CONFIG": PINNED_SOURCES["V2_CONFIG"],
        "HELPER": PINNED_SOURCES["V2_HELPER"],
        "CLI": PINNED_SOURCES["V2_CLI"],
        "TESTS": PINNED_SOURCES["V2_TESTS"],
    }
    if config.get("v2_implementation_pins") != expected_v2:
        raise BootstrapV3Error("frozen compatibility-v2 pins changed")
    runtime = config.get("canonical_runtime_contract", {})
    if (
        runtime.get("python_version") != "3.12.10"
        or runtime.get("required_cli_flags") != ["-I", "-S", "-B"]
        or runtime.get("stdlib_inventory", {}).get("directories")
        != STDLIB_EXPECTED["directories"]
        or runtime.get("stdlib_inventory", {}).get("files") != STDLIB_EXPECTED["files"]
        or runtime.get("stdlib_inventory", {}).get("file_bytes")
        != STDLIB_EXPECTED["file_bytes"]
        or runtime.get("stdlib_inventory", {}).get("sha256")
        != STDLIB_EXPECTED["sha256"]
        or set(runtime.get("third_party_distributions", {})) != set(DEPENDENCY_EXPECTED)
    ):
        raise BootstrapV3Error("v3 runtime trust contract changed")
    dependency_config = runtime["third_party_distributions"]
    for name, expected in DEPENDENCY_EXPECTED.items():
        (
            dist_info,
            version,
            record_bytes,
            record_sha,
            files,
            file_bytes,
            inventory_sha,
            native_files,
            native_bytes,
            native_sha,
        ) = expected
        if dependency_config.get(name) != {
            "version": version,
            "dist_info": dist_info,
            "record_bytes": record_bytes,
            "record_sha256": record_sha,
            "files": files,
            "file_bytes": file_bytes,
            "inventory_sha256": inventory_sha,
            "native_files": native_files,
            "native_bytes": native_bytes,
            "native_sha256": native_sha,
        }:
            raise BootstrapV3Error(f"dependency trust config changed: {name}")
    stable = config.get("stable_semantic_read_contract", {})
    if (
        stable.get("ancestor_link_and_reparse_containment") is not True
        or stable.get("held_windows_handles_until_final_verification") is not True
        or stable.get("strict_json_from_authenticated_buffer_only") is not True
        or stable.get("npy_from_io_bytesio_only") is not True
        or stable.get("parquet_from_pyarrow_buffer_reader_only") is not True
        or stable.get("protected_path_reopen_allowed") is not False
    ):
        raise BootstrapV3Error("v3 stable semantic read contract changed")


def main():
    global _FAST_HASH

    arguments = _parse_early_arguments()
    workspace = _plain_absolute(arguments["root"], "directory")
    canonical_config = _contained(workspace, V3_CONFIG_RELATIVE, True, "file")
    if arguments["config"] is not None:
        requested_config = arguments["config"]
        if not os.path.isabs(requested_config):
            requested_config = os.path.join(workspace, requested_config)
        if _norm(requested_config) != _norm(canonical_config):
            raise BootstrapV3Error("alternate compatibility-v3 config is forbidden")
    workspace_env = os.environ.get("P3_WORKSPACE_ROOT")
    data_env = os.environ.get("P3_DATA_DIR")
    if not workspace_env or not data_env:
        raise BootstrapV3Error("P3_WORKSPACE_ROOT and P3_DATA_DIR are required")
    if _norm(_plain_absolute(workspace_env, "directory")) != _norm(workspace):
        raise BootstrapV3Error("P3_WORKSPACE_ROOT differs from --root")
    data_root = _plain_absolute(data_env, "directory")
    thread_environment = {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
    }
    if any(os.environ.get(key) != value for key, value in thread_environment.items()):
        raise BootstrapV3Error("canonical thread environment changed")
    initial_report = _assert_initial_runtime(workspace)
    venv_root = _plain_absolute(os.path.join(workspace, ".venv-p1"), "directory")
    site_root = _plain_absolute(os.path.join(venv_root, "Lib", "site-packages"), "directory")
    base_root = _plain_absolute(sys.base_prefix, "directory")
    registry = _HeldRegistry(workspace, data_root)
    loaded_modules = []
    finder = None
    try:
        bootstrap_path = _contained(workspace, BOOTSTRAP_RELATIVE, True, "file")
        bootstrap_raw = registry.lock_absolute(
            bootstrap_path, None, "externally pinned bootstrap self", protect=True
        )
        hash_primitive_pins = (
            (
                os.path.join(base_root, "DLLs", "_hashlib.pyd"),
                68976,
                "03d233dbace599168eaffec823de6ed7beec8c2a4ebfe9b8a3d8e042a59af3ba",
            ),
            (
                os.path.join(base_root, "DLLs", "libcrypto-3.dll"),
                5231472,
                "ccfffddcd3defb8d899026298af9af43bc186130f8483d77e97c93233d5f27d7",
            ),
        )
        for path, size, digest in hash_primitive_pins:
            registry.lock_absolute(
                path,
                _pin(os.path.basename(path), size, digest),
                "preauthenticated hash primitive",
                protect=False,
            )
        import _hashlib

        if _norm(_hashlib.__spec__.origin) != _norm(hash_primitive_pins[0][0]):
            raise BootstrapV3Error("authenticated _hashlib origin changed")
        _FAST_HASH = _hashlib.openssl_sha256
        _AUDIT["phase"] = "STDLIB_AUTH"
        stdlib_report, stdlib_buffers, stdlib_native = _stdlib_inventory(base_root)
        for path in stdlib_native:
            if not registry.has_path(path):
                registry.hold_native(path)
        _SOURCE_PATH_BUFFERS.update(stdlib_buffers)
        _SOURCE_MODULES.update(
            _source_module_map(os.path.join(base_root, "Lib"), stdlib_buffers)
        )
        _NATIVE_MODULES.update(
            _native_module_map(os.path.join(base_root, "DLLs"), stdlib_native)
        )
        finder = _MemorySourceFinder()
        builtin = sys.modules["_frozen_importlib"].BuiltinImporter
        frozen = sys.modules["_frozen_importlib"].FrozenImporter
        sys.meta_path[:] = [builtin, frozen, finder]
        import hashlib
        import json

        if _norm(hashlib.__file__) not in _SOURCE_PATH_BUFFERS:
            raise BootstrapV3Error("hashlib did not execute from authenticated source buffer")
        _FAST_HASH = hashlib.sha256
        _verify_source_pin_schema()
        _AUDIT["phase"] = "SOURCE_AUTH"
        source_buffers = {}
        for role, pin in PINNED_SOURCES.items():
            source_buffers[role] = registry.lock_pin(
                pin, label=f"authenticated source {role}", protect=True
            )
        v3_config = _strict_json(source_buffers["V3_CONFIG"], "v3 config")
        _validate_v3_config(v3_config)
        runtime_contract = v3_config["canonical_runtime_contract"]
        registry.lock_pin(
            {
                "path": runtime_contract["python_relative_to_workspace"],
                **runtime_contract["python_pin"],
            },
            label="canonical python executable",
            protect=True,
        )
        registry.lock_pin(
            {
                "path": runtime_contract["pyvenv_cfg_relative_to_workspace"],
                **runtime_contract["pyvenv_cfg_pin"],
            },
            label="canonical pyvenv.cfg",
            protect=True,
        )
        dependency_report, dependency_sources, registered_dependency_paths = (
            _authenticate_dependencies(registry, venv_root, site_root)
        )
        _SOURCE_PATH_BUFFERS.update(dependency_sources)
        _SOURCE_MODULES.update(_dependency_module_map(site_root, dependency_sources))
        dependency_native = [
            entry["path"]
            for key, entry in registry.entries.items()
            if key in registered_dependency_paths and entry["path"].lower().endswith(".pyd")
        ]
        _NATIVE_MODULES.update(_native_module_map(site_root, dependency_native))
        _AUDIT["phase"] = "DEPENDENCIES_AUTHENTICATED"
        import numpy
        import pandas
        import pyarrow

        if (
            numpy.__version__ != "2.3.5"
            or pandas.__version__ != "3.0.1"
            or pyarrow.__version__ != "25.0.1"
        ):
            raise BootstrapV3Error("authenticated numerical package version changed")
        dependency_modules = _verify_dependency_module_origins(
            site_root, registered_dependency_paths
        )
        semantic_lineage = _build_semantic_registry(
            registry, source_buffers, v3_config
        )
        for relative in (
            "artifacts/p3_gen6_incumbent_preserving_residual_calibrator_20260823_v1r2",
            "artifacts/p3_gen6_incumbent_preserving_residual_calibrator_20260823_v1r2/blind",
            "artifacts/p3_gen6_incumbent_preserving_residual_calibrator_20260823_v1r2/commitments",
            "artifacts/p3_gen6_incumbent_preserving_residual_calibrator_20260823_v1r2_control",
            "artifacts/p3_meaningful_learning_curve_20260823_v1",
            "artifacts/p3_meaningful_learning_curve_20260823_v1/oof",
            "artifacts/p3/features_all20_v1",
            "artifacts/meaningful_score_goal_v9",
        ):
            registry.hold_directory(_contained(workspace, relative, True, "directory"))
        registry.hold_directory(data_root)
        registry.protect_all()
        _AUDIT["phase"] = "SEMANTIC_REPLAY"
        runtime_report = {
            **initial_report,
            "python_version": sys.version.split()[0],
            "python_executable": sys.executable,
            "stdlib_inventory": stdlib_report,
            "dependency_distributions": dependency_report,
            "dependency_module_origins": dependency_modules,
            "source_loader": "AUTHENTICATED_BUFFER_COMPILE_EXEC_NO_PYC",
            "native_loader": "AUTHENTICATED_ORIGIN_HELD_FILE_SHARE_READ_ONLY",
            "sys_path_unchanged": tuple(sys.path) == _INITIAL_SYS_PATH,
        }
        result, phases, loaded_modules = _run_authenticated_modules(
            registry,
            source_buffers,
            workspace,
            v3_config,
            runtime_report,
            dependency_report,
        )
        final_dependency_modules = _verify_dependency_module_origins(
            site_root, registered_dependency_paths
        )
        loaded_origin_report = _verify_all_loaded_origins(
            registry, registered_dependency_paths, bootstrap_path
        )
        registry.final_verify()
        final_stdlib, unused_buffers, unused_native = _stdlib_inventory(base_root)
        del unused_buffers, unused_native
        if final_stdlib != stdlib_report:
            raise BootstrapV3Error("stdlib inventory changed during compatibility replay")
        if ENGINE_MODULE in sys.modules:
            raise BootstrapV3Error("r2 execution engine entered sys.modules")
        if any(_AUDIT["attempts"].values()):
            raise BootstrapV3Error(f"zero-attempt audit firewall changed: {_AUDIT['attempts']}")
        result["trusted_runtime"] = {
            **runtime_report,
            "final_stdlib_inventory_verified": True,
            "audit_hook_installed_before_non_sys_import": True,
            "audit_attempts": dict(_AUDIT["attempts"]),
            "dependency_module_origins": final_dependency_modules,
            "all_loaded_module_origins": loaded_origin_report,
        }
        result["stable_semantic_registry"] = registry.report()
        result["stable_semantic_registry"]["transitive_pin_count"] = semantic_lineage[
            "pin_count"
        ]
        result["stable_semantic_registry"]["v9"] = semantic_lineage["v9"]
        result["trusted_bootstrap"] = {
            "identity": IDENTITY,
            "path": BOOTSTRAP_RELATIVE,
            "bytes": len(bootstrap_raw),
            "sha256": _sha256(bootstrap_raw),
            "externally_pinned_noncyclic_root": True,
            "authenticated_buffer_execution_only": True,
            "reverification_phases": phases,
            "r2_engine_imported": False,
        }
        rendered = json.dumps(
            result,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    finally:
        for module_name in reversed(loaded_modules):
            sys.modules.pop(module_name, None)
        if finder is not None and finder in sys.meta_path:
            sys.meta_path[:] = list(_INITIAL_META_PATH)
        registry.close()
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
