"""Authenticated in-memory CLI body for the P3 Gen6r2 v3 verifier."""

if (
    "__trusted_v3_context__" not in globals()
    or "__trusted_v3_token__" not in globals()
    or "__trusted_v3_helper__" not in globals()
):
    raise RuntimeError("compatibility-v3 CLI requires the trusted v3 bootstrap")

_CONTEXT = globals()["__trusted_v3_context__"]
_TOKEN = globals()["__trusted_v3_token__"]
_HELPER = globals()["__trusted_v3_helper__"]
if _CONTEXT.get("token") is not _TOKEN or _CONTEXT.get("helper") is not _HELPER:
    raise RuntimeError("compatibility-v3 CLI trust identity differs")


def run(*, root, requested_config=None, mode="check-only"):
    """Enter the authenticated helper exactly once."""

    _CONTEXT["claim_phase"]("V3_CLI_ONCE", _TOKEN)
    _CONTEXT["reverify"]("v3_cli_pre_helper")
    result = _HELPER.verify_trusted(
        root,
        requested_config=requested_config,
        mode=mode,
    )
    _CONTEXT["reverify"]("v3_cli_post_helper")
    return result


__all__ = ["run"]
