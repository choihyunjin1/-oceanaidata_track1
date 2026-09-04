"""Authenticated in-memory CLI body for the P3 Gen6r2 v4 verifier."""

if (
    "__trusted_v4_context__" not in globals()
    or "__trusted_v4_token__" not in globals()
    or "__trusted_v4_helper__" not in globals()
):
    raise RuntimeError("compatibility-v4 CLI requires the trusted v4 bootstrap")

_CONTEXT = globals()["__trusted_v4_context__"]
_TOKEN = globals()["__trusted_v4_token__"]
_HELPER = globals()["__trusted_v4_helper__"]
if _CONTEXT.get("token") is not _TOKEN or _CONTEXT.get("helper") is not _HELPER:
    raise RuntimeError("compatibility-v4 CLI trust identity differs")


def run(*, root, requested_config=None, mode="check-only"):
    """Enter the authenticated helper exactly once."""

    _CONTEXT["claim_phase"]("V4_CLI_ONCE", _TOKEN)
    _CONTEXT["reverify"]("v4_cli_pre_helper")
    result = _HELPER.verify_trusted(
        root,
        requested_config=requested_config,
        mode=mode,
    )
    _CONTEXT["reverify"]("v4_cli_post_helper")
    return result


__all__ = ["run"]
