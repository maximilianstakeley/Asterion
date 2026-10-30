# Add this helper to service.py near _executor_metadata:

def _assert_nonexecuting_result(executor_result: Any) -> dict[str, Any]:
    """Reject executor metadata that contradicts Asterion's disabled mode."""
    metadata = _executor_metadata(executor_result)
    if metadata.get("executed") is True:
        raise RuntimeError(
            "Executor reported executed=True while live execution is disabled; "
            "treating this as a safety-contract violation."
        )
    if metadata.get("execution_enabled") is True:
        raise RuntimeError(
            "Executor reported execution_enabled=True while live execution "
            "is disabled; treating this as a safety-contract violation."
        )
    return metadata


# In analyze(), immediately after:
#     executor_result = _executor.execute(selected, verdict)
# replace the next line with:
executor_metadata = _assert_nonexecuting_result(executor_result)

# Then remove the later duplicate assignment:
#     executor_metadata = _executor_metadata(executor_result)
#
# Keep "no_actions_executed": True only after _assert_nonexecuting_result()
# succeeds. A raised exception is caught by analyze() and passed to
# _audit_failure(), preserving fail-closed behavior.
