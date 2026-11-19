import time
import secrets

import pytest

from Asterion.Intelligence.security import (
    initialize_security,
    provision_agent,
    revoke_agent,
    authenticate_agent_signed,
)


def test_agent_authentication_and_replay(monkeypatch, tmp_path):
    import Asterion.Intelligence.security as security

    monkeypatch.setattr(
        security,
        "DB_PATH",
        tmp_path / "test.db"
    )

    initialize_security()

    credential = provision_agent(
        "host-01",
        ["test-host"]
    )

    body = b'{"agent_id":"host-01","host":"test-host"}'
    timestamp = str(int(time.time()))
    nonce = secrets.token_urlsafe(24)

    signature = security.signature_for(
        "POST",
        "/v2/agents/telemetry",
        timestamp,
        nonce,
        body,
        credential.signing_secret
    )

    arguments = dict(
        agent_id="host-01",
        token=credential.token,
        signing_secret=credential.signing_secret,
        timestamp=timestamp,
        nonce=nonce,
        signature=signature,
        method="POST",
        path="/v2/agents/telemetry",
        body=body,
        host="test-host"
    )

    assert authenticate_agent_signed(**arguments)

    with pytest.raises(PermissionError):
        authenticate_agent_signed(**arguments)

    revoke_agent("host-01")

    with pytest.raises(PermissionError):
        authenticate_agent_signed(
            **{
                **arguments,
                "nonce": secrets.token_urlsafe(24)
            }
        )
