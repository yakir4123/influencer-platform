"""Module to provision secrets in Google Cloud Secret Manager."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict

import pulumi
import pulumi_gcp as gcp


class SecretResources(TypedDict):
    """Resources created for one Secret Manager secret."""

    secret: gcp.secretmanager.Secret
    version: gcp.secretmanager.SecretVersion | None


def create_secrets(
    secret_values: Mapping[str, pulumi.Output[str] | None],
) -> dict[str, SecretResources]:
    """Create Secret Manager containers and optional initial versions.

    A Secret resource is always created for each configured secret ID.
    A SecretVersion is created only when the corresponding secret value
    is not None.

    Args:
        secret_values: Mapping from Secret Manager secret IDs to optional
            encrypted Pulumi outputs.

    Returns:
        Mapping from secret IDs to their Secret and optional SecretVersion
        resources.
    """
    created_secrets: dict[str, SecretResources] = {}

    for secret_id, secret_data in secret_values.items():
        secret = gcp.secretmanager.Secret(
            secret_id,
            secret_id=secret_id,
            replication={"auto": {}},
        )

        version: gcp.secretmanager.SecretVersion | None = None

        if secret_data is not None:
            version = gcp.secretmanager.SecretVersion(
                f"{secret_id}-version",
                secret=secret.id,
                secret_data=secret_data,
            )

        created_secrets[secret_id] = {
            "secret": secret,
            "version": version,
        }

    return created_secrets