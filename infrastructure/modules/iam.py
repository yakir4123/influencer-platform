"""Module to configure IAM policies and permissions with least privilege."""

from __future__ import annotations

from collections.abc import Mapping

import pulumi_gcp as gcp

from modules.secrets import SecretResources


def configure_iam(
    project_id: str,
    service_account: gcp.serviceaccount.Account,
    storage_bucket: gcp.storage.Bucket,
    secrets: Mapping[str, SecretResources],
) -> None:
    """Configure least-privilege IAM permissions for the Cloud Run account.

    Grants:
    - Storage Object Admin on the image bucket.
    - Secret Accessor on each designated Secret Manager secret.
    - Datastore User at the project level for Firestore Native mode.

    Args:
        project_id: Google Cloud project ID.
        service_account: Service account used by the Cloud Run service.
        storage_bucket: Cloud Storage bucket for Influencer Platform images.
        secrets: Mapping of secret IDs to their Secret Manager resources.
    """
    sa_member = service_account.email.apply(
        lambda email: f"serviceAccount:{email}"
    )

    gcp.storage.BucketIAMMember(
        "bucket-object-admin",
        bucket=storage_bucket.name,
        role="roles/storage.objectAdmin",
        member=sa_member,
    )

    for secret_id, secret_resources in secrets.items():
        gcp.secretmanager.SecretIamMember(
            f"{secret_id.lower()}-accessor",
            secret_id=secret_resources["secret"].id,
            role="roles/secretmanager.secretAccessor",
            member=sa_member,
        )

    gcp.projects.IAMMember(
        "project-firestore-user",
        project=project_id,
        role="roles/datastore.user",
        member=sa_member,
    )