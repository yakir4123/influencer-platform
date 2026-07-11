"""Module to declare Pulumi stack outputs."""

import pulumi
import pulumi_gcp as gcp


def export_outputs(
    cloud_run_service: gcp.cloudrunv2.Service,
    storage_bucket: gcp.storage.Bucket,
    artifact_registry: gcp.artifactregistry.Repository,
    service_account: gcp.serviceaccount.Account,
) -> None:
    """Declares Pulumi stack outputs for external access and verification.

    Exports:
    - Cloud Run URL
    - Storage Bucket name
    - Artifact Registry repository path
    - Service Account email

    Args:
        cloud_run_service: The provisioned Cloud Run service.
        storage_bucket: The provisioned storage bucket.
        artifact_registry: The provisioned Artifact Registry repository.
        service_account: The provisioned service account.
    """
    # Public HTTPS endpoint of the Cloud Run service
    pulumi.export("cloud_run_url", cloud_run_service.uri)

    # Name of the Cloud Storage bucket
    pulumi.export("storage_bucket_name", storage_bucket.name)

    # Fully-qualified name of the Artifact Registry repository
    # Format: projects/{project}/locations/{location}/repositories/{repository_id}
    pulumi.export("artifact_registry_repository", artifact_registry.name)

    # Email of the dedicated service account
    pulumi.export("service_account_email", service_account.email)
