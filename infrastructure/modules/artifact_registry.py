"""Module to provision Google Artifact Registry repositories."""

import pulumi
import pulumi_gcp as gcp


def create_repository(
    repository_id: str,
    location: str,
) -> gcp.artifactregistry.Repository:
    """Creates a Docker repository in Google Artifact Registry.

    Args:
        repository_id: The ID of the repository to create.
        location: The region/location where the repository will be hosted.

    Returns:
        The created Artifact Registry Repository resource.
    """
    return gcp.artifactregistry.Repository(
        repository_id,
        repository_id=repository_id,
        location=location,
        format="DOCKER",
        description="Docker repository for Telegram bot backend images",
        # Enable cleanup policies or mutability if needed, but defaults are standard
        # image_immutable_updates=False,
    )
