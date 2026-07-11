"""Module to provision Google Cloud Storage buckets."""

import pulumi
import pulumi_gcp as gcp


def create_bucket(
    bucket_name: str,
    location: str,
) -> gcp.storage.Bucket:
    """Creates a private, secure Google Cloud Storage bucket with lifecycle rules.

    Requirements:
    - Uniform bucket-level access enabled
    - Public access prevention enforced
    - Versioning disabled
    - Lifecycle rule to delete objects older than 90 days

    Args:
        bucket_name: Name of the bucket.
        location: Location/Region of the bucket.

    Returns:
        The created Storage Bucket resource.
    """
    return gcp.storage.Bucket(
        bucket_name,
        name=bucket_name,
        location=location,
        force_destroy=False,  # Prevent deletion if the bucket contains objects, unless explicitly run
        uniform_bucket_level_access=True,
        public_access_prevention="enforced",
        versioning=gcp.storage.BucketVersioningArgs(
            enabled=False,
        ),
        lifecycle_rules=[
            gcp.storage.BucketLifecycleRuleArgs(
                action=gcp.storage.BucketLifecycleRuleActionArgs(
                    type="Delete",
                ),
                condition=gcp.storage.BucketLifecycleRuleConditionArgs(
                    age=90,  # Delete objects older than 90 days
                ),
            )
        ],
    )
