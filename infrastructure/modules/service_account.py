"""Module to provision dedicated IAM Service Accounts."""

import pulumi
import pulumi_gcp as gcp


def create_service_account(
    account_id: str,
) -> gcp.serviceaccount.Account:
    """Creates a dedicated GCP Service Account for the Cloud Run service.

    Args:
        account_id: Service Account ID (e.g. 'influencer-platform-runner').

    Returns:
        The created Service Account resource.
    """
    return gcp.serviceaccount.Account(
        account_id,
        account_id=account_id,
        display_name="Cloud Run Influencer Platform Backend SA",
        description="Dedicated service account running the Influencer Platform backend on Cloud Run",
    )
