"""Configuration for the Influencer Platform backend infrastructure.

Loads and type-hints values from the selected Pulumi stack configuration.
"""

from typing import Final

import pulumi


config = pulumi.Config()
gcp_config = pulumi.Config("gcp")


# -----------------------------------------------------------------------------
# GCP core configuration
# -----------------------------------------------------------------------------

GCP_PROJECT: Final[str] = gcp_config.require("project")
GCP_REGION: Final[str] = gcp_config.get("region") or "us-central1"


# -----------------------------------------------------------------------------
# Storage configuration
# -----------------------------------------------------------------------------

STORAGE_BUCKET_NAME: Final[str] = config.require("storage_bucket_name")


# -----------------------------------------------------------------------------
# Artifact Registry configuration
# -----------------------------------------------------------------------------

ARTIFACT_REGISTRY_REPO_NAME: Final[str] = config.require(
    "artifact_registry_repo_name"
)


# -----------------------------------------------------------------------------
# Cloud Run configuration
# -----------------------------------------------------------------------------

CLOUD_RUN_SERVICE_NAME: Final[str] = (
    config.get("cloud_run_service_name") or "influencer-platform-backend"
)

CLOUD_RUN_MAX_INSTANCES: Final[int] = (
    config.get_int("cloud_run_max_instances") or 10
)

CLOUD_RUN_CPU: Final[str] = config.get("cloud_run_cpu") or "1"
CLOUD_RUN_MEMORY: Final[str] = config.get("cloud_run_memory") or "512Mi"

# Temporary public image that allows Cloud Run to be provisioned before the
# Influencer Platform backend application image is available.
CLOUD_RUN_IMAGE: Final[str] = (
    config.get("cloud_run_image")
    or "us-docker.pkg.dev/cloudrun/container/hello"
)


# -----------------------------------------------------------------------------
# Optional secret configuration
# -----------------------------------------------------------------------------

# get_secret() returns None when a secret has not been configured.
# Modules consuming these values must handle None and conditionally create
# Secret Manager versions and Cloud Run secret environment variables.
TELEGRAM_BOT_TOKEN: Final[pulumi.Output[str] | None] = config.get_secret(
    "telegram_bot_token"
)

OPENAI_API_KEY: Final[pulumi.Output[str] | None] = config.get_secret(
    "openai_api_key"
)

INSTAGRAM_COOKIES: Final[pulumi.Output[str] | None] = config.get_secret(
    "instagram_cookies"
)

TELEGRAM_BOT_WEBHOOK_URL: Final[str | None] = config.get(
    "telegram_bot_webhook_url"
)