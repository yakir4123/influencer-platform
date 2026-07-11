"""Main orchestration script for provisioning Telegram Bot Backend infrastructure on GCP."""

import config
from modules import (
    artifact_registry,
    cloud_run,
    firestore,
    iam,
    outputs,
    secrets,
    service_account,
    storage,
)

# 1. Provision Artifact Registry repository for Docker images
registry = artifact_registry.create_repository(
    repository_id=config.ARTIFACT_REGISTRY_REPO_NAME,
    location=config.GCP_REGION,
)

# 2. Provision GCS bucket for storing Telegram images
bucket = storage.create_bucket(
    bucket_name=config.STORAGE_BUCKET_NAME,
    location=config.GCP_REGION,
)

# 3. Provision Firestore Native database
database = firestore.create_firestore_database(
    location=config.GCP_REGION,
)

# 4. Provision dedicated Service Account for Cloud Run
sa = service_account.create_service_account(
    account_id="telegram-bot-runner",
)

# 5. Provision Secret Manager secrets using configurations
secret_resources = secrets.create_secrets(
    secret_values={
        "TELEGRAM_BOT_TOKEN": config.TELEGRAM_BOT_TOKEN,
        "OPENAI_API_KEY": config.OPENAI_API_KEY,
    }
)

# 6. Apply IAM permissions following the principle of least privilege
iam.configure_iam(
    project_id=config.GCP_PROJECT,
    service_account=sa,
    storage_bucket=bucket,
    secrets=secret_resources,
)

# 7. Provision Cloud Run service, mounting Secret Manager secrets
run_service = cloud_run.create_cloud_run_service(
    service_name=config.CLOUD_RUN_SERVICE_NAME,
    location=config.GCP_REGION,
    image_url=config.CLOUD_RUN_IMAGE,
    service_account_email=sa.email,
    max_instances=config.CLOUD_RUN_MAX_INSTANCES,
    cpu=config.CLOUD_RUN_CPU,
    memory=config.CLOUD_RUN_MEMORY,
    secrets=secret_resources,
)

# 8. Export stacked outputs
outputs.export_outputs(
    cloud_run_service=run_service,
    storage_bucket=bucket,
    artifact_registry=registry,
    service_account=sa,
)
