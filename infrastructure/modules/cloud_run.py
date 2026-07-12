"""Module to provision Google Cloud Run services."""

from __future__ import annotations

from collections.abc import Mapping

import pulumi
import pulumi_gcp as gcp

from modules.secrets import SecretResources


def create_cloud_run_service(
    service_name: str,
    location: str,
    image_url: str,
    service_account_email: pulumi.Input[str],
    max_instances: int,
    cpu: str,
    memory: str,
    secrets: Mapping[str, SecretResources],
    env: Mapping[str, pulumi.Input[str]] | None = None,
) -> gcp.cloudrunv2.Service:
    """Create a public Google Cloud Run service.

    The service scales to zero and injects only secrets that have an actual
    Secret Manager version.

    Args:
        service_name: Cloud Run service name.
        location: Google Cloud region.
        image_url: Docker image URL.
        service_account_email: Runtime service account email.
        max_instances: Maximum number of Cloud Run instances.
        cpu: Container CPU limit, such as ``"1"``.
        memory: Container memory limit, such as ``"512Mi"``.
        secrets: Mapping of environment-variable names to Secret Manager
            resources and optional secret versions.
        env: Optional mapping of environment-variable names to their values.

    Returns:
        The created Cloud Run service.
    """
    env_vars: list[gcp.cloudrunv2.ServiceTemplateContainerEnvArgs] = []

    if env:
        for env_name, env_val in env.items():
            env_vars.append(
                gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                    name=env_name,
                    value=env_val,
                )
            )

    for env_name, secret_resources in secrets.items():
        # Do not bind a secret that has no payload/version yet.
        if secret_resources["version"] is None:
            continue

        env_vars.append(
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                name=env_name,
                value_source=(
                    gcp.cloudrunv2.ServiceTemplateContainerEnvValueSourceArgs(
                        secret_key_ref=(
                            gcp.cloudrunv2.ServiceTemplateContainerEnvValueSourceSecretKeyRefArgs(
                                secret=secret_resources["secret"].secret_id,
                                version="latest",
                            )
                        ),
                    )
                ),
            )
        )

    service = gcp.cloudrunv2.Service(
        service_name,
        name=service_name,
        location=location,
        template=gcp.cloudrunv2.ServiceTemplateArgs(
            service_account=service_account_email,
            containers=[
                gcp.cloudrunv2.ServiceTemplateContainerArgs(
                    image=image_url,
                    resources=(
                        gcp.cloudrunv2.ServiceTemplateContainerResourcesArgs(
                            limits={
                                "cpu": cpu,
                                "memory": memory,
                            },
                        )
                    ),
                    envs=env_vars,
                )
            ],
            scaling=gcp.cloudrunv2.ServiceTemplateScalingArgs(
                min_instance_count=0,
                max_instance_count=max_instances,
            ),
        ),
    )

    # Commented out public invoker to enforce private access.
    # Only authenticated users via proxy can access the service.
    # gcp.cloudrunv2.ServiceIamMember(
    #     f"{service_name}-public-invoker",
    #     name=service.name,
    #     location=service.location,
    #     role="roles/run.invoker",
    #     member="allUsers",
    # )

    return service