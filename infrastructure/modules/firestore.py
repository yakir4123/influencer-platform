"""Module to provision Firestore database in Native mode."""

import pulumi
import pulumi_gcp as gcp


def create_firestore_database(
    location: str,
    database_id: str = "(default)",
) -> gcp.firestore.Database:
    """Creates a Firestore database in Native mode.

    Args:
        location: Location where the database is hosted.
        database_id: ID of the Firestore database. Defaults to "(default)" which is standard.

    Returns:
        The created Firestore Database resource.
    """
    # Note: Pulumi name must be unique within stack, but the GCP Firestore Database ID is passed as `name` in Pulumi GCP.
    return gcp.firestore.Database(
        "firestore-native-db",
        name=database_id,
        location_id=location,
        type="FIRESTORE_NATIVE",
        # OPTIONAL: "DISABLED" allows firestore to be managed independently of App Engine
        app_engine_integration_mode="DISABLED",
    )
