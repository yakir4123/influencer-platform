from pydantic import BaseModel


class HealthStatus(BaseModel):
    status: str
    environment: str
    version: str = "0.1.0"
