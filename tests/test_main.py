from unittest.mock import patch

def test_read_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "docs" in response.json()


def test_health_check(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "environment" in data
    assert "version" in data





@patch("app.api.routes.settings")
@patch("app.bot.get_bot_app")
def test_telegram_webhook_unauthorized(mock_get_bot_app, mock_settings, client):
    mock_settings.TELEGRAM_BOT_TOKEN = "12345:token"
    response = client.post(
        "/api/v1/telegram-webhook",
        json={"update_id": 123},
        headers={"X-Telegram-Bot-Api-Secret-Token": "wrong"}
    )
    assert response.status_code == 403
    assert "Invalid webhook secret token" in response.json()["detail"]


@patch("app.api.routes.settings")
@patch("app.bot.get_bot_app")
def test_telegram_webhook_success(mock_get_bot_app, mock_settings, client):
    import hashlib
    from unittest.mock import MagicMock, AsyncMock

    mock_settings.TELEGRAM_BOT_TOKEN = "12345:token"
    mock_bot_app = MagicMock()
    mock_bot_app.process_update = AsyncMock()
    mock_get_bot_app.return_value = mock_bot_app
    
    secret_token = hashlib.sha256(b"12345:token").hexdigest()
    response = client.post(
        "/api/v1/telegram-webhook",
        json={"update_id": 123},
        headers={"X-Telegram-Bot-Api-Secret-Token": secret_token}
    )
    assert response.status_code == 200
    assert response.json() == {"status": "success"}





