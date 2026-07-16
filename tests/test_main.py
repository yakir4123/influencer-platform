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


def test_generate_image(client):
    payload = {
        "prompt": "a futuristic city with flying cars",
        "number_of_images": 2,
        "aspect_ratio": "16:9"
    }
    response = client.post("/api/v1/generate-image", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["generated_images"] == ["mock_image_url_1.png"]
    assert data["message"] == "works!"

