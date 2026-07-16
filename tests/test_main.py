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
        "preset": "Normal",
        "prompt": "a futuristic city with flying cars",
        "count": 2,
        "retry_count": 3,
        "image_size": "2K",
        "is_selfie": True
    }
    response = client.post("/api/v1/generate-image", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["generated_images"]) == 2
    for img_path in data["generated_images"]:
        assert img_path.startswith("temp/generated_")
        assert img_path.endswith(".png")
    assert "Successfully generated images via" in data["message"]
    assert "a futuristic city with flying cars" in data["message"]
    assert "Picture 1 defines the identity" in data["message"]
    assert "It's a selfie" in data["message"]




