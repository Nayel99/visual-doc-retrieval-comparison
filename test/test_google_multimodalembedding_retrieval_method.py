import os
import pytest
import responses
import base64

from src.google_multimodalembedding_retrieval_method import GoogleMultimodalEmbeddingRetrievalMethod

@pytest.fixture
def setup_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_LOCATION", "us-central1")
    monkeypatch.setenv("GOOGLE_PROJECT_ID", "my-project")
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "fake-access-token")

@responses.activate
def test_query2vector(setup_env):
    retrieval_method = GoogleMultimodalEmbeddingRetrievalMethod()

    queries = ["What is AI?", "How does a neural network work?"]
    
    responses.add(
        responses.POST,
        "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google/models/multimodalembedding@001:predict",
        json={"predictions": [{"textEmbedding": [0.1, 0.2, 0.3]}]},
        status=200
    )
    
    result = retrieval_method.query2vector(queries)
    
    assert len(result) == 2
    assert "query" in result[0]
    assert "query_vector" in result[0]
    assert result[0]["query_vector"] == [0.1, 0.2, 0.3]

@responses.activate
def test_image2vector(setup_env):
    retrieval_method = GoogleMultimodalEmbeddingRetrievalMethod()

    with open("test/data/img_test.jpg", "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    images = [
        {"image_filename": "image1.jpg", "image_base64": "base64string1"},
        {"image_filename": "image2.jpg", "image_base64": "base64string2"}
    ]

    responses.add(
        responses.POST,
        "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google/models/multimodalembedding@001:predict",
        json={"predictions": [{"imageEmbedding": [0.4, 0.5, 0.6]}]},
        status=200
    )

    result = retrieval_method.image2vector(images)
    
    assert len(result) == 2
    assert "image_filename" in result[0]
    assert "image_vector" in result[0]
    assert result[0]["image_vector"] == [0.4, 0.5, 0.6]