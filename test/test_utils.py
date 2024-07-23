import base64
import responses
from src.utils import img_bytes_to_base64, check_access_token

def test_img_bytes_to_base64_bytes():
    data = b"test data"
    expected_result = base64.b64encode(data).decode('utf-8')
    result = img_bytes_to_base64(data)
    assert result == expected_result

def test_img_bytes_to_base64_dict():
    data = {"bytes": b"test data"}
    expected_result = base64.b64encode(data["bytes"]).decode('utf-8')
    result = img_bytes_to_base64(data)
    assert result == expected_result

@responses.activate
def test_check_access_token_valid():
    token = "valid_token"
    url = f"https://www.googleapis.com/oauth2/v1/tokeninfo?access_token={token}"

    responses.add(
        responses.GET,
        url,
        json={"audience": "your_audience"},
        status=200
    )

    result = check_access_token(token)
    assert result == True

@responses.activate
def test_check_access_token_invalid():
    token = "invalid_token"
    url = f"https://www.googleapis.com/oauth2/v1/tokeninfo?access_token={token}"

    responses.add(
        responses.GET,
        url,
        json={"error": "invalid_token"},
        status=400
    )

    result = check_access_token(token)
    assert result == False