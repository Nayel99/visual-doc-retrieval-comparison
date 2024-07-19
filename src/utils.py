import base64

def img_bytes_to_base64(img_bytes: bytes) -> str:
    """
    Convert image bytes to base64.
    """
    return base64.b64encode(img_bytes).decode('utf-8')