import base64
import requests

def img_bytes_to_base64(data: dict | bytes) -> str:
    """
    Convert a dict with 'bytes' key to base64 string or a bytes object to a base64 string.
    """
    if isinstance(data, bytes):
        return base64.b64encode(data).decode('utf-8') # Convert bytes to base64   
    else:  
        return base64.b64encode(data['bytes']).decode('utf-8')
    
def check_access_token(token : str) -> bool:
    url = f"https://www.googleapis.com/oauth2/v1/tokeninfo?access_token={token}"
    response = requests.get(url)
    if response.status_code == 200:
        return True
    else:
        return False