from src.utils import img_bytes_to_base64, jpg_to_pdf, check_access_token

from PIL import Image
import requests
import base64
import os
from dotenv import load_dotenv


def test_img_bytes_to_base64():

    img = Image.new('RGB', (100, 100), color = 'red')

    img_base64 = img_bytes_to_base64(img)

    assert isinstance(img_base64, str)

    print("Test img_bytes_to_base64 passed")

def test_jpg_to_pdf():

    jpg_path = "https://blog.hubspot.com/hs-fs/hub/53/file-250043455-jpg/Blog-Related_Images/KontestappInfographic.jpg?width=645&amp;name=KontestappInfographic.jpg"
    img_base64 = jpg_to_pdf(jpg_path)

    print("Test jpg path to pdf passed")
    response = requests.get(jpg_path)
    jpg_data = response.content
    img_base64 = base64.b64encode(jpg_data).decode('utf-8')
    pdf_path = jpg_to_pdf(img_base64)
    if os.path.exists(pdf_path):
        print("Test jpg_to_pdf passed.")
    else:
        print("Test jpg_to_pdf failed.")

def test_check_access_token():
    load_dotenv()
    token = os.getenv("GOOGLE_ACCESS_TOKEN")

    assert type(check_access_token(token)) == bool

if __name__ == "__main__":
    # test_img_bytes_to_base64()
    # test_jpg_to_pdf()
    test_check_access_token()