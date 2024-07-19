from src.utils import image_PIL_to_base64

from PIL import Image


def test_image_PIL_to_base64():

    img = Image.new('RGB', (100, 100), color = 'red')

    img_base64 = image_PIL_to_base64(img)

    assert isinstance(img_base64, str)

    print("Test image_PIL_to_base64 passed")


if __name__ == "__main__":
    test_image_PIL_to_base64()