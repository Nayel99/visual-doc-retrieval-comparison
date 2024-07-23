import base64
import img2pdf
import os
import requests
import logging

def img_bytes_to_base64(data: dict | bytes) -> str:
    """
    Convert a dict with 'bytes' key to base64 string or a bytes object to a base64 string.
    """
    if isinstance(data, bytes):
        return base64.b64encode(data).decode('utf-8') # Convert bytes to base64   
    else:  
        return base64.b64encode(data['bytes']).decode('utf-8')

def jpg_to_pdf(jpg_file_path: str, pdf_file_path: str = "to_delete/"):
    """
    Convert a JPG file from an online link or base64 string to a PDF file.
    Return the local path of the PDF file. None if error.
    """
    try:
        if jpg_file_path.startswith("http"):
            # JPG file path is an online link
            response = requests.get(jpg_file_path)
            response.raise_for_status()  # Check for any HTTP errors
            jpg_data = response.content
        else:
            # JPG file path is a base64 string
            jpg_data = base64.b64decode(jpg_file_path)

        pdf_data = img2pdf.convert(jpg_data)

        pdf_path = os.path.join(pdf_file_path, "to_delete.pdf")
        with open(pdf_path, 'wb') as pdf_file:
            pdf_file.write(pdf_data)

        return pdf_path
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return None
    
def check_access_token(token : str) -> bool:
    url = f"https://www.googleapis.com/oauth2/v1/tokeninfo?access_token={token}"
    response = requests.get(url)
    if response.status_code == 200:
        return True
    else:
        return False