from typing import Dict, List
from dotenv import load_dotenv
import base64

from openai import OpenAI
import pypdf

from src.retrieval_method import RetrievalMethod
from src.utils import img_bytes_to_base64, jpg_to_pdf

class PyPdfRetrievalMethod(RetrievalMethod):
    """
    Class for Retrieval Methods using PyPDF2.
    """

    def __init__(self):
        pass

    def query2vector(self, queries: List[str]) -> Dict[str, List]:
        """
        Convert a query to a vector.
        """
        load_dotenv()

        client = OpenAI()

        # Rate-Limit = 3.000 / min
        response = client.embeddings.create(
            input=queries,
            model="text-embedding-3-large"
        )

        try:
            data = response.data
            embeddings = map(lambda x: x.embedding, data)
            return dict(zip(queries, embeddings))
        except Exception as e:
            return {}

    def image2vector(self, images: List[Dict[str,str]]) -> Dict[str, List]:
        """
        Convert an image to a vector.
        images: List of Dict. Dict keys: image_filename and image_base64
        """

        # Add key and value image_text to each Dict
        # Remove image from images if error
        valid_images = []
        for image in images:
            try:
                # Image to pdf
                pdf_path = jpg_to_pdf(image['image_filename'])
                if pdf_path:
                    # Load the PDF file
                    r = pypdf.PdfReader(pdf_path)

                    # Extract the text from the first page (image = 1 page)
                    image['image_text'] = r.pages[0].extract_text()

                    # Add image to valid_images
                    valid_images.append(image)
            except:
                raise Exception(f"Error reading PDF file {image['image_filename']}")  

        print(valid_images)    
        # Text embedding
        load_dotenv()
        client = OpenAI()
        text = list(map(lambda x: x['image_text'], valid_images))

        print(len)

        # Rate-Limit = 3.000 / min
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-large"
        )

        try:
            data = response.data
            embeddings = map(lambda x: x.embedding, data)
            return dict(zip(text, embeddings))
        except Exception as e:
            return {}