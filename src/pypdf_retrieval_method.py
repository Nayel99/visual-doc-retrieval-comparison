from typing import Dict, List
from dotenv import load_dotenv
import base64

from openai import OpenAI
import pypdf

from src.retrieval_method import RetrievalMethod

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
        images_base64: Dict with keys image_name and image_base64
        """
        try:
            for image in images:
                r = pypdf.PdfReader(stream=image['image_base64'])
                image['image_text'] = r.pages[0].extract_text()
            print("A")
        except:
            print(f"Error reading PDF file {images[0]['image_base64']}")

        return [0.0, 1.0, 0.0]