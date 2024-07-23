import os
import time
from typing import Dict, List
import requests
from dotenv import load_dotenv
from typer import Option
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from src.retrieval_method import RetrievalMethod

class GoogleMultimodalEmbeddingRetrievalMethod(RetrievalMethod):
    """
    Class for Retrieval Methods using Google Multimodal Embedding.
    """

    def __init__(self):
        pass

    def query2vector(self, queries: List[str]) -> List[Dict[str, List]]:
        """
        Convert a query to a vector.
        """
        load_dotenv()

        LOCATION = os.getenv("GOOGLE_LOCATION")
        PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
        ACCESS_TOKEN = os.getenv("GOOGLE_ACCESS_TOKEN")

        URL = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}" \
            + f"/locations/{LOCATION}/publishers/google/models/multimodalembedding@001:predict"
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {ACCESS_TOKEN}"
        }

        def get_query_embedding(query):
            payload = {
                "instances": [
                    {
                        "text" : query
                    }
                ]
            }
            try:
                response = requests.post(URL, json=payload, headers=headers)
                response.raise_for_status()  # Check for any HTTP errors
                query_vector.append({
                    "query": query,
                    "query_vector": response.json()['predictions'][0]['textEmbedding']
                })
            except Exception as e:
                print(f"An error occurred: {e}")

        logging.info(f'query2vector: start for {len(queries)} queries')
        start_time = time.time()
        query_vector = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(get_query_embedding, query) for query in queries]

        logging.info(f'query2vector: {len(query_vector)} queries embedded in {time.time() - start_time} seconds')
        return query_vector

    def image2vector(self, images: List[Dict[str,str]]) -> Dict[str, List]:
        """
        Convert an image to a vector.
        images: List of Dict. Dict keys: image_filename and image_base64
        """
        load_dotenv()

        LOCATION = os.getenv("GOOGLE_LOCATION")
        PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
        ACCESS_TOKEN = os.getenv("GOOGLE_ACCESS_TOKEN")

        URL = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}" \
            + f"/locations/{LOCATION}/publishers/google/models/multimodalembedding@001:predict"
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {ACCESS_TOKEN}"
        }

        def get_image_embedding(image):
            payload = {
                "instances": [
                    {
                        "image" : {
                            "bytesBase64Encoded" : image['image_base64']
                        }                    
                    }
                ]
            }
            try:
                response = requests.post(URL, json=payload, headers=headers)
                response.raise_for_status()
                image_vector.append({
                    "image_filename": image['image_filename'],
                    "image_vector": response.json()['predictions'][0]['imageEmbedding']
                })
            except Exception as e:
                print(f"An error occurred: {e}")

        logging.info(f'image2vector: start for {len(images)} images.')
        start_time = time.time()
        image_vector = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(get_image_embedding, image) for image in images]

        logging.info(f'image2vector: {len(image_vector)} images embedded in {time.time() - start_time} seconds')
        return image_vector