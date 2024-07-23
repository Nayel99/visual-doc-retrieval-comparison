import time
from typing import Dict, List
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI

from retrieval_method import RetrievalMethod

from dotenv import load_dotenv

class Gpt40CaptioningRetrievalMethod(RetrievalMethod):
    """
    Class for Retrieval Methods using GPT-4.0 Captioning.
    """

    def __init__(self):
        pass

    def query2vector(self, queries: List[str]) -> Dict[str, List]:
        """
        Convert a query to a vector.
        """
        logging.info(f'query2vector: start for {len(queries)} queries')

        load_dotenv()
        client = OpenAI()

        start_time = time.time()
        response = client.embeddings.create(
            input=queries,
            model="text-embedding-3-large"
        )

        try:
            data = response.data
            embeddings = list(map(lambda x: x.embedding, data))
            logging.info(f'query2vector: {len(embeddings)} queries embedded in {time.time() - start_time} seconds')

            output = [
                {
                    "query": query,
                    "query_vector": embedding
                }
                for query, embedding in zip(queries, embeddings)
            ]
            return output
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise e

    def image2vector(self, images: List[Dict[str,str]]) -> List[Dict[str, List]]:
        """
        Convert an image to a vector.
        images: List of Dict. Dict keys: image_filename and image_base64
        """

        load_dotenv()
        client = OpenAI()

        logging.info(f'image2vector: start for {len(images)}.')

        def describe_image(client, image):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user",
                        "content": [
                            {"type": "text", "text": "Describe, in english, the image below."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpg;base64,{image['image_base64']}",
                                },
                            },
                        ],
                        }
                    ]
                )
                image['image_description'] = response.choices[0].message.content
            except Exception as e:
                logging.error(f"An error occurred: {e}")

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(describe_image, client, image) for image in images]

            for future in as_completed(futures):
                future.result()

        logging.info(f'image2vector: Embedding description starts.')
        start_time = time.time()
        response = client.embeddings.create(
            input=list(map(lambda x: x['image_description'], images)),
            model="text-embedding-3-large"
        )

        try:
            data = response.data
            embeddings = list(map(lambda x: x.embedding, data))
            logging.info(f"image2vector: {len(embeddings)} queries embedded in {time.time() - start_time} seconds")

            output = [
                {
                    "image_filename": image['image_filename'],
                    "image_vector": embedding
                }
                for image, embedding in zip(images, embeddings)
            ]
            return output
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return {}