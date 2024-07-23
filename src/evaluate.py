import typer
from typing import Annotated
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pytrec_eval
import numpy as np
import logging
import os
from dotenv import load_dotenv

from pypdf_retrieval_method import PyPdfRetrievalMethod
from google_multimodalembedding_retrieval_method import GoogleMultimodalEmbeddingRetrievalMethod
from gpt_4_0_captioning_retrieval_method import Gpt40CaptioningRetrievalMethod


from utils import img_bytes_to_base64, check_access_token
import time
from enum import Enum

app = typer.Typer(
    help="CLI for evaluating retrievers on the ViDoRe benchmark.",
    no_args_is_help=True,
)

def calculate_mean_ndcg(df):
    """
    Calculate the mean NDCG for a DataFrame.
    """
    def get_ground_truth(df):
        ground_truth = {}
        for _, row in df.iterrows():
            query = row['query']
            image = row['image_filename']
            
            if query not in ground_truth:
                ground_truth[query] = {}

            # If an image is associated with a query in the DataFrame, assume it is relevant (1)
            ground_truth[query][image] = 1

        return ground_truth

    # Prepare the results
    similarity_matrix = cosine_similarity(df['query_vector'].tolist(), df['image_vector'].tolist())

    results = {}
    for i, query in enumerate(df['query']):
        results[query] = {df['image_filename'][j]: float(similarity_matrix[i, j]) for j in range(len(df))}

    # Prepare the evaluator
    evaluator = pytrec_eval.RelevanceEvaluator(get_ground_truth(df), {'ndcg'})

    # Evaluate the results
    evaluation = evaluator.evaluate(results)

    # Calculate the mean NDCG
    ndcg_scores = [query_measures['ndcg'] for query_measures in evaluation.values()]
    mean_ndcg = np.mean(ndcg_scores)

    return mean_ndcg

class RetrievalMethod(Enum):
    GOOGLE_MULTIMODAL_EMBEDDING = "google_multimodal_embedding"
    GPT_4_0_CAPTIONING = "gpt_4_0_captioning"

@app.command()
def evaluate(
    hf_dataset_link_parquet: Annotated[str,typer.Option(help="Hugging Face dataset link")] = "hf://datasets/vidore/infovqa_test_subsampled/data/test-00000-of-00001.parquet",
    nbr_rows : Annotated[int,typer.Option(help="Number of rows to evaluate")] = 3,
    method : Annotated[RetrievalMethod,typer.Option(help="Retrieval method to evaluate")] = "gpt_4_0_captioning"
):
    """
    Evaluate a retrieval method with an Hugging Face dataset.
    """

    if method == RetrievalMethod.GOOGLE_MULTIMODAL_EMBEDDING:
        retrieval_method = GoogleMultimodalEmbeddingRetrievalMethod()
        load_dotenv()
        if not check_access_token(os.getenv("GOOGLE_ACCESS_TOKEN")):
            raise ValueError("Invalid google access token." \
                            + "Run 'gcloud auth print-access-token' and copy paste the token in the .env file (GOOGLE_ACCESS_TOKEN).")
    elif method == RetrievalMethod.GPT_4_0_CAPTIONING:
        retrieval_method = Gpt40CaptioningRetrievalMethod()
    else:
        raise ValueError("Invalid retrieval method")

    logging.info(f"Start getting data from {hf_dataset_link_parquet}.")
    start_time = time.time()
    df = pd.read_parquet(hf_dataset_link_parquet)
    logging.info(f"DataFrame read in {time.time() - start_time} seconds")

    df = df[:nbr_rows]

    # Keep only useful columns
    df = df[['query', 'image', 'image_filename']]

    # Add base64 img column
    df['img_base64'] = df['image'].apply(img_bytes_to_base64)

    # Add query2vector column
    query_vectors = retrieval_method.query2vector(df['query'].tolist())
    query_vectors_df = pd.DataFrame(query_vectors, columns=['query', 'query_vector'])
    df = pd.merge(df, query_vectors_df, on='query')

    # Add img column
    def to_dict(x):
        return {'image_filename': x['image_filename'], 'image_base64': img_bytes_to_base64(x['image']['bytes'])}
    img_dicts = df.apply(to_dict, axis=1).tolist()
    img_vectors = retrieval_method.image2vector(img_dicts)

    img_vectors_df = pd.DataFrame(img_vectors, columns=['image_filename', 'image_vector'])
    df = pd.merge(df, img_vectors_df, on='image_filename')
    
    logging.info(f"Mean NGCD : {calculate_mean_ndcg(df)}")
    logging.info(f"Program ended in {round(time.time() - start_time,2)} seconds")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app()