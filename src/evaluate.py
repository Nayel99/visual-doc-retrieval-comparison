import typer
from typing import Annotated

import pandas as pd

from pypdf_retrieval_method import PyPdfRetrievalMethod

from utils import img_bytes_to_base64

app = typer.Typer(
    help="CLI for evaluating retrievers on the ViDoRe benchmark.",
    no_args_is_help=True,
)

@app.command()
def evaluate(
    hf_dataset_link_parquet: Annotated[str,typer.Option(help="Hugging Face dataset link")] = "hf://datasets/vidore/infovqa_test_subsampled/data/test-00000-of-00001.parquet"
):
    """
    Evaluate a retrieval method with an Hugging Face dataset.
    """

    method = PyPdfRetrievalMethod()

    df = pd.read_parquet(hf_dataset_link_parquet)

    df = df[:3]

    # Add base64 img column
    # df['img_base64'] = df['img'].apply(img_bytes_to_base64)

    # Add query2vector column
    # query_vectors = method.query2vector(df['query'].tolist())
    # query_vectors_df = pd.DataFrame(query_vectors.items(), columns=['query', 'query_vector'])
    # df = pd.merge(df, query_vectors_df, on='query')

    # Add img column
    def to_dict(x):
        return {'image_filename': x['image_filename'], 'image_base64': img_bytes_to_base64(x['image']['bytes'])}
    img_dicts = df.apply(to_dict, axis=1).tolist()
    img_vectors = method.image2vector(img_dicts)
    img_vectors_df = pd.DataFrame(img_vectors.items(), columns=['image_filename', 'image_vector'])
    df = pd.merge(df, img_vectors_df, on='image_filename')
    print(df.head())

if __name__ == "__main__":
    app()