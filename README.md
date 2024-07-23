# Description

The aim of this project is to evaluate different methods for retrieving visual documents.

An article is forthcoming detailing the work carried out.

This work is inspired by work carried out in greater detail: https://huggingface.co/vidore

This repo tests different methods of retrieval.

Available methods :
- Google Multimodal Embedding
- Gpt4-0 Description (captioning) + ‘text-embedding-3-large’ embedding

Dataset available :
https://huggingface.co/datasets/vidore/infovqa_test_subsampled

The method used to compare the methods is NDCG. 
The average is displayed for each query.

# How to use

Clone the repository

```
git clone https://github.com/Nayel99/visual-doc-retrieval-comparison.git
cd visual-doc-retrieval-comparison
````

Creating a virtual environment and activate it

```
python -m venv venv
source venv/bin/activate
```

Install depencies

```
pip install -r requirements.txt
```

Update PYTHONPATH

```
export PYTHONPATH=$(pwd)
```

Create a .env file based on .env.example.

For multimodalembedding from Google, follow the instructions in [this documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-multimodal-embeddings?hl=fr). 

In particular, you need to activate the Vertex AI API.

To get your google access token (to copy/paste into your .env file) :

```
gcloud auth print-access-token
```

Try the code with: (gpt4-O is the default method)

```
python src/evaluate
```

# Discussion

Please do not hesitate to contact me to discuss the project: ferainayel@gmail.com