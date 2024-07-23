# from src.evaluate import calculate_metrics
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pytrec_eval
import numpy as np

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

def test_get_ground_truth():
    data = {
        'query': ['query1', 'query1', 'query2', 'query2', 'query3'],
        'query_vector': [[0.1, 0.2], [0.1, 0.2], [0.3, 0.4], [0.3, 0.4], [0.5, 0.6]],
        'image_filename': ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg', 'image5.jpg'],
        'image_vector': [[0.5, 0.6], [0.7, 0.8], [0.9, 1.0], [1.1, 1.2], [1.3, 1.4]]
    }
    df = pd.DataFrame(data)

    ground_truth = get_ground_truth(df)

    assert ground_truth == {
        'query1': {'image1.jpg': 1, 'image2.jpg': 1},
        'query2': {'image3.jpg': 1, 'image4.jpg': 1},
        'query3': {'image5.jpg': 1}
    }

def test_calculate_metrics():
    pass

def main():

    test_get_ground_truth()

    data = {
        'query': ['query1', 'query1', 'query2', 'query2', 'query3'],
        'query_vector': [[0.1, 0.2], [0.1, 0.2], [0.3, 0.4], [0.3, 0.4], [0.5, 0.6]],
        'image_filename': ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg', 'image5.jpg'],
        'image_vector': [[0.5, 0.6], [0.7, 0.8], [0.9, 1.0], [1.1, 1.2], [1.3, 1.4]]
    }
    df = pd.DataFrame(data)
    print(calculate_mean_ndcg(df))

if __name__ == "__main__":
    main()