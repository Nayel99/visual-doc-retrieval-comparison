from src.google_multimodalembedding_retrieval_method import GoogleMultimodalEmbeddingRetrievalMethod

import base64

def test_query_to_vector():
    query = "test"
    google_multimodal_embedding_retrieval_method = GoogleMultimodalEmbeddingRetrievalMethod()
    response = google_multimodal_embedding_retrieval_method.query2vector([query])
    print(response)

def test_image_to_vector():
    google_multimodal_embedding_retrieval_method = GoogleMultimodalEmbeddingRetrievalMethod()
    with open('test/data/img_test.jpg', 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    input = [{"image_filename": "img_test.jpg", "image_base64": encoded_image}]
    response = google_multimodal_embedding_retrieval_method.image2vector(input)

    assert response[0]['image_filename'] == "img_test.jpg"
    assert len(response[0]['vector']) > 0

    print("Test image_to_vector passed.")



def main():
    # test_query_to_vector()
    test_image_to_vector()

if __name__ == "__main__":
    main()