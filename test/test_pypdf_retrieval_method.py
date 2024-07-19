from src.pypdf_retrieval_method import PyPdfRetrievalMethod

def test_query2vector():
    pypdf_retrieval_method = PyPdfRetrievalMethod()
    queries = ["query1", "query2"]
    response = pypdf_retrieval_method.query2vector(queries)
    print(response)

def main():
    test_query2vector()

if __name__ == "__main__":
    main()