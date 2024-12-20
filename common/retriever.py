
from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer


def get_bm25_retriever(nodes, language):
# We can pass in the index, docstore, or list of nodes to create the retriever
    return BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=2,
        # Optional: We can pass in the stemmer and set the language for stopwords
        # This is important for removing stopwords and stemming the query + text
        # The default is english for both
        stemmer=Stemmer.Stemmer(language),
        language=language,
    )