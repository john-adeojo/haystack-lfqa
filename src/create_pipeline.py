from haystack.nodes import EmbeddingRetriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.pipelines import Pipeline


# retreiver relevant docs
def make_document_qa_pipeline(document_store):
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/all-mpnet-base-v2"
    )
    document_store.update_embeddings(retriever)
    
    # read relevant docs
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")
    
    document_qa = ExtractiveQAPipeline(reader=reader, retriever=retriever)
    return document_qa