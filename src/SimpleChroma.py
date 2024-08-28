import openai
import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.readers.box import BoxReaderTextExtraction
from modules.app_config import AppConfig
from modules.box import get_box_client


def main():
    app_config = AppConfig()

    # Load data using a Box reader
    box_client = get_box_client()
    box_reader = BoxReaderTextExtraction(box_client)
    documents = box_reader.load_data(file_ids=[app_config.get("DEMO_DOC_ID")])

    # Document metadata workaround
    # TODO: Fix this in the BoxReader API
    for box_document in documents:
        box_document.metadata["created_at"] = box_document.metadata["created_at"].isoformat()
        box_document.metadata["modified_at"] = box_document.metadata["modified_at"].isoformat()
        box_document.metadata["content_created_at"] = box_document.metadata["content_created_at"].isoformat()
        box_document.metadata["content_modified_at"] = box_document.metadata[
            "content_modified_at"
        ].isoformat()

    # Setup model
    openai.api_key = app_config.get("OPENAI_API_KEY")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    # Initialize ChromaDB (Vector store)
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("quickstart")

    # Set up ChromaVectorStore and load in data
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create a Chroma Index
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed_model
    )

    # Query Data
    query_engine = index.as_query_engine()
    query = "What to do in case of emergency?"
    response = query_engine.query(query)

    print("=" * 80)
    print(f"Query: {query}")
    print("-" * 80)
    print("Query result:")
    print(response)
    print("-" * 80)
    print()


if __name__ == "__main__":
    main()
