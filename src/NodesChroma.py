import openai
import chromadb

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.readers.box import BoxReader, BoxReaderTextExtraction

from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.schema import NodeWithScore
from typing import Optional
from llama_index.core.query_engine import RetrieverQueryEngine


from modules.app_config import AppConfig
from modules.box import get_box_client
from modules.retriver import VectorDBRetriever


def main():
    app_config = AppConfig()

    openai.api_key = app_config.get("OPENAI_API_KEY")

    # Load data using a Box reader
    box_client = get_box_client()

    # Using BoxReaderTextExtraction
    box_reader = BoxReaderTextExtraction(box_client)
    documents = box_reader.load_data(file_ids=[app_config.get("DEMO_DOC_ID")])

    # # Using BoxReader
    # box_reader = BoxReader(box_client)
    # documents = box_reader.load_data(file_ids=[app_config.get("DEMO_DOC_ID")])

    # Document metadata workaround
    # TODO: Fix this in the BoxReader API
    for box_document in documents:
        box_document.metadata["created_at"] = box_document.metadata["created_at"].isoformat()
        box_document.metadata["modified_at"] = box_document.metadata["modified_at"].isoformat()
        box_document.metadata["content_created_at"] = box_document.metadata["content_created_at"].isoformat()
        box_document.metadata["content_modified_at"] = box_document.metadata[
            "content_modified_at"
        ].isoformat()

    # Setup model and LLM
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    model_url = (
        "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"
    )
    llm = LlamaCPP(
        # You can pass in the URL to a GGML model to download it automatically
        model_url=model_url,
        # optionally, you can set the path to a pre-downloaded model instead of model_url
        model_path=None,
        temperature=0.1,
        max_new_tokens=256,
        # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
        context_window=3900,
        # kwargs to pass to __call__()
        generate_kwargs={},
        # kwargs to pass to __init__()
        # set to at least 1 to use GPU
        model_kwargs={"n_gpu_layers": 1},
        verbose=False,
    )

    # Initialize Chroma Vector Store
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("quickstart")

    # set up ChromaVectorStore and load in data
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Use a Text Splitter to Split Documents
    text_parser = SentenceSplitter(
        chunk_size=1024,
        # separator=" ",
    )

    text_chunks = []
    # maintain relationship with source doc index, to help inject doc metadata in (3)
    doc_indexes = []
    for doc_idx, doc in enumerate(documents):
        cur_text_chunks = text_parser.split_text(doc.text)
        text_chunks.extend(cur_text_chunks)
        doc_indexes.extend([doc_idx] * len(cur_text_chunks))

    # Manually Construct Nodes from Text Chunks
    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(
            text=text_chunk,
        )
        src_doc = documents[doc_indexes[idx]]
        node.metadata = src_doc.metadata
        nodes.append(node)

    # Generate Embeddings for each Node
    for node in nodes:
        node_embedding = embed_model.get_text_embedding(node.get_content(metadata_mode="all"))
        node.embedding = node_embedding

    # Load Nodes into a Vector Store
    vector_store.add(nodes)

    # Build Retrieval Pipeline from Scratch
    query_str = "what considerations should I have if I want to go diving?"

    # Generate a Query Embedding
    query_embedding = embed_model.get_query_embedding(query_str)

    # Query the Vector Database
    query_mode = "default"
    # query_mode = "sparse"
    # query_mode = "hybrid"

    vector_store_query = VectorStoreQuery(
        query_embedding=query_embedding, similarity_top_k=2, mode=query_mode
    )

    # returns a VectorStoreQueryResult
    query_result = vector_store.query(vector_store_query)

    # Parse Result into a Set of Nodes
    nodes_with_scores = []
    for index, node in enumerate(query_result.nodes):
        score: Optional[float] = None
        if query_result.similarities is not None:
            score = query_result.similarities[index]
        nodes_with_scores.append(NodeWithScore(node=node, score=score))

    # Put into a Retriever
    retriever = VectorDBRetriever(vector_store, embed_model, query_mode="default", similarity_top_k=2)

    # Plug this into our RetrieverQueryEngine to synthesize a response
    query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)

    query_str = "What equipment do I need to carry with me?"
    response = query_engine.query(query_str)

    print("=" * 80)
    print(f"Query: {query_str}")
    print("-" * 80)
    print("Query result:")
    print(response)
    print("-" * 80)
    print()


if __name__ == "__main__":
    main()
