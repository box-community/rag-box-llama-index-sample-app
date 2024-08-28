# Box and Llama-Index sample RAG application

This repository contains two Python scripts that demonstrate how to build a document query and retrieval system using `ChromaDB` as a vector store and `HuggingFace` or `Llama-CPP` models for embeddings and language modeling. The system leverages Box's text extraction capability to load documents, embeds them using a HuggingFace model, and queries them using a retrieval pipeline to provide relevant responses based on the input query.

## Overview

The repository includes two main scripts:

1. **Script 1: Simple Document Query System using ChromaDB**  
   This script sets up a basic document query system using `ChromaDB` for vector storage and a HuggingFace model for embeddings. It loads documents from Box, creates embeddings, stores them in a vector database, and uses an OpenAI model to query the database.

2. **Script 2: Advanced Document Query System with Llama-CPP**  
   This script builds upon the first by adding a more advanced retrieval pipeline that leverages `Llama-CPP` for natural language processing and retrieval query synthesis. It includes features like text splitting, embedding generation, node construction, and an enhanced retrieval process.

## Requirements

To run the code, ensure you have the following dependencies installed.
You can install these dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Configuration

Before running the scripts, you need to configure the application settings:

1. **Box Configuration**  
   Ensure you have a Box developer account and set up an application for Client Credentials Grant (CCG). Common configurations include:
   - App + Enterprise Access
   - Manage AI
   - Generate user access tokens
   - **Remember to Authorize your app**

2. **API Keys and Environment Variables**
   Copy the sample.env file to .env and update the values with your Box and OpenAI API keys.

## Code Structure

### 1. **Simple Document Query System (SimpleChroma.py)**

- **Steps**:
  1. Load document data from Box using `BoxReaderTextExtraction`.
  2. Convert document metadata to a compatible format.
  3. Set up the embedding model using `HuggingFaceEmbedding`.
  4. Initialize ChromaDB as a vector store.
  5. Create an index from the documents using the embeddings.
  6. Query the index and print the results.

### 2. **Advanced Document Query System with Llama-CPP (NodesChroma.py)**

- **Steps**:
  1. Load document data from Box using `BoxReaderTextExtraction`.
  2. Convert document metadata to a compatible format.
  3. Set up the embedding model using `HuggingFaceEmbedding`.
  4. Initialize the LLM using `LlamaCPP`.
  5. Use a text splitter to split documents into chunks.
  6. Construct nodes from the text chunks and generate embeddings.
  7. Load the nodes into a vector store.
  8. Build a retrieval pipeline and synthesize a response using `RetrieverQueryEngine`.
  9. Query the index and print the results.

## Usage

To run the scripts, execute them from the command line:

```bash
python SimpleChroma.py
```

Sample output:
```
================================================================================
Query: What to do in case of emergency?
--------------------------------------------------------------------------------
Query result:
If you become distressed on the surface during scuba diving, the appropriate action to take is to immediately drop your weight belt, inflate your buoyancy compensator (BC) for flotation, and signal for assistance.
--------------------------------------------------------------------------------
```

```bash
python script2.py
```

Sample Output:
```
================================================================================
Query: What equipment do I need to carry with me?
--------------------------------------------------------------------------------
Query result:


Based on the given context information, the answer to the query "What equipment do I need to carry with me?" is:

1. A snorkel
2. A submersible pressure gauge
3. A depth gauge
4. An alternate air source
5. A buoyancy compensator (BC) with an inflator hose and regulator.

This information can be found in the text under the section "Equipment Requirements".
--------------------------------------------------------------------------------
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any bugs or feature requests.
