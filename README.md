Creating RAG using Langchain and Mistral LLM, on articles pertaining to US Census
==========================================

Description
-----------

This project analyzes health insurance coverage data from the U.S. Census Bureau using Langchain, a Python package for natural language processing tasks. The project includes functionalities to load PDF documents, extract text, embed text using Hugging Face embeddings, perform similarity searches, and retrieve relevant information based on user queries.

Installation
------------

To install the required packages, run the following command:

bash

Copy code

`pip install -r requirements.txt`

The `requirements.txt` file includes the following packages:

*   `langchain_community`
*   `langchain`
*   `faiss-cpu`
*   `numpy`
*   `python-dotenv`

Usage
-----

### Step 1: Load Libraries and Packages

python

Copy code

`from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader from langchain.text_splitter import RecursiveCharacterTextSplitter from langchain_community.vectorstores import FAISS from langchain_community.embeddings import HuggingFaceBgeEmbeddings from langchain.prompts import PromptTemplate from langchain.chains import RetrievalQA`

### Step 2: Load PDF Documents

python

Copy code

`# Load PDF documents loader = PyPDFDirectoryLoader("/path/to/pdf_directory") documents = loader.load()  # Split documents into smaller chunks text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) final_documents = text_splitter.split_documents(documents)`

### Step 3: Embed Text using Hugging Face Embeddings

python

Copy code

`# Embed text using Hugging Face Bge Embeddings huggingface_embeddings = HuggingFaceBgeEmbeddings(     model_name="BAAI/bge-small-en-v1.5",     model_kwargs={"device": "cpu"},     encode_kwargs={'normalize_embeddings': True} )`

### Step 4: Create Vector Store

python

Copy code

`# Create Vector Store VectorStore = FAISS.from_documents(final_documents[:250], huggingface_embeddings)`

### Step 5: Query Using Similarity Search

python

Copy code

`# Perform similarity search query = "WHAT IS HEALTH INSURANCE COVERAGE?" relevant_documents = VectorStore.similarity_search(query) print(relevant_documents[0].page_content)`

### Step 6: Retrieve Relevant Answers

python

Copy code

`# Retrieve relevant answers retriever = VectorStore.as_retriever(search_type="similarity", search_kwargs={"k": 3}) print(retriever)  # Connect to Langchain LLM import os from dotenv import load_dotenv from langchain_community.llms import HuggingFaceHub  load_dotenv() api_key = os.environ.get("HF_API_TOKEN") hf = HuggingFaceHub(     repo_id="mistralai/Mistral-7B-Instruct-v0.2",     model_kwargs={"temperature": 0.1, "max_length": 500},     huggingfacehub_api_token=api_key, ) query = "What is health insurance coverage?" print(hf.invoke(query))`

### Step 7: Generate Helpful Answers

python

Copy code

`# Generate helpful answers prompt_template = """ Use the following piece of context to answer the question asked. Please try to provide the answer only based on the context  {context} Question: {question}  Helpful Answers:  """ prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"]) retrievalQA = RetrievalQA.from_chain_type(     llm=hf,     chain_type="stuff",     retriever=retriever,     return_source_documents=True,     chain_type_kwargs={"prompt": prompt} ) query = "DIFFERENCES IN THE UNINSURED RATE BY STATE IN 2022" result = retrievalQA.invoke({"query": query}) print(result['result'])`

License
-------

This project is licensed under the [MIT License](LICENSE).
