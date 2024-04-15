<img width="750" alt="title page" src="https://github.com/saimaansi13/DiscoverNZ-Chatbot-with-RAG-based-Retrieval/assets/125540201/b93b6f01-fc46-41e0-9e71-a3ab4f53eb98">


# Preview
![chatbot - GIF](https://github.com/saimaansi13/DiscoverNZ-Chatbot-with-RAG-based-Retrieval/assets/125540201/45fa51c7-a3be-4766-8cb5-2baf09d68ddb)

## Overview
DiscoverNZ is an AI-driven conversational chatbot engineered to enhance the exploration experience of New Zealand. Leveraging state-of-the-art large language models (LLMs) and advanced information retrieval methodologies, this project facilitates dynamic interactions with users, offering tailored recommendations for tourist attractions, cultural immersion, and comprehensive insights into New Zealand's diverse offerings. Through the integration of natural language processing (NLP) techniques and advanced retrieval algorithms, DiscoverNZ delivers personalized and contextually relevant information to users.

## Business Problem 
The tourism industry in New Zealand, once flourishing, faced a significant downturn due to the COVID-19 pandemic. Despite efforts to recover, challenges such as fluctuating visitor numbers and uncertain market conditions persist. DiscoverNZ seeks to be a part of the solution by offering a streamlined approach to assist travelers in navigating the evolving landscape of New Zealand's tourism sector.


## Structure and Technicalities
### 1) Document Loading and Parsing 
Started by loading the content from a PDF document (Nz_Info_Final.pdf). This is achieved using the PyPDFLoader module from langchain, which reads the content of the PDF file and prepares it for further processing.
```python
loader = PyPDFLoader("/content/Nz_Info_Final.pdf")
raw_documents = loader.load()
```
### 2) Text Splitting 
To facilitate efficient processing, the raw text from the PDF documents is split into smaller chunks. This is done using the RecursiveCharacterTextSplitter module from langchain, which divides the text into chunks of specified size with an overlap.
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, chunk_overlap=20, length_function=len
)
documents = text_splitter.split_documents(raw_documents)
```
### 3) Embeddings Generation
Next, I generated embeddings for each text chunk. Embeddings are numerical representations of text that capture semantic meaning. In this implementation, OpenAI embeddings are used, which are pre-trained on a large corpus of text data.
```python
openai_api_key = " " #replace with your own api key
embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
```
### 4) Vector Store Creation
The embeddings are then used to create a vector store using the FAISS library. FAISS is a library for efficient similarity search and clustering of dense vectors.
```python
db = FAISS.from_documents(documents, embeddings_model)
retriever = db.as_retriever()
```
### 5) Language Model Initialization
The chatbot utilizes a language model to understand and respond to user queries. In this implementation, the GPT-3.5 language model from OpenAI is used.
```python
llm_src = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k", openai_api_key=openai_api_key)
```
### 6) Conversational Retrieval Chain
The chatbot employs a conversational retrieval chain to provide relevant responses to user queries. This chain combines the language model with the vector store to retrieve and rank relevant documents based on their similarity to the user query.
```python
retrieval_qa = ConversationalRetrievalChain.from_llm(
    llm_src,
    retriever,
    return_source_documents=True,
)
```
### 7) Gradio Interface 
Finally, the chatbot is deployed using Gradio, which provides a simple web interface for users to interact with the chatbot.
```python
iface = gr.Interface(fn=answer_question, inputs="text", outputs="text", title="DiscoverNZ",
                     theme="light", description='Kia Ora! Ready to explore New Zealand?')
iface.launch(inline=True)
```

## Acknowledgements
- **OpenAI** for providing the GPT-3.5 language model.
- **gradio** for the easy-to-use interface creation.
- **FAISS** for efficient similarity search and clustering of dense vectors.

