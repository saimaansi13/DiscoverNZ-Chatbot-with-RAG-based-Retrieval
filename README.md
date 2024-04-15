<img width="750" alt="title page" src="https://github.com/saimaansi13/DiscoverNZ-Chatbot-with-RAG-based-Retrieval/assets/125540201/b93b6f01-fc46-41e0-9e71-a3ab4f53eb98">


# Preview
![chatbot - GIF](https://github.com/saimaansi13/DiscoverNZ-Chatbot-with-RAG-based-Retrieval/assets/125540201/45fa51c7-a3be-4766-8cb5-2baf09d68ddb)

## Overview
DiscoverNZ is an AI-driven conversational chatbot engineered to enhance the exploration experience of New Zealand. Leveraging state-of-the-art large language models (LLMs) and advanced information retrieval methodologies, this project facilitates dynamic interactions with users, offering tailored recommendations for tourist attractions, cultural immersion, and comprehensive insights into New Zealand's diverse offerings. Through the integration of natural language processing (NLP) techniques and advanced retrieval algorithms, DiscoverNZ delivers personalized and contextually relevant information to users.

## Business Problem 
Before the pandemic, tourism in New Zealand flourished, contributing NZ$44.7 billion to the economy and supporting over 230,000 jobs in 2019. However, the sector faced an 82% decrease in international visitor arrivals and a drop in total expenditure to NZ$6.8 billion in 2020. Despite recovery efforts, international tourists spent approximately NZ$1.9 billion in 2022, still below pre-pandemic levels. The industry also saw an 8.2% decrease in hotel and resort establishments from 2019 to 2022.

<img width="550" alt="chatbot - business problem" src="https://github.com/saimaansi13/DiscoverNZ-Chatbot-with-RAG-based-Retrieval/assets/125540201/7df93bb6-7b18-40dc-9e59-527f1046d0b0">

## Structure and Technicalities
### 1) Data Collection
The data collection process involved thorough research conducted on official travel and tourism websites of New Zealand, including sources such as NewZealand.com. Information relevant to the project was meticulously gathered from these websites, encompassing a wide range of tourism related content, such as destination guides, activity listings, and accommodation information.

![Untitled design](https://github.com/saimaansi13/DiscoverNZ-Chatbot-with-RAG-based-Retrieval/assets/125540201/cc379366-a195-44fd-bc98-18757a7f4dc9)

### 2) Data Storage 
Following the collection, the gathered data was meticulously organized and segregated into multiple PDF documents, each dedicated to a specific aspect of the travel and tourism landscape, such as activities, accommodation, transportation etc. Subsequently, all the sorted PDFs were consolidated into a single document for streamlined access, simplifying the uploading and utilization process.

### 2) Text Splitting 
To facilitate efficient processing, the raw text from the PDF documents is split into smaller chunks. This is done using the RecursiveCharacterTextSplitter module from langchain, which divides the text into chunks of specified size with an overlap.
![cb1](https://github.com/saimaansi13/DiscoverNZ-Chatbot-with-RAG-based-Retrieval/assets/125540201/4142f00e-2893-4e9c-ac4e-23a7cdad5d04)

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

