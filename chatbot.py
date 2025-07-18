from fastapi import FastAPI,Request,Form
# from twilio.twiml.messaging_response import MessagingResponse
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
from langchain.memory.buffer import ConversationBufferMemory
from get_text import preprocess_and_upsert
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import PromptTemplate
import os
import cohere
from langchain_core.embeddings import Embeddings


class CohereEmbedding(Embeddings):
    def __init__(self, cohere_api_key: str):
        self.client = cohere.Client(cohere_api_key)

    def embed_documents(self, texts):
        response = self.client.embed(
            texts=texts,
            model="embed-english-v3.0",
            input_type="search_document"
        )
        return response.embeddings

    def embed_query(self, text):
        response = self.client.embed(
            texts=[text],
            model="embed-english-v3.0",
            input_type="search_query"
        )
        return response.embeddings[0]


# embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
embedding = CohereEmbedding(cohere_api_key=os.getenv("COHERE"))

app = FastAPI()

groq_api = os.getenv("GROQ_API")

llm = ChatGroq(
     api_key=groq_api,
     model="llama-3.1-8b-instant",
)

with open("data.txt","r") as f:
        content = f.read()


def preprocess_and_upsert(text, chunk_size=500):
    # Chunk the text manually
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    # Convert chunks to LangChain Document format
    documents = [Document(page_content=chunk) for chunk in chunks]

    # Create Chroma vectorstore
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory="./chatbot_chroma"
    )

    print(f"✅ Upserted {len(chunks)} chunks to Chroma DB.")
    return vectorstore

vectorstore = preprocess_and_upsert(text=content)
user_sessions = {}
# vectorstore = Chroma(
#     persist_directory="./chatbot_chroma",
#     embedding_function=embedding
# )

@app.post("/webhook")
def whatsapp_webhook( Body: str = Form(...),From: str = Form(...)):
    incoming_msg = Body
    sender = From


    custom_prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
        Tum ek helpful assistant ho jo WhatsApp users ki madad karta hai.

        Instructions:
        - Reply hamesha Hinglish mein dena (Hindi + English mix).
        - Tone friendly aur casual honi chahiye but not disrespectful and in a casual way.
        - Bina context ke guess mat lagana.

        Context:
        {context}

        User ka sawaal:
        {question}

        Hinglish mein helpful jawab:
        """,
        )
    

    if sender not in user_sessions:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_type="similarity", k=3),
            combine_docs_chain_kwargs={"prompt": custom_prompt_template},
            memory=memory
        )
        user_sessions[sender] = qa_chain

    # Use existing user chain
    chain = user_sessions[sender]
    response = chain.invoke({"question": incoming_msg})
    return response["answer"]


