from fastapi import FastAPI,Request,Form
# from twilio.twiml.messaging_response import MessagingResponse
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
from langchain.memory.buffer import ConversationBufferMemory
from get_text import preprocess_and_upsert
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import PromptTemplate
import os
import cohere
from langchain_core.embeddings import Embeddings
from get_text import send_custom_messages,send_messages
from datetime import datetime, timedelta, timezone

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

# vectorstore = preprocess_and_upsert(text=content)
user_sessions = {}
vectorstore = Chroma(
    persist_directory="./chatbot_chroma",
    embedding_function=embedding
)

@app.post("/webhook")
async def whatsapp_webhook( request:Request):
    data = await request.json()
    print(data)
    message_data = data.get("data", {}).get("message", {})

    sender = message_data.get("phone_number")
    incoming_msg = message_data.get("message_content", {}).get("text")

    custom_prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
        Tum ek helpful assistant ho jo WhatsApp users ki madad karta hai.

        Instructions:
        Hamari company ek online chit fund platform hai. Humne notice kiya hai ki kai users registration ke baad hamari services ka actively use nahi karte.

✅ Chatbot ka kaam hai:

1. Aise users ko dynamically message bhejna jinhone register to kiya hai, lekin service use nahi kar rahe.
2. Har user ko contextually samajhne ke liye hum RAG (Retrieval-Augmented Generation) ka use karte hain — jaise ki unka past conversation ya common objections (e.g., trust issue, financial concern, loss, confusion, etc.).
3. Us context ke base pe chatbot ko relevant, empathetic aur convincing follow-up messages bhejne chahiye, taaki user wapas engage kare.
4. Common dispositions (e.g., “abhi interest nahi hai”, “paisa nahi hai”, “samajh nahi aaya”, etc.) pe chatbot ko automatically correct response dena chahiye.
5. Revival hamara main objective hai, to flow ko is goal ke around design kiya gaya hai — personalized messaging se user ko platform pe wapas lana.
6.Generate the message as a small whatsapp message
📌 Har interaction user-centric, revival-focused aur trust-building hona chahiye. Message professional yet friendly tone mein hona chahiye.
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
    delivered_at = data.get("data", {}).get("message", {}).get("delivered_at")
    delivered_time = datetime.fromtimestamp(delivered_at / 1000, tz=timezone.utc)
    five_minutes_ago = datetime.now(timezone.utc) - timedelta(minutes=5)
    from read_sheets import get_data
    extracted_data = get_data()
    phone_numbers = [f"91{data["User number"]}" for data in extracted_data]
    disposition = [f"{data["Disposition"]}" for data in extracted_data]
    if delivered_time<five_minutes_ago and incoming_msg:
        if sender in phone_numbers:
            response = chain.invoke({"question": incoming_msg})
            send_custom_messages(message=response["answer"],phone=sender)
            print({"status": "messages sent"})
            return response["answer"]
    else:
        return "Time not now"

@app.post("/send_messages")
async def send_msgs_app(request:Request):
    data = await request.json()
    problems = data.get("problems")
    names = data.get("names")
    phones = data.get("phones")
    phones = [f"91{phone}" for phone in phones]
    send_messages(problems,names,phones)


@app.get("/")
def ping():
    return {"status": "alive"}