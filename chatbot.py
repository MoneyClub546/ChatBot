from flask import Flask, request
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


embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

app = Flask(__name__)

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

@app.route("/webhook", methods=["POST"])
def whatsapp_webhook():
    incoming_msg = request.values.get('Body', '')
    sender = request.values.get('From', '')

    custom_prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
        Tum ek helpful assistant ho jo WhatsApp users ki madad karta hai.

        Instructions:
        - Reply hamesha Hinglish mein dena (Hindi + English mix).
        - Tone friendly aur casual honi chahiye.
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
    # Return to WhatsApp
    # twilio_resp = MessagingResponse()
    # twilio_resp.message(response)
    # return str(twilio_resp)

if __name__ == "__main__":
    app.run(port=5000)
