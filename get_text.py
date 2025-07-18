import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.config import Settings
from bs4 import BeautifulSoup
import os

groq_api = os.getenv("GROQ_API")

client = Groq(
     api_key=groq_api,
)


chroma_client = chromadb.PersistentClient(path="./chroma")
chroma_collection = chroma_client.get_or_create_collection(name="money_club")


def extract_text_from_website(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() 
        soup = BeautifulSoup(response.text, 'html.parser')
        for element in soup(['script', 'style', 'noscript']):
            element.decompose()
        text = soup.get_text(separator='\n')
        clean_text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())
        with open("structured_money_club_info.txt", "a", encoding="utf-8") as file:
            file.write(text)
        
        return clean_text

    except requests.RequestException as e:
        return f"Request error: {e}"
    except Exception as e:
        return f"An error occurred: {e}"


def preprocess_and_upsert(text, chunk_size=500):
    global chunks, vectorizer

    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    vectorizer = TfidfVectorizer().fit(chunks)

    chunk_vectors = vectorizer.transform(chunks).toarray()
    ids = [str(i) for i in range(len(chunks))]

    vectorstore = chroma_collection.upsert(
        documents=chunks,
        embeddings=chunk_vectors,
        ids=ids
    )
    print(f"Upserted {len(chunks)} chunks to vector DB.")
    return chunk_vectors
    


def query_most_relevant_chunk(query):
    global chunks, vectorizer

    if not chunks or vectorizer is None:
        raise ValueError("You must call preprocess_and_upsert() first with your text.")

    query_vector = vectorizer.transform([query]).toarray()

    results = chroma_collection.query(
        query_embeddings=query_vector,
        n_results=3
    )

    most_relevant_chunk = results['documents'][0][0]

    return most_relevant_chunk


def summarize_with_llm(prompt, chunk, name="Aakash"):
    """
    Generate a personalized WhatsApp marketing message in Hinglish
    
    Args:
        prompt: Context about what kind of message to create
        chunk: The problem/pain point to address
        name: Customer name for personalization
    """
    
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a friendly marketing assistant for MoneyClub. Your job is to write WhatsApp messages in Hinglish (Hindi + English mix) "
                    "that sound natural and conversational - like how friends talk to each other.\n\n"
                    
                    "IMPORTANT GUIDELINES:\n"
                    "1. Use Hinglish naturally - mix Hindi and English like: 'Paisa ki tension ho rahi hai? Don't worry!'\n"
                    "2. Keep it conversational and friendly - like talking to a friend\n"
                    "3. Address the specific problem mentioned\n"
                    "4. Mention MoneyClub as the solution\n"
                    "5. Use emojis sparingly (1-2 max)\n"
                    "6. Keep message between 2-4 lines\n"
                    "7. End with a call-to-action\n\n"
                    
                    "GOOD EXAMPLE:\n"
                    "'Hey! Emergency mein paisa chahiye aur koi option nahi dikh raha? MoneyClub try karo - safe hai aur easy bhi. App download karo aur dekho kitna simple hai! 💸'\n\n"
                    
                    "BAD EXAMPLE (too formal):\n"
                    "'Dear customer, if you need financial assistance, please consider MoneyClub application.'\n\n"
                    
                    "Now create a message based on the customer's problem:"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Customer Name: {name}\n"
                    f"Message Context: {prompt}\n"
                    f"Customer's Problem: {chunk}\n\n"
                    
                    "Create a personalized WhatsApp message that:\n"
                    "- Addresses their specific problem\n"
                    "- Suggests MoneyClub as solution\n"
                    "- Uses natural Hinglish (not formal English)\n"
                    "- Sounds like a friend giving advice\n"
                    "- Is ready to send on WhatsApp"
                )
            }
        ],
        temperature=0.7, 
        max_tokens=250, 
    )
    
    message = response.choices[0].message.content.strip()
    
    message = message.replace('"', '').replace("'", "") 
    print(message)
    return message


def send_custom_messages( message,phone,api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjY3NWFlNWJlZWM4NTExMGVlZWUzYjc5YiIsIm5hbWUiOiJUaGUgTW9uZXkgQ2x1YiIsImFwcE5hbWUiOiJBaVNlbnN5IiwiY2xpZW50SWQiOiI2NzVhZTViZWVjODUxMTBlZWVlM2I3OTMiLCJhY3RpdmVQbGFuIjoiQkFTSUNfTU9OVEhMWSIsImlhdCI6MTczNDAxMDMwMn0.XnKsryCNKLl67oml0kz20nQK6bkKCiwO5RqN40AlhBY"):

    url = "https://backend.aisensy.com/campaign/t1/api/v2"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    rows = []
    payload = {
        "apiKey": api_key,
        "campaignName": "Test_API_Campaign_2",
        "destination": phone,
        "userName":"The Money Club",
        "templateParams" : [message],
    }

    response = requests.post(url, json=payload,headers=headers)
    print(response)


def main():
    with open("data.txt","r") as f:
        content = f.read()
    preprocess_and_upsert(text=content)
    from read_sheets import get_data
    data = get_data()
    for row in data:
        problem = row["Common Call Disposition"]
        name = row["Name"]
        message = summarize_with_llm(prompt=problem,chunk=query_most_relevant_chunk(problem),name=name)
        send_custom_messages(message=message,phone="919640094070")



