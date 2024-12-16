import fitz  
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
from langchain.chat_models import ChatOpenAI

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DB_DIM = 384
CHUNK_SIZE = 500
PDF_PATH = "/mnt/data/sample_data.pdf"  


def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text


def extract_tables_from_pdf(pdf_path, page_number):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_number <= len(pdf.pages):
                page = pdf.pages[page_number - 1]
                return page.extract_table()
            else:
                raise ValueError("Page number out of range")
    except Exception as e:
        print(f"Error extracting table: {e}")
        return None


def extract_unemployment_data(text):
    lines = text.splitlines()
    unemployment_info = {}
    for line in lines:
        if "|" in line:
            parts = line.split("|")
            if len(parts) == 2:
                degree, rate = parts[0].strip(), parts[1].strip()
                unemployment_info[degree] = rate
    return unemployment_info


def chunk_text(text, chunk_size):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def generate_embeddings(chunks, model_name):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    return embeddings


def build_vector_db(embeddings):
    index = faiss.IndexFlatL2(VECTOR_DB_DIM)
    index.add(embeddings)
    return index


def query_vector_db(query, index, model_name, chunks, top_k=3):
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [(chunks[i], distances[j]) for j, i in enumerate(indices[0])]


def generate_response_with_llm(query, retrieved_chunks):
    context = "\n\n".join([chunk for chunk, _ in retrieved_chunks])
    prompt = f"Use the following context to answer the query:\n\n{context}\n\nQuery: {query}"
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)  # Replace with your OpenAI setup
    return llm.call_as_llm(prompt)


def main():
    pdf_text = extract_text_from_pdf(PDF_PATH)
    chunks = chunk_text(pdf_text, CHUNK_SIZE)
    embeddings = generate_embeddings(chunks, EMBEDDING_MODEL_NAME)
    vector_db = build_vector_db(embeddings)

    
    query = "What is the unemployment rate for Bachelor's degree?"
    retrieved_chunks = query_vector_db(query, vector_db, EMBEDDING_MODEL_NAME, chunks)
    response = generate_response_with_llm(query, retrieved_chunks)
    print("Generated Response:", response)

    
    unemployment_data = extract_unemployment_data(pdf_text)
    table_data = extract_tables_from_pdf(PDF_PATH, page_number=2)

    print("Unemployment Data:", unemployment_data)
    print("Table Data:", table_data)

if __name__ == "__main__":
    main()
