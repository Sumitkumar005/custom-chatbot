import os
import time
import PyPDF2
from datetime import datetime
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from gridfs import GridFS
import pickle
import tempfile

# Load environment variables from .env file
load_dotenv()

# Get environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")
if not MONGO_URI:
    raise ValueError("MONGO_URI not found in .env file")

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# MongoDB setup with connection timeout
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    db = client["university_db"]
    universities_collection = db["universities"]
    chunks_collection = db["chunks"]
    query_logs_collection = db["query_logs"]
    fs = GridFS(db)
    print("Connected to MongoDB successfully")
except ServerSelectionTimeoutError as e:
    raise RuntimeError(f"Failed to connect to MongoDB: {str(e)}. Ensure MongoDB is running or check MONGO_URI")
except Exception as e:
    raise RuntimeError(f"Unexpected MongoDB error: {str(e)}")

# Function to initialize MongoDB with university metadata
def initialize_university_metadata():
    """Initialize MongoDB with university metadata if not already present."""
    universities = [
        {
            "university_id": "24-25CA.WUCatalog",
            "gridfs_filename": "24-25CA.WUCatalog.pdf",
            "gridfs_faiss_index": "24-25CA.WUCatalog_index.faiss",
            "gridfs_faiss_pkl": "24-25CA.WUCatalog_index.pkl",
            "s3_path": None,
            "name": "Westcliff University"
        },
        {
            "university_id": "App Form MBA 2026",
            "gridfs_filename": "App Form MBA 2026.pdf",
            "gridfs_faiss_index": "App Form MBA 2026_index.faiss",
            "gridfs_faiss_pkl": "App Form MBA 2026_index.pkl",
            "s3_path": None,
            "name": "Istec Business School"
        }
    ]
    for uni in universities:
        existing = universities_collection.find_one({"university_id": uni["university_id"]})
        if not existing:
            universities_collection.insert_one(uni)
            print(f"Inserted metadata for {uni['university_id']}")
        elif not all(key in existing for key in ["gridfs_faiss_index", "gridfs_faiss_pkl", "s3_path"]):
            universities_collection.update_one(
                {"university_id": uni["university_id"]},
                {"$set": {
                    "gridfs_faiss_index": uni["gridfs_faiss_index"],
                    "gridfs_faiss_pkl": uni["gridfs_faiss_pkl"],
                    "s3_path": None
                }}
            )
            print(f"Updated metadata for {uni['university_id']}")
    # Create indexes for performance
    universities_collection.create_index("university_id", unique=True)
    chunks_collection.create_index([("university_id", 1), ("chunk_id", 1)])
    query_logs_collection.create_index("timestamp")
    print("MongoDB indexes created")

# Function to extract text from PDF stored in GridFS
def extract_text_from_pdf(gridfs_filename, fs):
    """Extract text from a PDF file stored in GridFS."""
    try:
        grid_file = fs.find_one({"filename": gridfs_filename})
        if not grid_file:
            raise ValueError(f"PDF not found in GridFS: {gridfs_filename}")
        pdf_data = grid_file.read()
        pdf_stream = BytesIO(pdf_data)
        reader = PyPDF2.PdfReader(pdf_stream)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if not text.strip():
            raise ValueError("No text extracted from PDF")
        return text
    except Exception as e:
        raise RuntimeError(f"Error extracting text from PDF: {str(e)}")

# Function to chunk text
def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """Split text into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

# Function to store chunks in MongoDB
def store_chunks_in_mongodb(university_id, chunks):
    """Store text chunks in MongoDB."""
    try:
        chunks_collection.delete_many({"university_id": university_id})
        for i, chunk in enumerate(chunks):
            chunks_collection.insert_one({
                "university_id": university_id,
                "chunk_id": i + 1,
                "text": chunk
            })
        print(f"Stored {len(chunks)} chunks for {university_id} in MongoDB")
    except Exception as e:
        raise RuntimeError(f"Error storing chunks in MongoDB: {str(e)}")

# Function to save FAISS index to GridFS
def save_faiss_to_gridfs(vectorstore, university_id, fs):
    """Save FAISS index and pkl files to GridFS."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = os.path.join(temp_dir, "index.faiss")
            pkl_path = os.path.join(temp_dir, "index.pkl")
            vectorstore.save_local(temp_dir)
            
            # Save index.faiss
            with open(index_path, 'rb') as f:
                fs.put(f, filename=f"{university_id}_index.faiss", metadata={"university_id": university_id})
            print(f"Saved FAISS index for {university_id} to GridFS")
            
            # Save index.pkl
            with open(pkl_path, 'rb') as f:
                fs.put(f, filename=f"{university_id}_index.pkl", metadata={"university_id": university_id})
            print(f"Saved FAISS pkl for {university_id} to GridFS")
    except Exception as e:
        raise RuntimeError(f"Error saving FAISS index to GridFS: {str(e)}")

# Function to load FAISS index from GridFS
def load_faiss_from_gridfs(university_id, gridfs_index_filename, gridfs_pkl_filename, embeddings, fs):
    """Load FAISS index from GridFS."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = os.path.join(temp_dir, "index.faiss")
            pkl_path = os.path.join(temp_dir, "index.pkl")
            
            # Load index.faiss
            grid_file = fs.find_one({"filename": gridfs_index_filename})
            if not grid_file:
                return None
            with open(index_path, 'wb') as f:
                f.write(grid_file.read())
            
            # Load index.pkl
            grid_file = fs.find_one({"filename": gridfs_pkl_filename})
            if not grid_file:
                return None
            with open(pkl_path, 'wb') as f:
                f.write(grid_file.read())
            
            vectorstore = FAISS.load_local(temp_dir, embeddings, allow_dangerous_deserialization=True)
            print(f"Loaded FAISS index for {university_id} from GridFS")
            return vectorstore
    except Exception as e:
        print(f"Failed to load FAISS index from GridFS for {university_id}: {str(e)}")
        return None

# Function to create or load FAISS index for a university
def create_or_load_faiss_index(university_id, gridfs_filename, gridfs_index_filename, gridfs_pkl_filename, fs):
    """Create a new FAISS index or load an existing one from GridFS."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        vectorstore = load_faiss_from_gridfs(university_id, gridfs_index_filename, gridfs_pkl_filename, embeddings, fs)
        if vectorstore:
            return vectorstore
        
        print(f"Creating new FAISS index for {gridfs_filename}...")
        text = extract_text_from_pdf(gridfs_filename, fs)
        chunks = chunk_text(text)
        print(f"Created {len(chunks)} chunks for {gridfs_filename}.")
        store_chunks_in_mongodb(university_id, chunks)
        vectorstore = FAISS.from_texts(chunks, embeddings)
        save_faiss_to_gridfs(vectorstore, university_id, fs)
        return vectorstore
    except Exception as e:
        raise RuntimeError(f"Error creating or loading FAISS index: {str(e)}")

# Function to search FAISS index
def search_faiss(query, vectorstore, k=3):
    """Search FAISS index for relevant chunks."""
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)
        return [doc.page_content for doc in docs]
    except Exception as e:
        raise RuntimeError(f"Error searching FAISS index: {str(e)}")

# Function to generate answer using Gemini API
def generate_answer(query, context, is_concise=False):
    """Generate an answer using Gemini API with context and concise option."""
    try:
        prompt = f"""
        You are an expert assistant tasked with providing accurate and helpful responses based on the provided university dataset. 
        Your goal is to assist users (prospective students, current students, or staff) by delivering clear, relevant, and contextually appropriate answers.
        Use the provided context from the university dataset to inform your responses. 
        If the information is not available in the context, explicitly state: "I don’t have enough information to answer this fully."

        ### Context:
        {context}

        ### Instructions:
        - Answer the user’s question based on the provided context.
        - Provide accurate and relevant information without assuming specific details about the university.
        - If the user requests a concise answer (e.g., "in one line," "briefly," or "shortly"), provide a short, precise response (one sentence).
        - Otherwise, give a detailed, well-structured answer with relevant details.
        - Avoid speculation; stick to the provided information.
        - If the query is unclear, ask the user for clarification politely.

        ### User Query:
        {query}

        ### Response Format:
        {"Provide a concise answer in one sentence." if is_concise else "Provide a detailed, well-structured answer."}
        """
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        raise RuntimeError(f"Error generating answer: {str(e)}")

# Function to log query in MongoDB
def log_query(university_id, query, answer, response_time):
    """Log query, answer, and response time in MongoDB."""
    try:
        query_logs_collection.insert_one({
            "university_id": university_id,
            "query": query,
            "answer": answer,
            "response_time": response_time,
            "timestamp": datetime.utcnow().isoformat()
        })
        print(f"Logged query for {university_id}")
    except Exception as e:
        raise RuntimeError(f"Error logging query in MongoDB: {str(e)}")

# Function to get vectorstore for a university
def get_vectorstore(university_id, fs):
    """Retrieve or create vectorstore for the given university ID."""
    try:
        uni = universities_collection.find_one({"university_id": university_id})
        if not uni:
            raise ValueError(f"University ID {university_id} not found in MongoDB")
        gridfs_filename = uni["gridfs_filename"]
        gridfs_index_filename = uni.get("gridfs_faiss_index", f"{university_id}_index.faiss")
        gridfs_pkl_filename = uni.get("gridfs_faiss_pkl", f"{university_id}_index.pkl")
        return create_or_load_faiss_index(university_id, gridfs_filename, gridfs_index_filename, gridfs_pkl_filename, fs)
    except Exception as e:
        raise RuntimeError(f"Error retrieving vectorstore: {str(e)}")

# Main function
def main():
    """Main function to run the chatbot."""
    try:
        # Initialize university metadata in MongoDB
        initialize_university_metadata()
        
        # Initialize vectorstores for all universities
        print("Initializing FAISS indexes for all universities...")
        for uni in universities_collection.find():
            create_or_load_faiss_index(
                uni["university_id"],
                uni["gridfs_filename"],
                uni.get("gridfs_faiss_index", f"{uni['university_id']}_index.faiss"),
                uni.get("gridfs_faiss_pkl", f"{uni['university_id']}_index.pkl"),
                fs
            )
        
        # Interactive chat loop
        print("\nChatbot is ready. Type 'exit' to quit.")
        while True:
            university_id = input("Enter university ID (e.g., 24-25CA.WUCatalog): ")
            if university_id.lower() == "exit":
                print("Exiting chatbot.")
                break
            query = input("Ask a question: ")
            if query.lower() == "exit":
                print("Exiting chatbot.")
                break
            try:
                # Start timing
                mongo_start = time.perf_counter()
                vectorstore = get_vectorstore(university_id, fs)
                mongo_time = time.perf_counter() - mongo_start
                
                # Search and generate answer
                start_time = time.perf_counter()
                is_concise = any(phrase in query.lower() for phrase in ["in one line", "briefly", "shortly"])
                relevant_chunks = search_faiss(query, vectorstore)
                context = "\n".join(relevant_chunks)
                answer = generate_answer(query, context, is_concise)
                end_time = time.perf_counter()
                response_time = end_time - start_time
                
                # Log query to MongoDB
                log_query(university_id, query, answer, response_time)
                
                # Print answer and times
                print(f"Answer: {answer}\nMongoDB query time: {mongo_time:.4f} seconds\nFAISS + Gemini response time: {response_time:.4f} seconds\n")
            except ValueError as e:
                print(f"Error: {str(e)}\n")
            except Exception as e:
                print(f"Unexpected error: {str(e)}\n")
    except Exception as e:
        print(f"Initialization error: {str(e)}")
    finally:
        client.close()

if __name__ == "__main__":
    main()