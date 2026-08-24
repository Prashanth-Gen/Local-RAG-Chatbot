from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Load documents (PDFs / txt files)
print("Loading documents...")
loader = DirectoryLoader(
    path="docs",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)
documents = loader.load()

# 2. Split documents into chunks
print("Splitting documents...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

# 3. Local embeddings
print("Loading embeddings model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# 4. Create vector store
print("Creating vector database...")
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="db"
)

# 5. Initialize retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# 6. Lightweight local LLM
print("Loading language model...")
llm = CTransformers(
    model="TheBloke/zephyr-7B-beta-GGUF",
    model_file="zephyr-7b-beta.Q4_K_M.gguf",
    model_type="mistral",
    config={'max_new_tokens': 256, 'temperature': 0.01, 'context_length': 1024}
)

# 7. Prompt template
template = """Answer using ONLY this context:
{context}

Question: {question}
Answer in 2-3 sentences:"""
prompt = ChatPromptTemplate.from_template(template)

# 8. Create chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 9. Chat interface
print("\nLocal RAG Chatbot is ready with yoyur documents! (Type 'quit' to exit)")
while True:
    try:
        question = input("\nYou: ")
        if question.lower().strip() in ["quit", "exit", "bye"]:
            vector_db.persist()
            print("Goodbye!")
            break
        
        print("Bot: ", end="", flush=True)  # Start printing response immediately
        for chunk in rag_chain.stream(question):
            print(chunk, end="", flush=True)  # Stream response
        print()  # New line after response
        
    except KeyboardInterrupt:
        vector_db.persist()
        print("\nGoodbye!")
        break
    except Exception as e:
        print(f"\nError: {str(e)}")
        continue