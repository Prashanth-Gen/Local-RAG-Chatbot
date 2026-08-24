A **local RAG (Retrieval-Augmented Generation)** chatbot that answers questions based on custom documents (PDFs) without needing an internet connection. 

**Key Features:**
1. 100% Local Execution: Runs entirely on your machine (CPU-only)
2. Document Intelligence: Reads and understands PDF files
3. Memory Efficient: Optimized for 8GB RAM systems
4. Conversational Interface: Interactive Q&A with streaming responses

**Screenshots:**
<img width="1574" height="386" alt="image" src="https://github.com/user-attachments/assets/612c9610-748c-4077-8c63-b3aa3a0a6ac4" />
<img width="1575" height="356" alt="image" src="https://github.com/user-attachments/assets/3a3e95a1-fba2-448e-8933-77fa586c95b1" />
<img width="1566" height="825" alt="image" src="https://github.com/user-attachments/assets/615ce19b-e350-40ca-b11e-19c6a7aa1003" />


**Primary Components in simple terms:**
1. Embeddings: Convert text into numerical features (vectors).
2. Vector DB: Database where the stored vectors are organised and searched. 
3. LLM: Generates human like response with help of the questions asked and the answer found.

**Technical Stack used:**
1. Framework: LangChain
2. Embeddings: HuggingFace all-MiniLM-L6-v2 (Lightweight (80MB), CPU-friendly)
3. Vector DB: Chroma
4. LLM: Zephyr-7B(4-bit quantized) (Balanced performance for 8GB RAM)
5. Document Loader: PyPDFLoader

**Choice of Model:**

| Model            | Size (4-bit) | RAM Needed | Strengths                    | Weaknesses                     |
|------------------|-------------|------------|------------------------------|--------------------------------|
| **Zephyr-7B** (Chosen) | ~3.5GB     | 6-8GB      | Best accuracy/size balance    | Requires 8GB RAM               |
| TinyLlama-1.1B   | ~0.7GB      | 3-4GB      | Runs on very weak hardware    | Less coherent answers          |
| Mistral-7B       | ~3.5GB      | 6-8GB      | Strong reasoning             | Slightly slower               |
| Phi-2            | ~1.6GB      | 4-5GB      | Excellent for coding tasks   | Weaker with long text         |
| Llama2-7B        | ~3.5GB      | 6-8GB      | Widely supported             | Outperformed by newer models  |

**Selection Criteria:**
1. **Hardware Compatibility**: Optimized for 8GB RAM systems
2. **Instruction Following**: Specifically fine-tuned for Q&A tasks
3. **Quantization Support**: Available in efficient GGUF format
4. **License**: Permissive MIT license for commercial use
5. **Performance**: Benchmarked 15% better than Llama2-7B on RAG tasks

**Working (Step-by-Step):**

**A. Document Processing Pipeline:**
1. Loading: Scans docs/ folder for PDFs using DirectoryLoader. Extracts text with PyPDFLoader.
2. Chunking: Splits documents into 300-character segments with 50-character overlaps. Uses RecursiveCharacterTextSplitter to preserve context.
3. Vectorization: Converts text to numerical vectors using all-MiniLM-L6-v2 embeddings. Stores vectors in ChromaDB for fast searching.

**B. Question-Answering System:**
1. Retrieval: When a question is asked => Converts question to vector. Finds  most relevant document chunks.
2. Generation: Combines question + retrieved context in the prompt by Zephyr-7B model.
3. Streaming Interface: Displays answers word-by-word for natural conversation feel.

**Technical Challenges & Solutions:**
1. Limited RAM: Used quantized Zephyr-7B (4-bit) and small embeddings.
2. Ollama connectivity issues: Switched to fully local CTransformers backend.
3. PDF parsing complexity: Implemented recursive text splitting with overlaps.
4. Slow responses: Optimized chunk size (300 chars) and limited to 2 context chunks

**Real-World Applications:**
1. Internal Knowledge Base: For companies with sensitive documents.
2. Academic Research Helper: Quickly query research papers.
3. Customer Support: Answer from product manuals

**Skills and concepts learned:**

**AI/ML**: RAG architectures, LLM quantization, prompt engineering  
**Tools**: LangChain, ChromaDB, HuggingFace Transformers
