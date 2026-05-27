from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Sample legal documents
legal_docs = [
    """Section 420 IPC - Cheating and dishonestly inducing delivery of property. 
    Whoever cheats and thereby dishonestly induces the person deceived to deliver any property, 
    shall be punished with imprisonment of either description for a term which may extend to 
    seven years, and shall also be liable to fine.""",
    
    """Section 302 IPC - Punishment for murder. 
    Whoever commits murder shall be punished with death, or imprisonment for life, 
    and shall also be liable to fine.""",
    
    """Section 375 IPC - Rape. A man is said to commit rape who has sexual intercourse 
    with a woman against her will, without her consent, or with her consent obtained 
    by putting her in fear of death or hurt.""",
    
    """Right to Information Act 2005 - Every citizen shall have the right to information 
    under the control of public authorities to promote transparency and accountability 
    in the working of every public authority.""",
    
    """Consumer Protection Act 2019 - A consumer can file a complaint against 
    unfair trade practices, defective goods, deficiency in services. 
    District Commission handles complaints up to 1 crore rupees.""",
]

print("Creating documents...")
documents = [Document(page_content=doc) for doc in legal_docs]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)
splits = text_splitter.split_documents(documents)
print(f"Created {len(splits)} chunks!")

print("Creating embeddings... (may take 2-3 minutes first time)")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

print("Building FAISS index...")
vectorstore = FAISS.from_documents(splits, embeddings)
print("FAISS index created!")

query = "What is punishment for murder?"
docs = vectorstore.similarity_search(query, k=2)
print(f"\nQuery: {query}")
print(f"Retrieved {len(docs)} relevant documents:")
for i, doc in enumerate(docs):
    print(f"\nDoc {i+1}: {doc.page_content[:200]}")

print("\nRAG Pipeline working!")