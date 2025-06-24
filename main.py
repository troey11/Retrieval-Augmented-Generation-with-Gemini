import os
import streamlit as st
import hashlib
import tempfile
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain, ConversationalRetrievalChain
from langchain.chains.summarize import load_summarize_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader

# Konfigurasi API
GOOGLE_API_KEY = "YOUR API KEY"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
url_vectorstore_path = "url_faiss_store_gemini.pkl"
pdf_storage_path = "stored_pdfs"

# Streamlit Setup
st.set_page_config(page_title="AI Assistant Gemini", layout="wide")
st.markdown("<h1 style='text-align: center;'>🤖 Personal AI Assistant (Gemini)</h1>", unsafe_allow_html=True)
st.sidebar.markdown("### 📚 Source Options")

# Fungsi Hash dan Cache PDF
def get_file_hash(uploaded_file):
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()

def save_vector_to_cache(file_hash, vectorstore):
    path = os.path.join("vector_cache", file_hash)
    vectorstore.save_local(path)

def load_vector_from_cache(file_hash):
    path = os.path.join("vector_cache", file_hash)
    if os.path.exists(path):
        try:
            return FAISS.load_local(
                path,
                GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY),
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            st.warning(f"Cache rusak untuk {file_hash}, dihapus. Error: {e}")
            import shutil
            shutil.rmtree(path)
    return None

def save_uploaded_pdf(file, file_hash):
    os.makedirs(pdf_storage_path, exist_ok=True)
    file_path = os.path.join(pdf_storage_path, file_hash + ".pdf")
    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            f.write(file.getvalue())

def load_saved_pdfs():
    pdf_files = []
    if os.path.exists(pdf_storage_path):
        for fname in os.listdir(pdf_storage_path):
            if fname.endswith(".pdf"):
                with open(os.path.join(pdf_storage_path, fname), "rb") as f:
                    pdf_files.append((fname.replace(".pdf", ""), f.read()))
    return pdf_files

# Load URL
num_links = st.sidebar.slider("Berapa URL?", 1, 5, 1)
urls = [st.sidebar.text_input(f"URL {i+1}", key=f"url{i}") for i in range(num_links)]

if urls and any(urls):
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "."], chunk_size=1000)
    url_docs = splitter.split_documents(data)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    url_vectorstore = FAISS.from_documents(url_docs, embeddings)

    with open(url_vectorstore_path, "wb") as f:
        pickle.dump(url_vectorstore, f)

# Upload PDF
uploaded_files = st.sidebar.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, google_api_key=GOOGLE_API_KEY)

all_docs = []
doc_hashes = set()

# Proses PDF tersimpan
saved_pdfs = load_saved_pdfs()
if saved_pdfs:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    for file_hash, file_content in saved_pdfs:
        doc_hashes.add(file_hash)
        vector = load_vector_from_cache(file_hash)
        if not vector:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name
            reader = PdfReader(tmp_path)
            all_text = "".join([page.extract_text() or "" for page in reader.pages])
            chunks = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "."], chunk_size=500).split_text(all_text)
            docs = [Document(page_content=c) for c in chunks]
            vector = FAISS.from_documents(docs, embeddings)
            save_vector_to_cache(file_hash, vector)
        all_docs.append(vector)

# Proses upload PDF baru
if uploaded_files:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    for uploaded_file in uploaded_files:
        file_hash = get_file_hash(uploaded_file)
        if file_hash not in doc_hashes:
            save_uploaded_pdf(uploaded_file, file_hash)
            vectorstore = load_vector_from_cache(file_hash)
            if not vectorstore:
                reader = PdfReader(uploaded_file)
                all_text = "".join([page.extract_text() or "" for page in reader.pages])
                chunks = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "."], chunk_size=500).split_text(all_text)
                docs = [Document(page_content=c) for c in chunks]
                vectorstore = FAISS.from_documents(docs, embeddings)
                save_vector_to_cache(file_hash, vectorstore)
            all_docs.append(vectorstore)

# Gabungkan semua vector
# Gabungkan semua vector tanpa duplikat ID
pdf_vectors = None
if all_docs:
    pdf_vectors = all_docs[0]
    for vs in all_docs[1:]:
        try:
            pdf_vectors.merge_from(vs)  # Hapus override
        except ValueError as e:
            st.warning(f"Gagal menggabungkan vectorstore karena ID duplikat. Diabaikan.")

# Memory untuk percakapan nyambung
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Query
source = st.selectbox("Tanya Berdasarkan:", ["URL", "PDF"])

if source == "URL":
    q = st.text_input("Pertanyaan tentang URL:")
    if q and os.path.exists(url_vectorstore_path):
        with open(url_vectorstore_path, "rb") as f:
            vectorstore = pickle.load(f)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": q}, return_only_outputs=True)
        st.subheader("Jawaban:")
        st.write(result["answer"])

elif source == "PDF":
    q = st.text_input("Pertanyaan tentang PDF:")
    if q and pdf_vectors:
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=pdf_vectors.as_retriever(),
            memory=memory
        )
        result = chain({"question": q})
        st.subheader("Jawaban:")
        st.write(result["answer"])

    if st.button("📝 Ringkas Semua PDF"):
        def summarize_all_pdfs(files):
            summaries = []
            for item in files:
                if hasattr(item, "getvalue"):
                    file_name = item.name
                    content = item.getvalue()
                else:
                    file_name = item[0] + ".pdf"
                    content = item[1]

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name

                loader = PyPDFLoader(tmp_path)
                docs = loader.load_and_split()
                chain = load_summarize_chain(llm=llm, chain_type="map_reduce")
                summary = chain.run(docs)
                summaries.append((file_name, summary))
                os.remove(tmp_path)

            return summaries

        files_to_summarize = uploaded_files if uploaded_files else saved_pdfs
        all_summaries = summarize_all_pdfs(files_to_summarize)
        for filename, summary in all_summaries:
            st.markdown(f"### 📄 Ringkasan: {filename}")
            st.write(summary)
