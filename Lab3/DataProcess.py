import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import CharacterTextSplitter


def load_data(file_path):
    loader = CSVLoader(file_path=file_path, encoding='utf-8')
    documents = loader.load()
    return documents

def text_split(documents):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500, 
        chunk_overlap=50
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"分割完成！原始文档有 {len(documents)} 条，分割后变成了 {len(split_docs)} 个分块。")
    return split_docs

def build_embeddingsdb(split_docs):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    model_name = "moka-ai/m3e-base"
    model_kwargs = {'device': 'cpu'} 
    encode_kwargs = {'normalize_embeddings': True} 
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    db = FAISS.from_documents(split_docs, embeddings)
    db.save_local("faiss_legal_db")