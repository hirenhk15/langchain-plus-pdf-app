import crud

from uuid import uuid4
from typing import List
from database import SessionLocal
from sqlalchemy.orm import Session
from schemas import PDFRequest, PDFResponse, QuestionRequest

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File


router = APIRouter(prefix="/pdfs")

# Define QA chain
# Define LLM
llm = ChatGroq(temperature=0, model="llama3-70b-8192")

promp_template = """
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.

    Question: {question} 
    Context: {context} 
    Answer:
"""
prompt = PromptTemplate.from_template(promp_template)

# Define embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_qa_chain(retriever):
    qa_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return qa_chain

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("", response_model=PDFResponse, status_code=status.HTTP_201_CREATED)
def create_pdf(pdf: PDFRequest, db: Session = Depends(get_db)):
    return crud.create_pdf(db, pdf)

@router.post("/upload", response_model=PDFResponse, status_code=status.HTTP_201_CREATED)
def upload_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    file_name = f"{uuid4()}-{file.filename}"
    return crud.upload_pdf(db, file, file_name)

@router.get("", response_model=List[PDFResponse])
def get_pdfs(selected: bool = None, db: Session = Depends(get_db)):
    return crud.read_pdfs(db, selected)

@router.get("/{id}", response_model=PDFResponse)
def get_pdfs(id: int, db: Session = Depends(get_db)):
    pdf = crud.read_pdf(db, id)
    if pdf is None:
        raise HTTPException(status_code=404, detail="PDF not found")
    return pdf

@router.put("/{id}", response_model=PDFResponse)
def get_pdfs(id: int, pdf: PDFRequest, db: Session = Depends(get_db)):
    updated_pdf = crud.update_pdf(db, id, pdf)
    if updated_pdf is None:
        raise HTTPException(status_code=404, detail="PDF not found")
    return updated_pdf

@router.delete("/{id}", status_code=status.HTTP_200_OK)
def delete_pdf(id: int, db: Session = Depends(get_db)):
    res = crud.delete_pdf(db, id)
    if res is None:
        raise HTTPException(status_code=404, detail="PDF not found")
    return {"message": "PDF successfully deleted"}

@router.post("/qa-pdf/{id}")
def qa_pdf_by_id(id: int, question_request: QuestionRequest, db: Session = Depends(get_db)):
    pdf = crud.read_pdf(db, id)
    if pdf is None:
        raise HTTPException(status_code=404, detail="PDF not found")
    
    # Load the pdf file
    loader = PyPDFLoader(pdf.file)
    docs = loader.load()

    # Split the text from the docs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)
    document_chunks = text_splitter.split_documents(docs)

    # Store into vector db
    vectorstore = FAISS.from_documents(document_chunks, embeddings)
    retriever = vectorstore.as_retriever()

    # Get the QA chain
    qa_chain = get_qa_chain(retriever)
    answer = qa_chain.invoke(question_request.question)

    return {"answer": answer}