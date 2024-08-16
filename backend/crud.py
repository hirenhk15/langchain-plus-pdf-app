from models import PDF
from config import Settings
from sqlalchemy.orm import Session
from schemas import PDFRequest, PDFResponse
from fastapi import UploadFile, HTTPException
from botocore.exceptions import NoCredentialsError, BotoCoreError


def create_pdf(db: Session, pdf: PDFRequest):
    db_pdf = PDF(name=pdf.name, selected=pdf.selected, file=pdf.file)
    db.add(db_pdf)
    db.commit()
    db.refresh(db_pdf)
    return db_pdf

def read_pdfs(db: Session, selected: bool=None):
    if selected is None:
        return db.query(PDF).all()
    else:
        return db.query(PDF).filter(PDF.selected == selected).all()

def read_pdf(db: Session, id: int):
    return db.query(PDF).filter(PDF.id == id).first()

def update_pdf(db: Session, id: int, pdf: PDFRequest):
    db_pdf = db.query(PDF).filter(PDF.id == id).first()
    if db_pdf is None:
        return None
    
    db_pdf.name = pdf.name
    db_pdf.selected = pdf.selected
    db_pdf.file = pdf.file
    # updated_data = pdf.dict(exclude_unset=True)
    # for key, value in updated_data.items():
    #     setattr(db_pdf, key, value)
    db.commit()
    db.refresh(db_pdf)
    return db_pdf

def delete_pdf(db: Session, id: int):
    db_pdf = db.query(PDF).filter(PDF.id == id).first()
    if db_pdf is None:
        return None
    db.delete(db_pdf)
    db.commit()
    return True

def upload_pdf(db: Session, file: UploadFile, file_name: str):
    s3_client = Settings.get_s3_client()
    BUCKET_NAME = Settings().AWS_S3_BUCKET
    
    try:
        s3_client.upload_fileobj(
            file.file,
            BUCKET_NAME,
            file_name,
            # ExtraArgs={'ACL': 'public-read'}
        )
        file_url = f'https://{BUCKET_NAME}.s3.amazonaws.com/{file_name}'

        db_pdf = PDF(name=file.filename, selected=False, file=file_url)
        db.add(db_pdf)
        db.commit()
        db.refresh(db_pdf)
        return db_pdf
    except NoCredentialsError:
        raise HTTPException(status_code=500, detail="Error in AWS credentials")
    # except BotoCoreError as e:
    #     raise HTTPException(status_code=500, detail=str(e))