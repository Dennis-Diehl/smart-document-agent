# Handles document loading from various file formats
from langchain_community.document_loaders import PyPDFLoader


# load the document from a PDF file
class DocumentLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        # Use PyPDFLoader to load the PDF document
        loader = PyPDFLoader(self.file_path)
        documents = loader.load()
        return documents
