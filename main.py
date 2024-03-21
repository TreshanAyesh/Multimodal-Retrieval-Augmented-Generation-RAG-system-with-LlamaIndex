from pydantic import BaseModel,Field
from PIL import Image
import matplotlib.pyplot as plt
import os
import google.generativeai as genai
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.core.program import MultiModalLLMCompletionProgram
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core import SimpleDirectoryReader
import time
from pathlib import Path
import random
from typing import Optional
import requests
from io import BytesIO
from IPython.display import Image,display
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.schema import TextNode
from typing import List
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import ServiceContext, VectorStoreIndex
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from chromadb.config import Settings
from dotenv import load_dotenv
load_dotenv('.env')


GOOGLE_API_KEY : str = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


#file imports
import classify_images
import vector_db


if __name__ == "__main__":
    
    # img_dir = "Wardrobe/2136.jpg"
    # nodes = classify_images.classify_single_image(img_dir)
    # classify_images.print_nodes(nodes)

    #img_dir = "Wardrobe/"
    #nodes = classify_images.classify_images_in_directory(img_dir, sample_size=0)
    # classify_images.print_nodes(nodes)

    # print(vector_db.get_count())
    #vector_db.index_vectors(nodes)

    # vector_db.erase_collection()
    #print(vector_db.get_count())
    
    results = vector_db.retrieve( "a good dress for party")
    for i in range(len(results["ids"][0])):
        print(results["metadatas"][0][i]["image_file"])
        
