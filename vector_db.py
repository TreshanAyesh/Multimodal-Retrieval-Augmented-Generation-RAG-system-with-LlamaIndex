import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import ServiceContext, VectorStoreIndex
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from chromadb.config import Settings
from chromadb import Documents, EmbeddingFunction, Embeddings
import google.generativeai as genai
from dotenv import load_dotenv
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
from llama_index.core.retrievers import VectorIndexAutoRetriever
import os
from IPython.display import Image
import IPython.display as display
from llama_index.core.schema import TextNode
from typing import List
load_dotenv('.env')

GOOGLE_API_KEY : str = os.getenv("GOOGLE_API_KEY")



class GeminiEmbeddingFunction(EmbeddingFunction):
  def __call__(self, input: Documents) -> Embeddings:
    model = 'models/embedding-001'
    title = "Custom query"
    return genai.embed_content(model=model,
                                content=input,
                                task_type="retrieval_document",
                                title=title)["embedding"]
  
#cromadb
chroma_client = chromadb.PersistentClient(path="./vdb", settings=Settings(
    anonymized_telemetry=False
))

chroma_collection = chroma_client.get_or_create_collection(
                    name = "quickstart",
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=GeminiEmbeddingFunction()   
)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

embed_model = GeminiEmbedding(
    model_name="models/embedding-001", api_key=GOOGLE_API_KEY
)

service_context = ServiceContext.from_defaults(
    llm=Gemini(api_key=GOOGLE_API_KEY),
    embed_model=embed_model,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)



# def create_database():
#     global service_context 
#     global storage_context 

#     vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

#     embed_model = GeminiEmbedding(
#         model_name="models/embedding-001", api_key=GOOGLE_API_KEY
#     )
#     service_context = ServiceContext.from_defaults(
#         llm=Gemini(api_key=GOOGLE_API_KEY),
#         embed_model=embed_model,
#     )

#     storage_context = StorageContext.from_defaults(vector_store=vector_store)



# def get_collection_client():
#     global chroma_collection
#     global chroma_client
    
#     #cromadb
#     chroma_client = chromadb.PersistentClient(path="./vdb", settings=Settings(
#         anonymized_telemetry=False
#     ))

#     chroma_collection = chroma_client.get_or_create_collection(
#                         name = "quickstart",
#                         metadata={"hnsw:space": "cosine"},
#                         embedding_function=GeminiEmbeddingFunction()                
#     )





def index_vectors(nodes):
    # get_collection_client()
    # create_database()
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        service_context=service_context,
    )



def get_count():
    """Get the count of the collection"""

    # get_collection_client()
    return chroma_collection.count()


def get_all_ids():
    """Returns all the ids in the collection"""
    # get_collection_client()
    documents = chroma_collection.get(limit = 0)
    all_ids = documents["ids"]
    return all_ids

def get_all_metadata():
    """Returns all the metadata in the collection"""
    # get_collection_client()
    documents = chroma_collection.get(limit = 0)
    all_m = documents["metadatas"]
    return all_m


def erase_collection():
    """Erase all the data in the collection"""

    documents = chroma_collection.get(limit=0)
    ids = documents['ids']
    chroma_collection.delete(ids)


vector_store_info = VectorStoreInfo(
    content_info="Fashion items",
    metadata_info=[
        MetadataInfo(
            name="cloth_type",
            description="Type of the cloth or fashion item",
            type="string",
        ),
        MetadataInfo(
            name="color",
            description="color of the item",
            type="string",
        ),
        MetadataInfo(
            name="season",
            description="The season which this cloth is more suitable",
            type="string",
        ),
        MetadataInfo(
            name="category",
            description="Is this a women's or men's item. unisex for anything other than those two",
            type="string",
        ),
        MetadataInfo(
            name="summary",
            description="a simple description including when can this be worn, suitable events to wear,  etc. ",
            type="string",
        ),
    ],
)

def define_retriever(index):
    retriever = VectorIndexAutoRetriever(
        index,
        vector_store_info=vector_store_info,
        similarity_top_k=3,
        empty_query_top_k=10,  # if only metadata filters are specified, this is the limit
        verbose=True,
    )
    return retriever




def display_response(nodes):
    """Display response."""
    for node in nodes:
        print(node.get_content(metadata_mode="all"))
        # img = Image.open(open(node.metadata["image_file"], 'rb'))
        display(Image(filename=node.metadata["image_file"], width=200))

def retrieve(query):
    results = chroma_collection.query(
    query_texts=[query],
    n_results=5,
    # where={"color": "red"}, # optional filter
)
    return results


def retrieve_similar(nodes: List[TextNode]):
    """Receive a single image and file similar images in the database"""

    cloth_type = nodes[0].metadata["cloth_type"]
    color = nodes[0].metadata["color"]
    season = nodes[0].metadata["season"]
    category = nodes[0].metadata["category"]

    summary = nodes[0].text

    print("Finding similar cloths to :" + color +" "+ cloth_type)

    results = chroma_collection.query(
    query_texts=[summary],
    n_results=5,
)
    return results