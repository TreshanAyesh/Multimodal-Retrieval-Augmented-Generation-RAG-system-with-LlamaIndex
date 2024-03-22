import os
from dotenv import load_dotenv
load_dotenv('.env')


GOOGLE_API_KEY : str = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


#file imports
import classify_images
import vector_db



if __name__ == "__main__":
    
    """Input a single image and get a description of what is in the image"""
    # img_dir = "Wardrobe\\1785.jpg"
    # nodes = classify_images.classify_single_image(img_dir)
    # classify_images.print_nodes(nodes)

    """Input a directory of images and get description of what is in each image"""
    # img_dir = "Wardrobe\\"
    # nodes = classify_images.classify_images_in_directory(img_dir, sample_size= 0)
    # classify_images.print_nodes(nodes)

    """Number of embeddings stored in the vector database"""
    # print(vector_db.get_count())

    """store the current image(s) in the vector database. 
    Should call classify_single_image() or classify_images_in_directory before this"""
    # vector_db.index_vectors(nodes)

    """Erase all the embeddings in the vector storage"""
    # vector_db.erase_collection()
    
    """Retrieve images that corresponds to the query text given"""
    # results = vector_db.retrieve( "red t-shirts")
    # for i in range(len(results["ids"][0])):
    #     print(results["metadatas"][0][i]["image_file"])
    
    """Retrieve simlar images to the given image. Should call classify_single_image() first"""
    # results = vector_db.retrieve_similar(nodes)
    # for i in range(len(results["ids"][0])):
    #     print(results["metadatas"][0][i]["image_file"])

    """Retrieve a matching cloth to the item in the image. Give the path of the image as img_dir"""
    # results = classify_images.find_matching(img_dir)
    # for i in range(len(results["ids"][0])):
    #     print(results["metadatas"][0][i]["image_file"])

    
