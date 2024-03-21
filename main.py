import os
from dotenv import load_dotenv
load_dotenv('.env')


GOOGLE_API_KEY : str = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


#file imports
import classify_images
import vector_db



if __name__ == "__main__":
    
    img_dir = "Wardrobe/2136.jpg"
    nodes = classify_images.classify_single_image(img_dir)
    classify_images.print_nodes(nodes)

    #img_dir = "Wardrobe/"
    #nodes = classify_images.classify_images_in_directory(img_dir, sample_size=0)
    # classify_images.print_nodes(nodes)

    #print(vector_db.get_count())
    # vector_db.index_vectors(nodes)

    # vector_db.erase_collection()
    #print(vector_db.get_count())
    
    # results = vector_db.retrieve( "blue tshirt")
    # for i in range(len(results["ids"][0])):
    #     print(results["metadatas"][0][i]["image_file"])
    
    results = vector_db.retrieve_similar(nodes)
    for i in range(len(results["ids"][0])):
        print(results["metadatas"][0][i]["image_file"])

    # results = classify_images.find_matching(img_dir)
    # for i in range(len(results["ids"][0])):
    #     print(results["metadatas"][0][i]["image_file"])

    
