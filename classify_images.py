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
from IPython.display import Image,display
from llama_index.core.schema import TextNode
from typing import List
from dotenv import load_dotenv
# from IPython.display import Image

import vector_db
load_dotenv('.env')


GOOGLE_API_KEY : str = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


#The gemini model
class ReceiptInfo(BaseModel):
    cloth_type: str = Field(..., description="Type of the cloth or fashion item")
    color: str = Field(..., description="color of the item")
    season: str = Field(..., description="The season which this cloth is more suitable")
    category: str = Field(..., description="Is this a women's or men's item. unisex for anything other than those two")
    summary: str = Field(
        ...,
        description="a simple description including when can this be worn, suitable events to wear,  etc. ",
    )

#The gemini model matching
class ReceiptInfo_matching(BaseModel):
    cloth_type: str = Field(..., description="Type of the cloth or fashion item")
    summary: str = Field(
        ...,
        description="description of the matching clothing item.",
    )

prompt_template_str = """\
    Can you summarize the cloting item in the image and return a response \
    with the following JSON format: \
"""

prompt_give_matching_cloths_str = """\
    Can you give information for a matching clothing items to wear with the cloth in the image\
    If this is top wear give a matching bottom wear and vice versa.\
    with the following JSON format: \
"""


def pydantic_gemini(output_class, image_documents, prompt_template_str):
    gemini_llm = GeminiMultiModal(
        api_key=GOOGLE_API_KEY, model_name="models/gemini-pro-vision"
    )

    llm_program = MultiModalLLMCompletionProgram.from_defaults(
        output_parser=PydanticOutputParser(output_class),
        image_documents=image_documents,
        prompt_template_str=prompt_template_str,
        multi_modal_llm=gemini_llm,
        verbose=True,
    )

    response = llm_program()
    return response


#Function to get images in a path
def get_image_files(
    dir_path, sample: Optional[int] = 10, shuffle: bool = False
):
    dir_path = Path(dir_path)
    image_paths = []
    for image_path in dir_path.glob("*.jpg"):
        image_paths.append(image_path)

    random.shuffle(image_paths)
    if sample:
        return image_paths[:sample]
    else:
        return image_paths


#process a single file 
def aprocess_image_file(image_file):
    # should load one file
    print(f"Image file: {image_file}")

    img_docs = SimpleDirectoryReader(input_files=[image_file]).load_data()
    output = pydantic_gemini(ReceiptInfo, img_docs, prompt_template_str)
    return output

#process a single file to get matching cloths
def aprocess_image_file_matching(image_file):
    # should load one file
    print(f"Image file: {image_file}")

    img_docs = SimpleDirectoryReader(input_files=[image_file]).load_data()
    output = pydantic_gemini(ReceiptInfo_matching, img_docs, prompt_give_matching_cloths_str)
    return output



#process several files in a path 
def aprocess_image_files(image_files):
    """Process metadata on image files."""

    new_docs = []
    tasks = []
    for image_file in image_files:
        time.sleep(3)
        task = aprocess_image_file(image_file)
        tasks.append(task)
    return tasks

def get_nodes_from_objs(
    objs: List[ReceiptInfo], image_files: List[str]
) -> TextNode:
    """Get nodes from objects."""
    nodes = []
    for image_file, obj in zip(image_files, objs):
        node = TextNode(
            text=obj.summary,
            metadata={
                "cloth_type": obj.cloth_type,
                "color": obj.color,
                "season": obj.season,
                "category": obj.category,
                "image_file": str(image_file),
            },
            excluded_embed_metadata_keys=["image_file"],
            excluded_llm_metadata_keys=["image_file"],
        )
        nodes.append(node)
    return nodes



def classify_images_in_directory(img_dir, sample_size):
    """Receive the path of images and return text node 
    objects of the classifications"""

    image_files = get_image_files(img_dir, sample= sample_size)
    outputs = aprocess_image_files(image_files)
    nodes = get_nodes_from_objs(outputs, image_files)
    return nodes


def classify_single_image(img_dir):
    """Receive the path of an image and return text node 
    object of the classification"""
    output = aprocess_image_file(img_dir)
    #add the description to the index
    node = get_nodes_from_objs([output], [img_dir])
    return node


def print_nodes(nodes):
    """receive text node object and print them"""
    for i in range(len(nodes)):
        print(nodes[i].get_content(metadata_mode="all"))


def find_matching(img_dir):
    """Receive the path of an image and return text node 
    object of the matching image classification"""
    output = aprocess_image_file_matching(img_dir)
    output = output.summary
    output = output.split("would")
    result = vector_db.retrieve(output[0])
    return result