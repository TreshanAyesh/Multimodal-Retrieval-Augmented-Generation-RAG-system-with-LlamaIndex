from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
# import requests
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import matplotlib.cm as cm
# from pathlib import Path
# import random
# from typing import Optional

processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

def save_image(tens,filename):
    pred_seg_np = tens.detach().cpu().numpy()
    cmap = cm.get_cmap('viridis')
    norm_pred_seg_np = (pred_seg_np - pred_seg_np.min()) / (pred_seg_np.max() - pred_seg_np.min())
    colored_seg_np = cmap(norm_pred_seg_np)[:, :, :3]
    colored_seg_pil = Image.fromarray((colored_seg_np * 255).astype(np.uint8))
    colored_seg_pil.save(filename)

    
def model_in(image):
    # image = Image.open(filename)
    inputs = processor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0]

    return pred_seg

def apply_mask_image(mask,image,filename,label,save):
    image = np.array(image)
    masked_image = np.zeros_like(image)
    masked_image[mask == 1] = image[mask == 1]
    if save:
        masked_image_pil = Image.fromarray(masked_image)
        masked_image_pil.save(filename)

    return masked_image

def background_rem(pred_seg,image,filename):
    # Convert the tensor to a numpy array
    segmented_img_numpy = pred_seg.detach().cpu().numpy()
    # Get the unique labels
    labels = np.unique(segmented_img_numpy)
    
    combined_mask = np.zeros_like(segmented_img_numpy, dtype=np.uint8)
    # Loop through each label
    for label in labels:
        if label==0:
            continue
        # Create a mask based on the label
        combined_mask[segmented_img_numpy == label] = 1
    _ = apply_mask_image(combined_mask, image, filename,"background_removed",True)