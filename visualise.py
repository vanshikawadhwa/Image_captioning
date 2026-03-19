"""
visualise.py — Utility helpers for displaying images and captions.
"""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from config import IMAGES_DIR


def show_images_with_captions(
    image_paths: list[str],
    captions_dict: dict[str, list[str]],
    n: int = 5,
) -> None:
    """
    Display the first *n* images from *image_paths* alongside their captions.

    Args:
        image_paths:   List of absolute image paths.
        captions_dict: { filename -> [caption, ...] }
        n:             Number of images to show.
    """
    for path in image_paths[:n]:
        filename = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get ground truth captions
        caps = captions_dict.get(filename, ["(no caption)"])
        all_caps = "\n".join([f"• {c}" for c in caps])
        
        plt.figure(figsize=(7, 7))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Filename: {filename}\n{all_caps}", fontsize=10, loc='left', pad=15)
        plt.tight_layout()
        plt.show()


def visualize_prediction(
    image_path: str, 
    predicted_caption: str, 
    reference_captions: list[str] = None
) -> None:
    """
    Overlays the predicted caption directly onto the image.
    """
    img = cv2.imread(image_path)
    if img is None:
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    
    plt.figure(figsize=(9, 7))
    plt.imshow(img)
    plt.axis("off")
    
    # ── Overlay Prediction onto photo ──
    # Draw a semi-transparent background box at the bottom for the prediction
    overlay_text = f"Predicted: {predicted_caption.strip()}"
    props = dict(boxstyle='round', facecolor='black', alpha=0.7)
    
    # Place text at the bottom of the image
    plt.text(w//2, h - (h*0.05), overlay_text, fontsize=12, color='white',
             ha='center', va='bottom', fontweight='bold', bbox=props, wrap=True)
    
    # ── Display Reference Captions in Title ──
    if reference_captions:
        title_text = f"Result: {os.path.basename(image_path)}\n"
        title_text += "Real: " + " | ".join(reference_captions[:3]) # show top 3
        plt.title(title_text, fontsize=10, loc='center', pad=15)
    
    plt.tight_layout()
    plt.show()


def show_feature_sample(
    image_features: dict[str, np.ndarray],
    captions_dict: dict[str, list[str]],
) -> None:
    """Show one image from the feature dictionary as a sanity check."""
    filename = next(iter(image_features))
    img_path = os.path.join(IMAGES_DIR, filename)
    img = cv2.imread(img_path)
    if img is None:
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    caps = captions_dict.get(filename, ["(no caption)"])
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Feature Sample: {filename}\nCaptions: {len(caps)}", fontsize=12)
    plt.xlabel("\n".join(caps), fontsize=9, wrap=True)
    plt.tight_layout()
    plt.show()
