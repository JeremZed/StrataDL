

import cv2
import hashlib
import numpy as np
from PIL import Image

EXTENSION_IMG = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def hash_frame(frame):
    """
    Permet de générer un hash SHA256 à partir d'une image (frame).
    """
    # Convertir l'image en bytes
    _, buffer = cv2.imencode('.jpg', frame)
    img_bytes = buffer.tobytes()

    # Calculer le hash SHA256
    return hashlib.sha256(img_bytes).hexdigest()

def get_resize_dim_img_pil(img, ratio):
    """
        Permet de retourner les nouvelles dimensions d'une image PIL après application d'un ratio
    """
    width, height = img.size
    new_w, new_h = int(width * ratio), int(height * ratio)

    return new_w, new_h

def get_resize_dim_img_np(img, ratio):
    """
        Permet de retourner les nouvelles dimensions d'une image numpy après application d'un ratio
    """
    height, width = img.shape[:2]
    new_w, new_h = int(width * ratio), int(height * ratio)

    return new_w, new_h

def resize_image(img, ratio: float = 1.0, target_size: tuple[int,int] | None = None):
    """
    Redimensionne une image (PIL ou NumPy) selon un ratio ou une taille cible.

    Args:
        img: Image PIL.Image ou np.ndarray.
        ratio: Facteur d'échelle (ex: 0.5 = réduire, 2.0 = agrandir).
               Ignoré si target_size est fourni.
        target_size: Tuple (width, height) pour redimensionnement fixe.
                     Si fourni, ratio sera ignoré.

    Returns:
        Image du même type que l'entrée.
    """
    if target_size is not None:
        new_w, new_h = target_size
    else:
        if ratio <= 0:
            raise ValueError("Le ratio doit être strictement positif")
        if isinstance(img, Image.Image):
            new_w, new_h = get_resize_dim_img_pil(img, ratio)
        elif isinstance(img, np.ndarray):
            new_w, new_h = get_resize_dim_img_np(img, ratio)
        else:
            raise TypeError("img doit être une instance de PIL.Image.Image ou numpy.ndarray")

    # Redimensionnement selon le type
    if isinstance(img, Image.Image):
        resample = Image.Resampling.LANCZOS if (ratio < 1 or (target_size and new_w*new_h < img.size[0]*img.size[1])) else Image.Resampling.BICUBIC
        return img.resize((new_w, new_h), resample=resample)

    elif isinstance(img, np.ndarray):
        interp = cv2.INTER_AREA if (ratio < 1 or (target_size and new_w*new_h < img.shape[1]*img.shape[0])) else cv2.INTER_CUBIC
        return cv2.resize(img, (new_w, new_h), interpolation=interp)

    else:
        raise TypeError("img doit être une instance de PIL.Image.Image ou numpy.ndarray")