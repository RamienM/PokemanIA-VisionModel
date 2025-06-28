import torch
import torch.nn as nn
import numpy as np
from PIL import Image

class RGBToMask(nn.Module):
    """
    Transformación personalizada para convertir una imagen RGB a una máscara de clases.
    """
    def __init__(self, color_to_class):
        super(RGBToMask, self).__init__()
        self.color_to_class = color_to_class

    def forward(self, rgb_image):
        """
        Args:
            rgb_image (PIL.Image or numpy.ndarray): Imagen RGB.
        
        Returns:
            mask (torch.Tensor): Máscara de etiquetas (2D) con valores enteros.
        """
        # Convertir la imagen a un array de NumPy si es PIL.Image
        if isinstance(rgb_image, Image.Image):
            rgb_image = np.array(rgb_image)
        
        h, w, _ = rgb_image.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Mapear cada color a una etiqueta
        for color, label in self.color_to_class.items():
            mask[np.all(rgb_image == np.array(color), axis=-1)] = label
        
        # Convertir a tensor de PyTorch
        return torch.from_numpy(mask).long()

class MaskToRGB(nn.Module):
    """
    Transformación personalizada para convertir una máscara de clases a una imagen RGB.
    """
    def __init__(self, color_to_class):
        super(MaskToRGB, self).__init__()
        self.color_to_class = color_to_class
        self.class_to_color = {v: k for k, v in color_to_class.items()}

    def forward(self, mask):
        """
        Args:
            mask (torch.Tensor): Máscara de etiquetas (2D) con valores enteros.
        
        Returns:
            rgb_image (PIL.Image): Imagen RGB.
        """
        # Convertir la máscara a un array de NumPy si es un tensor
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        
        print(mask.shape)
        if len(mask.shape) == 3:
            _,h, w = mask.shape
            mask = mask.argmax(axis=0)
        else:
            h, w = mask.shape
        
        rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Mapear cada etiqueta a su color correspondiente
        for label, color in self.class_to_color.items():
            rgb_image[mask == label] = color
        
        # Convertir a PIL.Image
        return rgb_image