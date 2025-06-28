from utils.model.STELLE_model import STELLE_Seg

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt

from tqdm import tqdm
import numpy as np

class STELLE_Trainer():
    def __init__(self, dataset, batch_size = 8):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = STELLE_Seg().to(self.device)
        self.model.load_state_dict(torch.load("models_STELLE_imp/STELLE_Seg_2.pth", weights_only=True))
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
    
    def train(self, num_epochs=10, log_freq=10):
        torch.cuda.empty_cache()
        self.model.train()

        writer = SummaryWriter()  # guarda logs en runs/

        global_step = 0
        for epoch in tqdm( range(num_epochs), desc=f"Epocas {num_epochs}" ):
            epoch_total_loss = 0.0

            for batch in self.dataloader:
                inputs, mask = batch
                inputs = inputs.to(self.device)
                mask = mask.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(inputs)
                loss = self.criterion(output, mask)
                loss.backward()
                self.optimizer.step()

                epoch_total_loss += loss.item()
                global_step += 1

                # Logs cada log_freq pasos
                if global_step % log_freq == 0:
                    # 1. Log scalar de pérdida
                    writer.add_scalar("Loss/train", loss.item(), global_step)

                    # 2. Log gradientes por capaten
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            writer.add_histogram(f"Gradients/{name}", param.grad.detach().cpu(), global_step)

                    # 3. Log imágenes: predicciones vs. target en color
                    with torch.no_grad():
                        # Asumimos salida [B, C, H, W] → convertir a clases [B, 1, H, W]
                        pred_class = torch.argmax(output, dim=1, keepdim=True)  # [B, 1, H, W]
                        gt_class = mask.unsqueeze(1)                            # [B, 1, H, W]

                        # Aplicar colormap a predicciones y ground truth
                        pred_rgb = self.apply_colormap(pred_class, num_classes=14)
                        gt_rgb = self.apply_colormap(gt_class, num_classes=14)

                        # Entrada normalizada
                        input_vis = inputs.clone()
                        if input_vis.max() > 1:
                            input_vis = input_vis / 255.0

                        # Log de imágenes
                        writer.add_images("Input", input_vis, global_step)
                        writer.add_images("Prediction_Color", pred_rgb, global_step)
                        writer.add_images("GroundTruth_Color", gt_rgb, global_step)

        writer.close()
        torch.cuda.empty_cache()
    
    def apply_colormap(self, mask, num_classes):
        """
        Convierte un tensor [B, 1, H, W] de clases en una imagen RGB [B, 3, H, W] con colores únicos para cada clase.
        
        Args:
            mask (Tensor): Tensor de clases, shape [B, 1, H, W].
            num_classes (int): Número de clases.
        
        Returns:
            rgb (Tensor): Imagen coloreada, shape [B, 3, H, W], con valores en [0,1].
        """
        # One-hot encoding: de [B, 1, H, W] a [B, num_classes, H, W]
        one_hot = F.one_hot(mask.squeeze(1).long(), num_classes=num_classes)  # [B, H, W, num_classes]
        one_hot = one_hot.permute(0, 3, 1, 2).float()  # [B, num_classes, H, W]

        # Crear un colormap personalizado usando Matplotlib (usamos 'tab20' que tiene colores distintos)
        cm = plt.get_cmap('tab20', num_classes)
        colors = [cm(i) for i in range(num_classes)]
        # Extraer solo los canales RGB y convertir a tensor (valores entre 0 y 1)
        colors = torch.tensor([list(color[:3]) for color in colors], dtype=torch.float32, device=mask.device)  # [num_classes, 3]

        # Usar tensordot: one_hot tiene shape [B, num_classes, H, W] y colors [num_classes, 3]
        # El resultado tendrá forma [B, H, W, 3]
        rgb = torch.tensordot(one_hot, colors, dims=([1], [0]))  # [B, H, W, 3]

        # Reordenar dimensiones a [B, 3, H, W] para compatibilidad con TensorBoard e imagenes
        rgb = rgb.permute(0, 3, 1, 2)
        return rgb
    
    def evaluate(self, dataset=None, global_step=0, tag_prefix="Eval"):
        self.model.eval()
        aux = DataLoader(dataset, batch_size=1, shuffle=True)
        dataloader = aux if dataset else self.dataloader

        total_pixels = 0
        correct_pixels = 0
        iou_per_class = torch.zeros(16).to(self.device)
        class_counts = torch.zeros(16).to(self.device)

        writer = SummaryWriter()

        with torch.no_grad():
            for i, (inputs, mask) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                mask = mask.to(self.device)

                output = self.model(inputs)
                pred = torch.argmax(output, dim=1)

                correct_pixels += (pred == mask).sum()
                total_pixels += mask.numel()

                for cls in range(16):
                    pred_inds = (pred == cls)
                    target_inds = (mask == cls)
                    intersection = (pred_inds & target_inds).sum()
                    union = (pred_inds | target_inds).sum()
                    if union > 0:
                        iou_per_class[cls] += intersection.float() / union
                        class_counts[cls] += 1

                if i == 0:
                    pred_rgb = self.apply_colormap(pred.unsqueeze(1), num_classes=16)
                    gt_rgb = self.apply_colormap(mask.unsqueeze(1), num_classes=16)
                    input_vis = inputs.clone()
                    if input_vis.max() > 1:
                        input_vis = input_vis / 255.0
                    writer.add_images(f"{tag_prefix}/Input", input_vis, global_step)
                    writer.add_images(f"{tag_prefix}/Prediction", pred_rgb, global_step)
                    writer.add_images(f"{tag_prefix}/GroundTruth", gt_rgb, global_step)

        pixel_acc = correct_pixels.float() / total_pixels
        mean_iou = (iou_per_class / class_counts.clamp(min=1)).mean()

        # Logging to TensorBoard
        writer.add_scalar(f"{tag_prefix}/Pixel_Accuracy", pixel_acc, global_step)
        writer.add_scalar(f"{tag_prefix}/mIoU", mean_iou, global_step)
        for cls in range(14):
            iou_value = iou_per_class[cls] / class_counts[cls].clamp(min=1)
            writer.add_scalar(f"{tag_prefix}/IoU_class_{cls}", iou_value, global_step)

        writer.close()

        # Print numeric results
        print(f"Pixel Accuracy: {pixel_acc.item():.4f}")
        print(f"Mean IoU: {mean_iou.item():.4f}")
        for cls in range(14):
            iou_value = (iou_per_class[cls] / class_counts[cls].clamp(min=1)).item()
            print(f"IoU class {cls}: {iou_value:.4f}")

    
    def predict(self, np_image):
        """
        Predice la máscara de clases a partir de una imagen en formato numpy.

        Args:
            np_image (np.ndarray): Imagen [H, W, C] o [C, H, W], valores en [0, 255] o [0, 1].

        Returns:
            pred_mask (np.ndarray): Máscara predicha [H, W], valores enteros de clase.
        """
        self.model.eval()

        # Asegurar [C, H, W] y tipo float32
        if np_image.ndim == 3 and np_image.shape[0] <= 4:  # [C, H, W]
            image_tensor = torch.from_numpy(np_image).float()
        elif np_image.ndim == 3 and np_image.shape[2] <= 4:  # [H, W, C]
            image_tensor = torch.from_numpy(np_image).permute(2, 0, 1).float()
        else:
            raise ValueError("Formato de imagen no reconocido. Usa [H,W,C] o [C,H,W].")

        # Normalizar si es necesario
        if image_tensor.max() > 1:
            image_tensor = image_tensor / 255.0

        image_tensor = image_tensor.unsqueeze(0).to(self.device)  # [1, C, H, W]

        # Predicción
        with torch.no_grad():
            output = self.model(image_tensor)  # [1, C, H, W]
            pred_class = torch.argmax(output, dim=1)  # [1, H, W]

        return pred_class.squeeze(0).cpu().numpy().astype(np.uint8)  # [H, W]