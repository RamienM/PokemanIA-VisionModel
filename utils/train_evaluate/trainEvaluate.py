from utils.model.STELLE_model import STELLE_Seg

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from tqdm import tqdm
import numpy as np
import time

class STELLE_Trainer():
    def __init__(self, dataset, batch_size = 8):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = STELLE_Seg().to(self.device)
        self.model.load_state_dict(torch.load("weights/STELLE_Seg.pth", weights_only=True))
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
    
    def train(self, num_epochs=10, log_freq=10):
        """
        Trains the segmentation model for a specified number of epochs.

        The training loop iterates over the dataset batches, performing forward passes,
        computing the loss, backpropagating gradients, and updating model weights.
        It also logs training loss, gradient histograms, and visualizations of inputs,
        predicted masks, and ground truth masks to TensorBoard at specified intervals.

        Args:
            num_epochs (int): Number of epochs to train the model.
            log_freq (int): Frequency (in steps) at which to log training metrics and images.

        Returns:
            None. Training metrics and images are logged to TensorBoard.
        """
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

                if global_step % log_freq == 0:
                    writer.add_scalar("Loss/train", loss.item(), global_step)

                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            writer.add_histogram(f"Gradients/{name}", param.grad.detach().cpu(), global_step)

                    with torch.no_grad():
                        pred_class = torch.argmax(output, dim=1, keepdim=True)  # [B, 1, H, W]
                        gt_class = mask.unsqueeze(1)                            # [B, 1, H, W]


                        pred_rgb = self.apply_colormap(pred_class, num_classes=14)
                        gt_rgb = self.apply_colormap(gt_class, num_classes=14)

                        input_vis = inputs.clone()
                        if input_vis.max() > 1:
                            input_vis = input_vis / 255.0

                        writer.add_images("Input", input_vis, global_step)
                        writer.add_images("Prediction_Color", pred_rgb, global_step)
                        writer.add_images("GroundTruth_Color", gt_rgb, global_step)

        writer.close()
        torch.cuda.empty_cache()
    
    def apply_colormap(self, mask, num_classes):
        """
        Converts a [B, 1, H, W] class tensor into an RGB image [B, 3, H, W] with unique colors for each class.
        
        Args:
            mask (Tensor): Class tensor, shape [B, 1, H, W].
            num_classes (int): Number of classes.
        
        Returns:
            rgb (Tensor): Colored image, shape [B, 3, H, W], with values in [0,1].
        """
        one_hot = F.one_hot(mask.squeeze(1).long(), num_classes=num_classes)  # [B, H, W, num_classes]
        one_hot = one_hot.permute(0, 3, 1, 2).float()  # [B, num_classes, H, W]

        cm = plt.get_cmap('tab20', num_classes)
        colors = [cm(i) for i in range(num_classes)]

        colors = torch.tensor([list(color[:3]) for color in colors], dtype=torch.float32, device=mask.device)  # [num_classes, 3]

        rgb = torch.tensordot(one_hot, colors, dims=([1], [0]))  # [B, H, W, 3]

        rgb = rgb.permute(0, 3, 1, 2)
        return rgb
    
    def _pixel_accuracy_multiclass(self, pred, target):
        # pred: [B, C, H, W] logits o probs
        # target: [B, H, W] con labels 0..C-1 (clase correcta por pixel)
        pred_labels = pred.argmax(dim=1)
        correct = (pred_labels == target).float()
        return correct.sum() / correct.numel()

    def _iou_multiclass(self,pred, target, num_classes=14):
        pred_labels = pred.argmax(dim=1)
        ious = []
        for cls in range(num_classes):
            pred_mask = (pred_labels == cls)
            target_mask = (target == cls)
            intersection = (pred_mask & target_mask).float().sum()
            union = (pred_mask | target_mask).float().sum()
            if union == 0:
                ious.append(torch.tensor(1.0))
            else:
                ious.append(intersection / union)
        return ious

    def _precision_multiclass(self,pred, target, num_classes=14):
        pred_labels = pred.argmax(dim=1)
        precisions = []
        for cls in range(num_classes):
            pred_mask = (pred_labels == cls)
            target_mask = (target == cls)
            tp = (pred_mask & target_mask).sum().float()
            fp = (pred_mask & (~target_mask)).sum().float()
            precisions.append(tp / (tp + fp + 1e-8))
        return precisions

    def _recall_multiclass(self, pred, target, num_classes=14):
        pred_labels = pred.argmax(dim=1)
        recalls = []
        for cls in range(num_classes):
            pred_mask = (pred_labels == cls)
            target_mask = (target == cls)
            tp = (pred_mask & target_mask).sum().float()
            fn = ((~pred_mask) & target_mask).sum().float()
            recalls.append(tp / (tp + fn + 1e-8))
        return recalls

    def _f1_score_multiclass(self, precisions, recalls):
        f1s = []
        for p, r in zip(precisions, recalls):
            f1s.append(2 * (p * r) / (p + r + 1e-8))
        return f1s

    def _measure_inference_time(self,  model, dataset, device, warmup=5):
        model.eval()
        times = []

        for i in range(len(dataset)):
            img, *_ = dataset[i]
            img = img.unsqueeze(0).to(device)  # (1, 1, H, W)

            if i < warmup:
                with torch.no_grad():
                    model(img)
                continue

            start = time.time()
            with torch.no_grad():
                model(img)
            end = time.time()
            times.append(end - start)

        avg_time = np.mean(times)
        return avg_time, 1.0 / avg_time

    def evaluate(self, dataset, num_classes=14):
        """
        Evaluates the segmentation model on a given dataset by computing multiple performance metrics.

        This method iterates over all samples in the dataset, performs model inference,
        and calculates pixel-wise accuracy as well as per-class metrics including IoU (Intersection over Union),
        precision, recall, and F1 score. Metrics are computed both as unweighted averages and weighted averages
        based on the number of pixels per class present in the ground truth.

        Additionally, it measures the average inference time per image and frames per second (FPS)
        to assess the model's efficiency.

        Finally, the function prints out the summarized metrics for easy analysis.

        Args:
            dataset: Dataset object providing images and ground truth segmentation masks.
            num_classes (int): Number of segmentation classes.

        Returns:
            None. Metrics are printed to standard output.
        """
        model = self.model
        device = self.device
        model.eval()

        overall_pixel_accuracies = []
        per_class_ious = [[] for _ in range(num_classes)]
        per_class_precisions = [[] for _ in range(num_classes)]
        per_class_recalls = [[] for _ in range(num_classes)]
        per_class_f1s = [[] for _ in range(num_classes)]

        total_class_pixels = torch.zeros(num_classes, dtype=torch.long)

        for i in tqdm(range(len(dataset))):
            img, map_gt, *_ = dataset[i]
            img = img.unsqueeze(0).to(device)

            if map_gt.ndim == 3 and map_gt.shape[0] == num_classes:
                target = map_gt.argmax(dim=0).to(device)
            else:
                target = map_gt.to(device)

            target_cpu = target.cpu()

            with torch.no_grad():
                output = model(img)
            output = output.cpu()

            unique_labels, counts = torch.unique(target_cpu, return_counts=True)
            for label, count in zip(unique_labels, counts):
                if 0 <= label < num_classes:
                    total_class_pixels[label] += count.item()

            pixel_acc = self._pixel_accuracy_multiclass(output, target_cpu)
            overall_pixel_accuracies.append(pixel_acc.item())

            ious_img = self._iou_multiclass(output, target_cpu, num_classes)
            prec_img = self._precision_multiclass(output, target_cpu, num_classes)
            rec_img = self._recall_multiclass(output, target_cpu, num_classes)
            f1_img = self._f1_score_multiclass(prec_img, rec_img)

            for cls_idx in range(num_classes):
                per_class_ious[cls_idx].append(ious_img[cls_idx].item())
                per_class_precisions[cls_idx].append(prec_img[cls_idx].item())
                per_class_recalls[cls_idx].append(rec_img[cls_idx].item())
                per_class_f1s[cls_idx].append(f1_img[cls_idx].item())

        mean_pixel_acc = np.mean(overall_pixel_accuracies)

        mean_per_class_ious = [np.mean(cls_scores) if cls_scores else 0.0 for cls_scores in per_class_ious]
        mean_per_class_precisions = [np.mean(cls_scores) if cls_scores else 0.0 for cls_scores in per_class_precisions]
        mean_per_class_recalls = [np.mean(cls_scores) if cls_scores else 0.0 for cls_scores in per_class_recalls]
        mean_per_class_f1s = [np.mean(cls_scores) if cls_scores else 0.0 for cls_scores in per_class_f1s]

        overall_mean_iou_unweighted = np.mean(mean_per_class_ious)
        overall_mean_precision_unweighted = np.mean(mean_per_class_precisions)
        overall_mean_recall_unweighted = np.mean(mean_per_class_recalls)
        overall_mean_f1_unweighted = np.mean(mean_per_class_f1s)

        total_pixels_sum = total_class_pixels.sum().item()
        if total_pixels_sum == 0:
            weighted_precision = 0.0
            weighted_recall = 0.0
            weighted_f1 = 0.0
        else:
            weighted_precisions_list = []
            weighted_recalls_list = []
            weighted_f1s_list = []
            valid_weights = []

            for cls_idx in range(num_classes):
                if total_class_pixels[cls_idx] > 0:
                    weighted_precisions_list.append(mean_per_class_precisions[cls_idx] * total_class_pixels[cls_idx].item())
                    weighted_recalls_list.append(mean_per_class_recalls[cls_idx] * total_class_pixels[cls_idx].item())
                    weighted_f1s_list.append(mean_per_class_f1s[cls_idx] * total_class_pixels[cls_idx].item())
                    valid_weights.append(total_class_pixels[cls_idx].item())

            weighted_precision = sum(weighted_precisions_list) / sum(valid_weights)
            weighted_recall = sum(weighted_recalls_list) / sum(valid_weights)
            weighted_f1 = sum(weighted_f1s_list) / sum(valid_weights)
        
        avg_time, fps = self._measure_inference_time(model, dataset, device)

        # Mean (Unweighted) Metrics
        print("Mean (Unweighted) Metrics:")
        print(f"  Pixel Accuracy: {mean_pixel_acc:.4f}")
        print(f"  Mean IoU: {overall_mean_iou_unweighted:.4f}")
        print(f"  Mean Precision: {overall_mean_precision_unweighted:.4f}")
        print(f"  Mean Recall: {overall_mean_recall_unweighted:.4f}")
        print(f"  Mean F1 Score: {overall_mean_f1_unweighted:.4f}")
        print()

        # Weighted Metrics
        print("Weighted Metrics:")
        print(f"  Precision: {weighted_precision:.4f}")
        print(f"  Recall: {weighted_recall:.4f}")
        print(f"  F1 Score: {weighted_f1:.4f}")
        print()

        # Time Metrics
        print("Time Metrics:")
        print(f"  Average Time per Image: {avg_time:.4f} seconds")
        print(f"  Frames Per Second (FPS): {fps:.2f}")

    

    
    def predict(self, np_image):
        """
        Predicts the class mask from a numpy image.

        Args:
            np_image (np.ndarray): Image [H, W, C] or [C, H, W], values in [0, 255] or [0, 1].

        Returns:
            pred_mask (np.ndarray): Predicted mask [H, W], integer class values.
        """
        self.model.eval()

        if np_image.ndim == 3 and np_image.shape[0] <= 4:  # [C, H, W]
            image_tensor = torch.from_numpy(np_image).float()
        elif np_image.ndim == 3 and np_image.shape[2] <= 4:  # [H, W, C]
            image_tensor = torch.from_numpy(np_image).permute(2, 0, 1).float()
        else:
            raise ValueError("Image format not recognized. Use [H,W,C] or [C,H,W].")

        if image_tensor.max() > 1:
            image_tensor = image_tensor / 255.0

        image_tensor = image_tensor.unsqueeze(0).to(self.device)  # [1, C, H, W]

        with torch.no_grad():
            output = self.model(image_tensor)  # [1, C, H, W]
            pred_class = torch.argmax(output, dim=1)  # [1, H, W]

        return pred_class.squeeze(0).cpu().numpy().astype(np.uint8)  # [H, W]