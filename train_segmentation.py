import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataloader import SegmentationDataset 
from model import UNet
import numpy as np
import os
import time

def calculate_iou(pred, target, num_classes=4):
    pred = pred.numpy().flatten()
    target = target.numpy().flatten()
    valid = target != 255
    return np.mean([
        (np.sum((pred[valid] == c) & (target[valid] == c)) + 1e-6) / 
        (np.sum((pred[valid] == c) | (target[valid] == c)) + 1e-6)
        for c in range(num_classes)
    ])

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_iou = 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        start_time = time.time()
        
        for images, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss, val_iou = 0, 0
        with torch.no_grad():
            for images, masks in val_loader:
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()
                val_iou += calculate_iou(torch.argmax(outputs, 1), masks)
        
        # Metrics
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        avg_iou = val_iou / len(val_loader)
        epoch_time = time.time() - start_time
        
        scheduler.step(avg_iou)
        
        print(f'Epoch {epoch+1:2d} | Time: {epoch_time:5.1f}s | '
              f'Train: {avg_train:.4f} | Val: {avg_val:.4f} | '
              f'IoU: {avg_iou:.4f} | LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Saved best model (IoU {avg_iou:.4f})')

def main():
    # CPU optimizations
    torch.set_num_threads(os.cpu_count() or 1)
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Verify dataset paths
    train_path = os.path.join('dataset', 'train')
    val_path = os.path.join('dataset', 'val')
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError("Dataset folders not found. Expected structure: dataset/{train,val}/{images,labels}")
    
    # Initialize datasets
    train_set = SegmentationDataset(train_path, train_transform)
    val_set = SegmentationDataset(val_path, val_transform)
    
    # Data loaders
    batch_size = 4
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(0, os.cpu_count())
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(0, os.cpu_count())
    )
    
    # Model setup
    model = UNet(n_classes=4)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    
    # Train
    train_model(model, train_loader, val_loader, criterion, optimizer, 50)

if __name__ == '__main__':
    main()