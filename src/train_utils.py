import random
import torch


def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for input_scan, mask_scan in loader:
        for slice_idx in range(len(input_scan)): 
            slice = input_scan[0][:, :, slice_idx]  # Access the slice
            slice, mask_scan = input_scan.to(device), mask_scan.to(device)
            optimizer.zero_grad()
            output = model(slice)
            loss = criterion(output, mask_scan, slice_idx)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    return running_loss/len(loader)
_

def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
    return val_loss / len(loader), correct / len(loader.dataset)
