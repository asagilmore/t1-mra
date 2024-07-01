import random
import torch


def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss/len(loader)


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


class RandomRotationTransform90:
    def __call__(self, img):
        rotations = random.randint(0, 3)
        return torch.rot90(img, rotations, [1, 2])


class RandomFlipTransform():
    def __call__(self, img):
        flip = random.random() > 0.5
        axis = random.randint(1, 2)
        if flip:
            return torch.flip(img, [axis])
        else:
            return img
