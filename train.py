from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from T1mra_dataset import T1w2MraDataset
from UNet import UNet
import torch
import torch.optim as optim
from torch.nn import MSELoss
from PerceptualLoss import VGG16FeatureExtractor, PerceptualLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

feature_extractor = VGG16FeatureExtractor()
feature_extractor.to(device)
perceptual_loss = PerceptualLoss(feature_extractor, MSELoss)

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

dataset = T1w2MraDataset("/Users/asagilmore/src/t1-mra/processed-data/T1W",
                         "/Users/asagilmore/src/t1-mra/processed-data/MRA",
                         transform=train_transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

model = UNet(1, 1)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 25  # Number of epochs

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in dataloader:
        images, masks = images.to(device).float(), masks.to(device).float()
        optimizer.zero_grad()
        outputs = model(images)
        loss = perceptual_loss.get_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")