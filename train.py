from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from T1mra_dataset import T1w2MraDataset
from UNet import UNet

dataset = T1w2MraDataset("/Users/jonahrudin/src/t1-mra/data/registered-mini/T1W",
                         "/Users/jonahrudin/src/t1-mra/data/registered-mini/MRA")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
model = UNet(512*512, 512*512, 1)

# num_epochs = 25  # Number of epochs

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for images, masks in dataloader:
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, masks)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#     print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")