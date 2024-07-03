import torch
import torchvision


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
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(loader)


def tensorboard_write(writer, train_loss, val_loss, epoch,
                      model, val_loader, num_images=4):
    writer.add_scalar('Loss/train', train_loss, global_step=epoch)
    writer.add_scalar('Loss/val', val_loss, global_step=epoch)

    # get validation images
    val_images, _ = next(iter(val_loader))
    val_images = val_images[:num_images]
    with torch.no_grad():
        gen_images = model(val_images)

    image_grid_original = torchvision.utils.make_grid(val_images)
    image_grid_pred = torchvision.utils.make_grid(gen_images)

    writer.add_image('Images/Original', image_grid_original, global_step=epoch)
    writer.add_image('Images/Predicted', image_grid_pred, global_step=epoch)
