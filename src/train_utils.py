import torch
import torchvision


class RandomRotation90(torch.nn.Module):
    def forward(self, img, mask):
        k = torch.randint(0, 4, (1,)).item()
        return torch.rot90(img, k, [1, 2]), torch.rot90(mask, k, [1, 2])


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
    return running_loss / len(loader)


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


def tensorboard_write(writer, device, train_loss, val_loss, epoch,
                      model, val_loader, num_images=4,
                      adam_optim=False):
    writer.add_scalar('Loss/train', train_loss, global_step=epoch)
    writer.add_scalar('Loss/val', val_loss, global_step=epoch)

    if adam_optim:
        for i, param_group in enumerate(adam_optim.param_groups):
            writer.add_scalar(f'Adam_optim/lr_{i}',
                              param_group['lr'], global_step=epoch)
            writer.add_scalar(f'Adam_optim/beta1_{i}',
                              param_group['betas'][0], global_step=epoch)
            writer.add_scalar(f'Adam_optim/beta2_{i}',
                              param_group['betas'][1], global_step=epoch)
            writer.add_scalar(f'Adam_optim/epsilon_{i}',
                              param_group['eps'], global_step=epoch)
            writer.add_scalar(f'Adam_optim/weight_decay_{i}',
                              param_group['weight_decay'], global_step=epoch)

    # get validation images
    input_images, mask_images = next(iter(val_loader))
    input_images, mask_images = input_images.to(device), mask_images.to(device)
    input_images = input_images[:num_images]
    mask_images = mask_images[:num_images]

    with torch.no_grad():
        gen_images = model(input_images)

    image_grid_original = grid_from_tensor(mask_images)
    image_grid_pred = grid_from_tensor(gen_images)
    image_grid_input = grid_from_tensor(input_images)

    writer.add_image('Images/Input', image_grid_input, global_step=epoch)
    writer.add_image('Images/Original', image_grid_original, global_step=epoch)
    writer.add_image('Images/Predicted', image_grid_pred, global_step=epoch)


def grid_from_tensor(image_tensor, nrow=5):
    # handle 3d slices
    if image_tensor.shape[1] != 1:
        middle_slice_idx = image_tensor.shape[2] // 2
        image_tensor = image_tensor[:, :, middle_slice_idx, :, :]

    image_grid = torchvision.utils.make_grid(image_tensor.cpu(), nrow=nrow)
    min_val = torch.min(image_grid)
    image_grid = (image_grid - min_val) / (torch.max(image_grid) - min_val)

    return image_grid
