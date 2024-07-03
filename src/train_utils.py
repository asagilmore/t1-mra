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
    val_images, input_images = next(iter(val_loader))
    input_images = input_images[:num_images]
    val_images = val_images[:num_images]

    with torch.no_grad():
        gen_images = model(val_images)

    image_grid_original = torchvision.utils.make_grid(val_images)
    image_grid_pred = torchvision.utils.make_grid(gen_images)
    image_grid_input = torchvision.utils.make_grid(input_images)

    writer.add_image('Images/Input', image_grid_input, global_step=epoch)
    writer.add_image('Images/Original', image_grid_original, global_step=epoch)
    writer.add_image('Images/Predicted', image_grid_pred, global_step=epoch)
