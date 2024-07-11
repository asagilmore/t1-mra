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


def train_scans(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, masks in loader:
        output_scan = []
        inputs = inputs.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        for i in range(inputs.shape[1]):
            scan_slice = inputs[:, i, :, :]
            scan_slice = scan_slice.unsqueeze(1)
            outputs = model(scan_slice)
            output_scan.append(outputs)
        stacked_outputs = torch.stack(output_scan)
        loss = criterion(stacked_outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_loss = running_loss / len(inputs)

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, masks in loader:
            output_scan = []
            inputs = inputs.to(device)
            masks = masks.to(device)
            for i in range(inputs.shape[1]):
                scan_slice = inputs[:, i, :, :]
                scan_slice = scan_slice.unsqueeze(1)
                outputs = model(scan_slice)
                output_scan.append(outputs)
            stacked_outputs = torch.stack(output_scan)
            loss = criterion(stacked_outputs, masks)
            running_loss += loss.item()
            running_loss = running_loss / len(inputs)

    return running_loss / len(loader)
'''
for scan in loader:
    output_volume
    for slice in scan:
        model_out = model(slice)
        output_volume.concat(model_out)

    output_volume_FM = get_feature_map(output_volume)
    mask_volume_FM = get_feature_map(mask_volume)

    for index in range(output_volume_FM.shape[0]):
        loss = criterion(output_volume_FM[index], mask_volume_FM[index])
        loss.backward()

    optimizer.step()
    running_loss += loss.item()


for scan in loader:
    output_volume
    for slice in scan:
        model_out = model(slice)
        output_volume.concat(model_out)

    output_volume_FM_axis1 = get_feature_map(output_volume)
    mask_volume_FM_axis1 = get_feature_map(mask_volume)

    for index in range(output_volume_FM.shape[0]):
        loss_axis1 = criterion(output_volume_FM_axis1[index],
        mask_volume_FM_axis1[index])
        loss_axis2 = criterion(output_volume_FM_axis2[index],
        mask_volume_FM_axis2[index])
        loss_axis3 = criterion(output_volume_FM_axis3[index],
        mask_volume_FM_axis3[index])

        loss = loss_axis1 + loss_axis2 + loss_axis3
        loss.backward()

    optimizer.step()
    running_loss += loss.item()

'''


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
    image_grid = torchvision.utils.make_grid(image_tensor.cpu(), nrow=nrow)
    min_val = torch.min(image_grid)
    image_grid = (image_grid - min_val) / (torch.max(image_grid) - min_val)

    return image_grid

