import torch
torch.autograd.set_detect_anomaly(True)

import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
sys.path.append('./scripts')

from scripts.localnet import Localnet
from FocalTverskyLoss import FocalTverskyLoss, gradient_loss, DiceLoss
from dataloader import MedicalImageDataset

f_path = 'dataset_path'
batch_size = 8
num_epochs = 300
lr = 0.005
moving_image_shape, fixed_image_shape = (1, 64, 64, 64), (1, 64, 64, 64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Localnet()
print(model)
model = model.to(device)


focal_tversky_loss = FocalTverskyLoss(alpha=0.5, beta=0.5, gamma=1.0)
mse_loss = nn.MSELoss()
dice_loss = DiceLoss()

dataset = MedicalImageDataset(f_path, moving_image_shape, fixed_image_shape, with_label_inputs=True)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


optimizer = optim.Adam(model.parameters(), lr=lr)


for epoch in range(num_epochs):
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
    epoch_loss = 0
    for inputs, outputs in pbar:

        moving_images, fixed_images, moving_labels, fixed_labels = inputs
        pre_fixed_image, _fixed_labels = outputs

        moving_images, fixed_images, moving_labels, fixed_labels = moving_images.to(device), fixed_images.to(
            device), moving_labels.to(device), fixed_labels.to(device)
        pre_fixed_image, _fixed_labels = pre_fixed_image.to(device), _fixed_labels.to(device)

        out_ddf, pred_labels = model(moving_images, fixed_images, moving_labels)

        loss = focal_tversky_loss(pred_labels, _fixed_labels) + 0.1 * gradient_loss(out_ddf)
        epoch_loss += loss.item()
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()

    average_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch + 1} Average Loss: {average_loss:.4f}")
