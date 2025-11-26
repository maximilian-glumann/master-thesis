"""author: Alexander Ortlieb"""

from tqdm import tqdm
import torch
from torch.nn import BCEWithLogitsLoss
from xray_fov.nn.models.utils import save_model, save_optimizer
from xray_fov.nn.metrics import check_accuracy

def train_epoch(loader, model, optimizer, loss_fn, scaler, device):
    model.to(device)
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        with torch.amp.autocast(device):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        
    return loss

def model_train(train_loader, validation_loader, base_dir, hash, model, optimizer, num_epochs, device):    
    loss_fn = BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(device)

    # training
    for epoch in range(1, num_epochs+1):
        loss = train_epoch(train_loader, model, optimizer, loss_fn, scaler, device)
        if epoch % 5 == 0:
            check_accuracy(epoch, validation_loader, model, device=device)

    save_model(model, hash, base_dir)
    save_optimizer(optimizer, hash, base_dir)

            