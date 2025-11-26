"""author: Alexander Ortlieb"""

from xray_fov.nn.data.dataset import LungDataset
from torch.utils.data import DataLoader, Subset

BATCH_SIZE = 8

def get_loaders(
    data_dir,
    transform,
    batch_size=BATCH_SIZE,
    num_workers=0,
    pin_memory=False,
    **kwargs
):
    import importlib
    mod = importlib.import_module("xray_fov.nn.data.transform."+transform)
    
    train_ds = LungDataset(
        data_dir['train-images'],
        data_dir['train-masks'],
        transform=mod.transform_train,
        **kwargs
    )
    test_ds = LungDataset(
        data_dir['test-images'],
        data_dir['test-masks'],
        transform=mod.transform_test,
        **kwargs
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    
    test_size =  test_ds.__len__()
    validation_indices = list(range(test_size))[:int(test_size/2)]
    test_indices = list(range(test_size))[int(test_size/2):]
    
    test_loader = DataLoader(
        Subset(test_ds, test_indices),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    validation_loader = DataLoader(
        Subset(test_ds, validation_indices),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, test_loader, validation_loader