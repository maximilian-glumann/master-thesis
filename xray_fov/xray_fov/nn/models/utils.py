"""author: Maximilian Glumann"""

from xray_fov.nn.data.dataloaders import get_loaders
from xray_fov.nn.defaults.hyperparameters import HyperParameters
from dict_hash import shake_128
import dataclasses
import json
import torch

def get_hash(source, model, data_dict, hp, **kwargs):
    kwargs["source"] = source
    kwargs["model"] = model
    kwargs["data_dict"] = data_dict
    kwargs["hp"] = dataclasses.asdict(hp)
    return shake_128(kwargs, hash_length=8)

def create_model(source, model, base_dir, data_dict, hp, **kwargs):
    import importlib
    mod = importlib.import_module(source)
    hash = get_hash(source, model, data_dict, hp, **kwargs)
    with open(base_dir+"/"+hash+'.json', 'w') as f:
        json.dump({"source": source, "model": model, "kwargs": kwargs}, f)
    with open(base_dir+"/"+hash+'_data.json', 'w') as f:
        json.dump(data_dict, f)
    with open(base_dir+"/"+hash+'_hp.json', 'w') as f:
        json.dump(dataclasses.asdict(hp), f)
    return getattr(mod, model)(**kwargs), hash

def save_model(model, hash, base_dir):
    torch.save(model.state_dict(), base_dir+"/"+hash+".pth")

def load_model(model, hash, base_dir):
    model.load_state_dict(torch.load(base_dir+"/"+hash+".pth"))

def create_model_from_hash(hash, base_dir):
    with open(base_dir+"/"+hash+'.json') as f:
        config = json.load(f)
    with open(base_dir+"/"+hash+'_data.json') as f:
        data_dict = json.load(f)
    with open(base_dir+"/"+hash+'_hp.json') as f:
        hp = HyperParameters(json.load(f))
    return create_model(config['source'], config['model'], base_dir, data_dict, hp, **config['kwargs'])[0]

def create_optimizer(source, optimizer, base_dir, hash, model, **kwargs):
    import importlib
    mod = importlib.import_module(source)
    with open(base_dir+"/"+hash+'_opt.json', 'w') as f:
        json.dump({"source": source, "optimizer": optimizer, "kwargs": kwargs}, f)
    return getattr(mod, optimizer)(model.parameters(), **kwargs)

def save_optimizer(optimizer, hash, base_dir):
    print('=> Saving optimizer')
    torch.save(optimizer.state_dict(), base_dir+"/"+hash+"_opt.pth")

def load_optimizer(optimizer, hash, base_dir):
    print('=> Loading optimizer')
    optimizer.load_state_dict(torch.load(base_dir+"/"+hash+"_opt.pth"))

def create_optimizer_from_hash(hash, base_dir, model):
    with open(base_dir+"/"+hash+'_opt.json') as f:
        config = json.load(f)
    return create_optimizer(config['source'], config['optimizer'], base_dir, hash, model, **config['kwargs'])

def create_loaders(data_dir, transform, batch_size=8, num_workers=20, pin_memory=False, **kwargs):
    return get_loaders(data_dir,transform,batch_size,num_workers,pin_memory,**kwargs)+({"data_dir": data_dir, "transform": transform, "kwargs": kwargs},)

def create_loaders_from_hash(hash, base_dir, batch_size=8, num_workers=20, pin_memory=False):
    with open(base_dir+"/"+hash+'_data.json') as f:
        data_dict = json.load(f)
    return get_loaders(data_dict["data_dir"],data_dict["transform"],batch_size,num_workers,pin_memory,**data_dict["kwargs"])