"""author: Maximilian Glumann"""

from xray_fov.nn.train import model_train
def train(model_dir, hp, ldrs, mdl, opt):
    model_train(ldrs.train, ldrs.validation, model_dir, mdl.hash, mdl.net, opt, hp.num_epochs, hp.device)