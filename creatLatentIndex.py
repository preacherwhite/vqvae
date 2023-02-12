import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import utils
import os
from models.vqvae import VQVAE, VQVAE1d
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Utility functions
"""


def load_model(model_filename):
    path = os.getcwd() + '/results/'

    if torch.cuda.is_available():
        data = torch.load(path + model_filename)
    else:
        data = torch.load(path + model_filename, map_location=lambda storage, loc: storage)

    params = data["hyperparameters"]

    model = VQVAE(params['n_hiddens'], params['n_residual_hiddens'],
                  params['n_residual_layers'], params['n_embeddings'],
                  params['embedding_dim'], params['beta']).to(device)

    model.load_state_dict(data['model'])

    return model, data


"""
End of utilities
"""

model_filename = 'vqvae_data_mon_feb_6_14_51_08_2023.pth'

model, vqvae_data = load_model(model_filename)
model.to(device)
model.eval()

training_data, validation_data, training_loader, validation_loader, x_train_var = utils.load_data_and_data_loaders('TANG', 1)

allLatents = []
for img in training_loader:

    encoding = model.encoder(img.to(device))
    _, _, _, _, index  = model.vector_quantization(model.pre_quantization_conv(encoding))
    allLatents.append(index.detach().cpu().numpy())
allLatents = numpy.concatenate(allLatents, axis=1)
np.save("LatentIndexM2S1", allLatents)

allLatentsVal = []
for img in validation_loader:

    encoding = model.encoder(img.to(device))
    _, _, _, _, index  = model.vector_quantization(model.pre_quantization_conv(encoding))
    allLatentsVal.append(index.detach().cpu().numpy())


allLatentsVal = numpy.concatenate(allLatentsVal, axis=1)

np.save("LatentIndexM2S1Val",allLatentsVal)
