import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os
from datasets.block import BlockDataset, LatentBlockDataset
import numpy as np
from models.vqvae import VQVAE

def load_tang():
    train = []
    val = []
    sites = ['m1s1', 'm1s2', 'm1s3', 'm2s1', 'm2s2', 'm3s1']
    transform = transforms.Normalize(0.5, 0.5)
    for site in sites:
        train.append(np.load(f'D:/school/research/vqvae/data/tang_img/train_img_{site}.npy'))
        val.append(np.load(f'D:/school/research/vqvae/data/tang_img/val_img_{site}.npy'))
    train = np.concatenate(train, axis=0)
    train = np.reshape(train, (train.shape[0], 1, 50, 50))
    train_new = np.zeros((train.shape[0], 1, 32, 32))
    for i in range(len(train)):
        res = train[i,0,9:41,9:41]
        train_new[i] = res
    val = np.concatenate(val, axis=0)
    val = np.reshape(val, (val.shape[0], 1, 50, 50))
    val_new = np.zeros((train.shape[0], 1, 32, 32))
    for i in range(len(val)):
        res = val[i,0,9:41,9:41]
        val_new[i] = res
    return torch.FloatTensor(train_new), torch.FloatTensor(val_new)

def load_tang_rsp():
    train = []
    val = []
    sites = ['m2s1']
    for site in sites:
        train.append(np.load(f'D:/school/research/vqvae/data/tang_rsp/trainRsp_{site}.npy'))
        val.append(np.load(f'D:/school/research/vqvae/data/tang_rsp/valRsp_{site}.npy'))
    train = np.concatenate(train, axis=0)
    train = np.reshape(train, (train.shape[0], 1, train.shape[1]))
    val = np.concatenate(val, axis=0)
    val = np.reshape(val, (val.shape[0], 1, val.shape[1]))
    return torch.FloatTensor(train[..., :296]), torch.FloatTensor(val[..., :296])

def load_cifar():
    train = datasets.CIFAR10(root="data", train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Grayscale(),
                                 # transforms.Normalize(
                                 #     (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                 transforms.Normalize(0.5, 0.5),

                             ]))

    val = datasets.CIFAR10(root="data", train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Grayscale(),
                               # transforms.Normalize(
                               #     (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               transforms.Normalize(0.5, 0.5),
                           ]))
    return train, val

def load_cifar100():
    train = datasets.CIFAR100(root="data", train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Grayscale(),
                                 # transforms.Normalize(
                                 #     (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                 transforms.Normalize(0.5, 0.5),

                             ]))

    val = datasets.CIFAR100(root="data", train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Grayscale(),
                               # transforms.Normalize(
                               #     (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               transforms.Normalize(0.5, 0.5),
                           ]))
    return train, val

def load_block():
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path + \
                     '/data/randact_traj_length_100_n_trials_1000_n_contexts_1.npy'

    train = BlockDataset(data_file_path, train=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(
                                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ]))

    val = BlockDataset(data_file_path, train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(
                               (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ]))
    return train, val


def load_latent_block():
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path + \
                     '/data/latent_e_indices.npy'

    train = LatentBlockDataset(data_file_path, train=True,
                               transform=None)

    val = LatentBlockDataset(data_file_path, train=False,
                             transform=None)
    return train, val


def data_loaders(train_data, val_data, batch_size):
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True)
    return train_loader, val_loader


def load_data_and_data_loaders(dataset, batch_size):
    if dataset == 'TANG':
        training_data, validation_data = load_tang()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        x_train_var = np.var(training_data.numpy() / 255.0)

    elif dataset == 'TANGRSP':
        training_data, validation_data = load_tang_rsp()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        x_train_var = np.var(training_data.numpy() / 255.0)

    elif dataset == 'CIFAR100':
        training_data, validation_data = load_cifar100()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        x_train_var = np.var(training_data.data / 255.0)

    elif dataset == 'CIFAR10':
        training_data, validation_data = load_cifar()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        x_train_var = np.var(training_data.data / 255.0)

    elif dataset == 'BLOCK':
        training_data, validation_data = load_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)

        x_train_var = np.var(training_data.data / 255.0)
    elif dataset == 'LATENT_BLOCK':
        training_data, validation_data = load_latent_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)

        x_train_var = np.var(training_data.data)

    else:
        raise ValueError(
            'Invalid dataset: only CIFAR10 and BLOCK datasets are supported.')

    return training_data, validation_data, training_loader, validation_loader, x_train_var


def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()


def load_model(model_filename):
    path = os.getcwd() + '/results/'

    if torch.cuda.is_available():
        data = torch.load(path + model_filename)
    else:
        data = torch.load(path + model_filename, map_location=lambda storage, loc: storage)

    params = data["hyperparameters"]

    model = VQVAE(params['n_hiddens'], params['n_residual_hiddens'],
                  params['n_residual_layers'], params['n_embeddings'],
                  params['embedding_dim'], params['beta']).to('cuda')

    model.load_state_dict(data['model'])

    return model, data

def save_model_and_results(model, results, hyperparameters, timestamp):
    SAVE_MODEL_PATH = os.getcwd() + '/results'

    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    torch.save(results_to_save,
               SAVE_MODEL_PATH + '/vqvae_data_' + timestamp + '.pth')
