import torch
import os
from src.dataset import Multimodal_Datasets


def get_data(args, dataset, split='train'):
    alignment = 'a' if args.aligned else 'na'
    trunc_factor = 20 if args.trunc else None
    data_path = os.path.join(args.data_path, dataset) + f"_{split}_{alignment}{('_trunc'+str(trunc_factor)) if args.trunc else ''}.dt"
    if not os.path.exists(data_path):
        print(f"  - Creating new {split} data")
        data = Multimodal_Datasets(args.data_path, dataset, split, args.aligned, trunc_factor=trunc_factor)
        torch.save(data, data_path)
    else:
        print(f"  - Found cached {split} data")
        data = torch.load(data_path)
    return data


def save_load_name(args, name=''):
    if args.aligned:
        name = name if len(name) > 0 else 'aligned_model'
    elif not args.aligned:
        name = name if len(name) > 0 else 'nonaligned_model'

    return name + '_' + args.model


def save_model(args, model, name=''):
    name = save_load_name(args, name)
    torch.save(model, f'pre_trained_models/{name}.pt')


def load_model(args, name=''):
    name = save_load_name(args, name)
    model = torch.load(f'pre_trained_models/{name}.pt')
    return model
