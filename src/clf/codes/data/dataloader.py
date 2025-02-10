import sys
from argparse import Namespace
from typing import Type

from torchvision import transforms
from torch.utils.data import DataLoader

sys.path.append("..")
from codes.data.dataset import BUETDataset, PrivatePCGDataset, SynPCGDataset
from common.transform_funcs import (
    ToTensor, 
    ScaleSignal,
    AlignLength,
    Downsampler,
    ExtractCenter
)
from common.augment import (
    RandomMask,
    RandomShift,
    RandomFlip, 
    AddBreathingSound,
    StretchSignal,
    RandomScaleSignal,
)

def apply_augmentation(key: str, params: Namespace) -> bool:
    """
    Args:

    Returns:

    """
    if key == "mask":
        return hasattr(params, "aug_mask_ratio")
    elif key == "shift":
        return hasattr(params, "max_shift_ratio")
    elif key == "flip":
        return hasattr(params, "flip_rate")
    elif key == "breath":
        return hasattr(params, "breathing_scale")
    elif key == "stretch":
        return hasattr(params, "stretch_ratio")
    elif key == "scale":
        return hasattr(params, "scale_ratio")
    else:
        raise ValueError

def prepare_preprocess(
    params: Namespace, 
    is_train: bool,
) -> Type[transforms.Compose]:
    """
    Prepare and compose transform functions.
    Args:
        params (Namespace): 
        is_train (bool): 
    Returns:
        composed
    """
    transformations = []
    transformations.append(
        Downsampler(params.target_freq))
    transformations.append(ScaleSignal())

    # Simple augmentations.
    if is_train:
        transformations.append(
            AlignLength(int(params.max_duration * params.target_freq))
        )
        if apply_augmentation("mask", params):
            transformations.append(
                RandomMask(params.aug_mask_ratio))
        if apply_augmentation("shift", params):
            transformations.append(
                RandomShift(params.max_shift_ratio))
        if apply_augmentation("flip", params):
            transformations.append(
                RandomFlip(params.flip_rate))
        if apply_augmentation("breath", params):
            transformations.append(
                AddBreathingSound(params.breathing_scale, params.target_freq))
        if apply_augmentation("stretch", params):
            transformations.append(
                StretchSignal(params.stretch_ratio))
        if apply_augmentation("scale", params):
            transformations.append(
                RandomScaleSignal(params.scale_ratio))
    else:
        transformations.append(
            ExtractCenter(int(params.max_duration * params.target_freq))
        )

    # ToTensor and compose.
    transformations.append(ToTensor())
    composed = transforms.Compose(transformations)
    return composed

def prepare_dataset(params: Namespace, data_split: str, transformations):

    if data_split == "train":
        data_lim = params.data_lim
    elif data_split == "val":
        data_lim = params.val_data_lim
    else:
        data_lim = None

    if params.dataset == "buet":
        dataset = BUETDataset(
            data_split, 
            params.seed, 
            params.target_dx,
            data_lim,
            transformations
        )
    elif params.dataset == "private":
        dataset = PrivatePCGDataset(
            data_split, 
            params.seed, 
            params.target_dx,
            data_lim,
            transformations
        )
    elif params.dataset == "syn":
        dataset = SynPCGDataset(
            data_split, 
            params.seed, 
            params.target_dx,
            params.dataset_ver_norm,
            params.dataset_ver_dx,
            data_lim,
            transformations
        )
    else:
        raise NotImplementedError

    return dataset

def prepare_dataloader(
    params: Namespace,
    data_split: str,
    is_train: bool,
) -> Type[DataLoader]:

    transformations = prepare_preprocess(params, is_train)
    
    dataset = prepare_dataset(params, data_split, transformations)

    if not is_train:
        drop_last = False
    else:
        if params.data_lim is not None:
            data_lim = params.data_lim
        else:
            data_lim = 1e10

        if params.batch_size > data_lim:
            drop_last = False
        else:
            drop_last = True

    batch_size = params.batch_size

    loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=is_train, 
        drop_last=drop_last, 
        num_workers=params.n_workers
    )
    return loader