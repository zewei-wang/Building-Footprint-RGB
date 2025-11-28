#!/usr/bin/env python3
"""Extract setup-to-training cells from Building-Footprint-RGB.ipynb."""

import json  # noqa: F401 - mirrored from notebook imports
import os
import random  # noqa: F401 - mirrored from notebook imports
import sys
import time  # noqa: F401 - mirrored from notebook imports
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
import torch  # noqa: F401 - mirrored from notebook imports
from p_tqdm import p_umap  # noqa: F401 - mirrored from notebook imports
from torch import nn  # noqa: F401 - mirrored from notebook imports
from tqdm import tqdm

import warnings

import libs.solaris as sol
from networks.vgg16_unet import get_modified_vgg16_unet

warnings.filterwarnings("ignore", category=FutureWarning)


def visualize_sample(img_dir: str, bldg_dir: str, prefix: str) -> None:
    """Render a representative RGB, LiDAR, and mask triple for quick inspection."""
    sample = 'img1423'
    img_file = prefix + 'RGB+ELEV_' + sample + '.tif'
    img_path = os.path.join(img_dir, img_file)
    img = skimage.io.imread(img_path)
    rgb = img[..., :3]
    elev = img[..., -1]

    bldg_file = prefix + 'geojson_buildings_' + sample + '.geojson'
    bldg_path = os.path.join(bldg_dir, bldg_file)
    mask = sol.vector.mask.footprint_mask(bldg_path, reference_im=img_path)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(rgb)
    ax[0].set_title('Satellite image')
    ax[1].imshow(elev, cmap='gray', vmin=0, vmax=5000)
    ax[1].set_title('LiDAR elevation')
    ax[2].imshow(mask, cmap='Blues')
    ax[2].set_title('Building footprint masks')
    fig.tight_layout()
    plt.show()


def ensure_mask_dir(img_dir: str, bldg_dir: str, mask_dir: str, prefix: str) -> None:
    """Build mask rasters from GeoJSON annotations when they are not available."""
    if os.path.exists(mask_dir):
        return

    os.mkdir(mask_dir)
    img_file_list = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
    for img_file in tqdm(img_file_list, desc='Generating masks'):
        chip_id = os.path.splitext(img_file)[0].split('_')[-1]
        bldg_file = prefix + 'geojson_buildings_' + chip_id + '.geojson'
        mask_file = prefix + 'mask_buildings_' + chip_id + '.tif'
        sol.vector.mask.footprint_mask(
            os.path.join(bldg_dir, bldg_file),
            out_file=os.path.join(mask_dir, mask_file),
            reference_im=os.path.join(img_dir, img_file),
        )


def build_dataframes(img_dir: str, mask_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Mirror the notebook logic for building train/test CSV splits."""
    img_file_list = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
    mask_file_list = [f for f in os.listdir(mask_dir) if f.endswith('.tif')]
    mask_file_subset = [
        f for f in mask_file_list if f.replace('mask_buildings_', 'RGB+ELEV_') in img_file_list
    ]

    img_path_list = sorted(os.path.join(img_dir, f) for f in img_file_list)
    mask_path_list = sorted(os.path.join(mask_dir, f) for f in mask_file_subset)
    assert len(img_path_list) == len(mask_path_list)

    total_df = pd.DataFrame({'image': img_path_list, 'label': mask_path_list})
    split_mask = np.random.rand(len(total_df)) < 0.7
    train_df = total_df[split_mask]
    test_df = total_df[~split_mask]

    train_csv_path = './data/buildings/split_train_data.csv'
    test_csv_path = './data/buildings/split_blind_test.csv'
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    train_subset_frac = 0.1
    train_small_df = train_df.sample(frac=train_subset_frac, random_state=42)
    train_small_csv_path = './data/buildings/split_train_data_small.csv'
    train_small_df.to_csv(train_small_csv_path, index=False)

    print(f"{len(total_df)} images in total, {len(train_df)} - train, {len(test_df)} - test.")
    print(
        'Training subset: {} images ({:.0f}% of training data).'.format(
            len(train_small_df), train_subset_frac * 100
        )
    )

    return train_df, test_df, train_small_df


def configure_training(train_small_csv_path: str, test_csv_path: str) -> Tuple[dict, sol.nets.train.Trainer]:
    """Prepare the solaris Trainer using the RGB-only configuration."""
    config = sol.utils.config.parse('./configs/buildings/RGB-only.yml')
    config['training_data_csv'] = train_small_csv_path
    config['inference_data_csv'] = test_csv_path

    model_ckpt_dir = Path(config['training']['callbacks']['model_checkpoint']['filepath']).parent
    model_ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_dest_dir = Path(config['training']['model_dest_path']).parent
    model_dest_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(config['inference']['output_dir']).parent
    results_dir.mkdir(parents=True, exist_ok=True)

    print('Checkpoint dir:', model_ckpt_dir)
    print('Model dir:', model_dest_dir)
    print('Results dir:', results_dir)

    custom_model = get_modified_vgg16_unet(in_channels=config['data_specs']['channels'])
    custom_model_dict = {
        'model_name': 'modified_vgg16_unet',
        'weight_path': None,
        'weight_url': None,
        'arch': custom_model,
    }

    trainer = sol.nets.train.Trainer(config, custom_model_dict=custom_model_dict)
    return config, trainer


def main(run_training: bool = False) -> None:
    print(sys.executable)
    plt.style.use('seaborn-notebook')

    data_dir = './data/buildings/'
    img_dir = os.path.join(data_dir, 'RGB+ELEV')
    bldg_dir = os.path.join(data_dir, 'geojson_buildings')
    mask_dir = os.path.join(data_dir, 'mask_buildings')
    prefix = 'SN2_buildings_train_AOI_2_Vegas_'

    # visualize_sample(img_dir, bldg_dir, prefix)
    ensure_mask_dir(img_dir, bldg_dir, mask_dir, prefix)

    build_dataframes(img_dir, mask_dir)
    train_small_csv_path = './data/buildings/split_train_data_small.csv'
    test_csv_path = './data/buildings/split_blind_test.csv'

    config, trainer = configure_training(train_small_csv_path, test_csv_path)

    if run_training:
        trainer.train()
    else:
        print('Training skipped. Pass run_training=True to main() to enable it.')


if __name__ == '__main__':
    main(run_training=True)
