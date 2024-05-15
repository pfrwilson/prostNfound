import json
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from medAI.datasets.nct2013 import data_accessor

DATA_ROOT = os.environ['DATA_ROOT']
PREPROCESSED_DATA_DIR = os.path.join(DATA_ROOT, 'preprocessed_data')


def load_or_create_resized_bmode_data(image_size): 
    dataset_dir = os.path.join(PREPROCESSED_DATA_DIR, f'images_{image_size[0]}x{image_size[1]}')
    if not os.path.exists(dataset_dir): 
        print(f'Creating preprocessed dataset at {dataset_dir}')

        core_ids = sorted(data_accessor.get_metadata_table().core_id.unique().tolist())
        data_buffer = np.zeros((len(core_ids), *image_size), dtype=np.uint8)
        for i, core_id in enumerate(tqdm(core_ids, desc="Preprocessing B-mode images")): 
            bmode = data_accessor.get_bmode_image(core_id, 0)
            bmode = bmode / 255.0
            from skimage.transform import resize
            bmode = resize(bmode, image_size, anti_aliasing=True)
            bmode = (bmode * 255).astype(np.uint8)
            data_buffer[i, ...] = bmode 

        print(f'Saving preprocessed dataset at {dataset_dir}')
        os.makedirs(dataset_dir, exist_ok=True)
        np.save(os.path.join(dataset_dir, 'bmode.npy'), data_buffer)
        core_id_2_idx = {core_id: idx for idx, core_id in enumerate(core_ids)}
        with open(os.path.join(dataset_dir, 'core_id_2_idx.json'), 'w') as f: 
            json.dump(core_id_2_idx, f)

    bmode_data = np.load(os.path.join(dataset_dir, 'bmode.npy'), mmap_mode='r')
    with open(os.path.join(dataset_dir, 'core_id_2_idx.json'), 'r') as f: 
        core_id_2_idx = json.load(f)

    return bmode_data, core_id_2_idx