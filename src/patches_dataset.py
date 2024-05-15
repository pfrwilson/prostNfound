from .nct2013.utils import load_or_create_resized_bmode_data
from .nct2013.data_access import data_accessor
from torch.utils.data import Dataset
from torchvision import transforms as T
import numpy as np 
from .utils import PatchView
import torch
from skimage.transform import resize


class BModePatchesDataset(Dataset):
    _bmode_data, _core_id_2_idx = load_or_create_resized_bmode_data((1024, 1024))
    _metadata_table = data_accessor.get_metadata_table()

    def __init__(
        self,
        core_ids,
        patch_size,
        stride,
        needle_mask_threshold,
        prostate_mask_threshold,
        transform=None,
    ):
        self.core_ids = sorted(core_ids)
        N = len(self.core_ids)

        self._images = [
            self._bmode_data[self._core_id_2_idx[core_id]] for core_id in core_ids
        ]
        self._prostate_masks = np.zeros((N, 256, 256))
        for i, core_id in enumerate(core_ids):
            self._prostate_masks[i] = data_accessor.get_prostate_mask(core_id)
        self._needle_masks = np.zeros((N, 512, 512))
        for i, core_id in enumerate(core_ids):
            self._needle_masks[i] = data_accessor.get_needle_mask(core_id)
        self._patch_views = PatchView.build_collection_from_images_and_masks(
            self._images,
            window_size=patch_size,
            stride=stride,
            align_to="topright",
            mask_lists=[self._prostate_masks, self._needle_masks],
            thresholds=[prostate_mask_threshold, needle_mask_threshold],
        )

        self._metadata_dicts = []
        for core_id in self.core_ids:
            metadata = (
                self._metadata_table[self._metadata_table.core_id == core_id]
                .iloc[0]
                .to_dict()
            )
            self._metadata_dicts.append(metadata)

        self._indices = []
        for i, pv in enumerate(self._patch_views):
            self._indices.extend([(i, j) for j in range(len(pv))])

        self.transform = transform

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        i, j = self._indices[idx]
        pv = self._patch_views[i]

        item = {}
        item["patch"] = pv[j] / 255.0

        metadata = self._metadata_dicts[i].copy()
        item.update(metadata)

        if self.transform is not None:
            item = self.transform(item)
        return item


class RFPatchesDataset(Dataset):
    _metadata_table = data_accessor.get_metadata_table()

    def __init__(
        self,
        core_ids,
        patch_size_mm=(5, 5),
        patch_stride_mm=(1, 1),
        needle_mask_threshold=0.6,
        prostate_mask_threshold=-1,
        transform=None,
    ):
        self.core_ids = core_ids
        im_size_mm = 28, 46.06
        im_size_px = data_accessor.get_rf_image(core_ids[0], 0).shape
        self.patch_size_px = int(patch_size_mm[0] * im_size_px[0] / im_size_mm[0]), int(
            patch_size_mm[1] * im_size_px[1] / im_size_mm[1]
        )
        self.patch_stride_px = int(
            patch_stride_mm[0] * im_size_px[0] / im_size_mm[0]
        ), int(patch_stride_mm[1] * im_size_px[1] / im_size_mm[1])

        self._images = []
        for core_id in core_ids:
            image = data_accessor.get_rf_image(core_id, 0) 
            if image.shape != im_size_px:
                image = resize(image, im_size_px)
            self._images.append(image)

        self._prostate_masks = [
            data_accessor.get_prostate_mask(core_id) for core_id in core_ids
        ]
        self._needle_masks = [
            data_accessor.get_needle_mask(core_id) for core_id in core_ids
        ]

        self._patch_views = PatchView.build_collection_from_images_and_masks(
            self._images,
            window_size=self.patch_size_px,
            stride=self.patch_stride_px,
            align_to="topright",
            mask_lists=[self._prostate_masks, self._needle_masks],
            thresholds=[prostate_mask_threshold, needle_mask_threshold],
        )
        self._indices = []
        for i, pv in enumerate(self._patch_views):
            self._indices.extend([(i, j) for j in range(len(pv))])

        self.transform = transform

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        i, j = self._indices[idx]
        metadata = (
            self._metadata_table[self._metadata_table.core_id == self.core_ids[i]]
            .iloc[0]
            .to_dict()
        )
        pv = self._patch_views[i]
        patch = pv[j]

        patch = patch.copy()
        resize(patch, (256, 256))    
        postition = pv.positions[j]

        data = {"patch": patch, **metadata, "position": postition}
        if self.transform is not None:
            data = self.transform(data)

        return data


class SSLTransform:
    def __call__(self, item):
        patch = item["patch"]
        patch = torch.from_numpy(patch).float()
        patch = patch.unsqueeze(0).repeat_interleave(3, dim=0)

        augs = [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        ]
        p1 = T.Compose(augs)(patch)
        p2 = T.Compose(augs)(patch)

        return p1, p2


class Transform:
    def __call__(self, item):
        patch = item["patch"]
        patch = torch.from_numpy(patch).float() 
        patch = patch.unsqueeze(0).repeat_interleave(3, dim=0)
        item["patch"] = patch
        return item

