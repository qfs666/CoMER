import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
from zipfile import ZipFile

import numpy as np
import pytorch_lightning as pl
import torch
from comer.datamodule.dataset import CROHMEDataset
# from dataset import CROHMEDataset
from PIL import Image
from torch import FloatTensor, LongTensor
from torch.utils.data.dataloader import DataLoader

from .vocab import vocab
# from vocab import vocab

Data = List[Tuple[str, Image.Image, List[str]]]

MAX_SIZE = 32e4  # change here accroading to your GPU memory

# load data
def data_iterator(
    data: Data,
    batch_size: int,
    batch_Imagesize: int = MAX_SIZE,
    maxlen: int = 1024,
    maxImagesize: int = MAX_SIZE,
):
    fname_batch = []
    feature_batch = []
    label_batch = []
    feature_total = []
    label_total = []
    fname_total = []
    biggest_image_size = 0

    data.sort(key=lambda x: x[1].size[0] * x[1].size[1])

    i = 0
    for fname, fea, lab in data:
        size = fea.size[0] * fea.size[1]
        fea = np.array(fea)
        if size > biggest_image_size:
            biggest_image_size = size
        batch_image_size = biggest_image_size * (i + 1)
        if len(lab) > maxlen:
            print("sentence", i, "length bigger than", maxlen, "ignore")
        elif size > maxImagesize:
            print(
                f"image: {fname} size: {fea.shape[0]} x {fea.shape[1]} =  bigger than {maxImagesize}, ignore"
            )
        else:
            if batch_image_size > batch_Imagesize or i == batch_size:  # a batch is full
                fname_total.append(fname_batch)
                feature_total.append(feature_batch)
                label_total.append(label_batch)
                i = 0
                biggest_image_size = size
                fname_batch = []
                feature_batch = []
                label_batch = []
                fname_batch.append(fname)
                feature_batch.append(fea)
                label_batch.append(lab)
                i += 1
            else:
                fname_batch.append(fname)
                feature_batch.append(fea)
                label_batch.append(lab)
                i += 1

    # last batch
    fname_total.append(fname_batch)
    feature_total.append(feature_batch)
    label_total.append(label_batch)
    print("total ", len(feature_total), "batch data loaded")
    return list(zip(fname_total, feature_total, label_total))


def extract_data(archive: str, dir_name: str) -> Data:
    """Extract all data need for a dataset from zip archive

    Args:
        archive (ZipFile):
        dir_name (str): dir name in archive zip (eg: train, test_2014......)

    Returns:
        Data: list of tuple of image and formula
    """
    with archive.open(f"data/{dir_name}/caption.txt", "r") as f:
        captions = f.readlines()
    data = []
    for line in captions:
        tmp = line.decode().strip().split()
        img_name = tmp[0]
        formula = tmp[1:]
        with archive.open(f"data/{dir_name}/img/{img_name}.bmp", "r") as f:
            # move image to memory immediately, avoid lazy loading, which will lead to None pointer error in loading
            img = Image.open(f).copy()
        data.append((img_name, img, formula))

    print(f"Extract data from: {dir_name}, with data size: {len(data)}")

    return data


@dataclass
class Batch:
    img_bases: List[str]  # [b,]
    imgs: FloatTensor  # [b, 1, H, W]
    mask: LongTensor  # [b, H, W]
    indices: List[List[int]]  # [b, l]

    def __len__(self) -> int:
        return len(self.img_bases)

    def to(self, device) -> "Batch":
        return Batch(
            img_bases=self.img_bases,
            imgs=self.imgs.to(device),
            mask=self.mask.to(device),
            indices=self.indices,
        )


def collate_fn(batch):
    # batch 是一个列表，里面每个元素是一个样本（fname, image, label）
    fnames = [item[0] for item in batch]
    images_x = [item[1] for item in batch]
    seqs_y = [vocab.words2indices(item[2]) for item in batch]

    heights_x = [img.size(1) for img in images_x]
    widths_x = [img.size(2) for img in images_x]

    n_samples = len(images_x)
    max_height_x = max(heights_x)
    max_width_x = max(widths_x)

    # 创建批量图像张量 [B, 1, H, W]，和 mask [B, H, W]
    x = torch.zeros(n_samples, 1, max_height_x, max_width_x)
    x_mask = torch.ones(n_samples, max_height_x, max_width_x, dtype=torch.bool)

    for idx, img in enumerate(images_x):
        h, w = heights_x[idx], widths_x[idx]
        x[idx, :, :h, :w] = img
        x_mask[idx, :h, :w] = 0  # mask 中 0 表示有效像素区域

    return Batch(fnames, x, x_mask, seqs_y)


def build_dataset(archive, folder: str, batch_size: int):
    data = extract_data(archive, folder)
    return data_iterator(data, batch_size)


class CROHMEDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        zipfile_path: str,
        train_batch_size: int = 8,
        eval_batch_size: int = 4,
        num_workers: int = 5,
        scale_aug: bool = False,
    ) -> None:
        super().__init__()
        self.zipfile_path = zipfile_path
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.scale_aug = scale_aug

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = CROHMEDataset(
                self.zipfile_path,
                True,
                self.scale_aug,
            )

            self.val_dataset = CROHMEDataset(
                self.zipfile_path,
                False,
                self.scale_aug,
            )
        if stage == "test" or stage is None:
            self.test_dataset = CROHMEDataset(
                self.zipfile_path,
                False,
                self.scale_aug,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )



if __name__=="__main__":
    path = "/mnt/pfs_l2/jieti_team/CV/qfs/project/server_project/std_project/CoMER-master/data"
    
    model = CROHMEDatamodule(zipfile_path=path) 
    model.setup("fit")
    model.setup("test")
    for i in model.train_dataloader():
        print(i.img_bases,i.imgs.shape,i.mask.shape,i.indices)