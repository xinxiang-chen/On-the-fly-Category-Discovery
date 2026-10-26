import os
import pandas as pd
import numpy as np
from copy import deepcopy

from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset

from data.data_utils import subsample_instances
from config import pets_root


class OxfordPetDataset(Dataset):
    """Oxford-IIIT Pet，从官方标注 split 文件解析。

    - 目录布局沿用 torchvision：root/oxford-iiit-pet/{images, annotations}。
    - class_id == CLASS-ID - 1（0..36），与 torchvision `OxfordIIITPet`（`int(label)-1`）
      及前人 `backups/data/pets.py` 完全一致，故 `train_classes=range(19)` 选到的 19 个
      known 品种与前人逐类相同。
    - 方案 B：只把 `trainval`(3680) 当训练池，不用官方 `test.txt`。
    """

    base_folder = 'oxford-iiit-pet'
    _RESOURCES = (
        ('https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz',
         '5c4f3ee8e5d25df40f4fd59a7f44e54c'),
        ('https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz',
         '95a8c909bbe2e81eed6a22bccdf3f68f'),
    )

    def __init__(self, root, split='trainval', transform=None, target_transform=None,
                 loader=default_loader, download=False):

        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        self._images_folder = os.path.join(self.root, self.base_folder, 'images')
        self._anns_folder = os.path.join(self.root, self.base_folder, 'annotations')

        if download:
            self._download()

        self._load_metadata()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.uq_idxs = np.array(range(len(self)))

    def _load_metadata(self):
        # split 文件每行： Image CLASS-ID SPECIES BREED-ID （#开头为注释；trainval/test 无注释行）
        split_file = os.path.join(self._anns_folder, f'{self.split}.txt')
        data = pd.read_csv(split_file, sep=' ', comment='#',
                           names=['img_id', 'target', 'species', 'breed_id'])
        data['target'] = data['target'] - 1  # CLASS-ID 1..37 -> 0..36
        data['filepath'] = data['img_id'].apply(
            lambda x: os.path.join(self._images_folder, f'{x}.jpg'))
        self.data = data

    def _check_integrity(self):
        for filepath in self.data['filepath']:
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        base = os.path.join(self.root, self.base_folder)
        if os.path.isdir(self._images_folder) and os.path.isdir(self._anns_folder):
            print('Files already downloaded and verified')
            return

        os.makedirs(base, exist_ok=True)
        for url, md5 in self._RESOURCES:
            filename = url.rpartition('/')[2]
            download_url(url, base, filename, md5)
            with tarfile.open(os.path.join(base, filename), 'r:gz') as tar:
                tar.extractall(path=base)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = sample['filepath']
        target = sample['target']
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.uq_idxs[idx]


def subsample_dataset(dataset, idxs):

    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.data = dataset.data[mask]
    dataset.uq_idxs = dataset.uq_idxs[mask]

    return dataset


def subsample_classes(dataset, include_classes=range(19)):

    # Pets 的 class_id 已是 0-based（0..36），不像 cub/scars 需要 +1 偏移
    include_classes_pets = np.array(include_classes)
    cls_idxs = [x for x, (_, r) in enumerate(dataset.data.iterrows())
                if int(r['target']) in include_classes_pets]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = np.unique(train_dataset.data['target'])

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.data['target'] == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_oxford_pets_datasets(train_transform, test_transform, train_classes=range(19),
                             prop_train_labels=0.5, split_train_val=False, seed=0):

    np.random.seed(seed)

    # 方案 B：整个 trainval split 作为训练池
    whole_training_set = OxfordPetDataset(root=pets_root, split='trainval', transform=train_transform)

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # 方案 B / 选项 2：无官方 test split，直接复用 unlabelled 余量作 'test'（配 test_transform）
    test_dataset = deepcopy(train_dataset_unlabelled)
    test_dataset.transform = test_transform

    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
    }

    return all_datasets


if __name__ == '__main__':

    x = get_oxford_pets_datasets(None, None, split_train_val=False,
                                 train_classes=range(19), prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].data["target"].values))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].data["target"].values))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')
