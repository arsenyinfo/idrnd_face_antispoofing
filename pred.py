import argparse
import os
from functools import lru_cache
from glob import glob

import albumentations as albu
import cv2
import numpy as np
import pandas as pd
import torch
from torch.jit import load
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

BATCH_SIZE = 32


def make_crops(img, target, idx):
    w, h, _ = img.shape
    assert w == h
    margin = w - target
    half = int(margin / 2)
    crops = [lambda img: img[:-margin, :-margin, :],
             lambda img: img[:-margin, margin:, :],
             lambda img: img[margin:, margin:, :],
             lambda img: img[margin:, :-margin, :],
             lambda img: img[half:-half, half:-half, :],
             lambda img: cv2.resize(img, (target, target))
             ]
    return crops[idx](img)


def get_normalize():
    normalize = albu.Normalize()

    def process(x):
        r = normalize(image=x)
        return r['image']

    return process


NORM_FN = get_normalize()


@lru_cache(8)
def read_img(x, target=384):
    x = cv2.imread(x)
    x = cv2.resize(x, (target, target))
    x = NORM_FN(x)
    return x


class TestAntispoofDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __getitem__(self, index):
        n_crops = 6
        img_idx = index // n_crops
        crop_idx = index % n_crops
        image_info = self.paths[img_idx]
        img = read_img(image_info['path'])
        img = make_crops(img, target=256, idx=crop_idx)
        return image_info['id'], image_info['frame'], np.transpose(img, (2, 0, 1))

    def __len__(self):
        return len(self.paths) * 6


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path-images-csv', type=str, required=True)
    parser.add_argument('--path-test-dir', type=str, required=True)
    parser.add_argument('--path-submission-csv', type=str, required=True)
    args = parser.parse_args()

    # prepare image paths
    test_dataset_paths = pd.read_csv(args.path_images_csv)
    path_test_dir = args.path_test_dir

    paths = [{'id': row.id,
              'frame': row.frame,
              'path': os.path.join(path_test_dir, row.path)}
             for _, row in test_dataset_paths.iterrows()]

    dataset = TestAntispoofDataset(paths=paths)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # predict
    samples, frames, probabilities = [], [], []

    models = [load(x).to(device) for x in glob('*.trcd')]

    with torch.no_grad():
        for video, frame, batch in tqdm(dataloader):
            batch = batch.to(device)

            acc = np.zeros((BATCH_SIZE, 4), dtype='float32')
            for model in models:
                acc += torch.softmax(model(batch), dim=1).cpu().numpy()
            acc /= len(models)
            proba = acc[:, :-1].sum(axis=1)
            samples.extend(video)
            frames.extend(frame.numpy())
            probabilities.extend(proba)

    # save
    predictions = pd.DataFrame.from_dict({
        'id': samples,
        'frame': frames,
        'probability': probabilities})

    predictions = predictions.groupby('id').probability.mean().reset_index()
    predictions['prediction'] = predictions.probability
    predictions[['id', 'prediction']].to_csv(args.path_submission_csv, index=False)
