import os
from glob import glob
from time import time

import numpy as np
import pandas as pd
import torch
from fire import Fire
from glog import logger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

pd.options.display.float_format = lambda x: '{:.0f}'.format(x) if round(x, 0) == x else '{:,.3f}'.format(x)


def parse_tb(path):
    _dir = os.path.dirname(path)
    files = sorted(glob(f'{_dir}/*tfevents*'))
    if not files:
        return {}
    ea = EventAccumulator(files[0])
    ea.Reload()

    res = {}
    for k in ('train_acc', 'train_loss', 'val_acc', 'val_loss'):
        try:
            vals = [x.value for x in ea.Scalars(k)]
            f = np.min if 'loss' in k else np.max
            res[k] = f(vals)
        except Exception:
            logger.exception(f'Can not process {k} from {files[0]}')
            res[k] = None
    return res


def explore_models(models, batch_size, img_size):
    batch = torch.rand((batch_size, 3, img_size, img_size)).to('cuda:0')

    for m in models:
        t0 = time()
        model = torch.jit.load(m).to('cuda:0')
        t1 = time()
        _ = model(batch)
        t2 = time()

        d = {'load time': t1 - t0,
             'predict time': t2 - t1,
             'name': m
             }
        tb_data = parse_tb(m)
        d.update(tb_data)
        yield d


def main(pattern="./**/*.trcd", batch_size=16, img_size=256):
    models = glob(pattern, recursive=True)
    data = []
    for x in explore_models(models, batch_size=batch_size, img_size=img_size):
        data.append(x)

    df = pd.DataFrame(data).sort_values('name')
    print(df)


if __name__ == '__main__':
    Fire(main)
