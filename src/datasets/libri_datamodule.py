import os
from typing import Optional

from torch.utils.data import DataLoader, Dataset

from src.datasets.libri_dataset import LibriSamples, LibriSamplesTest


class LibriDataModule():
    def __init__(
        self,
        data_dir: str = '/shared/youngkim/hw4p2_student_data/hw4p2_student_data',
        freq_mask: bool = False,
        add_noise: bool = False,
        batch_size: int = 128,
        num_workers: int = 0,
        pin_memory: bool = True,
        verbose: bool = False,
    ):

        self.data_dir = data_dir
        self.freq_mask = freq_mask
        self.add_noise = add_noise
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.verbose = verbose

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""

        if not os.path.exists(self.data_dir):
            print("No Data")

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`."""

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val:

            if stage in (None, 'fit'):
                self.data_train = LibriSamples(self.data_dir, 'train', self.freq_mask, self.add_noise)
                self.data_val = LibriSamples(self.data_dir, 'dev', False, False)
                if self.verbose: print('train/val dataset loaded.')

        if not self.data_test:
            if stage in (None, 'predict'):
                self.data_test = LibriSamplesTest(self.data_dir, 'test_order.csv')
                if self.verbose: print('test dataset loaded.')


    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=LibriSamples.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=LibriSamples.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=LibriSamplesTest.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
