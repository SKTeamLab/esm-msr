import unittest
import random
import torch
from typing import List, Dict, Any
from torch.utils.data import DataLoader, Dataset
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from esm_msr.data import ProteinCyclingBatchSampler


class MockProteinDataset(Dataset):
    """A minimal mock dataset providing pdb and subset_type for ProteinCyclingBatchSampler."""
    def __init__(self, pdb: str, n_items: int, subset_types=('single', 'double')):
        self.pdb = pdb
        self.items = []
        for i in range(n_items):
            st = subset_types[i % len(subset_types)]
            self.items.append({'pdb': pdb, 'subset_type': st, 'idx': i})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class TestResumableDataLoading(unittest.TestCase):
    def setUp(self):
        self.seed = 42
        self.batch_size = 32
        self.datasets = [
            MockProteinDataset('1A32', 200),
            MockProteinDataset('1ENH', 350),
            MockProteinDataset('1PSE', 180),
        ]
        self.train_list = ['1A32', '1ENH', '1PSE']

    def test_continuous_vs_resumed_batch_order(self):
        """Verifies that resuming a sampler at any epoch produces the exact same batch sequences as an uninterrupted run."""
        s_cont = ProteinCyclingBatchSampler(
            self.datasets, self.batch_size, self.train_list, rng_seed=self.seed
        )
        cont_epochs = []
        for ep in range(5):
            s_cont.set_epoch(ep)
            cont_epochs.append(list(s_cont))

        # Check each resume point from epoch 1 to 4
        for resume_ep in range(1, 5):
            s_resumed = ProteinCyclingBatchSampler(
                self.datasets, self.batch_size, self.train_list, rng_seed=self.seed
            )
            s_resumed.set_epoch(resume_ep)
            resumed_batches = list(s_resumed)

            self.assertEqual(
                cont_epochs[resume_ep],
                resumed_batches,
                f"Resumed batches at epoch {resume_ep} did not match uninterrupted epoch {resume_ep}!"
            )

    def test_plain_pytorch_advances_sequentially(self):
        """Verifies that plain iteration without set_epoch matches sequential set_epoch calls."""
        s_plain = ProteinCyclingBatchSampler(
            self.datasets, self.batch_size, self.train_list, rng_seed=self.seed
        )
        plain_epochs = [list(s_plain) for _ in range(3)]

        s_set = ProteinCyclingBatchSampler(
            self.datasets, self.batch_size, self.train_list, rng_seed=self.seed
        )
        set_epochs = []
        for ep in range(3):
            s_set.set_epoch(ep)
            set_epochs.append(list(s_set))

        for ep in range(3):
            self.assertEqual(plain_epochs[ep], set_epochs[ep], f"Plain loop failed to match set_epoch at epoch {ep}")

    def test_idempotent_set_epoch(self):
        """Verifies calling set_epoch multiple times before consumption does not advance or corrupt state."""
        s = ProteinCyclingBatchSampler(
            self.datasets, self.batch_size, self.train_list, rng_seed=self.seed
        )
        s.set_epoch(2)
        s.set_epoch(2)
        batches1 = list(s)

        s_ref = ProteinCyclingBatchSampler(
            self.datasets, self.batch_size, self.train_list, rng_seed=self.seed
        )
        s_ref.set_epoch(2)
        batches_ref = list(s_ref)

        self.assertEqual(batches1, batches_ref)

    def test_state_dict_load_state_dict(self):
        """Verifies state_dict and load_state_dict conform to DataLoader checkpointing."""
        s = ProteinCyclingBatchSampler(
            self.datasets, self.batch_size, self.train_list, rng_seed=self.seed
        )
        s.set_epoch(3)
        sd = s.state_dict()
        self.assertEqual(sd['epoch'], 3)
        self.assertEqual(sd['rng_seed'], self.seed)

        s_new = ProteinCyclingBatchSampler(
            self.datasets, self.batch_size, self.train_list, rng_seed=self.seed
        )
        s_new.load_state_dict(sd)
        self.assertEqual(s_new._completed_epochs, 3)

    def test_lightning_checkpoint_resume_exact_batch_match(self):
        """End-to-end integration test with Lightning Trainer.
        Verifies that resuming from last.ckpt delivers the exact identical sequence of batches
        as an uninterrupted run over all epochs."""
        import tempfile
        import os
        from torch.utils.data import ConcatDataset

        class DummyCollateModel(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.zeros(1))
                self.seen_indices = []

            def training_step(self, batch, batch_idx):
                # record the item indices in the order seen
                indices = [item['idx'] for item in batch]
                self.seen_indices.append(indices)
                return self.p.square()

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.01)

        def dummy_collate(items):
            return items

        concat_ds = ConcatDataset(self.datasets)

        # 1. Uninterrupted run for 4 epochs
        m_cont = DummyCollateModel()
        s_cont = ProteinCyclingBatchSampler(self.datasets, self.batch_size, self.train_list, rng_seed=self.seed)
        dl_cont = DataLoader(concat_ds, batch_sampler=s_cont, collate_fn=dummy_collate)
        t_cont = pl.Trainer(max_epochs=4, accelerator="cpu", logger=False, enable_progress_bar=False)
        t_cont.fit(m_cont, dl_cont)
        cont_seen = m_cont.seen_indices

        # 2. Interrupted run: train 2 epochs and save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            m1 = DummyCollateModel()
            s1 = ProteinCyclingBatchSampler(self.datasets, self.batch_size, self.train_list, rng_seed=self.seed)
            dl1 = DataLoader(concat_ds, batch_sampler=s1, collate_fn=dummy_collate)
            ckpt_cb = ModelCheckpoint(dirpath=tmpdir, save_last=True)
            t1 = pl.Trainer(max_epochs=2, callbacks=[ckpt_cb], accelerator="cpu", logger=False, enable_progress_bar=False)
            t1.fit(m1, dl1)

            # 3. Resume from last.ckpt for 2 more epochs (epochs 2 and 3)
            m2 = DummyCollateModel()
            s2 = ProteinCyclingBatchSampler(self.datasets, self.batch_size, self.train_list, rng_seed=self.seed)
            dl2 = DataLoader(concat_ds, batch_sampler=s2, collate_fn=dummy_collate)
            t2 = pl.Trainer(max_epochs=4, callbacks=[ckpt_cb], accelerator="cpu", logger=False, enable_progress_bar=False)
            t2.fit(m2, dl2, ckpt_path=os.path.join(tmpdir, "last.ckpt"))

            resumed_seen = m1.seen_indices + m2.seen_indices

            self.assertEqual(len(cont_seen), len(resumed_seen))
            self.assertEqual(cont_seen, resumed_seen, "Resumed Lightning batches did not match continuous batches!")


if __name__ == '__main__':
    unittest.main()
