import unittest
from collections import Counter
from esm_msr.data import ProteinCyclingBatchSampler


class DummyDataset:
    def __init__(self, pdb_id: str, items: list):
        self.pdb_id = pdb_id
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = dict(self.items[idx])
        item['pdb'] = self.pdb_id
        return item


class TestColumnAwareSampler(unittest.TestCase):
    def test_mixed_columns_and_singles(self):
        # 10 columns of 19 items each = 190 items
        # 110 singles = 110 items
        # Total = 300 items. Batch size = 128 -> 2 batches (256 items), 44 leftover
        items = []
        for col_id in range(10):
            for var_id in range(19):
                items.append({
                    'flip_key': f'PDB|12|{col_id}',
                    'subset_type': 'cond',
                    'val': var_id,
                })
        for s_id in range(110):
            items.append({
                'flip_key': '',
                'subset_type': 'single',
                'val': s_id,
            })

        ds = DummyDataset('1ABC', items)
        batch_size = 128
        sampler = ProteinCyclingBatchSampler(
            datasets=[ds],
            batch_size=batch_size,
            train_list=['1ABC'],
            strategy='all',
            rng_seed=42,
        )

        self.assertEqual(len(sampler), 2)
        batches = list(sampler)
        self.assertEqual(len(batches), 2)

        for b in batches:
            self.assertEqual(len(b), batch_size)
            # Check how many flip items and columns
            b_items = [ds[i] for i in b]
            f_keys = [it['flip_key'] for it in b_items if it['flip_key']]
            counts = Counter(f_keys)
            # All columns present in this batch should be complete (19 items)
            for k, count in counts.items():
                self.assertEqual(count, 19, f"Column {k} was split: has {count} items instead of 19")

    def test_only_singles(self):
        # Library with only single mutations (e.g. no flip columns)
        items = [{'flip_key': '', 'subset_type': 'single', 'val': i} for i in range(500)]
        ds = DummyDataset('1SNG', items)
        batch_size = 128
        sampler = ProteinCyclingBatchSampler(
            datasets=[ds],
            batch_size=batch_size,
            train_list=['1SNG'],
            strategy='all',
            rng_seed=123,
        )
        self.assertEqual(len(sampler), 500 // 128)
        for b in sampler:
            self.assertEqual(len(b), batch_size)

    def test_only_flip_columns(self):
        # Library with only flip columns and 0 singles
        items = []
        for col_id in range(20):
            for var_id in range(15):
                items.append({
                    'flip_key': f'PDB|10|{col_id}',
                    'subset_type': 'cond',
                    'val': var_id,
                })
        ds = DummyDataset('1FLP', items)
        batch_size = 120
        sampler = ProteinCyclingBatchSampler(
            datasets=[ds],
            batch_size=batch_size,
            train_list=['1FLP'],
            strategy='all',
            rng_seed=7,
        )
        self.assertEqual(len(sampler), len(items) // batch_size)
        for b in sampler:
            self.assertEqual(len(b), batch_size)

    def test_multi_dataset_cycling(self):
        # Two datasets with different proportions
        items1 = [{'flip_key': f'1A|{c}', 'subset_type': 'cond'} for c in range(10) for _ in range(19)]
        items1 += [{'flip_key': '', 'subset_type': 'single'} for _ in range(100)]

        items2 = [{'flip_key': '', 'subset_type': 'single'} for _ in range(300)]

        ds1 = DummyDataset('1A', items1)
        ds2 = DummyDataset('1B', items2)

        batch_size = 128
        sampler = ProteinCyclingBatchSampler(
            datasets=[ds1, ds2],
            batch_size=batch_size,
            train_list=['1A', '1B'],
            strategy='all',
            rng_seed=999,
        )

        expected_batches = (len(items1) // batch_size) + (len(items2) // batch_size)
        self.assertEqual(len(sampler), expected_batches)
        batches = list(sampler)
        self.assertEqual(len(batches), expected_batches)

        # Offsets check: items from ds1 must be in [0, len(items1)-1]
        # items from ds2 must be in [len(items1), len(items1)+len(items2)-1]
        offset2 = len(items1)
        for b in batches:
            self.assertEqual(len(b), batch_size)
            if b[0] < offset2:
                # ds1 batch: every item must be from ds1
                self.assertTrue(all(i < offset2 for i in b))
            else:
                # ds2 batch: every item must be from ds2
                self.assertTrue(all(i >= offset2 for i in b))

    def test_undersized_dataset(self):
        items = [{'flip_key': '', 'subset_type': 'single'} for _ in range(50)]
        ds = DummyDataset('1UND', items)
        sampler = ProteinCyclingBatchSampler(
            datasets=[ds],
            batch_size=128,
            train_list=['1UND'],
            strategy='all',
        )
        self.assertEqual(len(sampler), 0)
        self.assertEqual(list(sampler), [])

    def test_epoch_shuffling_and_reproducibility(self):
        items = [{'flip_key': f'P|1|{c}', 'subset_type': 'cond'} for c in range(8) for _ in range(15)]
        items += [{'flip_key': '', 'subset_type': 'single'} for _ in range(150)]
        ds = DummyDataset('1REP', items)
        batch_size = 64

        sampler1 = ProteinCyclingBatchSampler(
            datasets=[ds],
            batch_size=batch_size,
            train_list=['1REP'],
            strategy='all',
            rng_seed=42,
        )
        sampler2 = ProteinCyclingBatchSampler(
            datasets=[ds],
            batch_size=batch_size,
            train_list=['1REP'],
            strategy='all',
            rng_seed=42,
        )
        # Identical seeds must produce identical batches on epoch 1
        epoch1_s1 = list(sampler1)
        epoch1_s2 = list(sampler2)
        self.assertEqual(epoch1_s1, epoch1_s2)

        # Subsequent iteration (epoch 2) on sampler1 should produce different order
        epoch2_s1 = list(sampler1)
        self.assertEqual(len(epoch1_s1), len(epoch2_s1))
        self.assertNotEqual(epoch1_s1, epoch2_s1)

    def test_variable_column_lengths(self):
        # Varying column lengths: 5, 10, 18, 20
        items = []
        lengths = [5, 10, 18, 20, 8, 14, 19, 12]
        for col_id, length in enumerate(lengths):
            for _ in range(length):
                items.append({'flip_key': f'P|{col_id}', 'subset_type': 'cond'})
        items += [{'flip_key': '', 'subset_type': 'single'} for _ in range(200)]

        ds = DummyDataset('1VAR', items)
        batch_size = 64
        sampler = ProteinCyclingBatchSampler(
            datasets=[ds],
            batch_size=batch_size,
            train_list=['1VAR'],
            strategy='all',
            rng_seed=11,
        )
        expected = len(items) // batch_size
        self.assertEqual(len(sampler), expected)
        for b in sampler:
            self.assertEqual(len(b), batch_size)


if __name__ == '__main__':
    unittest.main()
