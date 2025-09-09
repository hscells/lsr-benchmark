import unittest
from pathlib import Path

from lsr_benchmark import register_to_ir_datasets
import ir_datasets

class TestIrdsIntegration(unittest.TestCase):
    def test_fails_for_non_existing_dataset(self):
        with self.assertRaises(Exception):
            register_to_ir_datasets("this-does-not-exist")

    def test_works_for_none_as_dataset(self):
        register_to_ir_datasets()

    def test_from_local_directory(self):
        resource_dir = str(Path(__file__).parent / "resources" / "example-dataset")
        register_to_ir_datasets(resource_dir)
        ds = ir_datasets.load(resource_dir)

        self.assertEqual(3, len(list(ds.queries_iter())))
        self.assertEqual(4, len(list(ds.docs_iter())))
