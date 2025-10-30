import unittest
import ir_datasets
from lsr_benchmark import register_to_ir_datasets

class TestLoadEmbeddings(unittest.TestCase):
    def test_load_fails_for_non_existing_dataset(self):
        with self.assertRaises(Exception):
            register_to_ir_datasets("lsr-benchmark")

    def test_load_webis_splade_on_dl_19(self):
        register_to_ir_datasets("msmarco-passage/trec-dl-2019/judged")
        irds = ir_datasets.load("lsr-benchmark/msmarco-passage/trec-dl-2019/judged")

        doc_embeddings = irds.doc_embeddings(model_name="lightning-ir/webis/splade")
        query_embeddings = irds.query_embeddings(model_name="lightning-ir/webis/splade")

        self.assertIsNotNone(doc_embeddings)
        self.assertIsNotNone(query_embeddings)

    def test_load_on_robust04(self):
        register_to_ir_datasets("disks45/nocr/trec-robust-2004/fold1")
        irds = ir_datasets.load("lsr-benchmark/disks45/nocr/trec-robust-2004/fold1")

        doc_embeddings = irds.doc_embeddings(model_name="lightning-ir/webis/splade")
        query_embeddings = irds.query_embeddings(model_name="lightning-ir/webis/splade")

        self.assertIsNotNone(doc_embeddings)
        self.assertIsNotNone(query_embeddings)