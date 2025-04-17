# The code is taken from https://github.com/webis-de/ecir25-corpus-subsampling/blob/main/tests/test_run_pool_corpus_sampler.py
import unittest

import pandas as pd
from trectools import TrecRun

from lsr_benchmark.corpus.corpus_subsampling import RunPoolCorpusSampler
DATASET_ID_FOR_TEST = "disks45/nocr/trec-robust-2004"
RUN_WITH_NO_OVERLAPPING_DOCUMENTS = TrecRun()
RUN_WITH_NO_OVERLAPPING_DOCUMENTS.run_data = pd.DataFrame(
    [
        {"query": "1", "docid": "does-not-exist", "rank": 1, "score": 1},
        {"query": "2", "docid": "does-not-exist", "rank": 1, "score": 1},
    ]
)
RUN_WITH_OVERLAPPING_DOCUMENTS = TrecRun()
RUN_WITH_OVERLAPPING_DOCUMENTS.run_data = pd.DataFrame(
    [
        {"query": "1", "docid": "FBIS4-57944", "rank": 1, "score": 1},
        {"query": "2", "docid": "FR940413-2-00131", "rank": 1, "score": 1},
        {"query": "3", "docid": "LA011890-0177", "rank": 1, "score": 1},
    ]
)


class TestJudgmentPoolCorpusSampler(unittest.TestCase):
    def test_with_empty_runs(self):
        expected = set()
        sampler = RunPoolCorpusSampler(depth=100)

        actual = sampler.sample_corpus(DATASET_ID_FOR_TEST, [])

        self.assertEqual(expected, actual)

    def test_with_run_without_overlapping_doc_ids(self):
        expected = set(["does-not-exist"])
        sampler = RunPoolCorpusSampler(depth=100)

        actual = sampler.sample_corpus(DATASET_ID_FOR_TEST, [RUN_WITH_NO_OVERLAPPING_DOCUMENTS])

        self.assertEqual(expected, actual)

    def test_with_run_with_overlapping_doc_ids(self):
        expected = set(["FBIS4-57944", "FR940413-2-00131", "LA011890-0177"])
        sampler = RunPoolCorpusSampler(depth=100)

        actual = sampler.sample_corpus(DATASET_ID_FOR_TEST, [RUN_WITH_OVERLAPPING_DOCUMENTS])

        self.assertEqual(expected, actual)

    def test_with_multiple_runs(self):
        expected = set(["FBIS4-57944", "FR940413-2-00131", "LA011890-0177", "does-not-exist"])
        sampler = RunPoolCorpusSampler(depth=100)

        actual = sampler.sample_corpus(
            DATASET_ID_FOR_TEST, [RUN_WITH_OVERLAPPING_DOCUMENTS, RUN_WITH_NO_OVERLAPPING_DOCUMENTS]
        )

        self.assertEqual(expected, actual)

    def test_string_representation_depth_10(self):
        expected = "top-10-run-pool"
        actual = str(RunPoolCorpusSampler(depth=10))

        self.assertEqual(expected, actual)

    def test_string_representation_depth_100(self):
        expected = "top-100-run-pool"
        actual = str(RunPoolCorpusSampler(depth=100))

        self.assertEqual(expected, actual)
