# The code is taken from https://github.com/webis-de/ecir25-corpus-subsampling/blob/main/tests/test_run_pool_corpus_sampler.py
import unittest

import pandas as pd
from trectools import TrecRun

from lsr_benchmark.corpus.corpus_subsampling import RunPoolCorpusSampler
DATASET_ID_FOR_TEST = "disks45/nocr/trec-robust-2004"
RUN_WITH_NO_OVERLAPPING_DOCUMENTS = TrecRun()
RUN_WITH_NO_OVERLAPPING_DOCUMENTS.run_data = pd.DataFrame(
    [
        {"query": "308", "docid": "does-not-exist", "rank": 1, "score": 1},
        {"query": "331", "docid": "does-not-exist", "rank": 1, "score": 1},
    ]
)
RUN_WITH_OVERLAPPING_DOCUMENTS = TrecRun()
RUN_WITH_OVERLAPPING_DOCUMENTS.run_data = pd.DataFrame(
    [
        {"query": "308", "docid": "FBIS4-57944-XXX", "rank": 1, "score": 1},
        {"query": "331", "docid": "FR940413-2-00131-XXX", "rank": 1, "score": 1},
        {"query": "425", "docid": "LA011890-0177-XXX", "rank": 1, "score": 1},
    ]
)

SIZE_POOL_ROBUST_04 = 174787

class TestJudgmentPoolCorpusSampler(unittest.TestCase):
    def test_with_empty_runs(self):
        sampler = RunPoolCorpusSampler(depth=100)

        actual = sampler.sample_corpus(DATASET_ID_FOR_TEST, [])

        # should be the complete judgment pool
        self.assertEqual(SIZE_POOL_ROBUST_04, len(actual))

    def test_with_run_without_overlapping_doc_ids(self):
        expected = set(["does-not-exist"])
        sampler = RunPoolCorpusSampler(depth=100)

        actual = sampler.sample_corpus(DATASET_ID_FOR_TEST, [RUN_WITH_NO_OVERLAPPING_DOCUMENTS])

        for i in expected:
            self.assertIn(i, actual)

        self.assertEqual(SIZE_POOL_ROBUST_04+1, len(actual))

    def test_with_run_with_overlapping_doc_ids(self):
        expected = set(["FBIS4-57944-XXX", "FR940413-2-00131-XXX", "LA011890-0177-XXX"])
        sampler = RunPoolCorpusSampler(depth=100)

        actual = sampler.sample_corpus(DATASET_ID_FOR_TEST, [RUN_WITH_OVERLAPPING_DOCUMENTS])

        for i in expected:
            self.assertIn(i, actual)
        self.assertEqual(SIZE_POOL_ROBUST_04+3, len(actual))

    def test_with_multiple_runs(self):
        expected = set(["FBIS4-57944-XXX", "FR940413-2-00131-XXX", "LA011890-0177-XXX", "does-not-exist"])
        sampler = RunPoolCorpusSampler(depth=100)

        actual = sampler.sample_corpus(
            DATASET_ID_FOR_TEST, [RUN_WITH_OVERLAPPING_DOCUMENTS, RUN_WITH_NO_OVERLAPPING_DOCUMENTS]
        )
        
        for i in expected:
            self.assertIn(i, actual)

        self.assertEqual(SIZE_POOL_ROBUST_04+4, len(actual))

    def test_string_representation_depth_10(self):
        expected = "top-10-run-pool"
        actual = str(RunPoolCorpusSampler(depth=10))

        self.assertEqual(expected, actual)

    def test_string_representation_depth_100(self):
        expected = "top-100-run-pool"
        actual = str(RunPoolCorpusSampler(depth=100))

        self.assertEqual(expected, actual)
