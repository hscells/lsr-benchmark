# The code is taken from https://github.com/webis-de/ecir25-corpus-subsampling/blob/main/tests/test_run_pool_corpus_sampler.py
import gzip
import json
from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm
from glob import glob

import ir_datasets
from trectools import TrecPoolMaker, TrecRun


class CorpusSampler(ABC):
    """Sample a set of documents from a large corpus."""

    @abstractmethod
    def sample_corpus(ir_datasets_id: str, runs: list[TrecRun]) -> set[str]:
        """Sample a corpus (returned as a set of document IDs)
        for a given dataset and a set of runs that were used
        to cunstruct the pool.

        Args:
            ir_datasets_id (str): The ir_datasets ID of the dataset.
            runs (list[TrecRun]): The runs used to construct the pool.

        Returns:
            set[str]: The sampled corpus as a set of document IDs.
        """
        pass


class JudgmentPoolCorpusSampler(CorpusSampler):
    def sample_corpus(self, ir_datasets_id: str, runs: list[TrecRun]) -> set[str]:
        """Sample a corpus (returned as a set of document IDs)
        by just returning all judged documents from the judgment pool.

        Args:
            ir_datasets_id (str): The ir_datasets ID of the dataset.
            runs (list[TrecRun]): The runs used to construct the pool.

        Returns:
            set[str]: The judged documents as sampled corpus.
        """
        qrels_iter = ir_datasets.load(ir_datasets_id).qrels_iter()
        ret = set()

        for qrel in qrels_iter:
            ret.add(qrel.doc_id)

        return ret

    def __str__(self) -> str:
        return "judgment-pool"


class RunPoolCorpusSampler(JudgmentPoolCorpusSampler):
    def __init__(self, depth: int):
        """Create a pool of the passed depth for the passed runs as sampled corpus.

        Args:
            depth (int): The depth of the pool
        """
        self.depth = depth

    def sample_corpus(self, ir_datasets_id: str, runs: list[TrecRun]) -> set[str]:
        """Sample a corpus (returned as a set of document IDs)
        by returning the top-k pool of all runs that formed
        the original judgment pool.

        Args:
            ir_datasets_id (str): The ir_datasets ID of the dataset.
            runs (list[TrecRun]): The runs used to construct the pool.

        Returns:
            set[str]: The top-k pool of the runs as sampled corpus
        """
        ret = super().sample_corpus(ir_datasets_id, runs)
        qrels_iter = ir_datasets.load(ir_datasets_id).qrels_iter()
        allowed_query_ids = set()

        for qrel in qrels_iter:
            allowed_query_ids.add(str(qrel.query_id))

        pool = TrecPoolMaker().make_pool(runs, strategy="topX", topX=self.depth).pool
        skipped = 0
        for qid in pool.keys():
            docids = pool[qid]
            if str(qid) not in allowed_query_ids:
                skipped += 1
                continue

            for doc_id in docids:
                ret.add(doc_id)

        print(f"Skipped {skipped} queries without relevance judgments")

        return ret

    def __str__(self) -> str:
        return f"top-{self.depth}-run-pool"

def create_subsample(run_dir, ir_datasets_id, depth, output_dir):
    if not (output_dir/"subsample.json").is_file():
        runs = []
        for i in tqdm(glob(f"{run_dir}/*"), "Load Runs"):
            try:
                runs.append(TrecRun(i))
            except:
                pass
        corpus = list(RunPoolCorpusSampler(depth).sample_corpus(ir_datasets_id, runs))
        with open(f"{output_dir}/subsample.json", "w") as f:
            f.write(json.dumps(corpus))
    ret = json.loads((output_dir/"subsample.json").read_text())
    return ret
    
