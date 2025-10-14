#!/usr/bin/env python3
import lsr_benchmark
import click
from tirex_tracker import tracking, ExportFormat, register_metadata
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from shutil import rmtree
from more_itertools import chunked
import ir_datasets
from lsr_benchmark.utils import ClickParamTypeLsrDataset
from math import floor
import gzip

import itertools
import json
import os
from typing import Iterable, List

# This needs to be in the middle of the imports, if not it breaks.
from pyserini import __file__ as pyserini_file
from pyserini.setup import configure_classpath

pyserini_path = os.path.dirname(pyserini_file)
configure_classpath(os.path.join(pyserini_path, "resources/jars"))

from pyserini.encode import QueryEncoder
from pyserini.search.lucene import LuceneImpactSearcher

AGGREGATION_TYPES = ["FirstP", "MaxP", "MaxToken"]

def convert_w_to_features(tokens, values):

    return {f"feature_{t}": w for t, w in zip(tokens, values)}


def write_to_files(docs: Iterable[dict], files):
    for idx, doc in enumerate(docs):
        f = files[idx % len(files)]
        f.write(json.dumps(doc) + "\n")


def quantize_vector(vector_dict: dict):
    return {k: int(round(100 * float(v), 0)) for k, v in vector_dict.items()}


class PassThroughEncoder(QueryEncoder):
    def __init__(self):
        pass

    def encode(self, vector, **kwargs):
        return vector


class AnseriniIndex():
    def __init__(self, local_root, index_batch_size=1000, query_batch_size=100, anserini_threads=16, flush_size=100000):
        self.local_path = os.path.join(local_root,"tmp","anserini/")
        self.index_batch_size = index_batch_size
        self.query_batch_size = query_batch_size
        self.reset_state()
        self.anserini_threads = anserini_threads
        self.flush_size = flush_size

    def create_files(self, data):
        files = [
            open(os.path.join(self.collection_path, "collection_{}.jsonl".format(idx)), "w")
            for idx in range(self.anserini_threads * 4)
        ]
        docs = list()
        for doc in data:
            docs.append(doc)
            if len(docs) >= self.flush_size:
                write_to_files(docs, files)
                docs = list()
        write_to_files(docs, files)
        for f in files:
            f.close()


    def create_index(self):
        from jnius import autoclass  # only for when we need to index
        JIndexCollection = autoclass("io.anserini.index.IndexCollection")
        args = [
            "-collection",
            "JsonVectorCollection",
            "-generator",
            "DefaultLuceneDocumentGenerator",
            "-index",
            self.index_path,
            "-threads",
            str(self.anserini_threads),
            "-impact",
            "-pretokenized",
            "-input",
            self.collection_path,
        ]
        JIndexCollection.main(args)

    def search_batch(self, queries: Iterable, top_k: int):
        self.impact_searcher = LuceneImpactSearcher(self.index_path, query_encoder=PassThroughEncoder())
        results = dict()
        for batch in chunked(queries, self.query_batch_size):

            local_results = self.impact_searcher.batch_search(
                [x["query_toks"] for x in batch],
                [str(x["qid"]) for x in batch],
                k=top_k,
                threads=self.anserini_threads,
            )
            local_results = {k: {d.docid: d.score for d in v} for k, v in local_results.items()}
            results.update(local_results)
        return results

    def reset_state(self):
        rmtree(self.local_path, ignore_errors=True)
        self.collection_path = os.path.join(self.local_path, "collection")
        os.makedirs(self.collection_path, exist_ok=True)
        self.index_path = os.path.join(self.local_path, "index")

def clean_docid(docid):
    if isinstance(docid, float):
        docid = int(docid)
    return str(docid)

@click.command()
@click.option("--dataset", type=ClickParamTypeLsrDataset(), required=True,help="The dataset id or a local directory.")
@click.option("--embedding", type=str, required=False, default="lightning-ir/naver/splade-v3", help="The embedding model.")
@click.option("--output", required=True, type=Path, help="The directory where the output should be stored.")
@click.option("--k", type=int, required=False, default=10, help="The retrieval depth.")
def main(dataset, output, embedding, k):
    output.mkdir(parents=True, exist_ok=True)
    lsr_benchmark.register_to_ir_datasets(dataset)
    ir_dataset = ir_datasets.load(f"lsr-benchmark/{dataset}")
    tag = f"pyserini-lsr-top-{k}"
    register_metadata({"actor": {"team": "reneuir-baselines"}, "tag": tag})
    index = AnseriniIndex(output)

    documents = ({"id": doc_id, "vector": quantize_vector(convert_w_to_features(tokens,values)), "content": ""} for (doc_id, tokens, values) in tqdm(ir_dataset.doc_embeddings(model_name=embedding), "transform dataset"))
    index.create_files(documents)

    with tracking(export_file_path=output / "index-metadata.yml", export_format=ExportFormat.IR_METADATA):
        index.create_index()

    rmtree(output / ".tirex-tracker")
    queries = ({"qid": query_id, "query_toks": quantize_vector(convert_w_to_features(query_components, query_values))} for (query_id, query_components, query_values) in  ir_dataset.query_embeddings(model_name=embedding))

    with tracking(export_file_path=output / "retrieval-metadata.yml", export_format=ExportFormat.IR_METADATA):
        run = index.search_batch(queries, k)

    with gzip.open(output/"run.txt.gz", "wt") as f:
        for qid, query_result in run.items():
            rank = 1
            for docno, score in query_result.items():
                f.write(f"{qid} Q0 {docno} {rank} {score} pyserini\n")
                rank += 1


if __name__ == "__main__":
    main()
