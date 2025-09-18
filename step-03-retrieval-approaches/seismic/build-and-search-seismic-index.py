#!/usr/bin/env python3
import ir_datasets
import lsr_benchmark
import numpy as np
import click
import seismic
from seismic import SeismicIndex, SeismicDataset
from tqdm import tqdm
from tirex_tracker import tracking, ExportFormat
from shutil import rmtree
from pathlib import Path
from lsr_benchmark.utils import ClickParamTypeLsrDataset
import gzip


@click.command()
@click.option(
    "--dataset",
    type=ClickParamTypeLsrDataset(),
    required=True,
    help="The dataset id or a local directory.",
)
@click.option("--output", required=True, type=Path, help="The directory where the output should be stored.",
)
@click.option("--embedding", type=str, required=False, default="naver/splade-v3", help="The embedding model.")
@click.option("--heap-factor", type=float, required=False, default=0.8, help="TBD.")
@click.option("--query-cut", type=int, required=False, default=10, help="TBD.")
@click.option("--k", type=int, required=False, default=10, help="TBD.")
def main(dataset, embedding, output, heap_factor, query_cut, k):
    output.mkdir(parents=True)
    lsr_benchmark.register_to_ir_datasets(dataset)
    ir_dataset = ir_datasets.load(f"lsr-benchmark/{dataset}")
    seismic_dataset = SeismicDataset()

    for (doc_id, tokens, values) in ir_dataset.doc_embeddings(model_name=embedding):
        seismic_dataset.add_document(doc_id, tokens, values)

    print("Documents added to the SeismicDataset. Now indexing..")
    with tracking(export_file_path=output / "index-metadata.yml", export_format=ExportFormat.IR_METADATA):
        index = SeismicIndex.build_from_dataset(seismic_dataset)

    query_embeddings = ir_dataset.query_embeddings(model_name=embedding)

    rmtree(output / ".tirex-tracker")
    results = []

    with tracking(export_file_path=output / "retrieval-metadata.yml", export_format=ExportFormat.IR_METADATA):
        for query_id, query_components, query_values in query_embeddings:
            current_res = index.search(query_id=query_id, query_components=query_components, query_values=query_values, k=k, query_cut=query_cut, heap_factor=heap_factor)
            results.append(current_res)

    rmtree(output / ".tirex-tracker")
    with gzip.open(output/"run.txt.gz", "wt") as f:
        for ranking_for_query in results:
            rank = 1
            for qid, score, docno in ranking_for_query:
                f.write(f"{qid} Q0 {docno} {rank} {score} seismic\n")
                rank += 1

if __name__ == "__main__":
    main()
