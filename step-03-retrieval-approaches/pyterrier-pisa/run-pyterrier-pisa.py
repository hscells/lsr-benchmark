#!/usr/bin/env python3
import lsr_benchmark
import click
from tirex_tracker import tracking, ExportFormat, register_metadata
from tqdm import tqdm
import pyterrier as pt
from pathlib import Path
from shutil import rmtree
import pandas as pd
from tira.third_party_integrations import ensure_pyterrier_is_loaded,  normalize_run
import ir_datasets
from lsr_benchmark.utils import ClickParamTypeLsrDataset
from math import floor
from pyterrier_pisa import PisaIndex


@click.command()
@click.option("--dataset", type=ClickParamTypeLsrDataset(), required=True, help="The dataset id or a local directory.")
@click.option("--output", required=True, type=Path, help="The directory where the output should be stored.")
@click.option("--k", type=int, required=False, default=10, help="The retrieval depth.")
def main(dataset, output, k):
    output.mkdir(parents=True, exist_ok=True)
    lsr_benchmark.register_to_ir_datasets(dataset)
    ir_dataset = ir_datasets.load(f"lsr-benchmark/{dataset}")
    ensure_pyterrier_is_loaded(boot_packages=())
    tag = f"pyterrier-splade-top-{k}"
    register_metadata({"actor": {"team": "reneuir-baselines"}, "tag": tag})

    documents = [{"docno": i.doc_id, "text": i.default_text()} for i in ir_dataset.docs_iter()]

    with tracking(export_file_path=output / "index-metadata.yml", export_format=ExportFormat.IR_METADATA):
        rmtree("/tmp/.ignored", ignore_errors=True)
        index = PisaIndex("/tmp/.ignored", stemmer='none')
        index.index(tqdm(documents, "Index docs"))

    rmtree(output / ".tirex-tracker")

    queries = []
    for i in ir_dataset.queries_iter():
        queries.extend([{"qid": i.query_id, "query": i.default_text()}])

    pipeline = index.bm25()
    with tracking(export_file_path=output / "retrieval-metadata.yml", export_format=ExportFormat.IR_METADATA):
        run = pipeline(pd.DataFrame(queries))

    pt.io.write_results(normalize_run(run, tag, k), f'{output}/run.txt')

if __name__ == "__main__":
    main()
