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
from lsr_benchmark.click import retrieve_command
from math import floor
from pyterrier_pisa import PisaIndex

def pyt_splade_encode(tokens, values, scale=100):
    return {str(k): floor(v*scale) for k,v in zip(tokens, values) if floor(v*scale) > 0}

def splade_query_to_pyterrier_query(toks):
    from pyt_splade import _matchop
    return ' '.join( _matchop(k, v) for k, v in sorted(toks.items(), key=lambda x: (-x[1], x[0])))

@retrieve_command()
def main(dataset, output, embedding, k):
    output.mkdir(parents=True, exist_ok=True)
    lsr_benchmark.register_to_ir_datasets(dataset)
    ir_dataset = ir_datasets.load(f"lsr-benchmark/{dataset}")
    ensure_pyterrier_is_loaded(boot_packages=())
    tag = f"pyterrier-splade-top-{k}"
    register_metadata({"actor": {"team": "reneuir-baselines"}, "tag": tag})

    documents = []

    for (doc_id, tokens, values) in tqdm(ir_dataset.doc_embeddings(model_name=embedding), "transform dataset"):
        documents.append({'docno' : doc_id, 'toks': pyt_splade_encode(tokens, values)})

    with tracking(export_file_path=output / "index-metadata.yml", export_format=ExportFormat.IR_METADATA):
        rmtree("/tmp/.ignored", ignore_errors=True)
        index = PisaIndex("/tmp/.ignored", stemmer='none')
        index.index(tqdm(documents, "Index docs"))

    rmtree(output / ".tirex-tracker")
    queries = []

    for query_id, query_components, query_values in  ir_dataset.query_embeddings(model_name=embedding):
        queries.extend([{"qid": query_id, "query_toks": pyt_splade_encode(query_components, query_values)}])

    pipeline =  index.quantized()
    with tracking(export_file_path=output / "retrieval-metadata.yml", export_format=ExportFormat.IR_METADATA):
        run = pipeline(pd.DataFrame(queries))

    pt.io.write_results(normalize_run(run, tag, k), f'{output}/run.txt')

if __name__ == "__main__":
    main()
