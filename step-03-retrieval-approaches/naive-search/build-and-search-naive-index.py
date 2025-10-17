#!/usr/bin/env python3
import ir_datasets
import lsr_benchmark
import click
from seismic import SeismicDataset
from tqdm import tqdm
from tirex_tracker import tracking, ExportFormat, register_metadata
from shutil import rmtree
from pathlib import Path
from lsr_benchmark.click import retrieve_command
import gzip


@retrieve_command()
def main(dataset, embedding, output, k):
    output.mkdir(parents=True, exist_ok=True)
    lsr_benchmark.register_to_ir_datasets(dataset)
    ir_dataset = ir_datasets.load(f"lsr-benchmark/{dataset}")
    seismic_dataset = SeismicDataset()
    register_metadata({"actor": {"team": "reneuir-baselines"}, "tag": f"naive_search-{embedding.replace('/', '-')}-{k}"})
    
    for (doc_id, tokens, values) in tqdm(ir_dataset.doc_embeddings(model_name=embedding), "create seismic dataset for naive search"):
        seismic_dataset.add_document(doc_id, tokens, values)

    with tracking(export_file_path=output / "index-metadata.yml", export_format=ExportFormat.IR_METADATA, ):
        print("There is no indexing with this technique.")

    query_embeddings = ir_dataset.query_embeddings(model_name=embedding)

    rmtree(output / ".tirex-tracker")
    results = []

    with tracking(export_file_path=output / "retrieval-metadata.yml", export_format=ExportFormat.IR_METADATA):
        for query_id, query_components, query_values in query_embeddings:
            current_res = seismic_dataset.search(query_id=query_id, query_components=query_components, query_values=query_values, k=k)
            results.append(current_res)

    rmtree(output / ".tirex-tracker")
    with gzip.open(output/"run.txt.gz", "wt") as f:
        for ranking_for_query in results:
            rank = 1
            for qid, score, docno in ranking_for_query:
                f.write(f"{qid} Q0 {docno} {rank} {score} naive_search\n")
                rank += 1

if __name__ == "__main__":
    main()
