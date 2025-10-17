#!/usr/bin/env python3
import ir_datasets
import lsr_benchmark
from lsr_benchmark.click import retrieve_command
import click
from seismic import SeismicIndex, SeismicDataset, SeismicDatasetLV, SeismicIndexLV
from tqdm import tqdm
from tirex_tracker import tracking, ExportFormat, register_metadata
from shutil import rmtree
from pathlib import Path
import gzip


@retrieve_command()
@click.option("--heap-factor", type=float, required=False, default=0.8, help="TBD.")
@click.option("--query-cut", type=int, required=False, default=10, help="Number of posting lists to explore when searching for candidates.")
@click.option("--use-u32", type=bool, required=False, default=False, help="Whether to use u32 for component ids, required for datasets with many components..")
def main(dataset, embedding, output, heap_factor, query_cut, k, use_u32):
    output.mkdir(parents=True, exist_ok=True)
    lsr_benchmark.register_to_ir_datasets(dataset)
    ir_dataset = ir_datasets.load(f"lsr-benchmark/{dataset}")
    seismic_dataset = SeismicDatasetLV() if use_u32 else SeismicDataset()
    
    index_class = SeismicIndexLV if use_u32 else SeismicIndex
    register_metadata({"actor": {"team": "reneuir-baselines"}, "tag": f"seismic-{embedding.replace('/', '-')}-{heap_factor}-{query_cut}-{k}"})
    for (doc_id, tokens, values) in tqdm(ir_dataset.doc_embeddings(model_name=embedding), "create seismic dataset"):
        seismic_dataset.add_document(doc_id, tokens, values)

    print("Documents added to the SeismicDataset. Now indexing..")
    with tracking(export_file_path=output / "index-metadata.yml", export_format=ExportFormat.IR_METADATA, ):
        index = index_class.build_from_dataset(seismic_dataset)

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
