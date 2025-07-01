#!/usr/bin/env python3
import ir_datasets
import lsr_benchmark
import numpy as np
import click
import seismic
from seismic import SeismicIndex, SeismicDataset
from tqdm import tqdm
from tirex_tracker import tracking
from shutil import rmtree
from pathlib import Path
import gzip


@click.command()
@click.option(
    "--dataset",
    type=click.Choice(lsr_benchmark.SUPPORTED_IR_DATASETS),
    required=True,
    help="The dataset id or a local directory.",
)
@click.option("--output", required=True, type=Path, help="The directory where the output should be stored.",
)
@click.option("--embedding", type=str, required=False, default="naver/splade-v3", help="The embedding model.")
@click.option("--passage_aggregation", required=False, default="--passage_aggregation", type=click.Choice(["first-passage"]), help="The passage aggregation to use.")
def main(dataset, embedding, passage_aggregation, output):
    output.mkdir(parents=True)
    lsr_benchmark.register_to_ir_datasets()
    dataset = ir_datasets.load(f"lsr-benchmark/{dataset}")
    seismic_dataset = SeismicDataset()
    string_type  = seismic.get_seismic_string()

    for doc in tqdm(dataset.docs_iter(embedding=embedding, passage_aggregation=passage_aggregation)):
        # The input types should be "string", array of f32, array of string_type    
        values = doc.embedding.values().numpy()
        keys = np.asarray(doc.embedding.indices().numpy(), dtype=string_type)
        
        seismic_dataset.add_document(doc.doc_id, keys, values)

    print("Documents added to the SeismicDataset. Now indexing..")
    with tracking(export_file_path=output / "index-metadata.yml"):
        index = SeismicIndex.build_from_dataset(seismic_dataset)

    rmtree(output / ".tirex-tracker")
    results = []

    with tracking(export_file_path=output / "index-metadata.yml"):
        for query in dataset.queries_iter(embedding=embedding, passage_aggregation=passage_aggregation):
            query_components = np.asarray(query.embedding.indices().numpy(), dtype=string_type)
            query_values = query.embedding.values().numpy()

            current_res = index.search(query_id=str(query.query_id), query_components=query_components, query_values=query_values, k=10, query_cut=10, heap_factor=0.8)
            results.append(current_res)

    rmtree(output / ".tirex-tracker")
    with gzip.open(output/"run.txt.gz", "wt") as f:
        for ranking_for_query in results:
            rank = 1
            for qid, score, docno in ranking_for_query:
                f.write(f"{qid} Q0 {docno} {rank} {score} seismic")
                rank += 1


if __name__ == "__main__":
    main()
