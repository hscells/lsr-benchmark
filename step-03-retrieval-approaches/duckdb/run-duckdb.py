#!/usr/bin/env python3
import duckdb
import ir_datasets
import lsr_benchmark
import click
from tqdm import tqdm
from tirex_tracker import tracking, ExportFormat, register_metadata
from shutil import rmtree
from pathlib import Path
from lsr_benchmark.utils import ClickParamTypeLsrDataset
import gzip
import pandas as pd


@click.command()
@click.option("--dataset", type=ClickParamTypeLsrDataset(), required=True, help="The dataset id or a local directory.")
@click.option("--output", required=True, type=Path, help="The directory where the output should be stored.",
)
@click.option("--embedding", type=str, required=False, default="naver/splade-v3", help="The embedding model.")
@click.option('--quantize', is_flag=True, help="Whether to quantize the index scores to integers.")
@click.option("--k", type=int, required=False, default=10, help="TBD.")
def main(dataset, embedding, output, quantize, k):
    output.mkdir(parents=True, exist_ok=True)
    lsr_benchmark.register_to_ir_datasets(dataset)
    ir_dataset = ir_datasets.load(f"lsr-benchmark/{dataset}")
    register_metadata({"actor": {"team": "reneuir-baselines"}, "tag": f"duckdb-{embedding.replace('/', '-')}-{'quantize-' if quantize else ''}{k}"})

    res = []
    for (doc_id, tokens, values) in tqdm(ir_dataset.doc_embeddings(model_name=embedding), "create documents dataframe"):
        res.append(pd.DataFrame({"doc_id": doc_id, "term_id": tokens, "score": values}))

    df = pd.concat(res)
    if quantize:
        df["score"] = (df["score"] * 100).round().astype(int)

    # Default is in-memory, may be adjusted if the index becomes too large
    conn = duckdb.connect(":memory:")

    print("Documents added to the DataFrame. Now indexing..")
    with tracking(export_file_path=output / "index-metadata.yml", export_format=ExportFormat.IR_METADATA, ):
        conn.execute("CREATE OR REPLACE TABLE index AS FROM df ORDER BY term_id, doc_id;")

    queries_dfs = []
    for (query_id, tokens, values) in tqdm(ir_dataset.query_embeddings(model_name=embedding), "create queries dataframe"):
        queries_dfs.append(pd.DataFrame({"query_id": query_id, "term_id": tokens, "score": values}))

    rmtree(output / ".tirex-tracker")
    results = []

    with tracking(export_file_path=output / "retrieval-metadata.yml", export_format=ExportFormat.IR_METADATA):
        for queries in queries_dfs:
            result = conn.query(
                """
                SELECT query_id, SUM(index.score * queries.score) AS score, doc_id
                FROM index
                JOIN queries USING (term_id)
                GROUP BY query_id, doc_id
                ORDER BY score DESC
                LIMIT ?
                """,
                params=[k]
            )

            results.append(result.fetchall())

    rmtree(output / ".tirex-tracker")
    with gzip.open(output/"run.txt.gz", "wt") as f:
        for ranking_for_query in results:
            rank = 1
            for qid, score, docno in ranking_for_query:
                f.write(f"{qid} Q0 {docno} {rank} {score} duckdb\n")
                rank += 1

if __name__ == "__main__":
    main()
