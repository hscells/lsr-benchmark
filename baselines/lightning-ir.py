#!/usr/bin/env python3
import lsr_benchmark
import ir_datasets
import click


@click.command()
@click.option("--dataset", type=click.Choice(lsr_benchmark.SUPPORTED_IR_DATASETS), required=True, help="The dataset id or a local directory.")
@click.option("--model", type=str, required=True, help="The lightning ir model.")
def main(dataset, model):
    lsr_benchmark.register_to_ir_datasets()
    dataset = ir_datasets.load(f"lsr-benchmark/{dataset}/segmented")

    # do something with the queries
    for query in dataset.queries_iter():
        print(query.query_id, "=>", query.default_text())
        break

    # do something with the documents
    for doc in dataset.docs_iter(embedding=None):
        print(f"Docid={doc.doc_id} segment.offset_start={doc.segment.offset_start} segment.offset_end={doc.segment.offset_end} segment.text={doc.default_text()[:150]}...")
        break


if __name__ == '__main__':
    main()

