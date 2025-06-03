#!/usr/bin/env python3
import lsr_benchmark
import lsr_benchmark.ir_datasets
import click


@click.command()
@click.option("--dataset", type=click.Choice(lsr_benchmark.ir_datasets.MAPPING_OF_DATASET_IDS.keys()), required=True, help="The dataset id or a local directory.")
@click.option("--model", type=str, required=True, help="The lightning ir model.")
def main(dataset, model):
    dataset = lsr_benchmark.load(dataset)
    
    # do something with the queries
    for query in dataset.queries_iter():
        print(query.query_id, "=>", query.default_text())
        break

    # do something with the documents
    for doc in dataset.docs_iter(embedding=None):
        for segment in doc.segments:
            print(f"Docid={doc.doc_id} segment.offset_start={segment.offset_start} segment.offset_end={segment.offset_end} segment.text={segment.text[:150]}...")
        break


if __name__ == '__main__':
    main()

