import click
import json
from pathlib import Path
from lsr_benchmark.corpus.corpus_subsampling import create_subsample
from lsr_benchmark.corpus import load_docs
from lsr_benchmark.corpus.segmentation import segmented_document
from trectools import TrecRun
from tqdm import tqdm
import ir_datasets
import gzip
import json


@click.command()
@click.argument('directory', type=Path)
def main(directory):
    config = json.loads((directory/"config.json").read_text())
    if (directory/"corpus.jsonl").is_file():
        return
    subsample = create_subsample(config["runs"], config["ir-datasets-id"], config["subsample_depth"], directory)
    docs = load_docs(config["ir-datasets-id"], subsample)
    docs = segmented_document(docs)
    with gzip.open(directory/"corpus.jsonl", 'wt') as f:
        for doc in docs.values():
            f.write(json.dumps(doc) + '\n')

if __name__ == '__main__':
    main()

