__version__ = "0.0.1"
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



@click.group()
def main():
    pass

@main.command()
def foo():
    print("foo")

@main.command()
@click.argument('directory', type=Path)
def create_lsr_corpus(directory):
    config = json.loads((directory/"config.json").read_text())
    if (directory/"corpus.jsonl").is_file():
        return
    ir_datasets_id = config["ir-datasets-id"]
    if ir_datasets_id.startswith("clueweb"):
        from ir_datasets_subsample import register_subsamples
        register_subsamples()
        ir_datasets_id = "corpus-subsamples/" + ir_datasets_id
    subsample = create_subsample(config["runs"], ir_datasets_id, config["subsample_depth"], directory)
    docs = load_docs(ir_datasets_id, subsample)
    docs = segmented_document(docs)
    with gzip.open(directory/"corpus.jsonl", 'wt') as f:
        for doc in docs.values():
            f.write(json.dumps(doc) + '\n')

if __name__ == '__main__':
    main()

