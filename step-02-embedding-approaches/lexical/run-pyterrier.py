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

def get_weights_of_texts(texts, weights):
    documents = [{"docno": str(k), "text": v} for k, v in zip(range(len(texts)), texts)]
                 
    index = pt.IterDictIndexer("ignored", meta= {'docno' : 100}, type=pt.IndexingType.MEMORY).index(tqdm(documents, "Index docs"))
    index = pt.IndexFactory.of(index)
    di = index.getDirectIndex()
    doi = index.getDocumentIndex()
    lex = index.getLexicon()
    wmodel = pt.java.autoclass("org.terrier.matching.models." + weights)()
    wmodel.setCollectionStatistics(index.getCollectionStatistics())
    ret = []
    for i in range(len(documents)):
        scores = {}
        for posting in di.getPostings(doi.getDocumentEntry(i)):
            lee = lex.getLexiconEntry(posting.getId())
            wmodel.setEntryStatistics(lex.getLexiconEntry(lee.getKey()))
            wmodel.setKeyFrequency(1)
            wmodel.prepare()
            scores[lee.getKey()] = wmodel.score(posting)
        ret += [scores]
    return ret

@click.command()
@click.option("--dataset", type=ClickParamTypeLsrDataset(), required=True, help="The dataset id or a local directory.")
@click.option("--output", required=True, type=Path, help="The directory where the output should be stored.")
@click.option("--weights", type=str, required=False, default="BM25", help="The retrieval model.")
def main(dataset, output, weights):
    output.mkdir(parents=True, exist_ok=True)
    lsr_benchmark.register_to_ir_datasets(dataset)
    ir_dataset = ir_datasets.load(f"lsr-benchmark/{dataset}")
    ensure_pyterrier_is_loaded(boot_packages=())

    register_metadata({"actor": {"team": "reneuir-baselines"}, "tag": f"pyterrier-lexical-embedding-{weights.lower()}"})
    documents = [{"docno": i.doc_id, "text": i.default_text()} for i in ir_dataset.docs_iter()]
    queries = [{"docno": i.query_id, "text": i.default_text()} for i in ir_dataset.queries_iter()]

    for text_type, texts in zip(["query", "doc"], [queries, documents]):
        text_type_save_dir = output / text_type
        with tracking(export_file_path=text_type_save_dir / f"{text_type}-ir-metadata.yml", export_format=ExportFormat.IR_METADATA):
            scores = get_weights_of_texts([i["text"] for i in texts], weights)
            if text_type == "query":
                # for queries we use always the weight 1
                scores = [{k: 1 for k in i.keys()} for i in scores]
            print(text_type, scores)

if __name__ == "__main__":
    main()
