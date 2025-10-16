#!/usr/bin/env python3
from pathlib import Path

import click
import ir_datasets
import numpy as np
import pyterrier as pt
from tira.third_party_integrations import ensure_pyterrier_is_loaded
from tirex_tracker import ExportFormat, register_metadata, tracking

import lsr_benchmark
from lsr_benchmark.utils import ClickParamTypeLsrDataset


def create_index(texts):
    indexer = pt.IterDictIndexer("ignored", meta={"docno": 100}, type=pt.IndexingType.MEMORY)
    index = indexer.index(texts)
    index_factory = pt.IndexFactory.of(index)
    return index_factory


def get_scores(doc_idx, term_lexicon, index_lexicon, doc_index, direct_index, wmodel):
    data = []
    indices = []
    postings = direct_index.getPostings(doc_index.getDocumentEntry(doc_idx))
    for posting in postings:
        term_id = posting.getId()
        lexicon_entry = term_lexicon.getLexiconEntry(term_id)
        term = lexicon_entry.getKey()
        lexicon_entry = index_lexicon.getLexiconEntry(term)
        if lexicon_entry is None:
            # term not in index
            continue
        term_id = lexicon_entry.getTermId()

        if wmodel is not None:
            wmodel.setEntryStatistics(lexicon_entry)
            wmodel.setKeyFrequency(1)
            wmodel.prepare()
            data.append(wmodel.score(posting))
        else:
            data.append(1.0)
        indices.append(term_id)

    return data, indices


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

    doc_save_dir = output / "doc"
    with tracking(export_file_path=doc_save_dir / "doc-ir-metadata.yml", export_format=ExportFormat.IR_METADATA):

        (doc_save_dir / "doc-ids.txt").write_text("\n".join([doc["docno"] for doc in documents]))

        data = list()
        indices = list()
        indptr = [0]

        index = create_index(documents)
        direct_index = index.getDirectIndex()
        doc_index = index.getDocumentIndex()
        doc_lexicon = index.getLexicon()
        wmodel = pt.java.autoclass("org.terrier.matching.models." + weights)()
        wmodel.setCollectionStatistics(index.getCollectionStatistics())
        for i in range(len(documents)):
            doc_data, doc_indices = get_scores(i, doc_lexicon, doc_lexicon, doc_index, direct_index, wmodel)
            data.extend(doc_data)
            indices.extend(doc_indices)
            indptr.append(len(data))

        np.savez_compressed(
            doc_save_dir / "doc-embeddings.npz",
            data=np.array(data, dtype=np.float32),
            indices=np.array(indices, dtype=np.int32),
            indptr=np.array(indptr, dtype=np.int64),
        )

    query_save_dir = output / "query"
    with tracking(export_file_path=query_save_dir / "query-ir-metadata.yml", export_format=ExportFormat.IR_METADATA):

        (query_save_dir / "query-ids.txt").write_text("\n".join([query["docno"] for query in queries]))

        data = list()
        indices = list()
        indptr = [0]

        index = create_index(queries)
        direct_index = index.getDirectIndex()
        doc_index = index.getDocumentIndex()
        query_lexicon = index.getLexicon()

        for i in range(len(queries)):
            query_data, query_indices = get_scores(i, query_lexicon, doc_lexicon, doc_index, direct_index, None)
            data.extend(query_data)
            indices.extend(query_indices)
            indptr.append(len(data))

        np.savez_compressed(
            query_save_dir / "query-embeddings.npz",
            data=np.array(data, dtype=np.float32),
            indices=np.array(indices, dtype=np.int32),
            indptr=np.array(indptr, dtype=np.int64),
        )


if __name__ == "__main__":
    main()
