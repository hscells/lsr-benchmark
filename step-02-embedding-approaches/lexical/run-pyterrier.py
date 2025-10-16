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
    queries = [{"qid": i.query_id, "text": i.default_text()} for i in ir_dataset.queries_iter()]

    doc_save_dir = output / "doc"
    with tracking(export_file_path=doc_save_dir / "doc-ir-metadata.yml", export_format=ExportFormat.IR_METADATA):

        (doc_save_dir / "doc-ids.txt").write_text("\n".join([doc["docno"] for doc in documents]))

        data = list()
        indices = list()
        indptr = [0]

        indexer = pt.IterDictIndexer("ignored", meta={"docno": 100}, type=pt.IndexingType.MEMORY)
        index = indexer.index(documents)
        index_factory = pt.IndexFactory.of(index)
        di = index_factory.getDirectIndex()
        doi = index_factory.getDocumentIndex()
        lex = index_factory.getLexicon()
        wmodel = pt.java.autoclass("org.terrier.matching.models." + weights)()
        wmodel.setCollectionStatistics(index_factory.getCollectionStatistics())
        for i in range(len(documents)):
            postings = di.getPostings(doi.getDocumentEntry(i))
            length = 0
            for posting in postings:
                term_id = posting.getId()
                lee = lex.getLexiconEntry(term_id)
                wmodel.setEntryStatistics(lex.getLexiconEntry(lee.getKey()))
                wmodel.setKeyFrequency(1)
                wmodel.prepare()
                data.append(wmodel.score(posting))
                indices.append(term_id)
                length += 1
            indptr.append(length)

        np.savez_compressed(
            doc_save_dir / "doc-embeddings.npz",
            data=np.array(data, dtype=np.float32),
            indices=np.array(indices, dtype=np.int32),
            indptr=np.array(indptr, dtype=np.int32),
        )

    query_save_dir = output / "query"
    with tracking(export_file_path=query_save_dir / "query-ir-metadata.yml", export_format=ExportFormat.IR_METADATA):

        (query_save_dir / "query-ids.txt").write_text("\n".join([query["qid"] for query in queries]))

        data = list()
        indices = list()
        indptr = [0]

        tokeniser = pt.java.autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()
        stemmer = pt.java.autoclass("org.terrier.terms.PorterStemmer")()

        for query in queries:
            query_text = query["text"]

            tokens = set(stemmer.stem(token) for token in tokeniser.getTokens(query_text))
            print(query["text"], tokens)
            length = 0
            for token in tokens:
                lee = lex.getLexiconEntry(token)
                if lee is None:
                    continue
                term_id = lee.getTermId()
                length += 1
                data.append(1.0)  # for queries we use always the weight 1
                indices.append(term_id)
                print(token, '->', term_id)
            indptr.append(length)

        np.savez_compressed(
            query_save_dir / "query-embeddings.npz",
            data=np.array(data, dtype=np.float32),
            indices=np.array(indices, dtype=np.int32),
            indptr=np.array(indptr, dtype=np.int32),
        )


if __name__ == "__main__":
    main()
