import zipfile
from pathlib import Path
from typing import List, NamedTuple
from ir_datasets.util import ZipExtractCache, ZipExtract, GzipExtract, home_path, Cache, _DownloadConfig

import torch
from ir_datasets.datasets.base import Dataset
from ir_datasets.formats import BaseDocs, BaseQueries, GenericQuery, TrecQrels
from tira.check_format import JsonlFormat, QueryProcessorFormat

MAPPING_OF_DATASET_IDS = {"clueweb09/en/trec-web-2009": "data/trec-18-web"}


DOWNLOAD_CONTENTS = {
    "clueweb09/en/trec-web-2009": {
        "inputs": {
            "url": "https://files.webis.de/data-in-progress/lsr-benchmark-delete-me-after-01-08-2025/inputs.zip",
            "expected_md5": "75a107e4c545a5d79c77942b5e863a16",
            "size_hint": 421574669,
            "cache_path": "trec-web-2009-inputs.zip",
        },
        "truths": {
            "url": "https://files.webis.de/data-in-progress/lsr-benchmark-delete-me-after-01-08-2025/truths.zip",
            "expected_md5": "f28e36759760c9520e7831aba86c4d23",
            "size_hint": 502691,
            "cache_path": "trec-web-2009-truths.zip",
        },
        "splade-v3-non-segmented": {
            "url": "https://files.webis.de/data-in-progress/lsr-benchmark-delete-me-after-01-08-2025/splade-v3-non-segmented.zip",
            "expected_md5": "f9131d00b4305744ac4ee0cae40bfeeb",
            "size_hint": 127198049,
            "cache_path": "trec-web-2009-splade-v3-non-segmented.zip",
        }
        
    }
}

DownloadConfig = _DownloadConfig(contents=DOWNLOAD_CONTENTS)

def extracted_resource(irds_id, f):
    zip_file = DownloadConfig.context(irds_id, home_path() / "lsr-benchmark" / irds_id.replace("/", "-"))[f].path()
    target_dir = Path(str(zip_file).replace(".zip", "") + "-extracted")
    extract_zip(zip_file, target_dir)
    return target_dir

class Segment(NamedTuple):
    offset_start: int
    offset_end: int
    text: str


class LsrBenchmarkDocument(NamedTuple):
    doc_id: str
    segments: List[Segment]

    @staticmethod
    def _from_json(json_doc):
        segments = [Segment(int(i["start"]), int(i["end"]), i["text"]) for i in json_doc["segments"]]
        return LsrBenchmarkDocument(json_doc["doc_id"], segments)

    def default_text(self):
        return "" if len(self.segments) == 0 else self.segments[0].text


class LsrBenchmarkDocumentEmbedding(NamedTuple):
    doc_id: str
    embedding: torch.Tensor


class LsrBenchmarkSegmentedDocument(NamedTuple):
    doc_id: str
    segment: Segment

    def default_text(self):
        return self.segment.text


class LsrBenchmarkQueries(BaseQueries):
    def __init__(self, queries_file):
        self.__queries_name = queries_file
        self.__irds_id = "clueweb09/en/trec-web-2009"

    def queries_iter(self, embedding=None, passage_aggregation=None):
        queries_file = extracted_resource(self.__irds_id, "truths") / self.__queries_name
        if embedding is None:
            for l in QueryProcessorFormat().all_lines(queries_file):
                yield GenericQuery(l["qid"], l["query"])
            return

        target_dir = None
        if embedding == "naver/splade-v3" and passage_aggregation == "first-passage":
            target_dir = extracted_resource(self.__irds_id, "splade-v3-non-segmented") / "queries"

        assert target_dir is not None

        embeddings = torch.load(target_dir / "query_embeddings.pt")
        query_ids = (target_dir / "query_ids.txt").read_text().strip().split("\n")
        query_id_to_embedding = {query_id: embedding for query_id, embedding in zip(query_ids, embeddings)}

        for l in QueryProcessorFormat().all_lines(queries_file):
            query_id = l["qid"]
            e = query_id_to_embedding[query_id]
            yield LsrBenchmarkQueryEmbedding(query_id, e)


class LsrBenchmarkQueryEmbedding(NamedTuple):
    query_id: str
    embedding: torch.Tensor


class LsrBenchmarkDocuments(BaseDocs):
    def __init__(self, corpus_file):
        self.__corpus_file = corpus_file
        self.__docs = None
        self.__irds_id = "clueweb09/en/trec-web-2009"

    def docs_iter(self, embedding=None, passage_aggregation=None):
        if embedding is None:
            for l in self.docs():
                yield LsrBenchmarkDocument._from_json(l)
            return

        target_dir = None
        if embedding == "naver/splade-v3" and passage_aggregation == "first-passage":
            target_dir = extracted_resource(self.__irds_id, "splade-v3-non-segmented") / "docs" / "lsr-benchmark" / self.__irds_id

        assert target_dir is not None

        embeddings = torch.load(target_dir / "index.pt")
        doc_ids = (target_dir / "doc_ids.txt").read_text().strip().split("\n")
        doc_id_to_embedding = {doc_id: embedding for doc_id, embedding in zip(doc_ids, embeddings)}

        for l in self.docs():
            doc = LsrBenchmarkDocument._from_json(l)
            doc_id = doc.doc_id
            e = doc_id_to_embedding[doc_id]
            yield LsrBenchmarkDocumentEmbedding(doc_id, e)

    def docs(self):
        if not self.__docs:
            docs_file = extracted_resource(self.__irds_id, "inputs") / self.__corpus_file
            reader = JsonlFormat()
            reader.apply_configuration_and_throw_if_invalid(
                {"required_fields": ["doc_id", "segments"], "max_size_mb": 2500}
            )
            self.__docs = reader.all_lines(docs_file)
        return self.__docs

    def docs_count(self):
        return len([1 for i in self.docs_iter()])


class LsrBenchmarkSegmentedDocuments(LsrBenchmarkDocuments):
    def docs_iter(self, embedding=None):
        for doc in super().docs_iter(embedding):
            for idx, segment in zip(range(len(doc.segments)), doc.segments):
                yield LsrBenchmarkSegmentedDocument(f"{doc.doc_id}___{idx}___", segment)

    def docs_count(self):
        return len([1 for i in self.docs_iter()])


class LsrBenchmarkDataset(Dataset):
    def __init__(self, docs=None, queries=None, qrels=None, segmented=False, documentation=None):
        if queries:
            queries = LsrBenchmarkQueries(queries)

        if qrels:
            qrels_file = qrels

            class QrelsObj:
                def stream(self):
                    return open(qrels_file, "rb")

            qrels = TrecQrels(QrelsObj(), {0: "Not Relevant", 1: "Relevant"})

        if docs and not segmented:
            docs = LsrBenchmarkDocuments(docs)
        if docs and segmented:
            docs = LsrBenchmarkSegmentedDocuments(docs)

        super().__init__(docs, queries, qrels, documentation)


def extract_zip(zip_file: Path, target_directory: Path):
    if target_directory.exists():
        return

    if not zipfile.is_zipfile(zip_file):
        raise ValueError(f"I expected that {zip_file} is not a valid ZIP archive.")

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        target_directory.mkdir(parents=True, exist_ok=True)
        zip_ref.extractall(target_directory)


def build_dataset(ir_datasets_id: str, segmented: bool):
    docs = "corpus.jsonl.gz"
    queries = "queries.jsonl"
    qrels = "qrels.txt"

    return LsrBenchmarkDataset(docs, queries, qrels, segmented)

