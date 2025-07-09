import zipfile
from pathlib import Path
from typing import List, NamedTuple, TYPE_CHECKING

import numpy as np
import torch
from ir_datasets.datasets.base import Dataset
from ir_datasets.formats import BaseDocs, BaseQueries, GenericQuery, TrecQrels
from ir_datasets.util import MetadataComponent, _DownloadConfig, home_path
from tira.check_format import JsonlFormat, QueryProcessorFormat

if TYPE_CHECKING:
    from typing import Optional

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
            "expected_md5": "fb05608785506a8047ba4a1e1fdee9f4",
            "size_hint": 127198049,
            "cache_path": "trec-web-2009-splade-v3-non-segmented.zip",
        },
    }
}

DownloadConfig = _DownloadConfig(contents=DOWNLOAD_CONTENTS)


def extracted_resource(irds_id: str, f) -> Path:
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

    def queries_iter(self):
        queries_file = extracted_resource(self.__irds_id, "truths") / self.__queries_name
        for l in QueryProcessorFormat().all_lines(queries_file):
            yield GenericQuery(l["qid"], l["query"])


class LsrBenchmarkQueryEmbedding(NamedTuple):
    query_id: str
    embedding: torch.Tensor


class LsrBenchmarkDocuments(BaseDocs):
    def __init__(self, corpus_file):
        self.__corpus_file = corpus_file
        self.__docs = None
        self.__irds_id = "clueweb09/en/trec-web-2009"

    def docs_iter(self, embedding=None, passage_aggregation=None):
        for l in self.docs():
            yield LsrBenchmarkDocument._from_json(l)

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
    def __init__(self, ir_datasets_id, docs=None, queries=None, qrels: "Optional[str]"=None, segmented=False, documentation=None):
        self.__irds_id = ir_datasets_id

        if queries:
            queries = LsrBenchmarkQueries(queries)

        if qrels is not None:
            class QrelsObj:
                def stream(self):
                    qrels_file = extracted_resource(ir_datasets_id, "truths") / qrels
                    return qrels_file.open("rb")

            qrels_obj = TrecQrels(QrelsObj(), {0: "Not Relevant", 1: "Relevant"})

        if docs and not segmented:
            docs = LsrBenchmarkDocuments(docs)
        if docs and segmented:
            docs = LsrBenchmarkSegmentedDocuments(docs)

        super().__init__(docs, queries, qrels_obj, documentation)
        self.metadata = MetadataComponent(ir_datasets_id, self)

    def embeddings(
        self, model_name: str, passage_aggregation: str, text_type: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        embedding_dir = extracted_resource(self.__irds_id, model_name) / text_type
        embeddings = np.load(embedding_dir / f"{text_type}-embeddings.npz")
        ids = (embedding_dir / f"{text_type}-ids.txt").read_text().strip().split("\n")
        return embeddings["data"], embeddings["indices"], embeddings["indptr"], ids

    def query_embeddings(
        self, model_name: str, passage_aggregation: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        ### hardcoded for now
        model_name = "splade-v3-non-segmented"
        ###

        return self.embeddings(model_name, passage_aggregation, "query")

    def doc_embeddings(
        self, model_name: str, passage_aggregation: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        ### hardcoded for now
        model_name = "splade-v3-non-segmented"
        ###

        return self.embeddings(model_name, passage_aggregation, "doc")


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

    return LsrBenchmarkDataset(
        ir_datasets_id=ir_datasets_id,
        docs=docs,
        queries=queries,
        qrels=qrels,
        segmented=segmented,
    )
