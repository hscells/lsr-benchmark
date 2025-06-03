from pathlib import Path
import zipfile
from tira.check_format import QueryProcessorFormat, JsonlFormat
from ir_datasets.formats import BaseDocs, TrecQrels, BaseQueries, GenericQuery
from ir_datasets.datasets.base import Dataset
from typing import List, NamedTuple

MAPPING_OF_DATASET_IDS = {
    "clueweb09/en/trec-web-2009": "data/trec-18-web"
}


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


class LsrBenchmarkSegmentedDocument(NamedTuple):
    doc_id: str
    segment: Segment

    def default_text(self):
        return self.segment.text


class LsrBenchmarkQueries(BaseQueries):
    def __init__(self, queries_file):
       self.__queries_file = queries_file

    def queries_iter(self):
        for l in QueryProcessorFormat().all_lines(self.__queries_file):
            yield GenericQuery(l["qid"], l["query"])


class LsrBenchmarkDocuments(BaseDocs):
    def __init__(self, corpus_file):
        self.__corpus_file = corpus_file
        self.__docs = None

    def docs_iter(self, embedding=None):
        if embedding is not None:
            raise ValueError("ToDo: implement this...")

        for l in self.docs():
            yield LsrBenchmarkDocument._from_json(l)

    def docs(self):
        if not self.__docs:
            reader = JsonlFormat()
            reader.apply_configuration_and_throw_if_invalid({"required_fields": ["doc_id", "segments"], "max_size_mb": 2500})
            self.__docs = reader.all_lines(self.__corpus_file)
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
            class QrelsObj():
                def stream(self):
                    return open(qrels_file, "rb")
            qrels = TrecQrels(QrelsObj(), {0: 'Not Relevant', 1: 'Relevant'})

        if docs and not segmented:
            docs = LsrBenchmarkDocuments(docs)
        if docs and segmented:
            docs = LsrBenchmarkSegmentedDocuments(docs)

        super().__init__(docs, queries, qrels, documentation)

def base_dir(ir_datasets_id: str):
    if ir_datasets_id not in MAPPING_OF_DATASET_IDS:
        raise ValueError(f"The dataset ID '{ir_datasets_id}' is not supported. Supported are: {MAPPING_OF_DATASET_IDS.keys()}.")
    return (Path(__file__).parent.parent / MAPPING_OF_DATASET_IDS[ir_datasets_id]).resolve().absolute()

def inputs_dir(ir_datasets_id: str):
    return base_dir(ir_datasets_id) / "inputs-extracted"

def truths_dir(ir_datasets_id: str):
    return base_dir(ir_datasets_id) / "truths-extracted"

def extract_zip(zip_file: Path, target_directory: Path):
    if target_directory.exists():
        return

    if not zipfile.is_zipfile(zip_file):
        raise ValueError(f"I expected that {zip_file} is not a valid ZIP archive.")

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        target_directory.mkdir(parents=True, exist_ok=True)
        zip_ref.extractall(target_directory)

def build_dataset_from_local_cache(ir_datasets_id: str, segmented: bool):
    docs = inputs_dir(ir_datasets_id) / "corpus.jsonl.gz"
    queries = inputs_dir(ir_datasets_id) / "queries.jsonl"
    qrels = truths_dir(ir_datasets_id) / "qrels.txt"

    return LsrBenchmarkDataset(docs, queries, qrels, segmented)


def ensure_corpus_is_extracted(ir_datasets_id: str):
    d = base_dir(ir_datasets_id)

    for src, extracted in [(d / "inputs.zip", inputs_dir(ir_datasets_id)), (d / "truths.zip", truths_dir(ir_datasets_id))]:
        extract_zip(src, extracted)

