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

class LsrBenchmarkQueries(BaseQueries):
    def __init__(self, queries):
       self.__queries = queries

    def queries_iter(self):
        for l in self.__queries:
            yield l

    @staticmethod
    def _from_file(corpus_file):
        queries = [GenericQuery(i["qid"], i["query"]) for i in QueryProcessorFormat().all_lines(corpus_file)]
        return queries

class LsrBenchmarkDocuments(BaseDocs):
    def __init__(self, docs):
       self.__docs = docs

    def docs_iter(self):
        for l in self.__docs:
            yield LsrBenchmarkDocument._from_json(l)

    @staticmethod
    def _from_file(corpus_file):
        reader = JsonlFormat()
        reader.apply_configuration_and_throw_if_invalid({"required_fields": ["doc_id", "segments"], "max_size_mb": 2500})
        docs = reader.all_lines(corpus_file)
        return LsrBenchmarkDocuments(docs)

class LsrBenchmarkDataset(Dataset):
    def __init__(self, docs=None, queries=None, qrels=None, documentation=None):
        if queries:
            queries = LsrBenchmarkQueries._from_file(queries)

        if qrels:
            qrels = TrecQrels(qrels, {0: 'Not Relevant', 1: 'Relevant'})

        if docs:
            docs = LsrBenchmarkDocuments._from_file(docs)

        super().__init__(docs, queries, qrels, documentation)

def base_dir(ir_datasets_id: str):
    if ir_datasets_id not in MAPPING_OF_DATASET_IDS:
        raise ValueError(f"The dataset ID '{ir_datasets_id}' is not supported. Supported are: {MAPPING_OF_DATASET_IDS.keys()}.")
    return Path(MAPPING_OF_DATASET_IDS[ir_datasets_id]).resolve().absolute()

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

def build_dataset_from_local_cache(ir_datasets_id: str):
    docs = inputs_dir(ir_datasets_id) / "corpus.jsonl.gz"
    queries = inputs_dir(ir_datasets_id) / "queries.jsonl"
    qrels = truths_dir(ir_datasets_id) / "qrels.txt"

    return LsrBenchmarkDataset(docs, queries, qrels)


def ensure_corpus_is_extracted(ir_datasets_id: str):
    d = base_dir(ir_datasets_id)

    for src, extracted in [(d / "inputs.zip", inputs_dir(ir_datasets_id)), (d / "truths.zip", truths_dir(ir_datasets_id))]:
        extract_zip(src, extracted)

