import zipfile
from pathlib import Path
from typing import List, NamedTuple, TYPE_CHECKING

import numpy as np
from ir_datasets.datasets.base import Dataset
from ir_datasets.formats import BaseDocs, BaseQueries, GenericQuery, TrecQrels
from ir_datasets.util import MetadataComponent
from tira.check_format import JsonlFormat, QueryProcessorFormat
from tira.third_party_integrations import in_tira_sandbox
from tqdm import tqdm
import os
from glob import glob

if TYPE_CHECKING:
    from typing import Optional

MAPPING_OF_DATASET_IDS = {"clueweb09/en/trec-web-2009": "data/trec-18-web"}
TIRA_LSR_TASK_ID = "lsr-benchmark"


_IR_DATASETS_FROM_TIRA = None

def embeddings(
        dataset_id: str, model_name: str, text_type: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    from tira.rest_api_client import Client
    tira = Client()
    team_and_model = model_name.split('/')
    team_name = team_and_model[0]
    model_name = '-'.join(team_and_model[1:])
    embedding_dir = tira.get_run_output(f"{TIRA_LSR_TASK_ID}/{team_name}/{model_name}", dataset_id) / text_type
    embeddings = np.load(embedding_dir / f"{text_type}-embeddings.npz")

    try:
        from tirex_tracker import register_file
        for i in glob(f"{embedding_dir}/*.yml") + glob(f"{embedding_dir}/*.yaml"):
            register_file(embedding_dir, i.split("/")[-1], subdirectory=".embedding")
    except:
        pass

    ids = (embedding_dir / f"{text_type}-ids.txt").read_text().strip().split("\n")

    ret = []
    indices = embeddings["indices"].astype("U30")
    data = embeddings["data"]

    ptr_start = 0
    for doc_id, ptr_end in tqdm(list(zip(ids, embeddings["indptr"][1:])), "load doc embeddings"):
        tokens = indices[ptr_start:ptr_end]
        values = data[ptr_start:ptr_end]
        ptr_start = ptr_end
        ret.append((doc_id, tokens, values))
    return ret


def ir_datasets_from_tira(force_reload=False):
    global _IR_DATASETS_FROM_TIRA
    
    if in_tira_sandbox():
        return []
    
    if _IR_DATASETS_FROM_TIRA is None or force_reload:
        from tira.rest_api_client import Client
        tira = Client()
        _IR_DATASETS_FROM_TIRA = list(tira.datasets(TIRA_LSR_TASK_ID, force_reload).keys())

    return _IR_DATASETS_FROM_TIRA


def extracted_resource(irds_id: str, f) -> Path:
    if os.path.isdir(irds_id):
        return Path(irds_id)
    else:
        raise ValueError("ToDo: finish refactoring")


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
    embedding: np.array


class LsrBenchmarkSegmentedDocument(NamedTuple):
    doc_id: str
    segment: Segment

    def default_text(self):
        return self.segment.text


class LsrBenchmarkQueries(BaseQueries):
    def __init__(self, queries_file, ir_datasets_id):
        self.__queries_name = queries_file
        self.__irds_id = ir_datasets_id

    def queries_iter(self):
        if os.path.isfile(self.__queries_name):
            queries_file = Path(self.__queries_name)
        else:
            queries_file = extracted_resource(self.__irds_id, "inputs") / self.__queries_name

        for l in QueryProcessorFormat().all_lines(queries_file):
            yield GenericQuery(l["qid"], l["query"])


class LsrBenchmarkQueryEmbedding(NamedTuple):
    query_id: str
    embedding: np.array


class LsrBenchmarkDocuments(BaseDocs):
    def __init__(self, corpus_file, irds_id):
        self.__corpus_file = corpus_file
        self.__docs = None
        self.__irds_id = irds_id

    def docs_iter(self):
        for l in self.docs():
            yield LsrBenchmarkDocument._from_json(l)

    def docs(self):
        if not self.__docs:
            if os.path.isfile(self.__corpus_file):
                docs_file = Path(self.__corpus_file)
            else:
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
            queries = LsrBenchmarkQueries(queries, ir_datasets_id)

        if qrels is None:
            qrels_obj = None
        else:
            class QrelsObj:
                def stream(self):
                    if os.path.isfile(qrels):
                        qrels_file = Path(qrels)
                    else:
                        qrels_file = extracted_resource(ir_datasets_id, "truths") / qrels
                    return qrels_file.open("rb")

            qrels_obj = TrecQrels(QrelsObj(), {0: "Not Relevant", 1: "Relevant"})

        if docs and not segmented:
            docs = LsrBenchmarkDocuments(docs, ir_datasets_id)
        if docs and segmented:
            docs = LsrBenchmarkSegmentedDocuments(docs, ir_datasets_id)

        super().__init__(docs, queries, qrels_obj, documentation)
        self.metadata = MetadataComponent(ir_datasets_id, self)

    def query_embeddings(self, model_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        return embeddings(self.__irds_id, model_name, "query")

    def doc_embeddings(self, model_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        return embeddings(self.__irds_id, model_name, "doc")


def extract_zip(zip_file: Path, target_directory: Path):
    if target_directory.exists():
        return

    if not zipfile.is_zipfile(zip_file):
        raise ValueError(f"I expected that {zip_file} is not a valid ZIP archive.")

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        target_directory.mkdir(parents=True, exist_ok=True)
        zip_ref.extractall(target_directory)


def build_dataset(ir_datasets_id: str, segmented: bool):
    try:
        from tirex_tracker import register_metadata
        register_metadata({"data": {"test collection": {"name": ir_datasets_id}}})
    except:
        pass

    if ir_datasets_id in ir_datasets_from_tira(True):
        from tira.rest_api_client import Client

        tira = Client()
        system_inputs = tira.download_dataset(task=None, dataset=ir_datasets_id, truth_dataset=False)

        docs = system_inputs / "corpus.jsonl.gz"
        queries = system_inputs / "queries.jsonl"
        qrels = None

        if not in_tira_sandbox():
            truths = tira.download_dataset(task=None, dataset=ir_datasets_id, truth_dataset=True)
            qrels = truths / "qrels.txt"
    else:
        docs = "corpus.jsonl.gz"
        queries = "queries.jsonl"
        qrels = "qrels.txt" if not in_tira_sandbox() else None

    return LsrBenchmarkDataset(
        ir_datasets_id=ir_datasets_id,
        docs=docs,
        queries=queries,
        qrels=qrels,
        segmented=segmented,
    )
