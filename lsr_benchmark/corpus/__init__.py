from tqdm import tqdm
import ir_datasets
import tempfile
from pathlib import Path
from .corpus_subsampling import create_subsample
from .segmentation import segmented_document
import gzip
import json
import shutil
from uuid import uuid4
from tira.ir_datasets_loader import IrDatasetsLoader

import zipfile
import os


def zip_directory(folder_path, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        len_prefix = len(os.path.abspath(folder_path)) + 1  # to remove the base folder path
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.abspath(abs_path)[len_prefix:]
                zipf.write(abs_path, arcname=rel_path)

def load_docs(ir_datasets_id, subsample):
    ret = {}
    docs_store = ir_datasets.load(ir_datasets_id).docs_store()
    irds_loader = IrDatasetsLoader()
    skipped = 0
    for doc in tqdm(subsample):
        try:
            ret[doc] = json.loads(irds_loader.map_doc(docs_store.get(doc)))
        except:
            skipped += 1
    print(f"Skipped {skipped} docs")
    return ret

def irds_id_from_config(config):
    ir_datasets_id = config["ir-datasets-id"]
    if ir_datasets_id.startswith("clueweb"):
        from ir_datasets_subsample import register_subsamples
        register_subsamples()
        ir_datasets_id = "corpus-subsamples/" + ir_datasets_id
    return ir_datasets_id

def materialize_raw_corpus(directory, subsample, config):
    ir_datasets_id = irds_id_from_config(config)
    docs = load_docs(ir_datasets_id, subsample)
    with gzip.open(directory / "corpus.jsonl.gz", 'wt') as f:
        for doc in docs.values():
            f.write(json.dumps(doc) + '\n')

def materialize_corpus(directory, config):
    ir_datasets_id = irds_id_from_config(config)
    if (directory/"document-mapping.json.gz").is_file():
        return
    subsample = create_subsample(config["runs"], ir_datasets_id, config["subsample_depth"], directory)
    docs = load_docs(ir_datasets_id, subsample)
    docs = segmented_document(docs, config.get("passage_size", 80))
    doc_mapping = {}
    with gzip.open(directory/"corpus.jsonl.gz", 'wt') as f:
        for doc in docs.values():
            old_id = doc["doc_id"]
            new_id = str(uuid4())
            assert new_id not in doc_mapping
            assert old_id not in doc_mapping.keys()
            doc_mapping[new_id] = old_id
            doc["doc_id"] = new_id
            f.write(json.dumps(doc) + '\n')

    with gzip.open(directory/"document-mapping.json.gz", "wt") as f:
        f.write(json.dumps(doc_mapping))

def materialize_queries(directory, config):
    from tira.ir_datasets_loader import IrDatasetsLoader
    
    irds_loader = IrDatasetsLoader()
    ir_datasets_id = irds_id_from_config(config)
    output_jsonl = directory / "queries.jsonl"
    output_xml = directory / "queries.xml"

    allowed_queries = set()
    for _, i in ir_datasets.load(ir_datasets_id).qrels_iter():
        allowed_queries.add(i.query_id)

    if not output_jsonl.exists():
        dataset = ir_datasets.load(ir_datasets_id)
        queries_mapped_jsonl = [irds_loader.map_query_as_jsonl(query, True) for query in dataset.queries_iter() if query.query_id in allowed_queries]
        with open(output_jsonl, 'w') as f:
            for l in queries_mapped_jsonl:
                f.write(l + '\n')

    if not output_xml.exists():
        dataset = ir_datasets.load(ir_datasets_id)
        queries_mapped_xml = [irds_loader.map_query_as_xml(query, True) for query in dataset.queries_iter() if query.query_id in allowed_queries]
        with open(output_xml, 'w') as f:
            for l in queries_mapped_xml:
                f.write(str(l) + '\n')

def materialize_qrels(source_directory, output_qrels, config):
    ir_datasets_id = irds_id_from_config(config)

    if output_qrels.exists():
        return

    dataset = ir_datasets.load(ir_datasets_id)
 
    with gzip.open(source_directory/"document-mapping.json.gz", "rt") as f:
        document_mapping = {v:k for k,v in json.loads(f.read()).items()}

    skipped = set()
    with open(output_qrels, 'w') as f:
        for qrel in dataset.qrels_iter():
            if qrel.doc_id not in document_mapping:
                skipped.add(qrel.doc_id)
                continue
            f.write(f"{qrel.query_id} 0 {document_mapping[qrel.doc_id]} {qrel.relevance}\n")
    print("skipped qrels", len(skipped))

def materialize_inputs(directory, config):
    output_zip = directory / "inputs.zip"
    if output_zip.exists():
        return
    with tempfile.TemporaryDirectory() as tmp_dir:
        materialize_queries(Path(tmp_dir), config)
        shutil.copy(directory/"corpus.jsonl.gz", Path(tmp_dir)/"corpus.jsonl.gz")
        zip_directory(tmp_dir, output_zip)


def materialize_truths(directory, config):
    output_zip = directory / "truths.zip"
    if output_zip.exists():
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        materialize_queries(Path(tmp_dir), config)
        materialize_qrels(directory, Path(tmp_dir)/"qrels.txt", config)
        zip_directory(tmp_dir, output_zip)
        
