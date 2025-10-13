import json
from pathlib import Path

def lsr_overview():
    return json.loads((Path(__file__).parent / "overview.json").read_text())

def all_embeddings():
    overview = lsr_overview()
    ret = set()

    for dataset_id, stats in overview.items():
        for embedding, embedding_size in stats['embedding-sizes'].items():
            ret.add(embedding)
    return sorted(list(ret))

def all_datasets():
    overview = lsr_overview()
    return sorted(list(overview.keys()))


TIRA_DATASET_ID_TO_IR_DATASET_ID = {
    'trec-18-web-20251008-test': 'clueweb09/en/trec-web-2009',
    'trec-19-web-20251008-test': 'clueweb09/en/trec-web-2010',
    'trec-20-web-20251008-test': 'clueweb09/en/trec-web-2011',
    'trec-21-web-20251008-test': 'clueweb09/en/trec-web-2012',
    'trec-22-web-20251008-test': 'clueweb12/trec-web-2013',
    'trec-23-web-20251008-test': 'clueweb12/trec-web-2014',
    'trec-28-deep-learning-passages-20250926-training': 'msmarco-passage/trec-dl-2019/judged',
    'trec-28-misinfo-20251008_1-test': 'clueweb12/b13/trec-misinfo-2019',
    'trec-29-deep-learning-passages-20250926-training': 'msmarco-passage/trec-dl-2020/judged',
    'trec-33-rag-20250926_1-training': 'msmarco-segment-v2.1/trec-rag-2024',
    'trec-robust-2004-fold-1-20250927-test': 'disks45/nocr/trec-robust-2004/fold1',
    'trec-robust-2004-fold-2-20250926-test': 'disks45/nocr/trec-robust-2004/fold2',
    'trec-robust-2004-fold-3-20250926-test': 'disks45/nocr/trec-robust-2004/fold3',
    'trec-robust-2004-fold-4-20250926-test': 'disks45/nocr/trec-robust-2004/fold4',
    'trec-robust-2004-fold-5-20250926-test': 'disks45/nocr/trec-robust-2004/fold5',
    'tiny-example-20251002_0-training': None
}

IR_DATASET_TO_TIRA_DATASET = {v:k for k, v in TIRA_DATASET_ID_TO_IR_DATASET_ID.items()}

def all_ir_datasets():
    return sorted([TIRA_DATASET_ID_TO_IR_DATASET_ID[i] for i in all_datasets() if i in TIRA_DATASET_ID_TO_IR_DATASET_ID])
