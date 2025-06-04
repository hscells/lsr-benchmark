#!/usr/bin/env python3
import ir_datasets

import lsr_benchmark

lsr_benchmark.register_to_ir_datasets()

dataset = ir_datasets.load(f"lsr-benchmark/clueweb09/en/trec-web-2009")

for doc in dataset.docs_iter(embedding="naver/splade-v3", passage_aggregation="first-passage"):
    print(doc.doc_id, doc.embedding)
