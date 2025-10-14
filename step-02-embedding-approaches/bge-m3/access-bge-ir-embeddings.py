#!/usr/bin/env python3
#STILL NEED TO BE TESTED

import ir_datasets

import lsr_benchmark

lsr_benchmark.register_to_ir_datasets()

dataset = ir_datasets.load("lsr-benchmark/trec-28-deep-learning-passages-20250926-training")

# embeddings are in csr format
# (values, column_indices, row_index_pointers)
# see https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html for details
*query_embeddings, query_ids = dataset.query_embeddings("tesdt/query") #Need to have started it before
for i in range(len(query_ids)):
    row_start, row_end = query_embeddings[2][i], query_embeddings[2][i + 1]
    column_idcs = query_embeddings[1][row_start:row_end]
    values = query_embeddings[0][row_start:row_end]
    print(f"Query ID: {query_ids[i]}, Embedding: {values}, Column Indices: {column_idcs}")
*doc_embeddings, doc_ids = dataset.doc_embeddings(model_name="naver/splade-v3", passage_aggregation="first-passage")
for i in range(10):
    row_start, row_end = doc_embeddings[2][i], doc_embeddings[2][i + 1]
    column_idcs = doc_embeddings[1][row_start:row_end]
    values = doc_embeddings[0][row_start:row_end]
    print(f"Doc ID: {doc_ids[i]}, Embedding: {values}, Column Indices: {column_idcs}")
