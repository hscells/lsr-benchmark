<img width="100%" src="assets/banner.png" alt="The lsr-benchmark banner image">
<h1 align="center">lsr-benchmark</h1>


[![CI](https://img.shields.io/github/actions/workflow/status/reneuir/lsr_benchmark/ci.yml?branch=master&style=flat-square)](https://github.com/reneuir/lsr_benchmark/actions/workflows/ci.yml)
[![Maintenance](https://img.shields.io/maintenance/yes/2025?style=flat-square)](https://github.com/reneuir/lsr_benchmark/graphs/contributors)
[![Code coverage](https://img.shields.io/codecov/c/github/reneuir/lsr_benchmark?style=flat-square)](https://codecov.io/github/reneuir/lsr_benchmark/)
\
[![Release](https://img.shields.io/github/v/tag/reneuir/lsr_benchmark?style=flat-square&label=library)](https://github.com/reneuir/lsr_benchmark/releases/)
[![PyPi](https://img.shields.io/pypi/v/lsr-benchmark?style=flat-square)](https://pypi.org/project/lsr-benchmark/)
[![Downloads](https://img.shields.io/pypi/dm/lsr-benchmark?style=flat-square)](https://pypi.org/project/lsr-benchmark/)
[![Commit activity](https://img.shields.io/github/commit-activity/m/reneuir/lsr_benchmark?style=flat-square)](https://github.com/reneuir/lsr_benchmark/commits)

[CLI](#command-line-tool)&emsp;•&emsp;[Python API](#cc-api)&emsp;•&emsp;[Citation](#citation)

**Attention: This is very early work in progress.**

The lsr-benchmark aims to support holisitc evaluations of the learned sparse retrieval paradigm to contrast efficiency and effectiveness accross diverse retrieval scenarios.

# Task

The learned sparse retrieval paradigm conducts retrieval in three steps:

1. Documents are segmented into passages so that the passages can be processed by pre-trained transformers.
2. Documents and queries are embedded into a sparse learned embedding.
3. Retrieval systems create an index of the document embeddings to return a ranking for each embedded query.

You can submit solutions to step 2 (i.e., models that embed documents and queries into sparse embeddings) and/or solutions to step 3 (i.e., retrieval systems). The idea is then to validate all combinations of embeddings with all retrieval systems to identify which solutions work well for which use case, taking different notions of efficiency/effectiveness trade-offs into consideration. The passage segmentation for step 1 is open source (i.e., created via `lsr-benchmark segment-corpus <IR-DATASETS-ID>`) but fixed for this task.

# Supported Corproa

ToDo: We move this list to [https://tira.io/datasets?query=lsr-benchmark](https://archive.tira.io/datasets?query=lsr-benchmark).

| Subsample | IRDS | Partitioned | Embedding |
|-----------|----------|----------|-----------|
|TREC RAG 24|OK|OK|OK|
|TREC DL 19 (Passage)|OK|OK|OK|
|TREC DL 20 (Passage)|OK|OK|OK|
|Robust04|OK|OK|OK|
|corpus-subsamples/clueweb09/en/trec-web-2009| OK | | |
|corpus-subsamples/clueweb09/en/trec-web-2010| OK | | |
|corpus-subsamples/clueweb09/en/trec-web-2011| OK | | |
|corpus-subsamples/clueweb09/en/trec-web-2012| OK | | |
|corpus-subsamples/clueweb12/en/trec-web-2013| OK | | |
|corpus-subsamples/clueweb12/en/trec-web-2014| OK | | |
|corpus-subsamples/clueweb12/b13/trec-misinfo-2019| OK | | |
|GOV Web track 2002||||
|GOV Web track 2003||||
|GOV Web track 2004||||
|GOV2 TB track 2004||||
|GOV2 TB track 2005||||
|GOV2 TB track 2006||||
|WaPo TREC Core 2018||||
|TREC DL 23 (Passage)||||
|Cranfield||||
|Argsme Touché 2020||||

- Potentially other datasets that we could add: TREC Covid, TREC DL 21 (Passage), TREC Precision Medicine, Argsme Touché 2021, TREC-7, TREC-8, TREC DL 19 (Document), TREC DL 20 (Document), TREC DL 21 (Document), TREC DL 23 (Document)


# ToDo: Documentation

- Maik: Write how to add new datasets, embeddings, retrieval, evaluation
  - short video

# Pre-computed Embeddings

- [x] [webis/splade](https://huggingface.co/webis/splade)
- [x] [naver/splade-v3](https://huggingface.co/naver/splade-v3)
- [x] [naver/splade-v3-distilbert](https://huggingface.co/naver/splade-v3-distilbert)
- [x] [naver/splade_v2_distil](https://huggingface.co/naver/splade_v2_distil)
- [x] [naver/splade-v3-lexical](https://huggingface.co/naver/splade-v3-lexical)
- [x] [naver/splade-v3-doc](https://huggingface.co/naver/splade-v3-doc)
- [x] [castorini/unicoil-noexp-msmarco-passage](https://huggingface.co/castorini/unicoil-noexp-msmarco-passage)
- [x] [opensearch-project/opensearch-neural-sparse-encoding-v2-distill](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-v2-distill)
- [x] [opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill)
- [x] [opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini)
- [x] [opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill)
- [ ] [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)

# Installation

To install the lsr-benchmark for development purposes, please clone the repository and then run:

```
pip3 install -e .
```

If you want to install from the main branch, please use:

```
pip3 install git+https://github.com/reneuir/lsr-benchmark.git
```

# Data

The formats for data inputs and outputs aim to support slicing and dicing diverse query and document distributions while enabling caching, allowing for GreenIR research.

You can slice and dice the document texts and document embeddings via the API. The document texts for private corpora are only available within the [TIRA sandbox](https://docs.tira.io/participants/python-client.html) whereas the document embeddings are publicly available for all corpora (as one can not re-construct the original documents from sparse embeddings).

```
dataset = lsr_benchmark.load('<IR-DATASETS-ID>')

# process the document embeddings:
for doc in dataset.docs_iter(embedding='<EMBEDDING-MODEL>', passage_aggregation="first-passage"):
    doc # namedtuple<doc_id, embedding>

# process the document embeddings for all segments:
for doc in dataset.docs_iter(embedding='<EMBEDDING-MODEL>'):
    doc # namedtuple<doc_id, segments.embedding>

# process the document texts:
for doc in dataset.docs_iter(embedding=None):
    doc # namedtuple<doc_id, segments.text>

# process the document texts via segmented versions in ir_datasets
lsr_benchmark.register_to_ir_datasets()
for segmented_doc in ir_datasets.load(f"lsr-benchmark/{dataset}/segmented")
    doc # namedtuple<doc_id, segment>
```

## Format of Document Texts

Inspired by the processing of [MS MARCO v2.1](https://trec-rag.github.io/annoucements/2024-corpus-finalization/), each document consists of a `doc_id` and a list of text `segments` that are short enough to be processed by pre-trained transformers. For instance, a document that consists of 4 passages (e.g., `"text-of-passage-1 text-of-passage-2 text-of-passage-3 text-of-passage-4"`) would be represented as:

- doc_id: 12fd3396-e4d7-4c0f-b468-5a82402b5336
- segments:
  - {"start": 1, "end": 2, "text": "text-of-passage-1 text-of-passage-2"}
  - {"start": 2, "end": 3, "text": "text-of-passage-2 text-of-passage-3"}
  - {"start": 3, "end": 4, "text": "text-of-passage-3 text-of-passage-4"}

## Format of Document Embeddings

Each document consists of a `doc_id` and a list of text `segments` that are short enough to be processed by pre-trained transformers. For instance, a document that consists of 4 passages would be represented as:

- doc_id: 12fd3396-e4d7-4c0f-b468-5a82402b5336
- segments:
  - {"start": 1, "end": 2, "embedding": {"term-1": 0.123, "term-2": 0.912}}
  - {"start": 2, "end": 3, "embedding": {"term-1": 0.421, "term-3": 0.743}}
  - {"start": 3, "end": 4, "embedding": {"term-2": 0.108, "term-4": 0.043}}

# Evaluation

The evaluation methodology encourages the development of diverse and novel measures, as a suitable interpretation of efficiency for a target task highly depends on the application and its context. Therefore, we aim to measure as many XY as possible in a standardized way with the [tirex-tracker](https://github.com/tira-io/tirex-tracker/) to ensure that XY. This methodology and related aspects were developed as part of the [ReNeuIR workshop series](https://reneuir.org/) held at SIGIR [2022](https://dl.acm.org/doi/abs/10.1145/3477495.3531704), [2023](https://dl.acm.org/doi/abs/10.1145/3539618.3591922), [2024](https://dl.acm.org/doi/abs/10.1145/3626772.3657994), and [2025](https://reneuir.org/).
