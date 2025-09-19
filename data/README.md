# Adding New Datasets

This README describs how to incorporate new datasets.

## Step 1: Materialize a Corpus

You can take inspiration from the existing directories. The command `lsr-benchmark create-lsr-corpus` applies the [Corpus Subsampling](https://webis.de/publications.html#froebe_2025c) and materializes the corpus into a directory. First, create a `config.json` file manually in your target directory that has the following fields:

- `runs`: The directory that contains all runs used for corpus subsampling (usually all runs submitted to TREC)
- `ir-datasets-id`: The ID of the dataset in ir-datstes
- `subsample_depth`: The subsampling depth, e.g., 100 or 200.

After the subsampling, the directory structure of your materialized corpus should be:

```
YOUR-DIRECTORY/
├── config.json
├── corpus.jsonl.gz
├── qrels.txt
├── queries.jsonl
└── README.md
```

You can find a minimum example in [the integration tests of TIRA](https://github.com/tira-io/tira/tree/main/python-client/tests/resources/example-datasets/learned-sparse-retrieval).

## Step 2: Ensure your TIRA Client works

We use [TIRA](https://archive.tira.io) as backend.

Install the tira client via:

```
pip3 install tira
```

Next, check that your TIRA client is correctly installed and that you are authenticated:

```
tira-cli verify-installation
```

If everything is as expected, the output should look like:

```
✓ You are authenticated against www.tira.io.
✓ TIRA home is writable.
✓ Docker/Podman is installed.
✓ The tirex-tracker works and will track experimental metadata.

Result:
✓ Your TIRA installation is valid.
```

## Step 3: Upload the Dataset

Accessing data (in progress)


tira-cli upload --dataset trec-web-2009-20250605-test --directory trec-18-web/truths-extracted/ --system truths

