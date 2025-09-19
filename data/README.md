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

Assuming you have materialized your corpus as above and you are authenticated against the TIRA backend as admin of the task, you can upload the dataset via:

```
tira-cli dataset-submission --path YOUR-DIRECTORY --task lsr-benchmark --split train
```

This will check that the system-inputs and the truths are valid, it will run a baseline on it, will check that the outputs of the basline are valid and will run the evaluation on the baseline to ensure that everything works. If so, it will upload it to TIRA. All of this is configured in the README.md of the dataset directory in the Hugging Face datasets format.

If everything worked, the output should look like:

```
TIRA Dataset Submission:
✓ Your tira installation is valid.
✓ The configuration of the dataset YOUR-DIRECTORY is valid.
✓ The system inputs are valid.
✓ The truth data is valid.
✓ Repository for the baseline is cloned from https://github.com/reneuir/lsr-benchmark.
✓ The baseline step-03-retrieval-approaches/pyterrier-naive is embedded in a Docker image.
✓ The evaluation of the baseline produced valid outputs: {'nDCG@10': 0.9077324383928644, 'P@10': 0.1}.
✓ Configuration for dataset learned-sparse-retrieval-20250919-training is uploaded to TIRA.
✓ inputs are uploaded to TIRA: Uploaded files ['corpus.jsonl.gz', 'queries.jsonl'] to dataset learned-sparse-retrieval-20250919-training. md5sum=d9853bbcec434be1db7410a3d8e3049e
✓ truths are uploaded to TIRA: Uploaded files ['qrels.txt', 'queries.jsonl'] to dataset learned-sparse-retrieval-20250919-training. md5sum=7a30c3370b098039b5439cbed60f16ce
```


