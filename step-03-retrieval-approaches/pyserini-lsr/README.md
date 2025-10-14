# Pyserini Baseline for LSR

This approach does retrieval as [https://github.com/castorini/pyserini](pyserini) would do, using pre-tokenized queries and embeddings to Pyserini.


## Submission

```
tira-cli code-submission \
    --path . \
    --task lsr-benchmark \
    --tira-vm-id reneuir-baselines \
    --dataset tiny-example-20251002_0-training \
    --command '/run-pyserini-lsr.py --dataset $inputDataset --output $outputDir' \
    --dry-run
```

## Development

This directory is [configured as DevContainer](https://code.visualstudio.com/docs/devcontainers/containers), i.e., you can open this directory with VS Code or some other DevContainer compatible IDE to work directly in the Docker container with all dependencies installed.

If you want to run it locally, please install the dependencies via `pip3 install -r requirements.txt`.

To make predictions on a dataset, run:

```
python run-pyserini-lsr.py --dataset trec-28-deep-learning-passages-20250926-training --output runs/naver-splade-v3/ --embeddings lightning-ir/naver/splade-v3
```

