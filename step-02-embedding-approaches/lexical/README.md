# Lexical embedding with PyTerrier

This is a naive baseline that embedds documents into the impact scores of a retrieval model (e.g., BM25).

## Submission

```
tira-cli code-submission \
    --path . \
    --task lsr-benchmark \
    --tira-vm-id reneuir-baselines \
    --dataset tiny-example-20251002_0-training \
    --command '/run-pyterrier.py --dataset $inputDataset --output $outputDir --weights BM25' \
    --dry-run
```

## Development

This directory is [configured as DevContainer](https://code.visualstudio.com/docs/devcontainers/containers), i.e., you can open this directory with VS Code or some other DevContainer compatible IDE to work directly in the Docker container with all dependencies installed.

If you want to run it locally, please install the dependencies via `pip3 install -r requirements.txt`.

To make predictions on a dataset, run:

```
./run-pyterrier.py --dataset tiny-example-20251002_0-training --weights BM25 --output output-dir
```
