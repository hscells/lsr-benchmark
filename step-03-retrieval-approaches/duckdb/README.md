# DuckDB Baseline for LSR

This is a simple baseline for the lsr-benchmark that stores document and query embeddings as tables in DuckDB and scores documents using SQL.

## Submission

```
tira-cli code-submission \
    --path . \
    --task lsr-benchmark \
    --tira-vm-id reneuir-baselines \
    --dataset tiny-example-20251002_0-training \
    --command '/run-duckdb.py --dataset $inputDataset --output $outputDir' \
    --dry-run
```

## Development

This directory is [configured as DevContainer](https://code.visualstudio.com/docs/devcontainers/containers), i.e., you can open this directory with VS Code or some other DevContainer compatible IDE to work directly in the Docker container with all dependencies installed.

If you want to run it locally, please install the dependencies via `pip3 install -r requirements.txt`.

To make predictions on a dataset, run:

```
./run-duckdb.py --dataset clueweb09/en/trec-web-2009 --output output-dir
```

The baseline also supports quantization (normalizing the floating point embedding values to integers between 0 and 100) using the `--quantize` flag.

```
./run-duckdb.py --dataset clueweb09/en/trec-web-2009 --output output-dir --quantize
```
