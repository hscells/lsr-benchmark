## Development

This directory is [configured as DevContainer](https://code.visualstudio.com/docs/devcontainers/containers), i.e., you can open this directory with VS Code or some other DevContainer compatible IDE to work directly in the Docker container with all dependencies installed.

If you want to run it locally, please install the dependencies via `pip3 install -r requirements.txt`.

To make predictions on a dataset, run:

```
./build-and-search-naive-index.py --dataset lsr-benchmark/clueweb09/en/trec-web-2009 --embedding naver/splade-v3 --output output-dir
```