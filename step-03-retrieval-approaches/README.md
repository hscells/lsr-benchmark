# Retrieval Engines in the lsr_benchmark

This directory contains the retrieval engines that we currently have in the lsr_benchmark. We aim to organize the lsr_benchmark as mono-repo that is fully self contained, for that reason, if you want to contribute new retrieval engines (we would be very happy about that), please make a pull request.

Currently, we have 8 retrieval engines that can run lsr retrieval:

- [duckdb](duckdb)
- [kannolo](kannolo)
- [naive-search](naive-search)
- [pyserini-lsr](pyserini-lsr)
- [pyterrier-splade](pyterrier-splade)
- [pyterrier-splade-pisa](pyterrier-splade-pisa)
- [pytorch-naive](pytorch-naive)
- [seismic](seismic)

Additionally, we have two lexical retrieval engines as baselines:

- [lexical/pyterrier-naive](lexical/pyterrier-naive)
- [lexical/pyterrier-pisa](lexical/pyterrier-pisa)

## Running all Retrieval Engines

The following code snippet runs all lsr retrieval engines on all embeddings and all datasets and stores the outputs in a directory `../runs`:

```
lsr-benchmark retrieval -o ../runs duckdb kannolo naive-search pyterrier-splade pyterrier-splade-pisa seismic pytorch-naive pyserini-lsr
```

The following snippet runs all lexical retrieval engines on all datasets and stores the outputs in a directory `../runs`:

```
lsr-benchmark retrieval -o ../runs pyterrier-naive/ pyterrier-pisa/ --embedding none
```

## Remaining Retrieval Engines

We are in the progress of adding the following remaining retrieval engines:

- [ ] anserini: Carlos
- [ ] naive with dictionaries or with rust: Cosimo
- [ ] opensearch (Maybe a testcontainer as starting point?): Carlos
- [ ] opensearch seismic (would be interesting to compare the plain seismic with a "production ready" variant"): Carlos
