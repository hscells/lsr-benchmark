<img width="100%" src="assets/banner.png" alt="The lsr_benchmark banner image">
<h1 align="center">lsr_benchmark</h1>


[![CI](https://img.shields.io/github/actions/workflow/status/reneuir/lsr_benchmark/ci.yml?branch=master&style=flat-square)](https://github.com/reneuir/lsr_benchmark/actions/workflows/ci.yml)
[![Maintenance](https://img.shields.io/maintenance/yes/2025?style=flat-square)](https://github.com/reneuir/lsr_benchmark/graphs/contributors)
[![Code coverage](https://img.shields.io/codecov/c/github/reneuir/lsr_benchmark?style=flat-square)](https://codecov.io/github/reneuir/lsr_benchmark/)
\
[![Release](https://img.shields.io/github/v/tag/reneuir/lsr_benchmark?style=flat-square&label=library)](https://github.com/reneuir/lsr_benchmark/releases/)
[![PyPi](https://img.shields.io/pypi/v/lsr-benchmark?style=flat-square)](https://pypi.org/project/lsr-benchmark/)
[![Downloads](https://img.shields.io/pypi/dm/lsr-benchmark?style=flat-square)](https://pypi.org/project/lsr-benchmark/)
[![Commit activity](https://img.shields.io/github/commit-activity/m/reneuir/lsr_benchmark?style=flat-square)](https://github.com/reneuir/lsr_benchmark/commits)

[CLI](#command-line-tool)&emsp;•&emsp;[Python API](#cc-api)&emsp;•&emsp;[Citation](#citation)

The lsr_benchmark is ...

# Task

Description of the task ...


# Data

The formats for data inputs and outputs aims to support slicing and dicing diverse query and document distributions while enabling caching, allowing for GreenIR research.


Document representation:

Default representation: first-passage

`lsr_benchmark.load(passages='first-passage')`
`lsr_benchmark.load(passages='passages-concatenated')`
`lsr_benchmark.load(passages='passages-stride-concatenated')`

...

# Evaluation

The evaluation methodology aims to encourage the development of diverse and novel measures, as a suitable intertpretation of efficiency for a target task highly depends on the application and its context. Therefore, we aim to measure as many XY as possible in a standardized way with the [tirex-tracker](https://github.com/tira-io/tirex-tracker/) to ensure that XY. This methodology and related aspects were developed as part of the [ReNeuIR workshop series](https://reneuir.org/) held at SIGIR [2022](https://dl.acm.org/doi/abs/10.1145/3477495.3531704), [2023](https://dl.acm.org/doi/abs/10.1145/3539618.3591922), [2024](https://dl.acm.org/doi/abs/10.1145/3626772.3657994), and [2025](https://reneuir.org/).
