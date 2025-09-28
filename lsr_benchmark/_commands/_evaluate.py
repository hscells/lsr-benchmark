import logging
import re
from collections import defaultdict
from glob import glob
from gzip import GzipFile
from io import TextIOWrapper
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Dict, Any
from zipfile import ZipFile
import gzip

import click
import ir_measures
import pandas as pd
import yaml
from tira.check_format import lines_if_valid
from ir_measures import parse_trec_measure

import lsr_benchmark

if TYPE_CHECKING:
    from typing import _KT, _T, _VT, Any, Callable, Literal, Optional, Union

    from ir_measures import Measure, ScoredDoc

    Metadata = dict[str, Any]


def __get_nested(
    d: "Mapping[_KT, Union[dict, _VT]]", keys: "list[_KT]"
) -> "Union[Mapping[_KT, Union[dict, _VT]], _VT]":
    """Recursively retrieves a value from a nested mapping using a list of keys.

    Args:
        d (Mapping[_KT, Union[dict, _VT]]): The dictionary to traverse.
        keys (list[_KT]): A list of keys representing the path to the desired value.

    Raises:
        TypeError: If an intermediate value in the path is not a mapping.
        KeyError: If any key in the path is not found in the corresponding mapping.

    Returns:
        The value located at the nested key path.
    """
    out: "Union[Mapping[_KT, Union[dict, _VT]], _VT]" = d
    for i, key in enumerate(keys):
        if not isinstance(out, Mapping):
            raise TypeError(f"The value at {'>'.join(map(str, keys[:i]))} is not a mapping. Have {out}.")
        if key not in out:
            raise KeyError(f"The key {'>'.join(map(str, keys[:i+1]))} could not be found. Have {out}.")
        out = out[key]
    return out


def __get_nested_or_default(d: "Mapping[_KT, Union[dict, _VT]]", keys: "list[_KT]", default: "_T" = None) -> "Union[_VT, _T]":
    try:
        return __get_nested(d, keys)
    except KeyError:
        return default



def __read_metrics(name: str) -> "tuple[dict[str, Metadata], list[ScoredDoc]]":
    metadata: "dict[str, Metadata]" = {}

    if Path(name).is_dir():
        for l in lines_if_valid(Path(name), "ir_metadata"):
            metadata[l['name'].replace('.', '').split("-")[0]] = l['content']
        if (Path(name) / "run.txt").is_file():
            run = list(ir_measures.read_trec_run((Path(name) / "run.txt").read_text()))
        else:
            run = list(ir_measures.read_trec_run(gzip.open(Path(name) / "run.txt.gz", "rt")))
    else:
        with ZipFile(name) as archive:
            for entry in archive.filelist:
                if (m := re.match(r"(\w+)-metadata.ya?ml", entry.filename)) is not None:
                    with archive.open(entry) as file:
                        metadata[m.group(1)] = yaml.safe_load(file)
            with archive.open("run.txt.gz", mode="r") as compressed:
                with GzipFile(fileobj=compressed, mode="r") as binary:
                    with TextIOWrapper(binary, encoding="utf-8") as file:
                        run = list(ir_measures.read_trec_run(file))
    
    if len(metadata) == 0:
        raise ValueError("I could not read any metadata")
    if len(run) == 0:
        raise ValueError("I could not load a run")
    return metadata, run


def __get_runtime(metadata: "Metadata", param: "Literal['system', 'user', 'wallclock']" = "wallclock") -> "Optional[str]":
    return __get_nested_or_default(metadata, ("resources", "runtime", param))


def __get_energy_usage(metadata: "Metadata", param: "Literal['total', 'cpu', 'gpu', 'ram']" = "total") -> "Optional[float]":
    def __get_energy(device: "str") -> float:
        try:
            energy_str = metadata["resources"][device]["energy used system"]
        except KeyError:
            logging.warning(f"Energy for {device} was not reported; using 0 Joules")
        try:
            if match := re.match(r'^(\S+)\s*J$', energy_str):
                return float(match[1])
            else:
                raise ValueError
        except ValueError:
            logging.error(f"Could not parse energy string: '{energy_str}'")
        return 0
    if param == "total":
        return __get_energy("cpu") + __get_energy("gpu") + __get_energy("ram")
    return __get_energy(param)


def __get_avg_cpu_usage(metadata: "Metadata") -> "Optional[int]":
    return __get_nested_or_default(metadata, ("resources", "cpu", "used process", "avg"))


def __get_max_ram_usage(metadata: "Metadata") -> "Optional[int]":
    return __get_nested_or_default(metadata, ("resources", "ram", "used process", "max"))


def __get_max_vram_usage(metadata: "Metadata") -> "Optional[int]":
    return __get_nested_or_default(metadata, ("resources", "gpu", "used vram process", "max"))


def __get_avg_gpu_usage(metadata: "Metadata") -> "Optional[int]":
    return __get_nested_or_default(metadata, ("resources", "gpu", "used process", "avg"))


__efficiency_measures: "dict[str, Callable]" = {
    "runtime": __get_runtime, "energy": __get_energy_usage, "cpu": __get_avg_cpu_usage, "ram": __get_max_ram_usage,
    "gpu": __get_avg_gpu_usage, "vram": __get_max_vram_usage
}


def __parse_tirex_measure(measure: "str") -> "Callable":
    name, *arg = measure.split('_', 2)
    func = __efficiency_measures[name]
    return lambda x: func(x, *arg)


def __parse_measure(measure: "str") -> "tuple[str, Literal['ir_measure', 'tirex'], Measure | Callable]":
    try:
        return (measure, 'ir_measure', parse_trec_measure(measure)[0])
    except ValueError:
        pass
    return (measure, 'tirex', __parse_tirex_measure(measure))


def __get_dataset_name(metadata: Dict[str, Any]) -> str:
    candidates = set()

    for k, m in metadata.items():
        if "data" in m and "test collection" in m["data"] and "name" in m["data"]["test collection"] and m["data"]["test collection"]["name"]:
            candidates.add(m["data"]["test collection"]["name"])

    if len(candidates) != 1:
        raise ValueError(f"I can not extract the dataset from the metadata. I found candidates: {list(candidates)}")

    return list(candidates)[0]


def __get_output_routine(specifier: str) -> "Callable[[pd.DataFrame], None]":
    suffix_to_routine: dict[str, Callable[[pd.DataFrame], None]] = {
        ".csv": lambda df: df.to_csv(specifier),
        ".xlsx": lambda df: df.to_excel(specifier),
        ".htm": lambda df: df.to_html(specifier),
        ".html": lambda df: df.to_html(specifier),
        ".json": lambda df: df.to_json(specifier),
        ".tex": lambda df: df.to_latex(specifier),
        ".md": lambda df: df.to_markdown(specifier),
        ".parquet": lambda df: df.to_parquet(specifier),
    }
    if specifier == "-":
        return print
    elif (routine := suffix_to_routine.get(Path(specifier).suffix, None)) is not None:
        return routine
    else:
        raise ValueError(f"The suffix of {specifier} is not known.")


@click.argument(
    "approaches",
    type=str,
    nargs=-1,
)
@click.option(
    "-m", "--measure",
    type=__parse_measure,
    required=False,
    multiple=True,
    default=["ndcg_cut.10",  "P_5", "map_cut.100", "runtime_wallclock", "energy_total"],
    help="The dataset id or a local directory.",
)
@click.option(
    "--upload",
    type=bool,
    default=False,
    is_flag=True,
    required=False,
    help="Upload to tira.",
)
@click.option(
    "-o", "--out",
    type=str,
    required=False,
    multiple=False,
    default="-",
    help="The output file to write to. Use - to print the results to stdout. Default: -",
)
def evaluate(approaches: list[str], measure: list[str], out: str, upload: bool) -> int:
    approaches = [x for xs in map(glob, approaches) for x in xs]
    output_routine = __get_output_routine(out)

    scores: dict[str, dict[str, float]] = defaultdict(dict)
    for approach in approaches:
        metadata, run = __read_metrics(approach)
        for group, meta in metadata.items():
            for name, typ, func in measure:
                if typ == 'tirex':
                    if (val := func(meta)) is not None:
                        scores[approach][f"{group}.{name}"] = val
                    else:
                        scores[approach][f"{group}.{name}"] = None
                        logging.warning(
                            f"Measure {name} could not be reported for {approach}.{group} as its metadata is not present")
        # Update with ir_measures effectiveness measures
        irmeasures = set(m for _, t, m in measure if t == 'ir_measure')

        dataset = __get_dataset_name(metadata)
        lsr_benchmark.register_to_ir_datasets(dataset)
        dset = lsr_benchmark.load(dataset)
        assert dset.has_qrels()

        scores[approach].update({str(k): v for k, v in ir_measures.calc_aggregate(irmeasures, dset.qrels, run).items()})

        if upload:
            from tira.tira_cli import upload_command
            from lsr_benchmark.irds import TIRA_LSR_TASK_ID
            upload_command(dataset=dataset, directory=approach, dry_run=False, system=approach, default_task=TIRA_LSR_TASK_ID)

    output_routine(pd.DataFrame(scores))
    return 0
