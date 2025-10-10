import click
import os
import sys
from tira.io_utils import _fmt, log_message, verify_tira_installation
from tira.third_party_integrations import temporary_directory
from tira.check_format import _fmt, check_format
from tira.rest_api_client import Client
from pathlib import Path
import shutil
import yaml
import json


def run_foo(docker_image, command, dataset_id, embedding, output_dir=None):
    if output_dir is not None and Path(output_dir).exists():
        return
    tira = Client()
    dataset_path = tira.download_dataset("lsr-benchmark", dataset_id)
    embeddings_dir = tira.get_run_output(f'lsr-benchmark/lightning-ir/{embedding}', dataset_id)
    tmp_dir = temporary_directory()
    tira.local_execution.run(
        image=docker_image,
        command=command,
        input_dir=dataset_path,
        output_dir=tmp_dir,
        allow_network=False,
        input_run=embeddings_dir
    )

    result, msg = check_format(Path(tmp_dir), ["run.txt"], {})
    if result != _fmt.OK:
        print(msg)
        raise ValueError(msg)

    tag = yaml.safe_load((Path(tmp_dir) / "retrieval-metadata.yml").read_text())["tag"]
    
    if output_dir is not None:
        from tira.io_utils import patch_ir_metadata
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(tmp_dir, output_dir)
        patch_ir_metadata(output_dir, {"data": {"test collection": {"name": "/tira-data/input"}}}, {"data": {"test collection": {"name": dataset_id}}})
    
    return tag

def all_embeddings():
    overview = json.loads((Path(__file__).parent.parent / "datasets" / "overview.json").read_text())
    ret = set()

    for dataset_id, stats in overview.items():
        for embedding, embedding_size in stats['embedding-sizes'].items():
            ret.add(embedding)
    return ret

def all_datasets():
    overview = json.loads((Path(__file__).parent.parent / "datasets" / "overview.json").read_text())
    return overview.keys()

@click.argument(
    "approaches",
    type=str,
    nargs=-1,
)
@click.option(
    "-o", "--out",
    type=str,
    required=True,
    multiple=False,
    help="The output directory to write to.",
)
@click.option(
    "--dataset",
    type=str,
    multiple=True,
    help="The output directory to write to.",
)
def retrieval(approaches: list[str], dataset: list[str], out: str) -> int:
    all_messages = []

    def print_message(message, level):
        all_messages.append((message, level))
        os.system("cls" if os.name == "nt" else "clear")
        print(' '.join([sys.argv[0].split('/')[-1]] + sys.argv[1:]))
        for m, l in all_messages:
            log_message(m, l)

    if dataset is none or not dataset:
        dataset = 

    status = verify_tira_installation()

    if status != _fmt.OK:
        print_message("Your TIRA installation is not valid. Please run 'tira-cli verify-installation' to resolve the problem", status)
        return 1

    print_message("Your TIRA installation is valid.", _fmt.OK)
    
    tira = Client()
    approach_to_execution = {}
    for approach in approaches:
        docker_tag, zipped_code, remotes, commit, active_branch = tira.build_docker_image_from_code(
            Path(approach), log_message, False
        )
        assert docker_tag not in approach_to_execution.values()
        cmd = (Path(approach) / "README.md").read_text().split("tira-cli code-submission")[1].split('--command')[1].split("'")[1]

        log_message(f"Approach {approach} is compiled.", _fmt.OK)
        system_tag = run_foo(docker_tag, cmd, 'tiny-example-20251002_0-training', 'naver-splade-v3')
        print_message(f"Approach {approach} compiled and produced valid outputs on example dataset (tag={system_tag}).", _fmt.OK)
        approach_to_execution[approach] = {"tag": docker_tag, "command": cmd}

    stats = {}
    for d in dataset:
        for embedding in all_embeddings():
            for approach in approaches:
                out_dir = Path(out) / d / embedding / approach
                try:
                    run_foo(approach_to_execution[approach]["tag"], approach_to_execution[approach]["command"], d, embedding, out_dir)
                except:
                    continue
                    log_message(f"Approach {approach} finished on {d} for embedding {embedding}", _fmt.OK)
                    if approach not in stats:
                        stats[approach] = {"datasets": set(), "embeddings": set()}
                    stats[approach]["datasets"].add(d)
                    stats[approach]["embeddings"].add(embedding)
    for approach in stats:
        print_message(f"Approach {approach} produced valid outputs on {len(stats[approach]['datasets'])} datasets for {len(stats[approach]['embeddings'])} embeddings.", _fmt.OK)
    
    return 0
