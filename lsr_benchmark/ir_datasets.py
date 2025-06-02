from pathlib import Path
import zipfile

MAPPING_OF_DATASET_IDS = {
    "clueweb09/en/trec-web-2009": "data/trec-18-web"
}

def base_dir(ir_datasets_id: str):
    if ir_datasets_id not in MAPPING_OF_DATASET_IDS:
        raise ValueError(f"The dataset ID '{ir_datasets_id}' is not supported. Supported are: {MAPPING_OF_DATASET_IDS.keys()}.")
    return Path(MAPPING_OF_DATASET_IDS[ir_datasets_id]).resolve().absolute()

def inputs_dir(ir_datasets_id: str):
    return base_dir(ir_datasets_id) / "inputs-extracted"

def truths_dir(ir_datasets_id: str):
    return base_dir(ir_datasets_id) / "truths-extracted"

def extract_zip(zip_file: Path, target_directory: Path):
    if target_directory.exists():
        return

    if not zipfile.is_zipfile(zip_file):
        raise ValueError(f"I expected that {zip_file} is not a valid ZIP archive.")

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        target_directory.mkdir(parents=True, exist_ok=True)
        zip_ref.extractall(target_directory)

def ensure_corpus_is_extracted(ir_datasets_id: str):
    d = base_dir(ir_datasets_id)

    for src, extracted in [(d / "inputs.zip", inputs_dir(ir_datasets_id)), (d / "truths.zip", truths_dir(ir_datasets_id))]:
        extract_zip(src, extracted)

