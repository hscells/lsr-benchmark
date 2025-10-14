#!/usr/bin/env python3
from pathlib import Path
import click
import numpy as np
import torch
from lightning_ir import DocDataset, LightningIRDataModule, QueryDataset
from tirex_tracker import tracking, register_metadata
from lsr_benchmark.utils import ClickParamTypeLsrDataset
import lsr_benchmark
from FlagEmbedding import BGEM3FlagModel
from more_itertools import chunked
from tqdm import tqdm

def create_iter(local_dataset, text_type):
    for x in tqdm(local_dataset, desc="Encoding"):
        yield [getattr(x, f"{text_type}_id") ,getattr(x, f"{text_type}") ]

@click.command()
@click.option("--dataset", type=ClickParamTypeLsrDataset(), required=True, help="The dataset id or a local directory.")
@click.option("--model", type=str, default="BAAI/bge-m3", help="The huggingface bge model.")
@click.option("--batch_size", type=int, default=4, help="Number of queries/documents to process in a batch.")
@click.option("--save_dir", type=Path, required=True, help="Directory to save output embeddings.")
def main(dataset: str, model: str, batch_size: int, save_dir: Path):
    # register the dataset with ir_datasets
    lsr_benchmark.register_to_ir_datasets(dataset)
    module = BGEM3FlagModel(model,  use_fp16=True) 

        # parse dataset id
    dataset_id = f"lsr-benchmark/{dataset}"

    register_metadata({"actor": {"team": "lightning-ir"}, "tag": model.replace('/', '-')})

    doc_dataset = DocDataset(dataset_id)
    query_dataset = QueryDataset(dataset_id)



    # embed queries and documents
    for text_type, local_dataset in zip(["query", "doc"], [query_dataset, doc_dataset]):
        local_dataset.prepare_data()
        text_type_save_dir = save_dir / text_type
        ids = list()
        all_data = list()
        all_amounts = list()
        all_columns = list()
        with tracking(export_file_path=text_type_save_dir / f"{text_type}-ir-metadata.yml"):
            for data in chunked(create_iter(local_dataset,text_type), batch_size):
                _id = [x[0] for x in data]
                text = [x[1] for x in data]
                output = module.encode(text, return_dense=False, return_sparse=True, return_colbert_vecs=False)

                data = [xs for x in output['lexical_weights'] for xs in list(x.values())]
                columns = [list(x.keys()) for x in output['lexical_weights']]
                amounts = [len(c) for c in columns]
                columns = [xs for x in columns for xs in x]      
                ids.extend(_id)
                all_amounts.extend(amounts)
                all_data.extend(data)
                all_columns.extend(columns)

        data = np.array(all_data)
        indices = np.array(all_columns)
        indptr = np.array(all_amounts).cumsum(0)    

        np.savez_compressed(
            text_type_save_dir / f"{text_type}-embeddings.npz",
            data=data,
            indices=indices,
            indptr=indptr,
        )
        (text_type_save_dir / f"{text_type}-ids.txt").write_text("\n".join(ids))


if __name__ == "__main__":
    main()
