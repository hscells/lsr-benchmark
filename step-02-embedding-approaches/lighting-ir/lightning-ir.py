#!/usr/bin/env python3
from pathlib import Path
import click
import numpy as np
import torch
from lightning_ir import BiEncoderModule, DocDataset, LightningIRDataModule, LightningIRTrainer, QueryDataset
from tirex_tracker import tracking, register_metadata
from lsr_benchmark.utils import ClickParamTypeLsrDataset
import lsr_benchmark


def convert_embeddings(embeddings: torch.Tensor):
    token_idcs, column_idcs = torch.nonzero(embeddings, as_tuple=True)
    row_indices = (token_idcs + 1).bincount().cumsum(0)
    values = embeddings[token_idcs, column_idcs]
    return values.numpy(), column_idcs.numpy(), row_indices.numpy()


@click.command()
@click.option("--dataset", type=ClickParamTypeLsrDataset(), required=True, help="The dataset id or a local directory.")
@click.option("--model", type=str, required=True, help="The lightning ir model.")
@click.option("--batch_size", type=int, default=4, help="Number of queries/documents to process in a batch.")
@click.option("--save_dir", type=Path, required=True, help="Directory to save output embeddings.")
def main(dataset: str, model: str, batch_size: int, save_dir: Path):
    # register the dataset with ir_datasets
    lsr_benchmark.register_to_ir_datasets(dataset)

    # load the model
    module = BiEncoderModule(model_name_or_path=model)

    # parse dataset id
    dataset_id = f"lsr-benchmark/{dataset}"

    trainer = LightningIRTrainer(logger=False)
    register_metadata({"actor": {"team": "lightning-ir"}, "tag": model.replace('/', '-')})

    # embed queries and documents
    for text_type, Dataset in zip(["query", "doc"], [QueryDataset, DocDataset]):
        datamodule = LightningIRDataModule(inference_datasets=[Dataset(dataset_id)], inference_batch_size=batch_size)
        # downloads dataset if not already downloaded
        datamodule.prepare_data()
        text_type_save_dir = save_dir / text_type

        with tracking(export_file_path=text_type_save_dir / f"{text_type}-ir-metadata.yml"):
            output = trainer.predict(model=module, datamodule=datamodule)

        embeddings = []
        ids = []
        for x in output:
            embeddings.append(getattr(x, f"{text_type}_embeddings").embeddings)
            ids.extend(getattr(x, f"{text_type}_embeddings").ids)

        data, indices, indptr = convert_embeddings(torch.cat(embeddings, dim=0).squeeze(1))

        np.savez_compressed(
            text_type_save_dir / f"{text_type}-embeddings.npz",
            data=data,
            indices=indices,
            indptr=indptr,
        )
        (text_type_save_dir / f"{text_type}-ids.txt").write_text("\n".join(ids))


if __name__ == "__main__":
    main()
