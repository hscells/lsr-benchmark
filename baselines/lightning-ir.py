#!/usr/bin/env python3
from pathlib import Path

import click
import torch
from lightning_ir import (
    BiEncoderModule,
    DocDataset,
    IndexCallback,
    LightningIRDataModule,
    LightningIRTrainer,
    QueryDataset,
    TorchSparseIndexConfig,
    TorchSparseIndexer,
)
from tirex_tracker import tracking

import lsr_benchmark


@click.command()
@click.option(
    "--dataset",
    type=click.Choice(lsr_benchmark.SUPPORTED_IR_DATASETS),
    required=True,
    help="The dataset id or a local directory.",
)
@click.option("--model", type=str, required=True, help="The lightning ir model.")
@click.option("--batch_size", type=int, default=4, help="Number of queries/documents to process in a batch.")
@click.option("--save_dir", type=Path, default=Path.cwd(), help="Directory to save output embeddings.")
def main(dataset: str, model: str, batch_size: int, save_dir: Path):
    # register the dataset with ir_datasets
    lsr_benchmark.register_to_ir_datasets()

    # load the model
    module = BiEncoderModule(model_name_or_path=model)

    # embed queries
    datamodule = LightningIRDataModule(
        inference_datasets=[QueryDataset(f"lsr-benchmark/{dataset}/segmented")], inference_batch_size=batch_size
    )
    trainer = LightningIRTrainer(logger=False)
    with tracking(export_file_path=save_dir / "queries" / "query-ir-metadata.yml"):
        output = trainer.predict(model=module, datamodule=datamodule)
    query_embeddings = torch.cat([x.query_embeddings.embeddings for x in output], dim=0).squeeze(1)
    sparse_query_embeddings = torch.sparse_csr_tensor(
        *TorchSparseIndexer.to_sparse_csr(query_embeddings), query_embeddings.shape
    )
    torch.save(sparse_query_embeddings, save_dir / "queries" / "query_embeddings.pt")
    del sparse_query_embeddings
    del query_embeddings

    # index documents
    datamodule = LightningIRDataModule(
        inference_datasets=[DocDataset(f"lsr-benchmark/{dataset}/segmented")], inference_batch_size=batch_size
    )
    index_callback = IndexCallback(index_dir=save_dir / "docs", index_config=TorchSparseIndexConfig())
    trainer = LightningIRTrainer(logger=False, callbacks=[index_callback])
    with tracking(export_file_path=save_dir / "docs" / "index-ir-metadata.yml"):
        trainer.index(model=module, datamodule=datamodule)


if __name__ == "__main__":
    main()
