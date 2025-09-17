from click import ParamType
from lsr_benchmark import SUPPORTED_IR_DATASETS, ir_datasets_from_tira
import os

class ClickParamTypeLsrDataset(ParamType):
    name = "dataset_or_dir"

    def convert(self, value, param, ctx):
        if value in SUPPORTED_IR_DATASETS:
            return value

        if os.path.isdir(value):
            return os.path.abspath(value)

        irds_from_tira = ir_datasets_from_tira()
        if value in irds_from_tira:
            return value

        irds_from_tira = ir_datasets_from_tira(force_reload=True)
        if value in irds_from_tira:
            return value

        available_datasets = list(SUPPORTED_IR_DATASETS)
        available_datasets += irds_from_tira

        msg = f"{value!r} is not a supported dataset " + \
        f"({', '.join(available_datasets)}) " + \
        "or a valid directory path"

        self.fail(msg, param, ctx)