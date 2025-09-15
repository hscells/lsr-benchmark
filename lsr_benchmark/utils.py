from click import ParamType
from lsr_benchmark import SUPPORTED_IR_DATASETS
import os

_IR_DATASETS_FROM_TIRA = None

def ir_datasets_from_tira():
    global _IR_DATASETS_FROM_TIRA
    if _IR_DATASETS_FROM_TIRA is None:
        from tira.rest_api_client import Client
        tira = Client()
        _IR_DATASETS_FROM_TIRA = list(tira.datasets("task_1").keys())
    return _IR_DATASETS_FROM_TIRA


class ClickParamTypeLsrDataset(ParamType):
    name = "dataset_or_dir"

    def convert(self, value, param, ctx):
        if value in SUPPORTED_IR_DATASETS:
            return value

        if os.path.isdir(value):
            return os.path.abspath(value)

        available_datasets = list(SUPPORTED_IR_DATASETS)
        available_datasets += ir_datasets_from_tira()

        msg = f"{value!r} is not a supported dataset " + \
        f"({', '.join(available_datasets)}) " + \
        "or a valid directory path"

        self.fail(msg, param, ctx)