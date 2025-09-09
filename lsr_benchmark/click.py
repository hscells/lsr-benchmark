from click import ParamType
from lsr_benchmark import SUPPORTED_IR_DATASETS
import os


class ClickParamTypeLsrDataset(ParamType):
    name = "dataset_or_dir"

    def convert(self, value, param, ctx):
        if value in SUPPORTED_IR_DATASETS:
            return value

        if os.path.isdir(value):
            return os.path.abspath(value)

        msg = f"{value!r} is not a supported dataset "
        f"({', '.join(SUPPORTED_IR_DATASETS)}) "
        "or a valid directory path"

        self.fail(msg, param, ctx)