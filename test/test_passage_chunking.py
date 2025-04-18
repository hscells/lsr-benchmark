import unittest
import json
from pathlib import Path
from approvaltests import verify_as_json
from lsr_benchmark.corpus.segmentation import segmented_document

def load_docs():
    ret = {}
    for i in ["1", "2"]:
        ret[i] = json.loads((Path(__file__).parent / "resources" / f"example-dl-0{i}.json").read_text())["text"]
    return ret

class TestPassageChunking(unittest.TestCase):
    def test_load_docs(self):
        docs = load_docs()
        self.assertIsNotNone(docs)

    def test_chunking(self):
        docs = load_docs()
        actual = segmented_document(docs)
        verify_as_json(actual)