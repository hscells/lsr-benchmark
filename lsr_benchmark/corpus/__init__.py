from tqdm import tqdm
import ir_datasets

def load_docs(ir_datasets_id, subsample):
    ret = {}
    docs_store = ir_datasets.load(ir_datasets_id).docs_store()
    skipped = 0
    for doc in tqdm(subsample):
        try:
            ret[doc] = docs_store.get(doc).default_text()
        except:
            skipped += 1
    print(f"Skipped {skipped} docs")
    return ret