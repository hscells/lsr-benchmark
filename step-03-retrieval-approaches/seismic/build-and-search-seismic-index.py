
import ir_datasets
import lsr_benchmark
import numpy as np

import seismic
from seismic import SeismicIndex, SeismicDataset
from tqdm import tqdm

lsr_benchmark.register_to_ir_datasets()


def main():
    dataset = ir_datasets.load(f"lsr-benchmark/clueweb09/en/trec-web-2009")
    seismic_dataset = SeismicDataset()
    string_type  = seismic.get_seismic_string()


    for doc in tqdm(dataset.docs_iter(embedding="naver/splade-v3", passage_aggregation="first-passage")):

        # The input types should be "string", array of f32, array of string_type    
        values = doc.embedding.values().numpy()
        keys = np.asarray(doc.embedding.indices().numpy(), dtype=string_type)
        
        seismic_dataset.add_document(doc.doc_id, keys, values)

    print("Documents added to the SeismicDataset. Now indexing..")
    index = SeismicIndex.build_from_dataset(seismic_dataset)
    
    results = []

    for query in dataset.queries_iter(embedding="naver/splade-v3", passage_aggregation="first-passage"):
        query_components = np.asarray(query.embedding.indices().numpy(), dtype=string_type)
        query_values = query.embedding.values().numpy()

        current_res = index.search(query_id=str(query.query_id), query_components=query_components, query_values=query_values, k=10, query_cut=10, heap_factor=0.8)
        results.append(current_res)

    print("Search completed.")
    print("Results for the first query: ", results[0])




if __name__ == "__main__":
    main()