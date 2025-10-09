
import numpy as np

class KannoloDatasetBuffer():
    
    def __init__(self):
        self.doc_ids = []
        self.tokens = []
        self.values = []
        self.offsets = [0]
            
    def add_document(self, doc_id, tokens, values):
        self.doc_ids.append(doc_id)
        self.tokens.append(np.fromiter(map(lambda x: int(x), tokens), dtype=np.int32))  
        self.values.append(np.fromiter(values, dtype=np.float32))
        self.offsets.append(self.offsets[-1] + len(tokens))
    
    def __len__(self):
        return len(self.doc_ids)

    def finalize(self):
        self.doc_ids = np.array(self.doc_ids)
        self.tokens = np.ascontiguousarray(np.concatenate(self.tokens, dtype=np.int32).flatten())
        self.values = np.ascontiguousarray(np.concatenate(self.values, dtype=np.float32).flatten())
        self.offsets = np.ascontiguousarray(np.array(self.offsets, dtype=np.int32).flatten())
        
        