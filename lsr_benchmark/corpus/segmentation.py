from operator import ge

from abc import ABC, abstractmethod
from tqdm import tqdm

from typing import Dict, List


class AbstractPassageChunker(ABC):
    @abstractmethod
    def process_batch(self, document_batch) -> List[Dict]:
        pass

    @staticmethod
    def chunk_document(document_sentences, sentences_word_count, passage_size) -> List[Dict]:
        """
        Creates the passage chunks for a given document
        """
        passages = []

        current_passage = ''
        current_passage_word_count = 0
        sub_id = 1

        for sentence, word_count in zip(document_sentences, sentences_word_count):
            if word_count >= passage_size:
                if current_passage:
                    passages.extend([{
                        "body": current_passage,
                        "id": sub_id
                    }, {
                        "body": sentence.text,
                        "id": sub_id+1
                    }])

                    sub_id += 2

                else:
                    passages.append({
                        "body": sentence.text,
                        "id": sub_id
                    })

                    sub_id += 1

            elif word_count + current_passage_word_count > passage_size:
                passages.append({
                    "body": current_passage,
                    "id": sub_id
                })

                current_passage = sentence.text
                current_passage_word_count = word_count
                sub_id += 1

            else:
                current_passage += ' ' + sentence.text + ' '
                current_passage_word_count += word_count

        if current_passage:
            passages.append({
                "body": current_passage,
                "id": sub_id
            })

        return passages

class SpacyPassageChunker(AbstractPassageChunker):
    def process_batch(self, document_batch, passage_size) -> None:
        import spacy
        nlp = spacy.load("en_core_web_sm", exclude=[
                         "parser", "tagger", "ner", "attribute_ruler", "lemmatizer", "tok2vec"])
        nlp.enable_pipe("senter")
        #python -m spacy download en_core_web_sm
        nlp.max_length = 2000000  # for documents that are longer than the spacy character limit
        doc_ids = list(document_batch.keys())
        ret = {}

        batch_document_texts = [document_batch[k] for k in doc_ids]
        processed_document_texts = nlp.pipe(batch_document_texts, n_process=1)

        for index, document in tqdm(enumerate(processed_document_texts), total=len(doc_ids)):
            document_sentences = list(document.sents)
            sentences_word_count = [
                len([token for token in sentence])
                for sentence in document_sentences
            ]

            generated_passages = self.chunk_document(document_sentences, sentences_word_count, passage_size)
            ret[doc_ids[index]] = generated_passages

        return ret

# code from https://github.com/grill-lab/trec-cast-tools
def segmented_document(documents, passage_size):
    print(f"Segment into passages of size {passage_size}.")
    chunker = SpacyPassageChunker()
    ret_passages = chunker.process_batch(documents, passage_size)
    ret = {}

    for k, p in ret_passages.items():
        segments = []
        if len(p) == 1:
            segments.append({"start": p[0]["id"], "end": p[0]["id"], "text": p[0]["body"].strip()})
        else:
            for i in range(len(p) - 1):
                start = p[i]
                end = p[i+1]

                segments.append({"start": start["id"], "end": end["id"], "text": (start["body"] + " " + end["body"]).strip()})

        ret[k] = {"doc_id": k, "segments": segments, "default_text": documents[k]}

    return ret
