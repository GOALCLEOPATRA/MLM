import torch
import flair
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings, BertEmbeddings, DocumentPoolEmbeddings

# set device
torch.cuda.set_device(2)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
flair.device = DEVICE

class DocEmbeddings(object):
    def __init__(self, model_name):
        super().__init__()
        assert model_name in ['flair', 'bert', 'use']
        self.model_name = model_name
        self.document_embeddings = self.load_embedding_model(model_name)

    def load_embedding_model(self, name):
        if name == 'flair':
            print('Loading Flair Embeddings...')
            flair_embedding_forward = FlairEmbeddings('news-forward')
            flair_embedding_backward = FlairEmbeddings('news-backward')
            return DocumentPoolEmbeddings([flair_embedding_forward, flair_embedding_backward])
        elif name == 'bert':
            print('Loading BERT Embeddings...')
            bert_embeddings = BertEmbeddings('bert-base-uncased') # bert-base-multilingual-cased
            return DocumentPoolEmbeddings([bert_embeddings])
        print('Done!')

    def embed(self, tokens):
        x = Sentence(tokens if self.model_name == 'flair' else tokens[:500]) # for bert we use the first 500 tokens
        self.document_embeddings.embed(x)
        return x.embedding