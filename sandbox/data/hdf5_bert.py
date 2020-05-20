


''' convert text field to document embeddings using flair'''

import torch 
from torch import tensor
import json 
from flair.embeddings import BertEmbeddings,DocumentPoolEmbeddings
import numpy as np 
from tqdm import tqdm 
from flair.data import Sentence
import os 
import h5py

# paths
text_dir = 'C:/Users/JasonA1/Documents/pythonscripts/wiki-mlm/pilot' # update path
train_text = 'C:/Users/JasonA1/Documents/pythonscripts/wiki-mlm/pilot/pilot_capitals_mini.json' # update path

# extract features and create h5f file
def extract_bert_features(train_text,text_dir,split="train"):
    capitals=json.load(open(train_text, encoding="utf8"))['capitals'] 

    entity_ids=[quest['itemLabel'] for quest in capitals]
    bert=BertEmbeddings('bert-base-uncased')
    doc_bert=DocumentPoolEmbeddings([bert])
    bert_embs_matrix=np.zeros((len(capitals),3072))
    print(bert_embs_matrix)
    print('Extracting bert features')
    
    for index,quest in tqdm(enumerate(capitals)):
        sentence=Sentence(quest['text'])
        doc_bert.embed(sentence)
        bert_embs_matrix[index]=sentence.embedding.detach().cpu().numpy()
    
    hdf5_file_path=os.path.join(text_dir,split+'_embs_text'+'.hdf5')
    h5f = h5py.File(hdf5_file_path, 'w')
    h5f.create_dataset('bert_embs', data=bert_embs_matrix)
    h5f.create_dataset('entity_ids', data=entity_ids)
    h5f.close()

extract_bert_features(train_text, text_dir)