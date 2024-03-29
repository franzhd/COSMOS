
import numpy as np
from sklearn.neighbors import NearestNeighbors
from langchain.embeddings import OpenAIEmbeddings
import os

class SemanticSearch:
    
    def __init__(self):
        import tensorflow_hub as hub
        self.use = hub.load("use")
        
        # self.use = OpenAIEmbeddings().embed_query
        self.fitted = False
    
    
    def fit(self, data, batch=512, n_neighbors=5):
        
        self.data = data
        self.embeddings = self.get_text_embedding(data, batch=batch)
        n_neighbors = min(n_neighbors, len(self.embeddings))
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        self.nn.fit(self.embeddings)
        self.fitted = True
    
    
    def __call__(self, text, return_data=True):
        
        inp_emb = self.use([text])
        neighbors = self.nn.kneighbors(inp_emb, return_distance=False)[0]
        
        if return_data:
            return [self.data[i] for i in neighbors]
        else:
            return neighbors
    
    
    def get_text_embedding(self, texts, batch=512):
        
        embeddings = []
        for i in range(0, len(texts), batch):
            text_batch = texts[i:(i+batch)]
            emb_batch = self.use(text_batch)
            embeddings.append(emb_batch)
        embeddings = np.vstack(embeddings)
        return embeddings