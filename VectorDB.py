import os
import json
from pathlib import Path
import copy
import warnings
from collections import defaultdict
from collections import OrderedDict
from collections import Counter
import datetime

import numpy as np
import pandas as pd

import faiss

# Define a custom FAISS class for dual search and custom IDs

class FAISSdb:
    
    def __init__(self, index_q, index_a, id_map, dataset = None):
        
        self.index_q = index_q
        self.index_a = index_a
        self.id_map = id_map
        self.counter = max(self.id_map.keys()) if len(self.id_map) > 0 else -1

        self.dataset = dataset

    def add(self, vectors_q, vectors_a, custom_ids):

        faiss.normalize_L2(vectors_q)
        faiss.normalize_L2(vectors_a)
        
        self.index_q.add(vectors_q)
        self.index_a.add(vectors_a)
        
        for _id in custom_ids:
            self.counter += 1
            self.id_map[self.counter] = _id

    def search(self, query, k = 5):
        
        query = np.ascontiguousarray(query.astype('float32'))
        faiss.normalize_L2(query)
        
        sim_q, ids_q = self.index_q.search(query, k)
        results_q = [[(self.id_map[docid], docsim) for docsim, docid in zip(qsim, qids) if docid != -1]
                           for qsim, qids in zip(sim_q, ids_q)]

        sim_a, ids_a = self.index_a.search(query, k)
        results_a = [[(self.id_map[docid], docsim) for docsim, docid in zip(asim, aids) if docid != -1]
                       for asim, aids in zip(sim_a, ids_a)]
        
        return results_q, results_a

    def rank(self, query, query_sim_weight = 0.5, k = 6, global_k = 100):

        id_list = []
        sim_list = []

        global_k = global_k if self.ntotal() >= global_k else self.ntotal()

        results_q, results_a = self.search(query, k = global_k)

        for i in range(len(results_q)):

            qids, qsims = zip(*results_q[i])
            aids, asims = zip(*results_a[i])

            aids_pos = {val : j for j, val in enumerate(aids)}
    
            aligned_asims = []
    
            for val in qids:
                if val in aids_pos:
                    aligned_asims.append(asims[aids_pos[val]])
                else:
                    aligned_asims.append(-1)
            
            weighted_sim = query_sim_weight * np.array(qsims) + (1 - query_sim_weight) * np.array(aligned_asims)
            
            sort_array = list(reversed(np.argsort(weighted_sim)))

            id_list.append(list(np.array(qids)[sort_array][:k + 1]))
            sim_list.append(list(weighted_sim[sort_array][:k + 1]))

        return id_list, sim_list

    def get_metadata(self, ids: list):
        
        if isinstance(self.dataset, pd.DataFrame):
            return self.dataset.loc[ids, :]
        else:
            print('No metadata to retrieve')
            return []
    
    def create_context_df(self, df, query_embed_col, query_sim_weight = 0.5, k = 6):

        query_embeds = np.ascontiguousarray(np.array(df[query_embed_col].tolist()).astype('float32'))
        faiss.normalize_L2(query_embeds)

        id_list, sim_list = self.rank(query_embeds, query_sim_weight = query_sim_weight, k = k)

        context_df = pd.DataFrame(
            {'context_ids' : id_list,
             'context_sims' : sim_list},
            index = df.index.tolist())

        return pd.concat([df, context_df], axis = 1)        
    
    def ntotal(self):
        return self.counter + 1

    def write(self, path, file_q, file_a, file_map):

        faiss.write_index(self.index_q, str(path.joinpath(file_q)))
        faiss.write_index(self.index_a, str(path.joinpath(file_a)))

        with open(path.joinpath(file_map), 'w') as file:
            file.write(json.dumps(self.id_map, indent = 2))

    @staticmethod
    def read(path, file_q, file_a, file_map, reinitialize_vector_index = False, embed_q_dim = 384, embed_a_dim = 384):

        path.mkdir(parents = True, exist_ok = True)
        
        if reinitialize_vector_index \
            or q_index_file_name not in os.listdir(faiss_subdir) \
                or a_index_file_name not in os.listdir(faiss_subdir) \
                    or index_db_id_map_file_name not in os.listdir(faiss_subdir):
        
            index_q = faiss.IndexFlatIP(embed_q_dim)
            index_a = faiss.IndexFlatIP(embed_a_dim)
        
            index_db_id_map = {}
        
        else:
        
            index_q = faiss.read_index(str(faiss_subdir.joinpath(q_index_file_name)))
            index_a = faiss.read_index(str(faiss_subdir.joinpath(a_index_file_name)))

            def json_import_dtypes(d, key_fn = int, value_fn = str):
                return {key_fn(k) if k.isdigit() else k: value_fn(v) for k, v in d.items()}

            with open(faiss_subdir.joinpath(index_db_id_map_file_name), 'r') as file:
                index_db_id_map = json.load(file, object_hook = json_import_dtypes)

        return index_q, index_a, index_db_id_map
        