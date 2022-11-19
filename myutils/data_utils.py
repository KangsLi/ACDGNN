import os
import pdb
import numpy as np
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.decomposition import PCA
from scipy.special import softmax
from tqdm import tqdm


def plot_rel_dist(adj_list, filename):
    rel_count = []
    for adj in adj_list:
        rel_count.append(adj.count_nonzero())

    fig = plt.figure(figsize=(12, 8))
    plt.plot(rel_count)
    fig.savefig(filename, dpi=fig.dpi)


def process_files(files, saved_relation2id=None):
    '''
    files: Dictionary map of file paths to read the triplets from.
    saved_relation2id: Saved relation2id (mostly passed from a trained model) which can be used to map relations to pre-defined indices and filter out the unknown ones.
    '''
    entity2id = {}
    relation2id = {} if saved_relation2id is None else saved_relation2id

    triplets = {}

    ent = 0
    rel = 0

    for file_type, file_path in files.items():

        data = []
        with open(file_path) as f:
            file_data = [line.split() for line in f.read().split('\n')[:-1]]

        for triplet in file_data:
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = ent
                ent += 1
            if triplet[2] not in entity2id:
                entity2id[triplet[2]] = ent
                ent += 1
            if not saved_relation2id and triplet[1] not in relation2id:
                relation2id[triplet[1]] = rel
                rel += 1

            # Save the triplets corresponding to only the known relations
            if triplet[1] in relation2id:
                data.append([entity2id[triplet[0]], entity2id[triplet[2]], relation2id[triplet[1]]])

        triplets[file_type] = np.array(data)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Construct the list of adjacency matrix each corresponding to eeach relation. Note that this is constructed only from the train data.
    adj_list = []
    for i in range(len(relation2id)):
        idx = np.argwhere(triplets['train'][:, 2] == i)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8), (triplets['train'][:, 0][idx].squeeze(1), triplets['train'][:, 1][idx].squeeze(1))), shape=(len(entity2id), len(entity2id))))

    return adj_list, triplets, entity2id, relation2id, id2entity, id2relation, rel

def process_files_ddi(files, triple_file, saved_relation2id=None, keeptrainone = False):
    entity2id = {}
    relation2id = {} if saved_relation2id is None else saved_relation2id

    triplets = {}
    kg_triple = []
    ent = 0
    rel = 0

    for file_type, file_path in files.items():
        data = []
        # with open(file_path) as f:
        #     file_data = [line.split() for line in f.read().split('\n')[:-1]]
        file_data = np.loadtxt(file_path)
        for triplet in file_data:
            #print(triplet)
            triplet[0], triplet[1], triplet[2] = int(triplet[0]), int(triplet[1]), int(triplet[2])
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = triplet[0]
                #ent += 1
            if triplet[1] not in entity2id:
                entity2id[triplet[1]] = triplet[1]
                #ent += 1
            if not saved_relation2id and triplet[2] not in relation2id:
                if keeptrainone:
                    triplet[2] = 0
                    relation2id[triplet[2]] = 0
                    rel = 1
                else:
                    relation2id[triplet[2]] = triplet[2]
                    rel += 1

            # Save the triplets corresponding to only the known relations
            if triplet[2] in relation2id and file_type=='train':
                data.append([entity2id[triplet[0]], entity2id[triplet[1]], relation2id[triplet[2]]])
            else:
                data.append([entity2id[triplet[0]], entity2id[triplet[1]], relation2id[triplet[2]], triplet[3]])

        triplets[file_type] = np.array(data)
    # print("rel:",rel)
    triplet_kg = np.loadtxt(triple_file)
    # print("np.max(triplet_kg[:, -1]):",np.max(triplet_kg[:, -1]))
    for (h, t, r) in triplet_kg:
        h, t, r = int(h), int(t), int(r)
        if h not in entity2id:
            entity2id[h] = h
        if t not in entity2id:
            entity2id[t] = t 
        if not saved_relation2id and rel+r not in relation2id:
            relation2id[rel+r] = rel + r
        kg_triple.append([h, t, r])
    kg_triple = np.array(kg_triple)
    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}
    #print(relation2id, rel)

    # Construct the list of adjacency matrix each corresponding to eeach relation. Note that this is constructed only from the train data.
    adj_list = []
    #print(kg_triple)
    #for i in range(len(relation2id)):
    shape = (max(entity2id.keys())+1,max(entity2id.keys())+1)
    # count = 0
    for i in range(rel):
        idx = np.argwhere(triplets['train'][:, 2] == i)
        # count = count + idx.shape[0]
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8), (triplets['train'][:, 0][idx].squeeze(1), triplets['train'][:, 1][idx].squeeze(1))), shape=shape))
    for i in range(rel, len(relation2id)):
        idx = np.argwhere(kg_triple[:, 2] == i-rel)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8), (kg_triple[:, 0][idx].squeeze(1), kg_triple[:, 1][idx].squeeze(1))), shape=shape))
    #print(adj_list)
    #assert 0
    return adj_list, triplets, entity2id, relation2id, id2entity, id2relation, rel

def process_files_decagon(files, triple_file, saved_relation2id=None, keeptrainone = True):
    entity2id = {}
    relation2id = {} if saved_relation2id is None else saved_relation2id

    triplets = {}
    triplets_mr = {}
    polarity_mr = {}
    kg_triple = []
    ent = 0
    rel = 0

    for file_type, file_path in files.items():
        data = []
        data_mr = []
        data_pol = []
        edges = {}
        # with open(file_path) as f:
        #     file_data = [line.split() for line in f.read().split('\n')[:-1]]
        #file_data = np.loadtxt(file_path)
        
        train = []
        train_edge = []
        with open(file_path, 'r') as f:
            for lines in f:
                x, y, z, w = lines.strip().split('\t')
                x, y = int(x), int(y)
                w = int(w) # pos/neg edge
                z1 = list(map(int, z.split(',')))

                
                z = [0] if keeptrainone else [i for i, _ in enumerate(z1) if _ == 1]  
                #train.append([x,y])
                #train_edge.append(z)
                for s in z:
                    #print(triplet)
                    triplet = [x,y,s]
                    triplet[0], triplet[1], triplet[2] = int(triplet[0]), int(triplet[1]), int(triplet[2])
                    if triplet[0] not in entity2id:
                        entity2id[triplet[0]] = triplet[0]
                        #ent += 1
                    if triplet[1] not in entity2id:
                        entity2id[triplet[1]] = triplet[1]
                        #ent += 1
                    if not saved_relation2id and triplet[2] not in relation2id:
                        if keeptrainone:
                            triplet[2] = 0
                            relation2id[triplet[2]] = 0
                            rel = 1
                        else:
                            relation2id[triplet[2]] = triplet[2]
                            rel += 1

                    # Save the triplets corresponding to only the known relations
                    if triplet[2] in relation2id:
                        data.append([entity2id[triplet[0]], entity2id[triplet[1]], relation2id[triplet[2]]])
                if keeptrainone:
                    #triplet[2] = 0
                    data_mr.append([entity2id[triplet[0]], entity2id[triplet[1]], 0])
                else:
                    data_mr.append([entity2id[triplet[0]], entity2id[triplet[1]], z1])
                data_pol.append(w)
        triplets[file_type] = np.array(data)
        triplets_mr[file_type] = data_mr
        polarity_mr[file_type] = np.array(data_pol)
    assert len(entity2id) == 604
    if not keeptrainone:
        assert rel == 200
    else:
        assert rel == 1
    #print(rel)
    triplet_kg = np.loadtxt(triple_file)
    print(np.max(triplet_kg[:, -1]))
    for (h, t, r) in triplet_kg:
        h, t, r = int(h), int(t), int(r)
        if h not in entity2id:
            entity2id[h] = h
        if t not in entity2id:
            entity2id[t] = t 
        if not saved_relation2id and rel+r not in relation2id:
            relation2id[rel+r] = rel + r
        kg_triple.append([h, t, r])
    kg_triple = np.array(kg_triple)
    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Construct the list of adjacency matrix each corresponding to eeach relation. Note that this is constructed only from the train data.
    adj_list = []
    #print(kg_triple)
    #for i in range(len(relation2id)):
    for i in range(rel):
        idx = np.argwhere(triplets['train'][:, 2] == i)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8), (triplets['train'][:, 0][idx].squeeze(1), triplets['train'][:, 1][idx].squeeze(1))), shape=(len(entity2id), len(entity2id))))
    for i in range(rel, len(relation2id)):
        idx = np.argwhere(kg_triple[:, 2] == i-rel)
        #print(len(idx), i)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8), (kg_triple[:, 0][idx].squeeze(1), kg_triple[:, 1][idx].squeeze(1))), shape=(len(entity2id), len(entity2id))))
    #print(adj_list)
    #assert 0
    return adj_list, triplets, entity2id, relation2id, id2entity, id2relation, rel, triplets_mr, polarity_mr

def save_to_file(directory, file_name, triplets, id2entity, id2relation):
    file_path = os.path.join(directory, file_name)
    with open(file_path, "w") as f:
        for s, o, r in triplets:
            f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')


def preprocess_adj_hete(adj):
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    adj = adj.T # transpose the adjacency matrix here
    indices = np.vstack((adj.row, adj.col)).transpose() 
    return indices, adj.data, adj.shape


def get_target_sim_matrix(target_matrix):
    def Jaccard(matrix):
        matrix = np.mat(matrix)
        numerator = matrix * matrix.T
        denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T
        return numerator / denominator
    sim_matrix = Jaccard(target_matrix)
    return sim_matrix

def get_pca_feature(feature,n_components):
    pca = PCA(n_components=n_components)  # PCA dimension
    pca.fit(feature)
    return pca.transform(feature)

def get_edge_count(adj_list):
    count = []
    for adj in adj_list:
        count.append(len(adj.tocoo().row.tolist()))
    return np.array(count)

#sample negatives
def sample_neg(adj_list, edges, num_neg_samples_per_link=1, max_size=1000000, constrained_neg_prob=0,drug_num=1710):
    pos_edges = edges.astype(np.int32)
    neg_edges = []
    train_drugs = list(set(edges[:,:-1].flatten()))

    # if max_size is set, randomly sample train links
    if max_size < len(pos_edges):
        perm = np.random.permutation(len(pos_edges))[:max_size]
        pos_edges = pos_edges[perm]

    # sample negative links for train/test
    n, r = drug_num, len(adj_list)

    # distribution of edges across reelations
    theta = 0.001
    edge_count = get_edge_count(adj_list)
    rel_dist = np.zeros(edge_count.shape)
    idx = np.nonzero(edge_count)
    rel_dist[idx] = softmax(theta * edge_count[idx])

    # possible head and tails for each relation
    valid_heads = [adj.tocoo().row.tolist() for adj in adj_list]
    valid_tails = [adj.tocoo().col.tolist() for adj in adj_list]

    pbar = tqdm(total=len(pos_edges))
    # count = 0
    while len(neg_edges) < num_neg_samples_per_link * len(pos_edges):
        neg_head, neg_tail, rel = pos_edges[pbar.n % len(pos_edges)][0], pos_edges[pbar.n % len(pos_edges)][1], pos_edges[pbar.n % len(pos_edges)][2]
        # token = 0
        if np.random.uniform() < constrained_neg_prob:
            # token = 1
            if np.random.uniform() < 0.5:
                neg_head = np.random.choice(valid_heads[rel])
            else:
                neg_tail = np.random.choice(valid_tails[rel])
        else:
            if np.random.uniform() < 0.5:
                neg_head = np.random.choice(len(train_drugs))
                neg_head = train_drugs[neg_head]
            else:
                neg_tail = np.random.choice(len(train_drugs))
                neg_tail = train_drugs[neg_tail]

        if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
            neg_edges.append([neg_head, neg_tail, rel])
            pbar.update(1)
            # if token == 1:
            #     count += 1

    pbar.close()

    neg_edges = np.array(neg_edges)
    return pos_edges, neg_edges
