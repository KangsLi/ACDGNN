import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.array_ops import concat
from tensorflow.python.ops.variables import trainable_variables
import myutils.data_utils as ut
from myutils.initialization_utils import initialize_experiment
import myutils.parser as pars
from myutils.dsraddi_func import _convert_sp_mat_to_sp_tensor,sp_hete_attn_head,sp_hete_attn_head1,get_output
from myutils.dsraddi_func import get_att_aggre_embedding
from functools import reduce
from myutils.eval import evaluate,evaluate1,evaluate_data
import time
import os
from scipy.sparse import csc_matrix

import pickle



#command line arguments processing
params = pars.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_id


def full_connection(seq, out_sz, activation=-1, in_drop=0.0, use_bias=True):
    with tf.name_scope('full_connection_layer'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, keep_prob=1.0 - in_drop)
        
        seq_fc = tf.layers.conv1d(seq, out_sz, 1, use_bias=use_bias)
        seq_fc = tf.squeeze(seq_fc) # remove the bach_size which is set as 1
        if activation!=-1:
            ret = activation(seq_fc)
        else:
            ret = seq_fc
        return ret


base_dir = os.getcwd()

print(params)
params.hid = [int(params.emb_dim/params.layers) for i in range(params.layers)]
print('params.hid:%s'%params.hid)
initialize_experiment(params, base_dir)

#dump prediction result to file
params.heads = params.heads
f = open("./result/myresult_%d_%d_%d_%d.txt"%(params.heads,params.emb_dim,params.fsia,\
    params.cdt),'a+',encoding='utf-8')  
params.fsia = bool(params.fsia)
params.cdt = bool(params.cdt)
heads = params.heads
print(params.split, file=f)
f.flush()
print('data/{}/{}/{}_triple.txt'.format(params.dataset, params.split, params.train_file))
#select training data
params.file_paths = {
        'train': os.path.join(params.main_dir, 'data/{}/{}/{}_triple.txt'.format(params.dataset, params.split, params.train_file)),
        'valid': os.path.join(params.main_dir, 'data/{}/{}/{}_triple.txt'.format(params.dataset, params.split, params.valid_file)),
        'test': os.path.join(params.main_dir, 'data/{}/{}/{}_triple.txt'.format(params.dataset, params.split, params.test_file))
    }
triple_file = 'data/{}/relations_2hop_data_new.txt'.format(params.dataset)
'''
print("total epoch:%d"%(params.epoch)) 
Dataset description：33765 nodes with 10 types;1690693 edges of 23 types
adj_list：adjacency matrices of DDI network and KG
drug related networks：(refer to https://github.com/hetio/hetionet/blob/master/describe/edges/metaedges.tsv)
87:Compound-resembles-drug/drug-resembles-drug
91:drug-treat-disease
92:drug-binds-gene
93:drug-upregulates-gene
96：drug-palliates-disease
104:drug-downregulates-gene
107:drug-causes-Side Effect
101:Pharmacologic-contains-drug


entity2id:mapping of entity to id
relation2id:mapping of relation to id
id2entity,id2relation：inverse of entity2id，relation2id
rel：type number of DDI events，rel=86
'''
print("emb dim:%d, rela dim:%d hid dim:%s"%(params.emb_dim,params.rel_dim,params.hid))
adj_list, triplets, entity2id, relation2id, id2entity, id2relation, rel = ut.process_files_ddi(params.file_paths, triple_file, None) #data reading
print("loading finish.", file=f) 
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),file=f)
f.flush()

n_components = params.hid[-1]
#calculate similarity of DDI 
ddi_adj = reduce(np.add,adj_list[:86])
adj_indices, adj_data, adj_shape = ut.preprocess_adj_hete(ddi_adj)
ddi_adj = csc_matrix((adj_data, (adj_indices[:,0],adj_indices[:,1])),shape=(1710,1710))
ddi_adj = ddi_adj.todense()
ddi_adj = ddi_adj + np.transpose(ddi_adj)
ddi_adj[np.where(ddi_adj>0)]=1
for i in range(len(ddi_adj)):
  ddi_adj[i,i] = 1
ddi_sim = ut.get_target_sim_matrix(ddi_adj) #calculate JACCARD similarity
ddi_sim = ut.get_pca_feature(ddi_sim,n_components)
ddi_sim_in = tf.constant(ddi_sim,dtype=tf.float32) 


sparse_ddi_adj = adj_list[:86]

#obtain masks of each DDI type
ddi_types = 86
ent_types = 10 
ent_type = np.loadtxt('./data/drugbank/entity.txt')
ent_mask = np.zeros((ent_types,ent_type.shape[0]),dtype=np.float32)
for i in range(ent_types):
    mask_i = np.where(ent_type==i)
    ent_mask[i][mask_i] = 1.0
ent_mask = np.expand_dims(ent_mask,axis=-1)
#109 adjacency matrices
sparse_adj_input = [ut.preprocess_adj_hete(a) for a in adj_list]

sparse_adj_input = sparse_adj_input[ddi_types:]

adj_type = [(1,1),(0,0),(2,1),(2,3),(2,4),(0,2),(0,1),(0,1),(2,2),(2,1),
        (0,2),(4,1),(4,1),(1,1),(1,5),(6,0),(1,7),(1,1),(0,1),(2,1),
        (1,8),(0,9),(4,1)]

sparse_adj_input_t = []
for item in sparse_adj_input:
    edge_list,value,dense_shape = item 
    sparse_adj_input_t.append((edge_list[:,-1::-1],value,dense_shape))

adj_type_t = np.array(adj_type)[:,-1::-1].tolist()

layers = params.layers

#load drug features and perform dimension reduction with PCA

feat = pickle.load(open("./data/drugbank/DB_molecular_feats.pkl", 'rb'), encoding='utf-8')
feat = np.array(feat["Morgan_Features"])
feat = np.array(feat.tolist())
feat[feat>0]=1
drug_fsim = ut.get_target_sim_matrix(feat)

drug_f = ut.get_pca_feature(drug_fsim,n_components*layers)

drug_f_sim = tf.constant(drug_f,dtype=tf.float32)

sparse_adj_input = sparse_adj_input + sparse_adj_input_t
adj_type = adj_type+adj_type_t

spa_adj = []
for item in sparse_adj_input:
    # spa_temp = csc_matrix((item[1], (item[0][:,0], item[0][:,1])), shape=item[2])
    spa_temp = tf.SparseTensor(indices=item[0], values=item[1], dense_shape=item[2])
    spa_temp = tf.sparse_reorder(spa_temp)
    spa_adj.append(spa_temp)

#load features of other entities which are obtained by TransE
ent_feature = np.loadtxt('./data/drugbank/ent_embeddings_DDI.txt')

if n_components*layers<ent_feature.shape[1]:
  ent_feature = ut.get_pca_feature(ent_feature,n_components*layers)
  
ent_feature = np.pad(ent_feature,((0,0),(0,n_components*layers-ent_feature.shape[1])),'constant',constant_values=(0,0))

ent_feature = np.vstack([drug_f,ent_feature[feat.shape[0]:]]) 
ent_f = np.array(np.expand_dims(ent_feature,0),dtype=np.float32)
hid = params.hid#[64,64]
batch_size = 1

hid_units = params.hid
nb_nodes = adj_list[0].shape[0]
activation = tf.nn.elu
# 0:drug，1:gene，2,disease，3,Symptom，4,Anatomy，5,Molecular Function
# 6,Pharmacologic Class，7,Cellular Component，8,Pathway，9,Side Effect

attn_drop = tf.placeholder(dtype=tf.float32, shape=())
ffd_drop = tf.placeholder(dtype=tf.float32, shape=())

ftr_in = tf.constant(ent_f,dtype=tf.float32)

#type mask,shape=(ent_types,nb_nodes,1)
type_mask =   tf.constant(ent_mask,dtype=tf.float32)         


attns = []


ent_mask = np.squeeze(ent_mask)
seq = tf.squeeze(ftr_in)
type_indexs = []
for i in range(ent_types):
    type_indexs.append(np.where(np.array(ent_mask[i])==1)[0])


ent_index_mapping = []
for i in range(ent_types):
    temp_dict = {}
    for j in range(len(type_indexs[i])):
        temp_dict[type_indexs[i][j]] = j
    ent_index_mapping.append(temp_dict)

initializer = tf.contrib.layers.xavier_initializer()

temp = tf.squeeze(ftr_in)
for _ in range(heads):
    ret = sp_hete_attn_head(seq=temp,spa_adj=spa_adj, adj_type=adj_type,
        out_sz=hid_units[0], activation=activation, nb_nodes=nb_nodes,in_drop=ffd_drop, coef_drop=attn_drop, 
        type_indexs=type_indexs,ent_index_mapping=ent_index_mapping,sparse_adj_input=sparse_adj_input,
        ent_types=ent_types,cdt=params.cdt)
    attns.append(ret)


h_11 = [tf.concat(attn, axis=-1) for attn in zip(*attns)]

h_pre = [h_11[i]*type_mask[i] for i in range(len(h_11))]
h_pre1 = reduce(tf.add, h_pre)
layer_fea = tf.concat([tf.expand_dims(temp,axis=0),h_pre1],axis=2)
layer_fea = full_connection(layer_fea, params.emb_dim, activation=tf.nn.leaky_relu, in_drop=ffd_drop, use_bias=True)
layer_fea = tf.expand_dims(layer_fea,axis=0)

for i in range(1, layers):
    h_old = layer_fea
    attns = []
    head_act = activation
    is_residual = False
    for _ in range(heads):
        ret1 =sp_hete_attn_head(seq=h_old,  spa_adj=spa_adj, adj_type=adj_type,out_sz=hid_units[i], 
                        activation=head_act, nb_nodes=nb_nodes,in_drop=ffd_drop, coef_drop=attn_drop, type_indexs=type_indexs,
                        ent_index_mapping=ent_index_mapping,sparse_adj_input=sparse_adj_input,ent_types=ent_types,
                        cdt=params.cdt)
        attns.append(ret1)
    h_1 = [tf.concat(attn, axis=-1) for attn in zip(*attns)]
   
    h_mid = [h_1[i]*type_mask[i] for i in range(len(h_1))]
    h_pre1 = reduce(tf.add, h_mid)
    layer_fea = tf.concat([h_old,h_pre1],axis=2)
  
    layer_fea = full_connection(layer_fea, params.emb_dim, activation=tf.nn.leaky_relu, in_drop=ffd_drop, use_bias=True)
    layer_fea = tf.expand_dims(layer_fea,axis=0)

h_1 = tf.squeeze(layer_fea) #final structural embedding


data_batch_size = params.batch_size
lr = params.lr
#placeholder
pos_drug1 = tf.placeholder(tf.int32, shape=[None], name='pos_drug1') #drug one in positive samples
pos_drug2 = tf.placeholder(tf.int32, shape=[None], name='pos_drug2') #drug two in positive samples
ddi_type = tf.placeholder(tf.int32, shape=[None], name='ddi_type') #DDI type
neg_drug1 = tf.placeholder(tf.int32, shape=[None], name='neg_drug1') #drug one in negative samples
neg_drug2 = tf.placeholder(tf.int32, shape=[None], name='neg_drug2') #drug two in positive samples
binary_label = tf.placeholder(tf.float32, shape=[None], name='binary_label') #ground truth label


aggre_shared_att = tf.Variable(initializer([1, layers*hid[-1]]), name='aggre_shared_att') #learnable parameter vector in formula 7
stru_embed_trans_matrix = tf.Variable(initializer([layers*hid[-1], layers*hid[-1]]), name='stru_embed_trans_matrix') #W_s in equation 6
# feat_embed_trans_matrix = tf.Variable(initializer([n_components, layers*hid[-1]]), name='feat_embed_trans_matrix')
feat_embed_trans_matrix = tf.Variable(initializer([n_components*layers, layers*hid[-1]]), name='feat_embed_trans_matrix') #W_f in equation 6
ddi_sim_trans_matrix = tf.Variable(initializer([n_components, layers*hid[-1]]), name='ddi_sim_trans_matrix') #W_11 in formula 11

rel_dim = 3*params.emb_dim if params.fsia else  2*params.emb_dim 
relation_vector = tf.Variable(initializer([ddi_types, rel_dim]), name='relation_vector')
batch_relation_matrix = tf.matrix_diag(tf.nn.embedding_lookup(relation_vector, ddi_type)) #M_r in Eq. 9
ddi_shared_matrix = tf.Variable(initializer([rel_dim, rel_dim]), name='ddi_shared_matrix') #R in Eq. 9

#Eq. 9：obtain prediction results
pos_output,pos_aggre_embed1,pos_aggre_embed2 = \
    get_output(pos_drug1,pos_drug2,drug_f_sim,h_1,stru_embed_trans_matrix,feat_embed_trans_matrix,aggre_shared_att,params,batch_relation_matrix,ddi_shared_matrix)
neg_output,neg_aggre_embed1,neg_aggre_embed2 = \
    get_output(neg_drug1,neg_drug2,drug_f_sim,h_1,stru_embed_trans_matrix,feat_embed_trans_matrix,aggre_shared_att,params,batch_relation_matrix,ddi_shared_matrix)


pred = tf.sigmoid(pos_output) 
output = tf.concat([pos_output,neg_output],axis=0)

base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=binary_label, logits=output)) #prediction loss

#constraint loss in Eq. 11
aggre_embed = tf.concat([pos_aggre_embed1,pos_aggre_embed2,neg_aggre_embed1,neg_aggre_embed2],axis=0)
drugs = tf.concat([pos_drug1,pos_drug2,neg_drug1,neg_drug2],axis=0)
ddi_sim_embed = tf.nn.embedding_lookup(ddi_sim_in, drugs)
ddi_sim_embed1_trans = ddi_sim_embed @ ddi_sim_trans_matrix
sim_mapping_loss = tf.square(ddi_sim_embed1_trans - aggre_embed)


sim_mapping_loss = tf.reduce_mean(sim_mapping_loss)

sim_mapping_weight = tf.placeholder(dtype=tf.float32, shape=(),name='sim_mapping_weight')
#final total loss
total_loss = base_loss + sim_mapping_weight * sim_mapping_loss
if params.fsia:
    opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss)
else:
    opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(base_loss)



triplets['train'] = triplets['train'].astype(np.int32)
triplets['valid'] = triplets['valid'].astype(np.int32)
threshold = 0.5 
feed_dict = {  'pos_drug1':pos_drug1,#drug one
          'pos_drug2':pos_drug2,#drug two
          'ddi_type':ddi_type,
          'ffd_drop':ffd_drop,
          'attn_drop':attn_drop,
          'pred':pred,
          'threshold':threshold
          }

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess = tf.Session()
sess.run(init_op)
# saver = tf.train.Saver()
print('negative sampling...')
pos_edges, neg_edges = ut.sample_neg(sparse_ddi_adj, triplets['train'], num_neg_samples_per_link=1, max_size=1000000, constrained_neg_prob=0.5,drug_num=1710)
pos_edges = pos_edges.astype(np.int32)
neg_edges = neg_edges.astype(np.int32)



for epoch in range(params.epoch):   
    permutation = np.random.permutation(pos_edges.shape[0])
    
    pos_edges = pos_edges[permutation] 
    neg_edges = neg_edges[permutation] 
    batch = len(permutation)//data_batch_size + 1
    total_base_loss = 0.
    total_sim_mapping_loss = 0.
    time1 = time.time()
    # saver.save(sess, "Model/model.ckpt")
    # loss_record = []
    for b in range(batch):
        start = b * data_batch_size
        if start == len(permutation):
            break
        end = (b+1) * data_batch_size
        #postive samples
        pos_train_batch = pos_edges[start:end]
        d1_feed_pos = pos_train_batch[:,0]
        d2_feed_pos = pos_train_batch[:,1]
        label_feed = pos_train_batch[:,2]
        #negative samples
        neg_train_batch = neg_edges[start:end]
        d1_feed_neg = neg_train_batch[:,0]
        d2_feed_neg = neg_train_batch[:,1]
        feed_dict_tra = {
            pos_drug1:d1_feed_pos,
            pos_drug2:d2_feed_pos,
            neg_drug1:d1_feed_neg,
            neg_drug2:d2_feed_neg,
            ddi_type:label_feed,
            # ftr_in:ent_f,
            ffd_drop:params.ffd_drop,
            attn_drop:params.attn_drop,
            binary_label: np.concatenate((np.ones((len(d1_feed_pos)),dtype=np.float32),np.zeros((len(d1_feed_neg)),dtype=np.float32)),axis=0),
            sim_mapping_weight:params.align_weight
            # type_mask:ent_mask
        }
        # feed_dict_tra.update({i: d for i, d in zip(spa_adj, sparse_adj_input)})
       
        _,batch_base_loss,batch_sim_mapping_loss = sess.run([opt,base_loss,sim_mapping_loss], feed_dict_tra)
        total_base_loss += batch_base_loss
        total_sim_mapping_loss += batch_sim_mapping_loss
        # loss_record.append(batch_base_loss)
        # print(b)
        # break
    
    time2 = time.time()
    print('epoch:%d/%d(time:%.4f),total_loss:%.4f + %.4f'%(epoch,params.epoch,time2-time1,total_base_loss,total_sim_mapping_loss),file=f)
    print('epoch:%d/%d(time:%.4f),total_loss:%.4f + %.4f'%(epoch,params.epoch,time2-time1,total_base_loss,total_sim_mapping_loss))
    f.flush()
    # -----------------training-------------------
    evaluate_data(triplets['train'],f,'训练',sess,feed_dict)
    # -----------------validation-------------
    valid_data = np.loadtxt(params.file_paths['valid'])
    evaluate_data(valid_data,f,'验证',sess,feed_dict)
   
    #-----------------testing----------------
    if epoch % 1 == 0:#每一个epoch就测试一次
      test_data = np.loadtxt(params.file_paths['test'])
      evaluate_data(test_data,f,'测试',sess,feed_dict)
    

f.close()
sess.close()
