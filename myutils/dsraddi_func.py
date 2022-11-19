import numpy as np
import tensorflow as tf
from functools import reduce
from operator import itemgetter

def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    if len(coo.data) == 0:
        return tf.SparseTensor([[1,2]], [0.], coo.shape)#生成空的稀疏矩阵
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

def _sparse_adj_list_process(all_sparse_adj_list):
    sparse_adj_list = []
    for adj in all_sparse_adj_list:
        convert_adj = _convert_sp_mat_to_sp_tensor(adj)#转tensor
        sparse_adj_list.append(convert_adj)
    return sparse_adj_list

def get_att_aggre_embedding(embedding_a,embedding_b,att_vec):
    #att 分子
    up_a = tf.exp(tf.nn.leaky_relu(embedding_a @ tf.transpose(att_vec)))
    up_b = tf.exp(tf.nn.leaky_relu(embedding_b @ tf.transpose(att_vec)))
    att_a = up_a / (up_a+up_b)
    att_b = up_b / (up_a+up_b)
    result = att_a * embedding_a + att_b * embedding_b
    return result


###
# seq:所有实体的初始嵌入（原有的顺序）
# out_sz：输出维度
# spa_adj：（tensor）异构邻接矩阵集合，包括正向和反向（23*2）
# adj_type：邻接矩阵的类型，包括正向和反向（23*2）
# activation：激活函数
# nb_nodes：所有实体的个数
# type_indexs：每一个类型节点的下标
# ent_index_mapping：下标到对应索引的映射
# ent_types：所有实体类别的数量
# sparse_adj_input：异构邻接矩阵集合，包括正向和反向（23*2）
# ###
def sp_hete_attn_head(seq, out_sz, spa_adj, adj_type, activation, nb_nodes,
            type_indexs,ent_index_mapping,ent_types,sparse_adj_input,cdt,in_drop=0.0, coef_drop=0.0):
    with tf.name_scope('sp_hete_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, keep_prob=1-in_drop)
        seq = tf.squeeze(seq)

        #根据type_index取对应类型实体的嵌入(按照类别先后)
        seqs = []
        for ti in type_indexs:
            temp = tf.gather_nd(seq,np.expand_dims(ti,axis=-1))
            seqs.append(tf.expand_dims(temp,axis=0))

        #1,为每一个矩阵下的特征乘以对应的转换矩阵，10*10---现在是正确的数量节点的特征
        if cdt:#使用跨域转换
            seq_fts =[[tf.layers.conv1d(s,
                out_sz, # out_sz_j
                1,
                use_bias=False) for s in seqs] for _ in range(ent_types) ]
        else:#不适用跨域转换，而是将每一种域进行统一的处理
            # seq_fts =[tf.layers.conv1d(s,
            #     out_sz, # out_sz_j
            #     1,
            #     use_bias=False) for s in seqs]
            # seq_fts = [seq_fts for i in range(len(seq_fts))]
            seq_fts =[seqs for _ in range(ent_types)]
        #2，对seq_fts每一个元素进行拼接，按照原有顺序排列
        # ###这里要注意对齐，不能按照type_indexs直接取数据,采用的方法是构建一个映射
        reorder1 = np.concatenate(type_indexs)#index:按照类型排列时节点的下标，value：节点原来的下标。
        reorder_dict = {}#按照类型排列节点索引与下标的映射 key:节点原来的下标，value：节点按照类型排列时节点的下标
        for i in range(len(reorder1)):
            reorder_dict[reorder1[i]] = i

        reorder2 = []#reorder_dict按照key，转换为list，index：节点原来的下标，value：节点按照类型排列时节点的下标
        for i in range(len(reorder_dict)):
            reorder2.append(reorder_dict[i])
        reorder2 = np.array(reorder2).reshape((-1,1))

        seq_fts_reorder = []#特征重排序(按照原有顺序)
        for i in range(len(seq_fts)):
            temp = tf.squeeze(tf.concat(seq_fts[i],1))
            temp = tf.gather_nd(temp,reorder2)
            seq_fts_reorder.append(temp)

        # 3，注意力稀疏集合
        coefs_lists = [[] for _ in range(ent_types)]# attention coefficient,长度为23

        for index in range(len(spa_adj)):
            i,j = adj_type[index]
            # self attention
            # adj_type[2]#这里用第三个邻接矩阵为例：disease-gene (2,1)
            # i,j = adj_type[2]
            f_1 = tf.layers.conv1d(seq_fts[j][j], 1, 1)#实体1特征（对应数量）
            f_2 = tf.layers.conv1d(seq_fts[j][i], 1, 1)#实体2特征（对应数量）

            f_1 = tf.reshape(f_1, (-1, 1))#shape=(34124,1)
            f_2 = tf.reshape(f_2, (-1, 1))#shape=(34124,1)

            #得到节点1原本的下标
            type_index0 = sparse_adj_input[index][0][:,0]
            #根据索引去字典中取值，如1710取0，1711取1
            type_index_mapped0 = np.expand_dims(np.array(itemgetter(*type_index0)(ent_index_mapping[j])),axis=-1)
            f_1_value = tf.gather_nd(f_1,type_index_mapped0)

            #得到节点2原本的下标
            type_index1 = sparse_adj_input[index][0][:,1]
            #根据索引去字典中取值，如1710取0，1711取1
            type_index_mapped1 = np.expand_dims(np.array(itemgetter(*type_index1)(ent_index_mapping[i])),axis=-1)
            f_2_value = tf.gather_nd(f_2,type_index_mapped1)

            #计算稀疏矩阵中对应位置的值，这是spa_adj[index]中对应位置的值
            f_value= f_1_value + f_2_value
            # 将f_value放入spa_adj[index] 中,,(公式3)
            coefs = tf.SparseTensor(indices=spa_adj[index].indices, 
                    values=tf.nn.leaky_relu(tf.squeeze(f_value)), 
                    dense_shape=spa_adj[index].dense_shape)

            if coef_drop != 0.0:
                coefs = tf.SparseTensor(indices=coefs.indices,
                      values=tf.nn.dropout(coefs.values, keep_prob=1-coef_drop),
                      dense_shape=coefs.dense_shape)
            coefs_lists[j].append(coefs)#保存传到j类型的注意力系数（稀疏矩阵）

        #coefs_lists，10个元素，每个元素存放了相关adj下的注意力权重，形状是稀疏的adj
        #seq_fts_lists，9个元素，每一种稀疏的adj对应的节点的特征
        #每个元素中相加
        coefs = [reduce(tf.sparse_add, coefs_item) for coefs_item in coefs_lists]
        coefs = [tf.sparse_softmax(coef) for coef in coefs]

        #式5
        vals = [tf.sparse_tensor_dense_matmul(coef, seq_ft) for coef, seq_ft 
                                                        in zip(coefs, seq_fts_reorder)]
        #拓展维度
        # vals = [tf.expand_dims(activation(val), axis=0) for val in vals]#这个激活函数可以添加
        vals = [tf.expand_dims(val, axis=0) for val in vals]
        for i, val in enumerate(vals):
            val.set_shape([1, nb_nodes, out_sz])
        return vals

def sp_hete_attn_head1(seq, out_sz, spa_adj, adj_type, activation, nb_nodes,
            type_indexs,ent_index_mapping,ent_types,sparse_adj_input,in_drop=0.0, coef_drop=0.0):
    # input adjacency matrices are TRANSPOSED before feeding!
    with tf.name_scope('sp_hete_attn'):
        #一次attention
        #1,先经过关系转换,这里只考虑86-108的这些邻接矩阵
        # heter_adj_list = sparse_adj_input
        # if in_drop != 0.0:
        #     seq = tf.nn.dropout(seq, keep_prob=1-in_drop)

        #这里的操作太耗时，用下标来提取每一个域中的实体的嵌入
        
        # #取每一种类型实体对应的下标
        # ent_mask = np.squeeze(ent_mask)
        seq = tf.squeeze(seq)

        # #根据type_index取对应类型实体的嵌入
        # seqs = []
        # for ti in type_indexs:
        #     temp = tf.gather_nd(seq,np.expand_dims(ti,axis=-1))
        #     seqs.append(tf.expand_dims(temp,axis=0))

        # #2,为每一个矩阵下的特征乘以对应的转换矩阵，10*10---现在是正确的数量节点的特征
        # seq_fts =[[tf.layers.conv1d(s,
        #       out_sz, # out_sz_j
        #       1,
        #       use_bias=False) for s in seqs] for _ in range(ent_types) ]
        # #3，对seq_fts每一个元素进行拼接，按照对应的索引
        # # reorder = np.concatenate(type_indexs).reshape((-1,1))
        # # ###这里要注意对齐，不能按照type_indexs直接取数据,采用的方法是构建一个映射
        # reorder = np.concatenate(type_indexs)
        # reorder_dict = {}
        # for i in range(len(reorder)):
        #     reorder_dict[reorder[i]] = i

        # reorder = []
        # for i in range(len(reorder_dict)):
        #     reorder.append(reorder_dict[i])
        # reorder = np.array(reorder).reshape((-1,1))

        
        # seq_fts_reorder = []
        # for i in range(len(seq_fts)):
        #     temp = tf.squeeze(tf.concat(seq_fts[i],1)) 
        #     temp = tf.gather_nd(temp,reorder)
        #     seq_fts_reorder.append(temp)

        
        #自己填充卷积后的特征，不做注意力转换
        vals = [seq for i in range(10)]

        #拓展维度
        vals = [tf.expand_dims(activation(val), axis=0) for val in vals]
        for i, val in enumerate(vals):
            val.set_shape([1, nb_nodes, out_sz])
        return vals


def get_output(drug1,drug2,drug_f_sim,h_1,stru_embed_trans_matrix,feat_embed_trans_matrix,aggre_shared_att,params,batch_relation_matrix,ddi_shared_matrix):
    #获取特征的嵌入
    drug1_emb = tf.nn.embedding_lookup(h_1, drug1)
    drug1_feat = tf.nn.embedding_lookup(drug_f_sim, drug1)
    drug2_emb = tf.nn.embedding_lookup(h_1, drug2)
    drug2_feat = tf.nn.embedding_lookup(drug_f_sim, drug2)

    drug1_emb_trans = drug1_emb @ stru_embed_trans_matrix#药物1：获取公式6的 e_d^L′
    drug1_feat_trans = drug1_feat @ feat_embed_trans_matrix#药物1：获取公式6的 e_d^f′
    drug2_emb_trans = drug2_emb @ stru_embed_trans_matrix#药物1：获取公式6的 e_d^L′
    drug2_feat_trans = drug2_feat @ feat_embed_trans_matrix#药物1：获取公式6的 e_d^f′

    #药物1：获取公式6的聚合特征
    aggre_embed1 = get_att_aggre_embedding(drug1_emb_trans,drug1_feat_trans,aggre_shared_att)
    #药物2：获取公式6的聚合特征
    aggre_embed2 = get_att_aggre_embedding(drug2_emb_trans,drug2_feat_trans,aggre_shared_att)

    #公式8：采用串联的方式(FSIA同时不考虑聚合特征以及对齐损失)
    if params.fsia:
        con_emb_drug1 = tf.concat([drug1_emb,drug1_feat,aggre_embed1], axis=1)
        con_emb_drug2 = tf.concat([drug2_emb,drug2_feat,aggre_embed2], axis=1)
    else:
        con_emb_drug1 = tf.concat([drug1_emb,drug1_feat], axis=1)
        con_emb_drug2 = tf.concat([drug2_emb,drug2_feat], axis=1)

    #公式9，基于因式分解的预测器
    output = tf.expand_dims(con_emb_drug1,axis=1) @ batch_relation_matrix 
    output = tf.squeeze(output) @ ddi_shared_matrix
    output = tf.expand_dims(output,axis=1) @ batch_relation_matrix
    output = tf.nn.l2_normalize(output,dim=0)
    output = output @ tf.expand_dims(con_emb_drug2,axis=-1)
    output = tf.squeeze(output) #1024

    return output,aggre_embed1,aggre_embed2






  