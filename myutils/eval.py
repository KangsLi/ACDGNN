from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve,cohen_kappa_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd


def evaluate(pred_type, pred_score, y_test, event_num=80):
    all_eval_type = 11
    result_all = {}
    each_eval_type = 6
    result_eve = np.zeros((event_num, each_eval_type), dtype=float)
    y_one_hot = label_binarize(y_test, classes = range(1,event_num+1))
    pred_one_hot = label_binarize(pred_type, classes = range(1,event_num+1))
    result_all['accuracy'] = accuracy_score(y_test, pred_type)
    result_all['aupr(micro)'] = roc_aupr_score(y_one_hot, pred_score, average='micro')
    result_all['aupr(macro)'] = roc_aupr_score(y_one_hot, pred_score, average='macro')
    result_all['auc(micro)'] = roc_auc_score(y_one_hot, pred_score, average='micro')
    # result_all['auc(macro)'] = roc_auc_score(y_one_hot, pred_score, average='macro')
    result_all['auc(macro)'] = 0
    result_all['f1(micro)'] = f1_score(y_test, pred_type, average='micro')
    result_all['f1(macro)'] = f1_score(y_test, pred_type, average='macro')
    result_all['precision(micro)'] = precision_score(y_test, pred_type, average='micro')
    result_all['precision(macro)'] = precision_score(y_test, pred_type, average='macro')
    result_all['recall(micro)'] = recall_score(y_test, pred_type, average='micro')
    result_all['recall(macro)'] = recall_score(y_test, pred_type, average='macro')

    # for i in range(event_num):
    #     result_eve[i, 0] = accuracy_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel())#one hot的accuracy没有借鉴意义
    #     result_eve[i, 1] = roc_aupr_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),average=None)
    #     try:
    #       result_eve[i, 2] = roc_auc_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),average=None)
    #     except:
    #       result_eve[i, 2] = 0.
    #     result_eve[i, 3] = f1_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),average='binary')
    #     result_eve[i, 4] = precision_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),average='binary')
    #     result_eve[i, 5] = recall_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),average='binary')
    return [result_all, result_eve]

def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        order = np.lexsort((recall,precision))
        return auc(precision[order], recall[order])

    def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)

    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)

def evaluate1(labels,scores,pred_type):
    result = {}
    result['acc'] = accuracy_score(labels, pred_type)
    # result['auc'] = f1_score(labels, pred_type, average='macro')
    result['auroc'] = roc_auc_score(labels, scores)
    precision, recall, _ = precision_recall_curve(labels-1, scores)
    result['auprc'] = auc(recall, precision)
    # result['auc_pr'] = f1_score(labels, pred_type, average='micro')
    result['f1'] = np.mean(f1_score(labels, pred_type, average=None))
    result['precision'] = precision_score(labels, pred_type)
    result['recall'] = recall_score(labels, pred_type)    
    result['kappa'] = cohen_kappa_score(labels, pred_type)
    print("report:",classification_report(labels-1,pred_type-1))
    return result

def evaluate_data(triplets,f,test_token,sess,feed_dict):
    pred_score = []
    batch_test_size = 10000
    batch_test_num = len(triplets)/batch_test_size+1
    for batch_i in range(int(batch_test_num)):
        start = batch_test_size*batch_i
        end = batch_test_size*(batch_i+1)
        d1_feed = triplets[start:end,0]
        d2_feed = triplets[start:end,1]
        label_true = triplets[start:end,2]
        feed_dict_test = {
                feed_dict['pos_drug1']:d1_feed,#药物1
                feed_dict['pos_drug2']:d2_feed,#药物2
                feed_dict['ddi_type']:label_true,
                feed_dict['ffd_drop']:0.,
                feed_dict['attn_drop']:0.
                }
        pred_score += list(sess.run(feed_dict['pred'], feed_dict_test))
    y_test_cat = triplets[:,3]#类型
    pred_score = np.array(pred_score)
    pred_type = pred_score.copy()
    print('max:',np.max(pred_type),'min:',np.min(pred_type),'mean',np.mean(pred_type))
    # np.save("pred_type_%s"%(test_token),pred_type)
    pred_type[np.where(pred_type>=feed_dict['threshold'])] =1
    pred_type[np.where(pred_type<feed_dict['threshold'])] =0
    # pred = np.array(pred_score).flatten()

    y_test = np.array(y_test_cat)+1
    pred_type = np.array(pred_type)+1
    result_all = evaluate1(y_test,pred_score,pred_type)

    print("==================模型%s结果=================="%(test_token))
    print("==================模型%s结果=================="%(test_token),file=f)
    print(result_all)
    print(result_all,file=f)
    # if test_token == '测试':
    #   for threshold in [0.1,0.3,0.5,0.7,0.9]:
    #     pred_type = pred_score.copy()
    #     print('max:',np.max(pred_type),'min:',np.min(pred_type),'mean',np.mean(pred_type),threshold)
    #     # np.save("pred_type",pred_type)
    #     pred_type[np.where(pred_type>=threshold)] =1
    #     pred_type[np.where(pred_type<threshold)] =0
    #     # pred = np.array(pred_score).flatten()

    #     y_test = np.array(y_test_cat)+1
    #     pred_type = np.array(pred_type)+1
    #     result_all = evaluate1(y_test,pred_score,pred_type)

    #     print("==================threshold=%.1f**模型%s结果=================="%(threshold,test_token))
    #     print(result_all)
    #     print("==================模型%s结果=================="%(test_token),file=f)
    #     print(result_all,file=f)
    
    f.flush()


def case_study(triplets,f,test_token,sess,feed_dict):
    pred_score = []
    batch_test_size = 10000
    batch_test_num = len(triplets)/batch_test_size+1
    for batch_i in range(int(batch_test_num)):
        start = batch_test_size*batch_i
        end = batch_test_size*(batch_i+1)
        d1_feed = triplets[start:end,0]
        d2_feed = triplets[start:end,1]
        label_true = triplets[start:end,2]
        feed_dict_test = {
                feed_dict['pos_drug1']:d1_feed,#药物1
                feed_dict['pos_drug2']:d2_feed,#药物2
                feed_dict['ddi_type']:label_true,
                feed_dict['ffd_drop']:0.,
                feed_dict['attn_drop']:0.
                }
        pred_score += list(sess.run(feed_dict['pred'], feed_dict_test))
    # y_test_cat = triplets[:,3]#类型
    pred_score = np.array(pred_score)
    # print(pred_score)
    # print(triplets)
    results = {'node_one':triplets[:,0],
                'node_two':triplets[:,1],
                'label':triplets[:,2],
                'prob':pred_score}
    pd.DataFrame(results).to_csv('case.csv')
    # pred_type = pred_score.copy()
    # print('max:',np.max(pred_type),'min:',np.min(pred_type),'mean',np.mean(pred_type))
    # np.save("pred_type_%s"%(test_token),pred_type)
    # pred_type[np.where(pred_type>=feed_dict['threshold'])] =1
    # pred_type[np.where(pred_type<feed_dict['threshold'])] =0
    # pred = np.array(pred_score).flatten()
    
    # y_test = np.array(y_test_cat)+1
    # pred_type = np.array(pred_type)+1
    # result_all = evaluate1(y_test,pred_score,pred_type)

    print("==================模型%s结果=================="%(test_token))
    print("==================模型%s结果=================="%(test_token),file=f)
    # print(result_all)
    # print(result_all,file=f)
    # if test_token == '测试':
    #   for threshold in [0.1,0.3,0.5,0.7,0.9]:
    #     pred_type = pred_score.copy()
    #     print('max:',np.max(pred_type),'min:',np.min(pred_type),'mean',np.mean(pred_type),threshold)
    #     # np.save("pred_type",pred_type)
    #     pred_type[np.where(pred_type>=threshold)] =1
    #     pred_type[np.where(pred_type<threshold)] =0
    #     # pred = np.array(pred_score).flatten()

    #     y_test = np.array(y_test_cat)+1
    #     pred_type = np.array(pred_type)+1
    #     result_all = evaluate1(y_test,pred_score,pred_type)

    #     print("==================threshold=%.1f**模型%s结果=================="%(threshold,test_token))
    #     print(result_all)
    #     print("==================模型%s结果=================="%(test_token),file=f)
    #     print(result_all,file=f)
    
    f.flush()
