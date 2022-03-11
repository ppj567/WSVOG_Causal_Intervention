import numpy as np
import json
import csv
import os
import errno
import torch
from tqdm import tqdm


def mkdir(path):

	folder = os.path.exists(path)

	if not folder:
		os.makedirs(path)
		print("---  new folder...  ---")
		print("---  OK  ---")

	else:
		print("---  There is this folder!  ---")

def pause():
    programPause = input("Press the <ENTER> key to continue...")

def test(model, test_loader, args, paras):
    box_result_list = []
    query_result_list = []
    model = set_eval(model)
    ged_result={}

    for batch_cnt, batch in enumerate(tqdm(test_loader)):
        pos_feature, pos_label, rpn, box= batch
        residual = 0.0
        obj_num = torch.sum(torch.eq(torch.eq(pos_label, 0), 0))
        pos_feature = pos_feature.cuda()
        pos_label = pos_label.cuda()
        coef = args.coef[paras.model_type]
        obj_num = obj_num.cuda()
        original_frm_num = pos_feature.size(1)
        T=10

        if original_frm_num % T == 0:
           pos_feature = pos_feature.view(int(original_frm_num / T), T, pos_feature.size(-2), pos_feature.size(-1))
           pos_label = pos_label.expand(int(original_frm_num / T), pos_label.size(-1))

        else:
           x= T-original_frm_num % T
           if original_frm_num < x:
              repeated_input_feature = pos_feature.expand(T, pos_feature.size(1), pos_feature.size(-2), pos_feature.size(-1)).contiguous().view(1, T*pos_feature.size(1), pos_feature.size(-2), pos_feature.size(-1))
              pos_feature=torch.cat([pos_feature, repeated_input_feature[:, T*original_frm_num-x:]], dim=1)
           else:
              pos_feature=torch.cat([pos_feature, pos_feature[:, original_frm_num-x:]], dim=1)

           pos_feature = pos_feature.view(int(original_frm_num / T)+1, T, pos_feature.size(-2), pos_feature.size(-1))
           pos_label = pos_label.expand(int(original_frm_num / T)+1, pos_label.size(-1))

        for i,component in enumerate(model):
            residual += component._test_score(pos_label, pos_feature, obj_num, original_frm_num, False)*coef[i]
        rpn_score = residual
        'accuracy compute'
        test_score_ = rpn_score.data.cpu().numpy()

        rpn_ = rpn.data.cpu().numpy()
        box_ = box.data.cpu().numpy()

        box_result, det_result = box_acc_compute(test_score_[0], rpn_[0], box_[0], args.accu_thresh)

        box_result_list.append(box_result)

        query_result = query_acc_compute(test_score_[0], rpn_[0], box_[0], args.accu_thresh)
        query_result_list.append(query_result)

    epoch_box_result_list, macro_box_acc, micro_box_acc = acc_count(box_result_list)
    epoch_query_result_list, macro_qry_acc, micro_qry_acc = acc_count(query_result_list)

    print('macro_box_acc: %lf' %(macro_box_acc))
    print('micro_box_acc: %lf' %(micro_box_acc))
    print('macro_qry_acc: %lf' %(macro_qry_acc))
    print('micro_qry_acc: %lf' %(micro_qry_acc))

    result_list = epoch_box_result_list + epoch_query_result_list[-2:]
    return result_list, macro_box_acc, micro_box_acc, macro_qry_acc, micro_qry_acc


def set_eval(item):
    for i in range(len(item)):
        item[i] = item[i].cuda().eval()
    return item


def iou(rec1, rec2):
    '''

    :param rec1: box1[x1,y1,x2,y2]
    :param rec2: box2[x1,y1,x2,y2]
    :return: IoU of box1,box2
    '''
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[0], rec2[0])
    right_line = min(rec1[2], rec2[2])
    top_line = max(rec1[1], rec2[1])
    bottom_line = min(rec1[3], rec2[3])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return intersect / (sum_area - intersect)
def query_acc_compute(rpn_score,rpn,gt_box,accu_thresh=0.5):
    '''

    :param rpn_score: [object_num,num_frm,num_proposal]
    :param rpn: [num_frm,num_proposal,4]
    :param gt_box: [object_num,num_frm,5]
    :param accu_thresh:testing thresh of the IoU in rpn_box,gt_box,default=0.5
    :return: segment result ,a list like: [(cls_ind,hit_num,miss_num),...]
    '''
    'select detect box'
    box_index=np.argmax(rpn_score,axis=2).tolist()
    object_num=gt_box.shape[0]
    num_frm=gt_box.shape[1]
    det_box = []
    for object_box_index in box_index:
        object_box=[]
        for i in range(len(object_box_index)):
            box=rpn[i][object_box_index[i]]
            object_box.append(box)
        det_box.append(object_box)
    det_box=np.array(det_box)

    seg_result_list = []
    for i in range(object_num):

        cls = -1
        flag_list = [-1] * num_frm
        for t in range(num_frm):
            if gt_box[i][t][0] == -1:
                continue
            if iou(det_box[i, t, :], gt_box[i, t, 1:]) > accu_thresh:
                flag_list[t]=1
            else:
                flag_list[t] = 0
            cls = gt_box[i][t][0]
        seg_result_list.append((cls,flag_list))
    cls_dict = {}
    for flag_tuple in seg_result_list:
        cls_dict[flag_tuple[0]]=[]
    for flag_tuple in seg_result_list:
        cls_dict[flag_tuple[0]].append(flag_tuple[1])
    for key in cls_dict:
        temp = np.array(cls_dict[key]).transpose()
        cls_dict[key] = temp.tolist()
    seg_tuple_list_new = []
    for key in cls_dict:
        hit = 0
        miss = 0
        for t_flag in cls_dict[key]:
            if 1 in t_flag:
                hit += 1
            elif 0 in t_flag:
                miss += 1
        seg_tuple_list_new.append((key, hit, miss))
    return seg_tuple_list_new
def box_acc_compute(rpn_score, rpn, gt_box, accu_thresh=0.5):
    '''

    :param rpn_score: [object_num,num_frm,num_proposal]
    :param rpn: [num_frm,num_proposal,4]
    :param gt_box: [object_num,num_frm,5]
    :param accu_thresh: testing thresh of the IoU in rpn_box,gt_box,default=0.5
    :return: segment result ,a list like: [(cls_ind,hit_num,miss_num),...]
    '''
    'select detect box'
    box_index=np.argmax(rpn_score, axis=2).tolist()

    object_num=gt_box.shape[0]
    num_frm=gt_box.shape[1]
    det_box = []

    for object_box_index in box_index:
        object_box=[]
        for i in range(len(object_box_index)):
            box=rpn[i][object_box_index[i]]
            object_box.append(box)
        det_box.append(object_box)
    det_box=np.array(det_box)

    seg_result_list = []
    for i in range(object_num):
        hit = 0
        miss = 0
        cls = -1
        for t in range(num_frm):
            if gt_box[i][t][0] == -1:
                continue
            if iou(det_box[i, t, :], gt_box[i, t, 1:]) > accu_thresh:
                hit += 1
            else:
                miss += 1
            cls = gt_box[i][t][0]
        seg_result_list.append((cls, hit, miss))

    return seg_result_list, det_box

def acc_count(result_list):

    'count all result and compute accuracy'
    result_dict={}

    'count all classes hit and miss'
    for seg_results in result_list:
        for result_tuple in seg_results:
            if result_tuple[0] in result_dict:
                result_dict[result_tuple[0]]['positive']+=result_tuple[1]
                result_dict[result_tuple[0]]['negtive'] += result_tuple[2]
            elif result_tuple[0]!=-1:
                result_dict[result_tuple[0]]={'positive':0, 'negtive':0}
                result_dict[result_tuple[0]]['positive'] += result_tuple[1]
                result_dict[result_tuple[0]]['negtive'] += result_tuple[2]

    acc_sum=0
    actually_class=0
    epoch_result_list=[]

    'macro_box_acc'
    for key in result_dict:
        if(result_dict[key]['positive'] + result_dict[key]['negtive'])==0:
            continue
        actually_class+=1
        acc=result_dict[key]['positive'] / (result_dict[key]['positive'] + result_dict[key]['negtive'])
        epoch_result_list.append((key,acc))
        acc_sum+=acc

    if actually_class!=0:

        epoch_result_list.append(('macro_box_acc:', acc_sum/actually_class))
    else:
        print('macro_box_acc:', 0)
        epoch_result_list.append(('macro_box_acc:', 0))


    macro_acc = acc_sum/actually_class
    'micro_box_acc'
    p=0
    n=0
    for key in result_dict:
        p+=result_dict[key]['positive']
        n+=result_dict[key]['negtive']
    if p+n!=0:

        epoch_result_list.append(('micro_box_acc:',p/(p+n)))
    else:
        print('micro_box_acc:', 0)
        epoch_result_list.append(('micro_box_acc:', 0))

    micro_acc = p/(p+n)

    return epoch_result_list, macro_acc, micro_acc