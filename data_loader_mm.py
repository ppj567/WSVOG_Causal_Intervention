import torch
import torch.utils.data.dataset as Dataset
import numpy as np
import json
import os
import csv
import h5py

from config import TfConfig

def pause():
    programPause = input("Press the <ENTER> key to continue...")

def collate_data(data):

    frm_feat, obj_id, obj_id_np = zip(*data)
    frm_feat = torch.stack(frm_feat,0)
    obj_id = torch.stack(obj_id, 0)
    bt = obj_id.size(0)
    
    mask = torch.zeros(bt, bt)
    lengths = obj_id.ne(0).long().sum(-1)
    
    for i in range(bt):
        q_i = set(obj_id_np[i][0:lengths[i]])
        for j in range(i+1,bt):
            q_j = set(obj_id_np[j][0:lengths[j]])
            if len(q_i.intersection(q_j))==0:
                mask[i,j] = 1
                mask[j,i] = 1
    result = (frm_feat, obj_id, mask.long())
    return result


class YC2_train_data(Dataset.Dataset):
    def __init__(self,
                 label_file,
                 roi_pooled_feat_root,
                 data_split,
                 rpn_proposal_root,
                 num_proposals,
                 num_frm,
                 max_num_obj,
                 num_classes,
                 rpn_feature_dims,
                 num_threads, dataset_name='bmvc'
                ):

        self.label_file = label_file
        self.feature_path = roi_pooled_feat_root
        self.num_frm = num_frm
        self.num_proposals=num_proposals
        self.max_num_obj=max_num_obj
        self.num_classes = num_classes
        self.rpn_feature_dims=rpn_feature_dims
        self.data_split=data_split
        self.num_threads=num_threads
        self.dataset_name=dataset_name
        self.proposal_root = rpn_proposal_root
        cfg = TfConfig()

        self.yc2_sentence_dict_path = cfg.yc2_sentence_dict_file
        self.yc2_word_emb_dict_path = cfg.yc2_word_emb_dict_file      
        self.yc2_training_vid_file = cfg.yc2_yc2_training_vid
        self.yc2_class_label_dict_path = cfg.yc2_class_label_dict

        file_name_list=[]
       
        
        with open(self.yc2_training_vid_file, 'r')as f:
            self.yc2_training_vid = json.load(f)

        with open(self.yc2_class_label_dict_path, 'r')as f:
            self.yc2_class_label_dict = json.load(f)    

        with open(self.label_file, 'r')as f:
            self.label_dict = json.load(f)
        

        for file_name in self.label_dict:
            file_name_list.append(file_name)

        self.file_name_list = file_name_list
        self.file_num = len(self.file_name_list)

        
        if self.dataset_name=='bmvc':
            'load rpn box index dict and corresponding rpn box'
            self.rpn_dict = {}
            self.rpn_chunk = []
            total_num_proposals = 100
            rpn_lst_file = os.path.join(self.proposal_root, self.data_split + '-box-' + str(total_num_proposals) + '.txt')
            rpn_chunk_file = os.path.join(self.proposal_root, self.data_split + '-box-' + str(total_num_proposals) + '.npy')

            key_counter = len(self.rpn_dict)
            with open(rpn_lst_file) as f:
                rpn_lst = f.readline().split(',')
                
                self.rpn_dict.update({r.strip(): (i + key_counter) for i, r in enumerate(rpn_lst)})
                
                self.rpn_chunk.append(np.load(rpn_chunk_file))
                
            self.rpn_chunk = np.concatenate(self.rpn_chunk)
            assert (self.rpn_chunk.shape[0] == len(self.rpn_dict))
            assert (self.rpn_chunk.shape[2] == 4)         
        
    def calc_class_balance(self):
        class_obj = dict.fromkeys([i for i in range(self.num_classes)], 0)
        class_obj_list = []
        for key in self.label_dict.keys():
            for obj in self.label_dict[key][1]:
                class_obj[obj] = class_obj[obj] + 1
        with open("class_freq.json",'w',encoding='utf-8') as json_f:
            json.dump(class_obj,json_f,ensure_ascii=False)
        sorted_class_obj = list(class_obj.values())
        sorted_class_obj.sort()
        return class_obj



    def __len__(self):
        return len(self.file_name_list)
        
    def get_sample(self, file_name):
    
        sample=self.label_dict[file_name]
        split = sample[0][0]
        rec = sample[0][1]
        vid = sample[0][2]
        seg = sample[0][3]

        seg_name = split+'_-_'+ rec+'_-_'+vid+'_-_'+seg

        entity_ind = np.zeros(self.max_num_obj)
        entity_ind[0:len(self.label_dict[file_name][1])] = np.array(self.label_dict[file_name][1])+1
           
        rpn_feature=np.load(self.feature_path+'/'+self.data_split+'/'+file_name+'.npy')

        
        T=rpn_feature.shape[0]
        step=1

        itv = T * 1. / self.num_frm
        ind = [min(T - 1, int((i + np.random.rand()) * itv)) for i in range(self.num_frm)]
        
        frm_id = ind  
        rpn_feature = rpn_feature[ind,:self.num_proposals, :]

        sample_label  = torch.LongTensor(entity_ind)
        sample_feature= torch.FloatTensor(rpn_feature)
        
        
        return sample_feature, sample_label
        
    def __getitem__(self, index):
      
        file_name = self.file_name_list[index]
        pos_feature, pos_label = self.get_sample(file_name)

        neg_index = np.random.randint(self.file_num)
        
        while len(set(self.label_dict[file_name][1]).intersection(
                set(self.label_dict[self.file_name_list[neg_index]][1]))) != 0:  
            neg_index = np.random.randint(self.file_num)

        neg_feature, neg_label= self.get_sample(self.file_name_list[neg_index])
        result = [pos_feature, pos_label, neg_feature, neg_label]        
        return result

class YC2_test_data(Dataset.Dataset):
    def __init__(self,label_file, gt_box_file, class_file, data_split, rpn_proposal_root, roi_pooled_feat_root, num_proposals=20, max_num_obj=16, num_classes=67, rpn_feature_dims=4096, dataset_name='bmvc'):

        self.label_file=label_file
        self.feature_path=roi_pooled_feat_root
        self.num_proposals=num_proposals
        self.max_num_obj=max_num_obj
        self.num_classes=num_classes
        self.rpn_feature_dims=rpn_feature_dims
        self.data_split=data_split
        self.gt_box_file = gt_box_file
        self.proposal_root = rpn_proposal_root
        self.max_frm = 200
        self.dataset_name=dataset_name
        cfg = TfConfig()
        'load class label dict,class_name: class_ind'
        self.class_dict = self.get_class_labels(class_file)
        self.yc2_class_label_dict_path = cfg.yc2_class_label_dict

        'load data and  file list'
        self.file_name_list=[]
        with open(self.label_file, 'r')as f:
            self.label_dict = json.load(f)
        for file_name in self.label_dict:
            self.file_name_list.append(file_name)
        self.file_num = len(self.file_name_list)

        with open(self.yc2_class_label_dict_path, 'r')as f:
           self.yc2_class_label_dict = json.load(f)    

        if self.dataset_name=='bmvc':
            'load rpn box index dict and corresponding rpn box'
            self.rpn_dict = {}
            self.rpn_chunk = []
            total_num_proposals = 100
            rpn_lst_file = os.path.join(self.proposal_root, self.data_split + '-box-' + str(total_num_proposals) + '.txt')
            rpn_chunk_file = os.path.join(self.proposal_root, self.data_split + '-box-' + str(total_num_proposals) + '.npy')

            key_counter = len(self.rpn_dict)
            with open(rpn_lst_file) as f:
                rpn_lst = f.readline().split(',')
                self.rpn_dict.update({r.strip(): (i + key_counter) for i, r in enumerate(rpn_lst)})
                self.rpn_chunk.append(np.load(rpn_chunk_file))
            self.rpn_chunk = np.concatenate(self.rpn_chunk)
            assert (self.rpn_chunk.shape[0] == len(self.rpn_dict))
            assert (self.rpn_chunk.shape[2] == 4)

        with open(self.gt_box_file, 'r') as f:
            self.data_all = json.load(f)

    def get_class_labels(self,class_file):
        class_dict = {}
        with open(class_file) as f:
            cls = csv.reader(f, delimiter=',')
            for i, row in enumerate(cls):
                for r in range(1, len(row)):
                    if row[r]:
                        class_dict[row[r]] = int(row[0])

        return class_dict
        
    def __len__(self):
        return len(self.file_name_list)

    def get_sample(self, file_name):
        sample=self.label_dict[file_name]
        split = sample[0][0]
        rec = sample[0][1]
        vid = sample[0][2]
        seg = sample[0][3]
        object_label=self.label_dict[file_name][1]
        
        seg_name = split+'_-_'+ rec+'_-_'+vid+'_-_'+seg
        'load seg gt box ,None box filled -1'
        gt_box = self.data_all['database'][vid]['annotations'][str(int(seg))]
        width = self.data_all['database'][vid]['width']
        height = self.data_all['database'][vid]['height']
        
        assert (len(gt_box) > 0)
        num_frames_1fps = len(gt_box['0']['boxes'])
        box = np.ones((len(gt_box), num_frames_1fps, 5)) * -1.0
        for i in gt_box:
            cls_label = self.class_dict[gt_box[i]['label']]
            object_label[int(i)] = cls_label
            for j in gt_box[i]['boxes']:
                if gt_box[i]['boxes'][j]['occluded'] == 0 and gt_box[i]['boxes'][j]['outside'] == 0:
                    box[int(i)][int(j)][0] = cls_label
                    box[int(i)][int(j)][1] = gt_box[i]['boxes'][j]['xtl']
                    box[int(i)][int(j)][2] = gt_box[i]['boxes'][j]['ytl']
                    box[int(i)][int(j)][3] = gt_box[i]['boxes'][j]['xbr']
                    box[int(i)][int(j)][4] = gt_box[i]['boxes'][j]['ybr']

        box = box.astype(np.float32)

        entity_ind = np.zeros(self.max_num_obj)
        entity_ind[0:len(self.label_dict[file_name][1])] = np.array(self.label_dict[file_name][1])+ 1

        'load seg feature'
        rpn_feature = np.load(self.feature_path+'/'+self.data_split+'/'+file_name+'.npy')

        rpn_feature = rpn_feature[:,:self.num_proposals,:]
                
        frm_num_ = rpn_feature.shape[0]
               
        seg_label   = torch.LongTensor(entity_ind)
        seg_feature = torch.FloatTensor(rpn_feature)
        

        if self.dataset_name=='cvpr':
            rpn_box = np.load(self.proposal_root + '/' + self.data_split + '/' + file_name + '.npy')
            rpn_box = rpn_box[:, :self.num_proposals, :]
            
            w_r = width / 224.
            h_r = height / 224.
            
            rpn_box[:, :, :1] = rpn_box[:, :, :1] * w_r
            rpn_box[:, :, 1:2] = rpn_box[:, :, 1:2] * h_r
            rpn_box[:, :, 2:3] = rpn_box[:, :, 2:3] * w_r
            rpn_box[:, :, 3:] = rpn_box[:, :, 3:] * h_r
        else:
            'load seg rpn box,according the rpn box index'
            rpn = []
            spatio_box=[]
            frm = 1
            img_name = vid + '_' + seg + '_' + str(frm).zfill(4) + '.jpg'

            while self.rpn_dict.get(img_name, -1) > -1:
                ind = self.rpn_dict[img_name]
                rpn.append(self.rpn_chunk[ind])
                frm += 1
                img_name = vid + '_' + seg + '_' + str(frm).zfill(4) + '.jpg'

            rpn_box = (np.array(rpn)[:, :self.num_proposals, :] - 1.0).astype(np.float32)
            
        ind_ = torch.linspace(0, frm_num_-1, frm_num_)
           
        return seg_feature, seg_label, rpn_box, box
                    
        
    def __getitem__(self, index):
        file_name = self.file_name_list[index]
        seg_feature, seg_label, rpn_box, box= self.get_sample(file_name)
        result = [seg_feature, seg_label, rpn_box, box]
        return result
