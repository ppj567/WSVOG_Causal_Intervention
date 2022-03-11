class TfConfig(object):
    #######################################################################
   
    yc2_class_label_dict='./mm_data/vg/video_grounding/data/youcook2/annotations/class_label_dict.json'
    yc2_yc2_training_vid='./mm_data/vg/video_grounding/data/youcook2/annotations/yc2_training_vid.json'
    yc2_train_file='./mm_data/vg/video_grounding/data/youcook2/annotations/train_dict.json'
    yc2_val_file='./mm_data/vg/video_grounding/data/youcook2/annotations/val_dict.json'
    yc2_test_file='./mm_data/vg/video_grounding/data/youcook2/annotations/test_dict.json'
    yc2_class_file='./mm_data/vg/video_grounding/data/youcook2/annotations/class_file.csv'
    yc2_val_gt_box='./mm_data/vg/video_grounding/data/youcook2/annotations/yc2_bb_val_annotations.json'
    yc2_test_gt_box = './mm_data/vg/video_grounding/data/youcook2/annotations/yc2_bb_test_annotations.json'
        
        
    yc2_sentence_dict_file = './mm_data/vg/video_grounding/data/youcook2/annotations/seg_sentence_dict.json'#sentence of segment
    yc2_cls_dict_file = './mm_data/vg/video_grounding/data/youcook2/annotations/cls_dict.json'
    yc2_word_emb_dict_file = './mm_data/vg/video_grounding/data/youcook2/annotations/word_embedding_glove.json'#embedding of word

    cvpr_rpn_feat_root= './datasets/yc2_cvpr'
    cvpr_proposal_root= './datasets/yc2_cvpr/rpn_box'

    cvpr_rpn_feat_dims=4096


    
    '''dataset info'''
    train_split='training'
    val_split='validation'
    test_split='testing'
    num_classes= 67#
    num_proposals=20
    num_frm = 5
    max_num_obj=16
    rpn_feature_dims=4096#2048
    input_feature_dims = 4096

    max_num_word = 46
    max_frm = 264
    word_embedding_dims = 300
    bn_feature_dims = 1024
    resnet_feature_dims = 2048

    # '''model select'''
    # model_name="WSTOG"

    '''model setting'''
    margin = 10
    mem_slots_num = 10
    input_label_dims = 67
    embedding_size = 512
    drop_crob= 0.9
    dataset = "YC2"
    '''training setting'''
    seed=123
    num_threads = 6
    max_epochs= 80
    train_batch_size= 1 # only training can select batchsize,if testing all load one segment
    optim='adam'
    learning_rate= 0.0001
    accu_thresh=0.5 # testing thresh of the IoU in rpn_box,gt_box

    '''save setting'''
    save_checkpoint_every= 1 # num of epoch to save model
    checkpoint_path='./checkpoint' # model save path
    start_from='' # if it is not None,will restore the model,and training from this model.
    result_save_path='./result' # testing submission result to save

    '''save settings'''
    model_name = {'full':['ACL_Spatial','ACL_Temporal','Deconfounded'],
                  'backbone':['STVG'],
                  'acl_spatial':['ACL_Spatial'],
                  'acl_temporal':['ACL_Temporal'],
                  'deconfounded':['Deconfounded'],}
    coef = {'full':[0.5,0.5,1.0],
            'backbone':[1.0],
            'acl_spatial':[1.0],
            'acl_temporal':[1.0],
            'deconfounded':[1.0],}
    fusion_coef = 0.9
    glove_dir = './mm_data/vg/video_grounding/data/youcook2/annotations/YC2_glove_dict.pth'
    glove_dir_emb = './mm_data/vg/video_grounding/data/youcook2/annotations/YC2_glove_dict_512dim.pkl'
    class_times = "./mm_data/vg/video_grounding/data/youcook2/annotations/yc2_class_times.json"
    mem_buffer_path = './datasets/mem_bank_buffer/'

