import torch
import numpy as np
import random
import argparse
import os
import os.path as osp
import sys
import time
import json
from mmcv import Config

from dataset import build_data_loader
from models import build_model
from models.utils import fuse_module
from utils import ResultFormat, AverageMeter, Corrector

# import file_util
import Polygon as plg
import numpy as np
import mmcv

import os

from os import listdir
from scipy import io
import numpy as np

from script import main_test_ic15

torch.manual_seed(123456)
torch.cuda.manual_seed(123456)
np.random.seed(123456)
random.seed(123456)



def read_dir(root):
	file_path_list = []
	for file_path, dirs, files in os.walk(root):
		for file in files:
			file_path_list.append(os.path.join(file_path, file).replace('\\', '/'))
	file_path_list.sort()
	return file_path_list

def read_file(file_path):
	file_object = open(file_path, 'r')
	file_content = file_object.read()
	file_object.close()
	return file_content

def get_pred(path):
    # lines = file_util.read_file(path).split('\n')
    lines = read_file(path).split('\n')
    bboxes = []
    for line in lines:
        if line == '':
            continue
        bbox = line.split(',')
        if len(bbox) % 2 == 1:
            print(path)
        bbox = [int(x) for x in bbox]
        bboxes.append(bbox)
    return bboxes


def get_gt(path):
    # lines = file_util.read_file(path).split('\n')
    lines = read_file(path).split('\n')
    bboxes = []
    for line in lines:
        if line == '':
            continue
        # line = util.str.remove_all(line, '\xef\xbb\xbf')
        # gt = util.str.split(line, ',')
        gt = line.split(',')

        x1 = np.int(gt[0])
        y1 = np.int(gt[1])

        bbox = [np.int(gt[i]) for i in range(4, 32)]
        bbox = np.asarray(bbox) + ([x1, y1] * 14)

        bboxes.append(bbox)
    return bboxes


def get_union(pD, pG):
    areaA = pD.area()
    areaB = pG.area()
    return areaA + areaB - get_intersection(pD, pG);


def get_intersection(pD, pG):
    pInt = pD & pG
    if len(pInt) == 0:
        return 0
    return pInt.area()

# Total_text 数据集测试函数
def main_test_tt():

    # project_root = '../../'
    project_root = './'

    input_dir = project_root + 'outputs/submit_tt/'
    gt_dir = project_root + 'data/total_text/Groundtruth/Polygon/Test/'
    fid_path = project_root + 'outputs/res_tt.txt'

    allInputs = listdir(input_dir)

    def get_union(pD, pG):
        areaA = pD.area()
        areaB = pG.area()
        return areaA + areaB - get_intersection(pD, pG)


    def get_intersection(pD, pG):
        pInt = pD & pG
        if len(pInt) == 0:
            return 0
        return pInt.area()


    def input_reading_mod(input_dir, input):
        """This helper reads input from txt files"""
        with open('%s/%s' % (input_dir, input), 'r') as input_fid:
            pred = input_fid.readlines()
        det = [x.strip('\n') for x in pred]
        return det


    def gt_reading_mod(gt_dir, gt_id):
        """This helper reads groundtruths from mat files"""
        gt_id = gt_id.split('.')[0]
        gt = io.loadmat('%s/poly_gt_%s.mat' % (gt_dir, gt_id))
        gt = gt['polygt']
        return gt


    def detection_filtering(detections, groundtruths, threshold=0.5):
        for gt_id, gt in enumerate(groundtruths):
            if (gt[5] == '#') and (gt[1].shape[1] > 1):
                gt_x = map(int, np.squeeze(gt[1]))
                gt_y = map(int, np.squeeze(gt[3]))
                gt_x = list(gt_x)
                gt_y = list(gt_y)

                gt_p = np.concatenate((np.array(gt_x), np.array(gt_y)))
                gt_p = gt_p.reshape(2, -1).transpose()
                gt_p = plg.Polygon(gt_p)

                for det_id, detection in enumerate(detections):
                    detection = detection.split(',')
                    # detection = map(int, detection[0:-1])
                    detection = map(int, detection)
                    detection = list(detection)
                    
                    det_y = detection[0::2]
                    det_x = detection[1::2]

                    det_p = np.concatenate((np.array(det_x), np.array(det_y)))
                    det_p = det_p.reshape(2, -1).transpose()
                    det_p = plg.Polygon(det_p)

                    try:
                        # det_gt_iou = iod(det_x, det_y, gt_x, gt_y)
                        det_gt_iou = get_intersection(det_p, gt_p) / det_p.area()
                    except:
                        print(det_x, det_y, gt_x, gt_y)
                    if det_gt_iou > threshold:
                        detections[det_id] = []

                detections[:] = [item for item in detections if item != []]
        return detections


    # def sigma_calculation(det_x, det_y, gt_x, gt_y):
    #     """
    #     sigma = inter_area / gt_area
    #     """
    #     return np.round((area_of_intersection(det_x, det_y, gt_x, gt_y) / area(gt_x, gt_y)), 2)

    # def tau_calculation(det_x, det_y, gt_x, gt_y):
    #     """
    #     tau = inter_area / det_area
    #     """
    #     return np.round((area_of_intersection(det_x, det_y, gt_x, gt_y) / area(det_x, det_y)), 2)

    def sigma_calculation(det_p, gt_p):
        """
        sigma = inter_area / gt_area
        """
        # return np.round((area_of_intersection(det_x, det_y, gt_x, gt_y) / area(gt_x, gt_y)), 2)
        return get_intersection(det_p, gt_p) / gt_p.area()


    def tau_calculation(det_p, gt_p):
        """
        tau = inter_area / det_area
        """
        return get_intersection(det_p, gt_p) / det_p.area()


    ##############################Initialization###################################
    global_tp = 0
    global_fp = 0
    global_fn = 0
    global_sigma = []
    global_tau = []
    tr = 0.7
    tp = 0.6
    fsc_k = 0.8
    k = 2
    ###############################################################################


    for input_id in allInputs:
        if (input_id != '.DS_Store'):
            # print('input_id', input_id)
            detections = input_reading_mod(input_dir, input_id)
            # from IPython import embed;
            groundtruths = gt_reading_mod(gt_dir, input_id)
            detections = detection_filtering(detections, groundtruths)  # filters detections overlapping with DC area
            dc_id = np.where(groundtruths[:, 5] == '#')
            groundtruths = np.delete(groundtruths, (dc_id), (0))

            local_sigma_table = np.zeros((groundtruths.shape[0], len(detections)))
            local_tau_table = np.zeros((groundtruths.shape[0], len(detections)))

            for gt_id, gt in enumerate(groundtruths):
                if len(detections) > 0:
                    for det_id, detection in enumerate(detections):
                        detection = detection.split(',')
                        # print (len(detection))

                        # detection = map(int, detection[:-1])
                        detection = map(int, detection)
                        detection = list(detection)
                        # print (len(detection))

                        # from IPython import embed;embed()
                        # detection = list(detection)
                        #print(np.squeeze(gt[1]))
                        #print(np.squeeze(gt[3]))
                        gt_x = map(int, np.squeeze(gt[1]))
                        gt_y = map(int, np.squeeze(gt[3]))
                        gt_x = list(gt_x)
                        gt_y = list(gt_y)
                        #print(np.array(gt_x))
                        #print(np.array(gt_y))

                        gt_p = np.concatenate((np.array(gt_x), np.array(gt_y)))
                        gt_p = gt_p.reshape(2, -1).transpose()
                        #print(gt_p)
                        gt_p = plg.Polygon(gt_p)

                        det_y = detection[0::2]
                        det_x = detection[1::2]

                        det_p = np.concatenate((np.array(det_x), np.array(det_y)))
                        # print (det_p.shape)
                        det_p = det_p.reshape(2, -1).transpose()
                        det_p = plg.Polygon(det_p)

                        # gt_x = list(map(int, np.squeeze(gt[1])))
                        # gt_y = list(map(int, np.squeeze(gt[3])))
                        # try:
                        #     local_sigma_table[gt_id, det_id] = sigma_calculation(det_x, det_y, gt_x, gt_y)
                        #     local_tau_table[gt_id, det_id] = tau_calculation(det_x, det_y, gt_x, gt_y)
                        # except:
                        #     embed()
                        local_sigma_table[gt_id, det_id] = sigma_calculation(det_p, gt_p)
                        local_tau_table[gt_id, det_id] = tau_calculation(det_p, gt_p)
            # if input_id == 'img1199.txt':
            #    embed()
            global_sigma.append(local_sigma_table)
            global_tau.append(local_tau_table)

    global_accumulative_recall = 0
    global_accumulative_precision = 0
    total_num_gt = 0
    total_num_det = 0


    def one_to_one(local_sigma_table, local_tau_table, local_accumulative_recall,
                local_accumulative_precision, global_accumulative_recall, global_accumulative_precision,
                gt_flag, det_flag):
        for gt_id in range(num_gt):
            qualified_sigma_candidates = np.where(local_sigma_table[gt_id, :] > tr)
            num_qualified_sigma_candidates = qualified_sigma_candidates[0].shape[0]
            qualified_tau_candidates = np.where(local_tau_table[gt_id, :] > tp)
            num_qualified_tau_candidates = qualified_tau_candidates[0].shape[0]

            if (num_qualified_sigma_candidates == 1) and (num_qualified_tau_candidates == 1):
                global_accumulative_recall = global_accumulative_recall + 1.0
                global_accumulative_precision = global_accumulative_precision + 1.0
                local_accumulative_recall = local_accumulative_recall + 1.0
                local_accumulative_precision = local_accumulative_precision + 1.0

                gt_flag[0, gt_id] = 1
                matched_det_id = np.where(local_sigma_table[gt_id, :] > tr)
                det_flag[0, matched_det_id] = 1
        return local_accumulative_recall, local_accumulative_precision, global_accumulative_recall, global_accumulative_precision, gt_flag, det_flag


    def one_to_many(local_sigma_table, local_tau_table, local_accumulative_recall,
                    local_accumulative_precision, global_accumulative_recall, global_accumulative_precision,
                    gt_flag, det_flag):
        for gt_id in range(num_gt):
            # skip the following if the groundtruth was matched
            if gt_flag[0, gt_id] > 0:
                continue

            non_zero_in_sigma = np.where(local_sigma_table[gt_id, :] > 0)
            num_non_zero_in_sigma = non_zero_in_sigma[0].shape[0]

            if num_non_zero_in_sigma >= k:
                ####search for all detections that overlaps with this groundtruth
                qualified_tau_candidates = np.where((local_tau_table[gt_id, :] >= tp) & (det_flag[0, :] == 0))
                num_qualified_tau_candidates = qualified_tau_candidates[0].shape[0]

                if num_qualified_tau_candidates == 1:
                    if ((local_tau_table[gt_id, qualified_tau_candidates] >= tp) and (
                            local_sigma_table[gt_id, qualified_tau_candidates] >= tr)):
                        # became an one-to-one case
                        global_accumulative_recall = global_accumulative_recall + 1.0
                        global_accumulative_precision = global_accumulative_precision + 1.0
                        local_accumulative_recall = local_accumulative_recall + 1.0
                        local_accumulative_precision = local_accumulative_precision + 1.0

                        gt_flag[0, gt_id] = 1
                        det_flag[0, qualified_tau_candidates] = 1
                elif (np.sum(local_sigma_table[gt_id, qualified_tau_candidates]) >= tr):
                    gt_flag[0, gt_id] = 1
                    det_flag[0, qualified_tau_candidates] = 1

                    global_accumulative_recall = global_accumulative_recall + fsc_k
                    global_accumulative_precision = global_accumulative_precision + num_qualified_tau_candidates * fsc_k

                    local_accumulative_recall = local_accumulative_recall + fsc_k
                    local_accumulative_precision = local_accumulative_precision + num_qualified_tau_candidates * fsc_k

        return local_accumulative_recall, local_accumulative_precision, global_accumulative_recall, global_accumulative_precision, gt_flag, det_flag


    def many_to_many(local_sigma_table, local_tau_table, local_accumulative_recall,
                    local_accumulative_precision, global_accumulative_recall, global_accumulative_precision,
                    gt_flag, det_flag):
        for det_id in range(num_det):
            # skip the following if the detection was matched
            if det_flag[0, det_id] > 0:
                continue

            non_zero_in_tau = np.where(local_tau_table[:, det_id] > 0)
            num_non_zero_in_tau = non_zero_in_tau[0].shape[0]

            if num_non_zero_in_tau >= k:
                ####search for all detections that overlaps with this groundtruth
                qualified_sigma_candidates = np.where((local_sigma_table[:, det_id] >= tp) & (gt_flag[0, :] == 0))
                num_qualified_sigma_candidates = qualified_sigma_candidates[0].shape[0]

                if num_qualified_sigma_candidates == 1:
                    if ((local_tau_table[qualified_sigma_candidates, det_id] >= tp) and (
                            local_sigma_table[qualified_sigma_candidates, det_id] >= tr)):
                        # became an one-to-one case
                        global_accumulative_recall = global_accumulative_recall + 1.0
                        global_accumulative_precision = global_accumulative_precision + 1.0
                        local_accumulative_recall = local_accumulative_recall + 1.0
                        local_accumulative_precision = local_accumulative_precision + 1.0

                        gt_flag[0, qualified_sigma_candidates] = 1
                        det_flag[0, det_id] = 1
                elif (np.sum(local_tau_table[qualified_sigma_candidates, det_id]) >= tp):
                    det_flag[0, det_id] = 1
                    gt_flag[0, qualified_sigma_candidates] = 1

                    global_accumulative_recall = global_accumulative_recall + num_qualified_sigma_candidates * fsc_k
                    global_accumulative_precision = global_accumulative_precision + fsc_k

                    local_accumulative_recall = local_accumulative_recall + num_qualified_sigma_candidates * fsc_k
                    local_accumulative_precision = local_accumulative_precision + fsc_k
        return local_accumulative_recall, local_accumulative_precision, global_accumulative_recall, global_accumulative_precision, gt_flag, det_flag


    for idx in range(len(global_sigma)):
        # print(allInputs[idx])
        local_sigma_table = global_sigma[idx]
        local_tau_table = global_tau[idx]

        num_gt = local_sigma_table.shape[0]
        num_det = local_sigma_table.shape[1]

        total_num_gt = total_num_gt + num_gt
        total_num_det = total_num_det + num_det

        local_accumulative_recall = 0
        local_accumulative_precision = 0
        gt_flag = np.zeros((1, num_gt))
        det_flag = np.zeros((1, num_det))

        #######first check for one-to-one case##########
        local_accumulative_recall, local_accumulative_precision, global_accumulative_recall, global_accumulative_precision, \
        gt_flag, det_flag = one_to_one(local_sigma_table, local_tau_table,
                                    local_accumulative_recall, local_accumulative_precision,
                                    global_accumulative_recall, global_accumulative_precision,
                                    gt_flag, det_flag)

        #######then check for one-to-many case##########
        local_accumulative_recall, local_accumulative_precision, global_accumulative_recall, global_accumulative_precision, \
        gt_flag, det_flag = one_to_many(local_sigma_table, local_tau_table,
                                        local_accumulative_recall, local_accumulative_precision,
                                        global_accumulative_recall, global_accumulative_precision,
                                        gt_flag, det_flag)

        #######then check for many-to-many case##########
        local_accumulative_recall, local_accumulative_precision, global_accumulative_recall, global_accumulative_precision, \
        gt_flag, det_flag = many_to_many(local_sigma_table, local_tau_table,
                                        local_accumulative_recall, local_accumulative_precision,
                                        global_accumulative_recall, global_accumulative_precision,
                                        gt_flag, det_flag)
        # for det_id in xrange(num_det):
        #     # skip the following if the detection was matched
        #     if det_flag[0, det_id] > 0:
        #         continue
        #
        #     non_zero_in_tau = np.where(local_tau_table[:, det_id] > 0)
        #     num_non_zero_in_tau = non_zero_in_tau[0].shape[0]
        #
        #     if num_non_zero_in_tau >= k:
        #         ####search for all detections that overlaps with this groundtruth
        #         qualified_sigma_candidates = np.where((local_sigma_table[:, det_id] >= tp) & (gt_flag[0, :] == 0))
        #         num_qualified_sigma_candidates = qualified_sigma_candidates[0].shape[0]
        #
        #         if num_qualified_sigma_candidates == 1:
        #             if ((local_tau_table[qualified_sigma_candidates, det_id] >= tp) and (local_sigma_table[qualified_sigma_candidates, det_id] >= tr)):
        #                 #became an one-to-one case
        #                 global_accumulative_recall = global_accumulative_recall + 1.0
        #                 global_accumulative_precision = global_accumulative_precision + 1.0
        #                 local_accumulative_recall = local_accumulative_recall + 1.0
        #                 local_accumulative_precision = local_accumulative_precision + 1.0
        #
        #                 gt_flag[0, qualified_sigma_candidates] = 1
        #                 det_flag[0, det_id] = 1
        #         elif (np.sum(local_tau_table[qualified_sigma_candidates, det_id]) >= tp):
        #             det_flag[0, det_id] = 1
        #             gt_flag[0, qualified_sigma_candidates] = 1
        #
        #             global_accumulative_recall = global_accumulative_recall + num_qualified_sigma_candidates * fsc_k
        #             global_accumulative_precision = global_accumulative_precision + fsc_k
        #
        #             local_accumulative_recall = local_accumulative_recall + num_qualified_sigma_candidates * fsc_k
        #             local_accumulative_precision = local_accumulative_precision + fsc_k

        fid = open(fid_path, 'a+')
        try:
            local_precision = local_accumulative_precision / num_det
        except ZeroDivisionError:
            local_precision = 0

        try:
            local_recall = local_accumulative_recall / num_gt
        except ZeroDivisionError:
            local_recall = 0

        temp = ('%s______/Precision:_%s_______/Recall:_%s\n' % (allInputs[idx], str(local_precision), str(local_recall)))
        fid.write(temp)
        fid.close()
    try:
        recall = global_accumulative_recall / total_num_gt
    except ZeroDivisionError:
        recall = 0

    try:
        precision = global_accumulative_precision / total_num_det
    except ZeroDivisionError:
        precision = 0

    try:
        f_score = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        f_score = 0

    fid = open(fid_path, 'a')
    hmean = 2 * precision * recall / (precision + recall)
    temp = ('Precision:_%s_______/Recall:_%s/Hmean:_%s\n' % (str(precision), str(recall), str(hmean)))
    # print(temp)
    fid.write(temp)
    fid.close()

    # print('pb')

    return {'precision':precision, 'recall':recall, 'hmean':hmean}

# CTW1500 数据集测试函数
def main_test_ctw():
    th = 0.5
    # pred_list = file_util.read_dir(pred_root)
    project_root = './'
    pred_root = project_root + 'outputs/submit_ctw'
    gt_root = project_root + 'data/ctw1500/test/text_label_circum/'

    pred_list = read_dir(pred_root)

    tp, fp, npos = 0, 0, 0

    for pred_path in pred_list:
        preds = get_pred(pred_path)
        gt_path = gt_root + pred_path.split('/')[-1]
        gts = get_gt(gt_path)
        npos += len(gts)

        cover = set()
        for pred_id, pred in enumerate(preds):
            pred = np.array(pred)
            pred = pred.reshape(int(pred.shape[0]) // 2, 2)[:, ::-1]

            pred_p = plg.Polygon(pred)

            flag = False
            for gt_id, gt in enumerate(gts):
                gt = np.array(gt)
                gt = gt.reshape(int(gt.shape[0]) // 2, 2)
                gt_p = plg.Polygon(gt)

                union = get_union(pred_p, gt_p)
                inter = get_intersection(pred_p, gt_p)

                if inter * 1.0 / union >= th:
                    if gt_id not in cover:
                        flag = True
                        cover.add(gt_id)
            if flag:
                tp += 1.0
            else:
                fp += 1.0

    # print tp, fp, npos
    precision = tp / (tp + fp)
    recall = tp / npos
    hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)

    # print('p: %.4f, r: %.4f, f: %.4f' % (precision, recall, hmean))

    return {'precision':precision, 'recall':recall, 'hmean':hmean}


def report_speed(outputs, speed_meters):
    total_time = 0
    for key in outputs:
        if 'time' in key:
            total_time += outputs[key]
            speed_meters[key].update(outputs[key])
            print('%s: %.4f' % (key, speed_meters[key].avg))

    speed_meters['total_time'].update(total_time)
    print('FPS: %.1f' % (1.0 / speed_meters['total_time'].avg))


# 测试功能 主函数
def test(test_loader, model, cfg):
    model = model.cuda()
    model.eval()

    with_rec = hasattr(cfg.model, 'recognition_head')
    if with_rec:
        pp = Corrector(cfg.data.test.type, **cfg.test_cfg.rec_post_process)
    rf = ResultFormat(cfg.data.test.type, cfg.test_cfg.result_path)

    if cfg.report_speed:
        speed_meters = dict(
            backbone_time=AverageMeter(500),
            neck_time=AverageMeter(500),
            det_head_time=AverageMeter(500),
            det_pa_time=AverageMeter(500),
            rec_time=AverageMeter(500),
            total_time=AverageMeter(500)
        )

    for idx, data in enumerate(test_loader):
        # print('Testing %d/%d' % (idx, len(test_loader)))
        sys.stdout.flush()

        # prepare input
        data['imgs'] = data['imgs'].cuda()
        data.update(dict(
            cfg=cfg
        ))
        # print(data)
        # forward
        with torch.no_grad():
            outputs = model(**data)

        if cfg.report_speed:
            report_speed(outputs, speed_meters)
        # post process of recognition
        if with_rec:
            outputs = pp.process(outputs)

        # save result
        image_name, _ = osp.splitext(osp.basename(test_loader.dataset.img_paths[idx]))
        rf.write_result(image_name, outputs)

# 测试功能入口
def main_test(args, model, checkpoint_test_file):
    cfg = Config.fromfile(args.config)
    for d in [cfg, cfg.data.test]:
        d.update(dict(
            report_speed=args.report_speed
        ))
    # print(json.dumps(cfg._cfg_dict, indent=4))
    sys.stdout.flush()

    # data loader
    data_loader = build_data_loader(cfg.data.test)
    test_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )
    # model
    if hasattr(cfg.model, 'recognition_head'):
        cfg.model.recognition_head.update(dict(
            voc=data_loader.voc,
            char2id=data_loader.char2id,
            id2char=data_loader.id2char,
        ))
    model = build_model(cfg.model)

    model = model.cuda()

    if checkpoint_test_file is not None:
        if os.path.isfile(checkpoint_test_file):
            # print("Loading model and optimizer from checkpoint '{}'".format(checkpoint_test_file))
            sys.stdout.flush()

            checkpoint_test = torch.load(checkpoint_test_file)

            d = dict()
            for key, value in checkpoint_test['state_dict'].items():
                tmp = key[7:]
                d[tmp] = value
            model.load_state_dict(d)
        else:
            print("No checkpoint_test found at '{}'".format(args.resume))
            raise

    # fuse conv and bn
    model = fuse_module(model)

    # test
    test(test_loader, model, cfg)

    # icdar2015
    # resDict = main_test_ic15()
    # result = resDict['method']
    
    # ctw
    # result = main_test_ctw()
    # tt
    result = main_test_tt()

    return result


# 训练功能函数
def train(train_loader, model, optimizer, epoch, start_iter, cfg):
    model.train()

    # meters
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    losses_text = AverageMeter()
    losses_kernels = AverageMeter()
    losses_emb = AverageMeter()
    losses_rec = AverageMeter()

    ious_text = AverageMeter()
    ious_kernel = AverageMeter()
    accs_rec = AverageMeter()

    with_rec = hasattr(cfg.model, 'recognition_head')
    # start time
    start = time.time()
    for iter, data in enumerate(train_loader):
        # skip previous iterations
        if iter < start_iter:
            print('Skipping iter: %d' % iter)
            sys.stdout.flush()
            continue

        # time cost of data loader
        data_time.update(time.time() - start)

        # adjust learning rate
        adjust_learning_rate(optimizer, train_loader, epoch, iter, cfg)

        # prepare input
        data.update(dict(
            cfg=cfg
        ))

        # forward
        outputs = model(**data)

        # detection loss
        loss_text = torch.mean(outputs['loss_text'])
        losses_text.update(loss_text.item())

        loss_kernels = torch.mean(outputs['loss_kernels'])
        losses_kernels.update(loss_kernels.item())
        if 'loss_emb' in outputs.keys():
            loss_emb = torch.mean(outputs['loss_emb'])
            losses_emb.update(loss_emb.item())
            loss = loss_text + loss_kernels + loss_emb
        else:
            loss = loss_text + loss_kernels

        iou_text = torch.mean(outputs['iou_text'])
        ious_text.update(iou_text.item())
        iou_kernel = torch.mean(outputs['iou_kernel'])
        ious_kernel.update(iou_kernel.item())

        # recognition loss
        if with_rec:
            loss_rec = outputs['loss_rec']
            valid = loss_rec > 0.5
            if torch.sum(valid) > 0:
                loss_rec = torch.mean(loss_rec[valid])
                losses_rec.update(loss_rec.item())
                loss = loss + loss_rec

                acc_rec = outputs['acc_rec']
                acc_rec = torch.mean(acc_rec[valid])
                accs_rec.update(acc_rec.item(), torch.sum(valid).item())

        losses.update(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - start)

        # update start time
        start = time.time()

        # print log
        if iter % 20 == 0:
            output_log = f'({iter + 1}/{len(train_loader)}) LR: {optimizer.param_groups[0]["lr"]:.6f} | ' \
                         f'Batch: {batch_time.avg:.3f}s | Total: {batch_time.avg * iter / 60.0:.0f}min | ' \
                         f'ETA: {batch_time.avg * (len(train_loader) - iter) / 60.0:.0f}min | ' \
                         f'Loss: {losses.avg:.3f} | ' \
                         f'Loss(text/kernel/emb{"/rec" if with_rec else ""}): {losses_text.avg:.3f}/{losses_kernels.avg:.3f}/' \
                         f'{losses_emb.avg:.3f}{"/" + format(losses_rec.avg, ".3f") if with_rec else ""} | ' \
                         f'IoU(text/kernel): {ious_text.avg:.3f}/{ious_kernel.avg:.3f}' \
                         f'{" | ACC rec: " + format(accs_rec.avg, ".3f") if with_rec else ""}'

            # output_log = '({batch}/{size}) LR: {lr:.6f} | Batch: {bt:.3f}s | Total: {total:.0f}min | ' \
            #              'ETA: {eta:.0f}min | Loss: {loss:.3f} | ' \
            #              'Loss(text/kernel/emb/rec): {loss_text:.3f}/{loss_kernel:.3f}/{loss_emb:.3f}/{loss_rec:.3f} ' \
            #              '| IoU(text/kernel): {iou_text:.3f}/{iou_kernel:.3f} | Acc rec: {acc_rec:.3f}'.format(
            #     batch=iter + 1,
            #     size=len(train_loader),
            #     lr=optimizer.param_groups[0]['lr'],
            #     bt=batch_time.avg,
            #     total=batch_time.avg * iter / 60.0,
            #     eta=batch_time.avg * (len(train_loader) - iter) / 60.0,
            #     loss_text=losses_text.avg,
            #     loss_kernel=losses_kernels.avg,
            #     loss_emb=losses_emb.avg,
            #     loss_rec=losses_rec.avg,
            #     loss=losses.avg,
            #     iou_text=ious_text.avg,
            #     iou_kernel=ious_kernel.avg,
            #     acc_rec=accs_rec.avg,
            # )
            print(output_log)
            sys.stdout.flush()

# 调整学习率
def adjust_learning_rate(optimizer, dataloader, epoch, iter, cfg):
    schedule = cfg.train_cfg.schedule

    if isinstance(schedule, str):
        assert schedule == 'polylr', 'Error: schedule should be polylr!'
        cur_iter = epoch * len(dataloader) + iter
        max_iter_num = cfg.train_cfg.epoch * len(dataloader)
        lr = cfg.train_cfg.lr * (1 - float(cur_iter) / max_iter_num) ** 0.9
    elif isinstance(schedule, tuple):
        lr = cfg.train_cfg.lr
        for i in range(len(schedule)):
            if epoch < schedule[i]:
                break
            lr = lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# 保存模型
def save_checkpoint(state, checkpoint_path, cfg):
    file_path = osp.join(checkpoint_path, 'checkpoint.pth.tar')
    torch.save(state, file_path)

    if cfg.data.train.type in ['synth'] or \
            (state['iter'] == 0 and state['epoch'] > cfg.train_cfg.epoch - 100 and state['epoch'] % 10 == 0):
        file_name = 'checkpoint_%dep.pth.tar' % state['epoch']
        file_path = osp.join(checkpoint_path, file_name)
        torch.save(state, file_path)

# 主函数
def main(args):
    cfg = Config.fromfile(args.config)
    print(json.dumps(cfg._cfg_dict, indent=4))

    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        cfg_name, _ = osp.splitext(osp.basename(args.config))
        # checkpoint_path = osp.join('checkpoints', cfg_name)
        checkpoint_path = osp.join('checkpoints', cfg_name + '_resnet34_Tucker_Tucker_CAB_pretrain_False')
    if not osp.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    print('Checkpoint path: %s.' % checkpoint_path)
    sys.stdout.flush()

    print("================ Start Loading data ...=====================\n")
    # data loader
    data_loader = build_data_loader(cfg.data.train)
    train_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True
    )

    print("================ End Loading data ...=====================\n")

    # model
    if hasattr(cfg.model, 'recognition_head'):
        cfg.model.recognition_head.update(dict(
            voc=data_loader.voc,
            char2id=data_loader.char2id,
            id2char=data_loader.id2char,
        ))
    model = build_model(cfg.model)
    model = torch.nn.DataParallel(model).cuda()
    # model = model.cuda()
    print("================ Start Loading model ...=====================\n")

    # Check if model has custom optimizer / loss
    if hasattr(model.module, 'optimizer'):
        optimizer = model.module.optimizer
    else:
        if cfg.train_cfg.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=cfg.train_cfg.lr, momentum=0.99, weight_decay=5e-4)
        elif cfg.train_cfg.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train_cfg.lr)

    start_epoch = 0
    start_iter = 0
    if hasattr(cfg.train_cfg, 'pretrain'):
        assert osp.isfile(cfg.train_cfg.pretrain), 'Error: no pretrained weights found!'
        print('Finetuning from pretrained model %s.' % cfg.train_cfg.pretrain)
        checkpoint = torch.load(cfg.train_cfg.pretrain)
        model.load_state_dict(checkpoint['state_dict'])
    if args.resume:
        assert osp.isfile(args.resume), 'Error: no checkpoint directory found!'
        print('Resuming from checkpoint %s.' % args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        start_iter = checkpoint['iter']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    best_result = {'precision':0, 'recall':0, 'hmean':0}
    result = {'precision':0, 'recall':0, 'hmean':0}

    for epoch in range(start_epoch, cfg.train_cfg.epoch):
        print('\nEpoch: [%d | %d]' % (epoch + 1, cfg.train_cfg.epoch))
        print("Training of {}".format(cfg.data.train.type))

        train(train_loader, model, optimizer, epoch, start_iter, cfg)
        

        state = dict(
            epoch=epoch + 1,
            iter=0,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict()
        )
        save_checkpoint(state, checkpoint_path, cfg)

        # test
        file_path = osp.join(checkpoint_path, 'checkpoint.pth.tar')        

        # if epoch %5 ==0:
        result = main_test(args, model, file_path)
        # else:
        #     result = main_test(args, model, file_path)

        if result['hmean'] > best_result["hmean"]:
            best_result['hmean'] = result['hmean']
            best_result['precision'] = result['precision']
            best_result['recall'] = result['recall']
        print('cur model result: precision:{} recall:{} hmean:{}'.format(result['precision'], result['recall'], result['hmean']))
        print('best model result: precision:{} recall:{} hmean:{}'.format(best_result['precision'], best_result['recall'], best_result['hmean']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--checkpoint', nargs='?', type=str, default=None)
    parser.add_argument('--report_speed', action='store_true')
    # parser.add_argument('--checkpoint_test', nargs='?', type=str, default=None)
    parser.add_argument('--resume', nargs='?', type=str, default=None)
    args = parser.parse_args()

    main(args)
