import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from lib.utils import square_distance
from sklearn.metrics import precision_recall_fscore_support
import pdb

class MetricLoss(nn.Module):
    """
    We evaluate both contrastive loss and circle loss
    """
    def __init__(self,configs,log_scale=16, pos_optimal=0.1, neg_optimal=1.4):
        super(MetricLoss,self).__init__()
        self.log_scale = log_scale
        self.pos_optimal = pos_optimal
        self.neg_optimal = neg_optimal

        self.pos_margin = configs.pos_margin
        self.neg_margin = configs.neg_margin
        self.max_points = configs.max_points

        # self.safe_radius = configs.safe_radius 
        self.global_neg_radius = configs.global_neg_radius
        self.global_pos_radius = configs.global_pos_radius 
        self.local_pos_radius = configs.local_pos_radius
        self.local_neg_radius = configs.local_neg_radius
    
    def global_circle_loss(self, coords_dist, feats_dist):
        """
        Modified from: https://github.com/XuyangBai/D3Feat.pytorch
        """
        pos_mask = coords_dist < self.global_pos_radius
        neg_mask = coords_dist > self.global_neg_radius

        ## get anchors that have both positive and negative pairs
        row_sel = ((pos_mask.sum(-1)>0) * (neg_mask.sum(-1)>0)).detach()
        col_sel = ((pos_mask.sum(-2)>0) * (neg_mask.sum(-2)>0)).detach()

        # get alpha for both positive and negative pairs
        pos_weight = feats_dist - 1e5 * (~pos_mask).float() # mask the non-positive 
        pos_weight = (pos_weight - self.pos_optimal) # mask the uninformative positive
        pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight).detach() 

        neg_weight = feats_dist + 1e5 * (~neg_mask).float() # mask the non-negative
        neg_weight = (self.neg_optimal - neg_weight) # mask the uninformative negative
        neg_weight = torch.max(torch.zeros_like(neg_weight),neg_weight).detach()

        furthest_positive1, _ = torch.max(feats_dist * pos_mask.float(), dim=1)
        closest_negative1, _ = torch.min(feats_dist + 1e5 * (~neg_mask).float(), dim=1)
        diff1 = (furthest_positive1 - closest_negative1) < 0
        accuracy = torch.true_divide(diff1.sum(), diff1.shape[0])
        furthest_positive2, _ = torch.max(feats_dist * pos_mask.float(), dim=0)
        closest_negative2, _ = torch.min(feats_dist + 1e5 * (~neg_mask).float(), dim=0)
        diff2 = (furthest_positive2 - closest_negative2) < 0
        diff = torch.cat((diff1,diff2),dim=0).detach()

        lse_pos_row = torch.logsumexp(self.log_scale * (feats_dist - self.pos_margin) * pos_weight,dim=-1)
        lse_pos_col = torch.logsumexp(self.log_scale * (feats_dist - self.pos_margin) * pos_weight,dim=-2)

        lse_neg_row = torch.logsumexp(self.log_scale * (self.neg_margin - feats_dist) * neg_weight,dim=-1)
        lse_neg_col = torch.logsumexp(self.log_scale * (self.neg_margin - feats_dist) * neg_weight,dim=-2)

        loss_row = F.softplus(lse_pos_row + lse_neg_row)/self.log_scale
        loss_col = F.softplus(lse_pos_col + lse_neg_col)/self.log_scale

        circle_loss = (loss_row[row_sel].mean() + loss_col[col_sel].mean()) / 2

        return circle_loss, diff

    def local_circle_loss(self, feats, feats_corr, neighbor_feats, knn_mask): 
        pos_feats_dist = torch.sqrt(((feats - feats_corr)**2).sum(dim=-1)).view(-1,1)  # [N, 1]
        pos_weight = (pos_feats_dist - self.pos_optimal)  # [N, 1]
        pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight).detach() # detach
        lse_pos = torch.logsumexp(self.log_scale * (pos_feats_dist - self.pos_margin) * pos_weight, dim=-1)
        
        neg_feats_sub = feats.unsqueeze(1)-neighbor_feats  # [N, K, d]
        neg_feats_dist = torch.sqrt((neg_feats_sub**2).sum(dim=-1))
        neg_weight = (self.neg_optimal - neg_feats_dist)  # [N, K]
        neg_weight = torch.max(torch.zeros_like(neg_weight), neg_weight).detach() 
        neg_weight[~knn_mask] = 0 
        lse_neg = torch.logsumexp(self.log_scale * (self.neg_margin - neg_feats_dist) * neg_weight, dim=-1)
        circle_loss = (F.softplus(lse_pos + lse_neg) / self.log_scale).mean()

        pos_feats_dist_ = pos_feats_dist.clone()
        neg_feats_dist_ = neg_feats_dist.clone()
        # neighbor_num = (knn_mask>0).sum(dim=1)  # [N]
        # neighbor_num = torch.max(neighbor_num, torch.ones_like(neighbor_num)) # [N]
        # neg_feats_dist_[~knn_mask] = 0. # [N, K]
        # mean_neg_dist = neg_feats_dist_.sum(dim=1) / neighbor_num
        # l_label = (pos_feats_dist_.view(-1) - mean_neg_dist) < 0
        # tau = torch.quantile(mean_neg_dist, 0.6)
        # l_label = mean_neg_dist > tau  
        # pdb.set_trace()
        neg_feats_dist_[~knn_mask] = 1e5
        tmp = pos_feats_dist_.view(-1) - neg_feats_dist_.min(dim=-1)[0]
        recall = (tmp < 0).sum() / len(tmp)

        # tau = torch.quantile(tmp, 0.4)  
        l_label = tmp < 0
        
        # pdb.set_trace()
        return circle_loss, l_label, recall

    def get_neighbor_features(self, pcd_down, pcd, feats, neighbor_radius=0.1, k=32, neg_radius=0.05):
        sq_dist_mat = torch.sqrt(square_distance(pcd_down[None, :, :], pcd[None, :, :]).squeeze(0))
        knn_sq_distances, knn_indices = sq_dist_mat.topk(dim=1, k=k+1, largest=False)
        knn_sq_distances, knn_indices = knn_sq_distances[:,1:], knn_indices[:,1:]
        knn_masks = (knn_sq_distances<neighbor_radius) & (knn_sq_distances>neg_radius)  # (N, k)
        sentinel_indices = torch.full_like(knn_indices, pcd.shape[0])  # (N, k)
        feats_padded = torch.cat((feats, torch.zeros_like(feats)), dim=0)
        knn_indices = torch.where(knn_masks, knn_indices, sentinel_indices)  # (N, k)
        neighbor_feats = feats_padded[knn_indices]
        return neighbor_feats, knn_masks

    def get_recall(self,coords_dist,feats_dist, thresh=0.1):
        """
        Get feature match recall, divided by number of true inliers
        """
        pos_mask = coords_dist < thresh
        n_gt_pos = (pos_mask.sum(-1)>0).float().sum()+1e-12
        _, sel_idx = torch.min(feats_dist, -1)
        sel_dist = torch.gather(coords_dist,dim=-1,index=sel_idx[:,None])[pos_mask.sum(-1)>0]
        n_pred_pos = (sel_dist < thresh).float().sum()
        recall = n_pred_pos / n_gt_pos
        return recall

    def get_weighted_bce_loss(self, prediction, gt):
        loss = nn.BCELoss(reduction='none')
        class_loss = loss(prediction, gt) 

        weights = torch.ones_like(gt)
        w_negative = gt.sum()/gt.size(0) 
        w_positive = 1 - w_negative  
        
        weights[gt >= 0.5] = w_positive
        weights[gt < 0.5] = w_negative
        w_class_loss = torch.mean(weights * class_loss)

        #######################################
        # get classification precision and recall
        predicted_labels = prediction.detach().cpu().round().numpy()
        PT_num = np.logical_and(gt.cpu().numpy(), predicted_labels).sum()
        P = PT_num / (predicted_labels.sum() + 1)
        R = PT_num / (gt.sum() + 1)
        return w_class_loss, P, R
            

    def forward(self, src_raw, tgt_raw, l_feats, h_feats, scores_overlap, len_src,
                correspondence, rot, trans, l_saliency, h_saliency):

        src_raw = (torch.matmul(rot, src_raw.transpose(0, 1)) + trans).transpose(0, 1)
        ##############################################################################
        # 1. Overlap Loss
        stats = dict()
        src_idx = list(set(correspondence[:, 0].int().tolist()))
        tgt_idx = list(set(correspondence[:, 1].int().tolist()))
        # get BCE loss for overlap, here the ground truth label is obtained from correspondence information
        src_gt = torch.zeros(src_raw.size(0))
        src_gt[src_idx] = 1.
        tgt_gt = torch.zeros(tgt_raw.size(0))
        tgt_gt[tgt_idx] = 1.
        gt_labels = torch.cat((src_gt, tgt_gt)).to(torch.device('cuda'))

        class_loss, cls_precision, cls_recall = self.get_weighted_bce_loss(scores_overlap, gt_labels)
        stats['overlap_loss'] = class_loss
        stats['overlap_precision'] = cls_precision

        ##############################################################################
        # 2. Feature Loss
        c_dist = torch.norm(src_raw[correspondence[:,0]] - tgt_raw[correspondence[:,1]], dim = 1)
        c_select = c_dist < self.local_pos_radius 
        correspondence = correspondence[c_select]
        if (correspondence.size(0) > self.max_points):
            choice = np.random.permutation(correspondence.size(0))[:self.max_points]
            correspondence = correspondence[choice]
        src_idx = correspondence[:, 0]
        tgt_idx = correspondence[:, 1] 
        src_pcd, tgt_pcd = src_raw[src_idx], tgt_raw[tgt_idx]
        coords_dist = torch.sqrt(square_distance(src_pcd[None, :, :], tgt_pcd[None, :, :]).squeeze(0))
        # low_feat_loss
        src_l_feats, tgt_l_feats = l_feats[:len_src], l_feats[len_src:]
        src_lf, tgt_lf = src_l_feats[src_idx], tgt_l_feats[tgt_idx]
        src_neighbor_feats, src_knn_masks = self.get_neighbor_features(src_pcd, tgt_raw, tgt_l_feats,
                                neighbor_radius=self.global_neg_radius, k=32, neg_radius=self.local_neg_radius)
        tgt_neighbor_feats, tgt_knn_masks = self.get_neighbor_features(tgt_pcd, src_raw, src_l_feats, 
                                neighbor_radius=self.global_neg_radius, k=32, neg_radius=self.local_neg_radius)
        src_low_feat_loss, src_low_label, src_recall = self.local_circle_loss(src_lf, tgt_lf, src_neighbor_feats, src_knn_masks)
        tgt_low_feat_loss, tgt_low_label, tgt_recall = self.local_circle_loss(tgt_lf, src_lf, tgt_neighbor_feats, tgt_knn_masks)
        # pdb.set_trace()
        low_feat_loss = (src_low_feat_loss + tgt_low_feat_loss)/2
        low_label = torch.cat((src_low_label, tgt_low_label), dim=0).view(-1)
        stats['low_feat_loss'] = low_feat_loss
        stats['low_feat_recall'] = (src_recall + tgt_recall) / 2
        # high_feat_loss
        src_h_feats, tgt_h_feats = h_feats[:len_src], h_feats[len_src:]
        src_hf, tgt_hf = src_h_feats[src_idx], tgt_h_feats[tgt_idx]
        feats_dist = torch.sqrt(square_distance(src_hf[None, :, :], tgt_hf[None, :, :], normalised=True)).squeeze(0)
        stats['high_feat_recall'] = self.get_recall(coords_dist, feats_dist)
        high_feat_loss, high_label = self.global_circle_loss(coords_dist, feats_dist)
        stats['high_feat_loss'] = high_feat_loss
        ###################################################
         # Saliency loss
        src_l_saliency, tgt_l_saliency = l_saliency[:len_src][src_idx], l_saliency[len_src:][tgt_idx]
        src_h_saliency, tgt_h_saliency = h_saliency[:len_src][src_idx], h_saliency[len_src:][tgt_idx]
        l_saliency = torch.cat((src_l_saliency, tgt_l_saliency), dim=0)
        h_saliency = torch.cat((src_h_saliency, tgt_h_saliency), dim=0)
        # l_saliency_loss, l_precision, l_recall = self.get_weighted_bce_loss(l_saliency, low_label.float())
        # h_saliency_loss, h_precision, h_recall = self.get_weighted_bce_loss(h_saliency, high_label.float())
       
        low_rating_labels = torch.zeros_like(l_saliency).detach()
        low_rating_labels[low_label&high_label] = 1.0
        low_rating_labels[low_label&(~high_label)] = 0.75
        low_rating_labels[(~low_label)&high_label] = 0.25
        low_rating_labels[(~low_label)& (~high_label)] = 0.0
        low_num1 = (low_rating_labels == 1.0).sum()
        low_num2 = (low_rating_labels == 0.75).sum()
        low_num3 = (low_rating_labels == 0.25).sum()
        low_num4 = (low_rating_labels == 0.0).sum()
        low_weights = torch.zeros_like(low_rating_labels)
        low_weights[low_rating_labels == 1.0] = 512 / low_num1  
        low_weights[low_rating_labels == 0.75] = 512 / low_num2
        low_weights[low_rating_labels == 0.25] = 512 / low_num3
        low_weights[low_rating_labels == 0.0] = 512 / low_num4

        # pdb.set_trace()
        stats['l_saliency_loss'] = torch.mean(low_weights * ((low_rating_labels - l_saliency) ** 2))

        high_rating_labels = torch.zeros_like(h_saliency).detach()
        high_rating_labels[high_label & low_label] = 1.0
        high_rating_labels[high_label & (~low_label)] = 0.75
        high_rating_labels[(~high_label) & low_label] = 0.25
        high_rating_labels[(~high_label)& (~low_label)] = 0.0
        high_num1 = (high_rating_labels == 1.0).sum()
        high_num2 = (high_rating_labels == 0.75).sum()
        high_num3 = (high_rating_labels == 0.25).sum()
        high_num4 = (high_rating_labels == 0.0).sum()
        high_weights = torch.zeros_like(high_rating_labels)
        high_weights[high_rating_labels == 1.0] = 512 / high_num1  
        high_weights[high_rating_labels == 0.75] = 512 / high_num2
        high_weights[high_rating_labels == 0.25] = 512 / high_num3
        high_weights[high_rating_labels == 0.0] = 512 / high_num4
        stats['h_saliency_loss'] = torch.mean(high_weights * ((high_rating_labels - h_saliency) ** 2))

        # stats['l_saliency_loss'] = l_saliency_loss
        # stats['h_saliency_loss'] = h_saliency_loss
        # stats['l_saliency_precision'] = l_precision
        # stats['h_saliency_precision'] = h_precision

        return stats
