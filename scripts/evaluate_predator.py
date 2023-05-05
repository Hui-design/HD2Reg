"""
Scripts for pairwise registration with RANSAC and our probabilistic sampling

Author: Shengyu Huang
Last modified: 30.11.2020
"""

import torch, os, sys, glob
cwd = os.getcwd()
sys.path.append(cwd)
from tqdm import tqdm
import numpy as np
from lib.utils import load_obj, natural_key,setup_seed 
from lib.benchmark_utils import ransac_pose_estimation, get_inlier_ratio, get_scene_split, write_est_trajectory, to_tsfm, to_tensor,get_corr, get_neighbor
import open3d as o3d
from lib.benchmark import read_trajectory, write_trajectory, benchmark
from lib.utils import square_distance
import argparse
import pdb
setup_seed(0)

def sample_interest_points(method, scores, N):
    """
    We can do random sampling, probabilistic sampling, or top-k sampling
    """
    assert method in ['prob','topk', 'random']
    n = scores.size(0)
    if n < N:
        choice = np.random.choice(n, N)
    else:
        if method == 'random':
            choice = np.random.permutation(n)[:N]
        elif method =='topk':
            choice = torch.topk(scores, N, dim=0)[1]
        elif method =='prob':
            idx = np.arange(n)
            probs = (scores / scores.sum()).numpy().flatten()
            choice = np.random.choice(idx, size= N, replace=False, p=probs)
            choice = torch.from_numpy(choice)
    
    return choice


def adj_std(scores, sigma):
    src_mean, src_std = scores.mean(), scores.std()
    scores = (scores - src_mean) / src_std * sigma + src_mean
    scores[scores < 0] = 0.01  # 防止非负值
    return scores


def rigid_transform_3d(A, B, weights=None, weight_threshold=0):
    """
    Input:
        - A:       [bs, num_corr, 3], source point cloud
        - B:       [bs, num_corr, 3], target point cloud
        - weights: [bs, num_corr]     weight for each correspondence
        - weight_threshold: float,    clips points with weight below threshold
    Output:
        - R, t
    """
    bs = A.shape[0]
    if weights is None:
        weights = torch.ones_like(A[:, :, 0])
    weights[weights < weight_threshold] = 0
    # weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-6)
    # find mean of point cloud
    centroid_A = torch.sum(A * weights[:, :, None], dim=1, keepdim=True) / (torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)
    centroid_B = torch.sum(B * weights[:, :, None], dim=1, keepdim=True) / (torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    # construct weight covariance matrix
    Weight = torch.diag_embed(weights)
    H = Am.permute(0, 2, 1) @ Weight @ Bm

    # find rotation
    U, S, Vt = torch.svd(H.cpu())
    U, S, Vt = U.to(weights.device), S.to(weights.device), Vt.to(weights.device)
    delta_UV = torch.det(Vt @ U.permute(0, 2, 1))
    eye = torch.eye(3)[None, :, :].repeat(bs, 1, 1).to(A.device)
    eye[:, -1, -1] = delta_UV
    R = Vt @ eye @ U.permute(0, 2, 1)
    t = centroid_B.permute(0,2,1) - R @ centroid_A.permute(0,2,1)
    # warp_A = transform(A, integrate_trans(R,t))
    # RMSE = torch.sum( (warp_A - B) ** 2, dim=-1).mean()
    est = np.zeros((4, 4))
    est[:3, :3] = R[0].numpy()
    est[:3, 3] = t[0].reshape(-1).numpy()
    est[3, 3] = 1
    return est

def benchmark_predator(feats_scores,n_points,exp_dir,whichbenchmark,sample_method,ransac_with_mutual=False, inlier_ratio_threshold = 0.05):
    gt_folder = f'configs/benchmarks/{whichbenchmark}'
    exp_dir = f'{exp_dir}/{whichbenchmark}_{n_points}_{sample_method}'
    if(not os.path.exists(exp_dir)):
        os.makedirs(exp_dir)
    print(exp_dir)

    results = dict()
    results['w_mutual'] = {'inlier_ratios':[], 'distances':[]}
    results['wo_mutual'] = {'inlier_ratios':[], 'distances':[]}
    tsfm_est = []
    for eachfile in tqdm(feats_scores):
        ########################################
        # 1. take the input point clouds
        data = torch.load(eachfile)
        len_src =  data['len_src']
        pcd =  data['pcd']
        l_feats = data['l_feats']
        feats =  data['h_feats']
        rot, trans = data['rot'], data['trans']
        l_saliency, h_saliency, overlap = data['l_saliency'], data['h_saliency'], data['overlaps']

        src_raw = pcd[:len_src]
        tgt_raw = pcd[len_src:]
        src_feats = feats[:len_src]
        tgt_feats = feats[len_src:]
        src_overlap, src_l_saliency, src_h_saliency = overlap[:len_src], l_saliency[:len_src], h_saliency[:len_src]
        tgt_overlap, tgt_l_saliency, tgt_h_saliency = overlap[len_src:], l_saliency[len_src:], h_saliency[len_src:]
        src_h_saliency, tgt_h_saliency = adj_std(src_h_saliency, 0.37), adj_std(tgt_h_saliency, 0.37)
        ########################################
        # 2. do probabilistic sampling guided by the score
        src_scores = src_h_saliency * src_overlap 
        tgt_scores = tgt_h_saliency * tgt_overlap 

        src_idx = sample_interest_points(sample_method, src_scores, n_points)
        tgt_idx = sample_interest_points(sample_method, tgt_scores, n_points)
        src_pcd, src_feats = src_raw[src_idx], src_feats[src_idx]
        tgt_pcd, tgt_feats = tgt_raw[tgt_idx], tgt_feats[tgt_idx]
        
        # 3. global matching
        coarse = ransac_pose_estimation(src_pcd, tgt_pcd, src_feats, tgt_feats, mutual=False)
        est_rot, est_trans = coarse[:3, :3], coarse[:3, 3].reshape(-1, 1)
        # calculate inlier ratios
        inlier_ratio_results = get_inlier_ratio(src_pcd, tgt_pcd, src_feats, tgt_feats, rot, trans)
        #  tsfm_est.append(coarse)
        gt_rot, gt_trans = rot.numpy(), trans.numpy()
        # re0 = np.arccos(np.clip((np.trace(est_rot.T @ gt_rot) - 1) / 2.0, a_min=-1, a_max=1)) * 180 / np.pi
        # 4. local matching
        src_scores = src_overlap * src_l_saliency
        tgt_scores = tgt_overlap * tgt_l_saliency
        src_idx = sample_interest_points('prob', src_scores, 1000)
        tgt_idx = sample_interest_points('prob', tgt_scores, 1000)
        src_r, src_feats = src_raw[src_idx], l_feats[:len_src][src_idx]  
        tgt_r, tgt_feats = tgt_raw[tgt_idx], l_feats[len_src:][tgt_idx]

        src = (torch.matmul(to_tensor(est_rot.copy()), src_r.transpose(0,1)) + to_tensor(est_trans.copy()).view(-1,1)).transpose(0,1)
        # find knn in coordinate space, and find nn in feature space
        coord_dist = torch.sqrt(square_distance(src[None, :, :], tgt_r[None, :, :]).squeeze(0))
        candit = torch.topk(coord_dist, dim=1, k=50, largest=False)[1]
        feat_dist = ((src_feats[:,None,:] - tgt_feats[candit])**2).sum(dim=-1)
        index = feat_dist.argmin(dim=1).view(-1,1)
        src_corr = torch.gather(candit, index=index, dim=1).view(-1)
        tgt = tgt_r[src_corr]
        mask = torch.sqrt(((src-tgt)**2).sum(1)) < 0.1
        # src_p, tgt_p, weights = src[mask], tgt[mask], weights[mask]
        src_p, tgt_p = src[mask], tgt[mask]
        # weights = src_scores[src_idx][mask] * tgt_scores[tgt_idx][src_corr][mask]
        # weights = weights / weights.sum()
        # est = weighted_svd(src_p.unsqueeze(0),tgt_p.unsqueeze(0),weights.unsqueeze(0))
        est = rigid_transform_3d(src_p.unsqueeze(0),tgt_p.unsqueeze(0))
        f_rot, f_trans = est[:3,:3], est[:3,3].reshape(-1,1)
        est_rot = f_rot @ est_rot
        est_trans = f_rot @ est_trans + f_trans
        # re = np.arccos(np.clip((np.trace(est_rot.T @ gt_rot) - 1) / 2.0, a_min=-1, a_max=1)) * 180 / np.pi
        tsfm_est.append(to_tsfm(est_rot, est_trans))
        ########################################
        
        results['w_mutual']['inlier_ratios'].append(inlier_ratio_results['w']['inlier_ratio'])
        results['w_mutual']['distances'].append(inlier_ratio_results['w']['distance'])
        results['wo_mutual']['inlier_ratios'].append(inlier_ratio_results['wo']['inlier_ratio'])
        results['wo_mutual']['distances'].append(inlier_ratio_results['wo']['distance'])

    tsfm_est = np.array(tsfm_est)

    ########################################
    # wirte the estimated trajectories
    write_est_trajectory(gt_folder, exp_dir, tsfm_est)
    
    ########################################
    # evaluate the results, here FMR and Inlier ratios are all average twice
    benchmark(exp_dir, gt_folder)
    split = get_scene_split(whichbenchmark)

    for key in['w_mutual','wo_mutual']:
        inliers =[]
        fmrs = []

        for ele in split:
            c_inliers = results[key]['inlier_ratios'][ele[0]:ele[1]]
            inliers.append(np.mean(c_inliers))
            fmrs.append((np.array(c_inliers) > inlier_ratio_threshold).mean())

        with open(os.path.join(exp_dir,'result'),'a') as f:
            f.write(f'Inlier ratio {key}: {np.mean(inliers):.3f} : +- {np.std(inliers):.3f}\n')
            f.write(f'Feature match recall {key}: {np.mean(fmrs):.3f} : +- {np.std(fmrs):.3f}\n')
        f.close()


if __name__=='__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source_path', default='snapshot/indoor/3DMatch', type=str, help='path to precomputed features and scores')
    parser.add_argument(
        '--benchmark', default='3DMatch', type=str, help='[3DMatch, 3DLoMatch]')
    parser.add_argument(
        '--n_points', default=500, type=int, help='number of points used by RANSAC')
    parser.add_argument(
        '--exp_dir', default='est_traj', type=str, help='export final results')
    parser.add_argument(
        '--sampling', default='prob', type = str, help='interest point sampling')
    args = parser.parse_args()

    feats_scores = sorted(glob.glob(f'{args.source_path}/*.pth'), key=natural_key)

    benchmark_predator(feats_scores, args.n_points, args.exp_dir, args.benchmark, args.sampling)
