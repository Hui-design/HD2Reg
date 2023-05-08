import pdb

import torch, os, sys, glob
cwd = os.getcwd()
sys.path.append(cwd)
from tqdm import tqdm
import numpy as np
from lib.utils import load_obj, natural_key,setup_seed, square_distance
from lib.benchmark_utils import ransac_pose_estimation, to_tsfm, get_inlier_ratio, get_scene_split, write_est_trajectory, mutual_selection
import open3d as o3d
from lib.benchmark import read_trajectory, write_trajectory, benchmark
import argparse
import pdb
import cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors
from datasets.dataloader import batch_grid_subsampling_kpconv
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

    return choice

def my_nms(pts, feats, score, radius=0.1, sample_n=500):
    # pdb.set_trace()
    raw_list = torch.tensor([len(pts)]).int()
    pool_p, pool_b, pool_f = batch_grid_subsampling_kpconv(pts, raw_list, features=feats, sampleDl=radius)
    pool_list = torch.tensor([len(pool_p)]).int()
    neighbor = cpp_neighbors.batch_query(pool_p.to(torch.device("cpu")), pts.to(torch.device("cpu")), pool_list,
                                         raw_list, radius=radius)

    score = np.concatenate((score, [-1]), axis=0)
    neighbor_score = score[neighbor]  # [N,n']

    column_idx = np.argmax(neighbor_score,axis=1)  # N
    local_idx = neighbor[np.arange(neighbor.shape[0]), column_idx] # N

    nms_score = np.max(neighbor_score, axis=1)
    global_idx = np.argsort(nms_score)[::-1][:sample_n]  # sample_n
    idx = local_idx[global_idx]

    return pts[idx], feats[idx]


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
    #rep_num_list = [4,8,16,32,64,128,256,512]
    rep_num_list = [500]
    rep_list = []
    for rep_num in rep_num_list:
        rep_list_i = []
        for i, eachfile in tqdm(enumerate(feats_scores)):
            ########################################
            # 1. take the input point clouds
            data = torch.load(eachfile)
            len_src =  data['len_src']
            pcd =  data['pcd']
            feats =  data['feats']
            rot, trans = data['rot'], data['trans']
            l_saliency, h_saliency, overlap = data['l_saliency'], data['h_saliency'], data['overlaps']

            src_pcd = pcd[:len_src]
            tgt_pcd = pcd[len_src:]
            src_feats = feats[:len_src]
            tgt_feats = feats[len_src:]
            src_overlap, src_l_saliency, src_h_saliency = overlap[:len_src], l_saliency[:len_src], h_saliency[:len_src]
            tgt_overlap, tgt_l_saliency, tgt_h_saliency = overlap[len_src:], l_saliency[len_src:], h_saliency[len_src:]
            src_score = src_overlap * src_h_saliency
            tgt_score = tgt_overlap * tgt_h_saliency
            src_idx = sample_interest_points('topk', src_score, rep_num)
            tgt_idx = sample_interest_points('topk', tgt_score, rep_num)
            src_pcd_s = src_pcd[src_idx]
            src_pcd_s = (torch.matmul(rot, src_pcd_s.transpose(0, 1)) + trans).transpose(0, 1)
            tgt_pcd_s = tgt_pcd[tgt_idx]
            coords_dist_s = torch.sqrt(square_distance(src_pcd_s[None, :, :], tgt_pcd_s[None, :, :]).squeeze(0))
            pos_mask_s = coords_dist_s < 0.1
            rep = (pos_mask_s.sum(1) > 0).sum() / pos_mask_s.shape[0]
            rep_list_i.append(rep)
   
        rep_list.append(np.mean(rep_list_i))
        print(rep_list)
        print(np.percentile(rep_list_i, 20))
        print(np.percentile(rep_list_i, 10))
        print(np.percentile(rep_list_i, 5))
    pdb.set_trace()
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
        '--source_path', default='snapshot/indoor/3DLoMatch', type=str, help='path to precomputed features and scores')
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
