"""optimize over a network structure."""

import argparse
import logging
import os
import copy

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Neural_Prior
import config
from data import (ArgoverseSceneFlowDataset, KITTISceneFlowDataset,
                  NuScenesSceneFlowDataset, FlyingThings3D)
from utils import scene_flow_metrics, Timers, GeneratorWrap, EarlyStopping
from loss import my_chamfer_fn
from visualize import show_flows, flow_to_rgb, custom_draw_geometry_with_key_callback


device = torch.device("cuda:0")


def solver(
    pc1: torch.Tensor,
    pc2: torch.Tensor,
    flow: torch.Tensor,
    options: argparse.Namespace,
    net: torch.nn.Module,
    i: int,
):

    for param in net.parameters():
        param.requires_grad = True
    
    if options.backward_flow:
        net_inv = copy.deepcopy(net)
        params = [{'params': net.parameters(), 'lr': options.lr, 'weight_decay': options.weight_decay},
                {'params': net_inv.parameters(), 'lr': options.lr, 'weight_decay': options.weight_decay}]
    else:
        params = net.parameters()
    
    if options.optimizer == "sgd":
        print('using SGD.')
        optimizer = torch.optim.SGD(params, lr=options.lr, momentum=options.momentum, weight_decay=options.weight_decay)
    elif options.optimizer == "adam":
        print("Using Adam optimizer.")
        optimizer = torch.optim.Adam(params, lr=options.lr, weight_decay=0)

    total_losses = []
    chamfer_losses = []

    early_stopping = EarlyStopping(patience=options.early_patience, min_delta=0.0001)

    if options.time:
        timers = Timers()
        timers.tic("solver_timer")

    pc1 = pc1.cuda().contiguous()
    pc2 = pc2.cuda().contiguous()
    flow = flow.cuda().contiguous()

    normal1 = None
    normal2 = None

    # ANCHOR: initialize best metrics
    best_loss_1 = 10.
    best_flow_1 = None
    best_epe3d_1 = 1.
    best_acc3d_strict_1 = 0.
    best_acc3d_relax_1 = 0.
    best_angle_error_1 = 1.
    best_outliers_1 = 1.
    best_epoch = 0
    
    for epoch in range(options.iters):
        optimizer.zero_grad()

        flow_pred_1 = net(pc1)
        pc1_deformed = pc1 + flow_pred_1
        loss_chamfer_1, _ = my_chamfer_fn(pc2, pc1_deformed, normal2, normal1)
        
        if options.backward_flow:
            flow_pred_1_prime = net_inv(pc1_deformed)
            pc1_prime_deformed = pc1_deformed - flow_pred_1_prime
            loss_chamfer_1_prime, _ = my_chamfer_fn(pc1_prime_deformed, pc1, normal2, normal1)
        
        if options.backward_flow:
            loss_chamfer = loss_chamfer_1 + loss_chamfer_1_prime
        else:
            loss_chamfer = loss_chamfer_1

        loss = loss_chamfer

        flow_pred_1_final = pc1_deformed - pc1
        
        if options.compute_metrics:
            EPE3D_1, acc3d_strict_1, acc3d_relax_1, outlier_1, angle_error_1 = scene_flow_metrics(flow_pred_1_final, flow)
        else:
            EPE3D_1, acc3d_strict_1, acc3d_relax_1, outlier_1, angle_error_1 = 0, 0, 0, 0, 0

        # ANCHOR: get best metrics
        if loss <= best_loss_1:
            best_loss_1 = loss.item()
            best_epe3d_1 = EPE3D_1
            best_flow_1 = flow_pred_1_final
            best_epe3d_1 = EPE3D_1
            best_acc3d_strict_1 = acc3d_strict_1
            best_acc3d_relax_1 = acc3d_relax_1
            best_angle_error_1 = angle_error_1
            best_outliers_1 = outlier_1
            best_epoch = epoch
            
        if epoch % 50 == 0:
            logging.info(f"[Sample: {i}]"
                        f"[Ep: {epoch}] [Loss: {loss:.5f}] "
                        f" Metrics: flow 1 --> flow 2"
                        f" [EPE: {EPE3D_1:.3f}] [Acc strict: {acc3d_strict_1 * 100:.3f}%]"
                        f" [Acc relax: {acc3d_relax_1 * 100:.3f}%] [Angle error (rad): {angle_error_1:.3f}]"
                        f" [Outl.: {outlier_1 * 100:.3f}%]")
            
        total_losses.append(loss.item())
        chamfer_losses.append(loss_chamfer)

        if options.animation:
            yield flow_pred_1_final.detach().cpu().numpy()

        if early_stopping.step(loss):
            break
        
        loss.backward()
        optimizer.step()

    if options.time:
        timers.toc("solver_timer")
        time_avg = timers.get_avg("solver_timer")
        logging.info(timers.print())

    # ANCHOR: get the best metrics
    info_dict = {
        'loss': best_loss_1,
        'EPE3D_1': best_epe3d_1,
        'acc3d_strict_1': best_acc3d_strict_1,
        'acc3d_relax_1': best_acc3d_relax_1,
        'angle_error_1': best_angle_error_1,
        'outlier_1': best_outliers_1,
        'time': time_avg,
        'epoch': best_epoch
    }

    # NOTE: visualization
    if options.visualize:
        fig = plt.figure(figsize=(13, 5))
        ax = fig.gca()
        ax.plot(total_losses, label="loss")
        ax.legend(fontsize="14")
        ax.set_xlabel("Iteration", fontsize="14")
        ax.set_ylabel("Loss", fontsize="14")
        ax.set_title("Loss vs iterations", fontsize="14")
        plt.show()

        idx = 0   
        show_flows(pc1[idx], pc2[idx], best_flow_1[idx])
        
        # ANCHOR: new plot style
        pc1_o3d = o3d.geometry.PointCloud()
        colors_flow = flow_to_rgb(flow[0].cpu().numpy().copy())
        pc1_o3d.points = o3d.utility.Vector3dVector(pc1[0].cpu().numpy().copy())
        pc1_o3d.colors = o3d.utility.Vector3dVector(colors_flow / 255.0)
        custom_draw_geometry_with_key_callback([pc1_o3d])  # Press 'k' to see with dark background.
        
    return info_dict


def optimize_neural_prior(options, data_loader):
    if options.time:
        timers = Timers()
        timers.tic("total_time")

    save_dir_path = f"checkpoints/{options.exp_name}"

    outputs = []
    
    if options.model == 'neural_prior':
        net = Neural_Prior(filter_size=options.hidden_units, act_fn=options.act_fn, layer_size=options.layer_size).cuda()
    else:
        raise Exception("Model not available.")    

    for i, data in tqdm(enumerate(data_loader), total=len(data_loader), smoothing=0.9):
        logging.info(f"# Working on sample: {data_loader.dataset.datapath[i]}...")

        pc1, pc2, flow = data
        
        if options.visualize:
            idx = 0
            # NOTE: ground truth flow
            show_flows(pc1[idx], pc2[idx], flow[idx])

        solver_generator = GeneratorWrap(solver(pc1, pc2, flow, options, net, i))
        
        if options.animation:
            #TODO: save frames to make video.
            info_dict = solver_generator.value
        else:
            for _ in solver_generator: pass
            info_dict = solver_generator.value

        # Collect results.
        info_dict['filepath'] = data_loader.dataset.datapath[i]
        outputs.append(info_dict)

        print(info_dict)

    if options.time:
        timers.toc("total_time")
        time_avg = timers.get_avg("total_time")
        logging.info(timers.print())

    df = pd.DataFrame(outputs)
    df.loc['mean'] = df.mean()
    logging.info(df.mean())
    df.loc['total time'] = time_avg
    df.to_csv('{:}.csv'.format(f"{save_dir_path}/results"))

    logging.info("Finish optimization!")
    
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Scene Flow Prior.")
    config.add_config(parser)
    options = parser.parse_args()

    exp_dir_path = f"checkpoints/{options.exp_name}"
    if not os.path.exists(exp_dir_path):
        os.makedirs(exp_dir_path)

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[logging.FileHandler(filename=f"{exp_dir_path}/run.log"), logging.StreamHandler()])
    logging.info(options)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    logging.info('---------------------------------------')
    print_options = vars(options)
    for key in print_options.keys():
        logging.info(key+': '+str(print_options[key]))
    logging.info('---------------------------------------')

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(options.seed)
    torch.cuda.manual_seed_all(options.seed)
    np.random.seed(options.seed)

    if options.dataset == "KITTISceneFlowDataset":
        data_loader = DataLoader(
            KITTISceneFlowDataset(options=options, train=False),
            batch_size=options.batch_size, shuffle=False, drop_last=False, num_workers=12
        )
    elif options.dataset == "FlyingThings3D":
        data_loader = DataLoader(
            FlyingThings3D(options=options, partition="test"),
            batch_size=options.batch_size, shuffle=False, drop_last=False, num_workers=12
        )
    elif options.dataset == "ArgoverseSceneFlowDataset":
        data_loader = DataLoader(
            ArgoverseSceneFlowDataset(options=options, partition=options.partition),
            batch_size=options.batch_size, shuffle=False, drop_last=False, num_workers=12
        )
    elif options.dataset == "NuScenesSceneFlowDataset":
        data_loader = DataLoader(
            NuScenesSceneFlowDataset(options=options, partition="val"),
            batch_size=options.batch_size, shuffle=False, drop_last=False, num_workers=12
        )

    optimize_neural_prior(options, data_loader)
