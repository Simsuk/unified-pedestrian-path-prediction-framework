import argparse
import logging
import os
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
# import utils 

from irl.utils import *
import irl.utils as utils
from irl.data.loader import data_loader
from irl.model_stgat import TrajectoryGenerator
# from irl.utils import (
#     displacement_error,
#     final_displacement_error,
#     get_dset_path,
#     int_tuple,
#     l2_loss,
#     relative_to_abs,
# )

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", default="./", help="Directory containing logging file")
"""arguments"""
parser = argparse.ArgumentParser(description='PyTorch Unified PPP framework')
parser.add_argument("--gpu-index", type=int, default=1, metavar='N')

parser.add_argument("--pretraining", default=True, help="pretraining in first 2 phases or not")

parser.add_argument("--model", default="stgat", help="The learning model method. Current models: original or stgat")

parser.add_argument('--randomness_definition', default='deterministic',  type=str, help='either stochastic or deterministic')
parser.add_argument('--step_definition', default='single',  type=str, help='either single or multi')
parser.add_argument('--loss_definition', default='discriminator',  type=str, help='either discriminator or l2')
parser.add_argument('--disc_type', default='original', type=str, help='either stgat or original')
parser.add_argument('--discount_factor', type=float, default=0.0, help='discount factor gamma, value between 0.0 and 1.0')
parser.add_argument('--multiple_executions', type=bool, default=False, help='turn multiple runs on or off')
parser.add_argument('--all_datasets', type=bool, default=True, help='run the script for all 5 datasets at once or not')

parser.add_argument('--log-std', type=float, default=-2.99, metavar='G', help='log std for the policy (default=-0.0)')


parser.add_argument("--dataset_name", default="hotel", type=str)
parser.add_argument("--delim", default="\t")
parser.add_argument("--loader_num_workers", default=16, type=int)
parser.add_argument("--obs_len", default=8, type=int)
parser.add_argument("--pred_len", default=8, type=int)
parser.add_argument("--skip", default=1, type=int)

parser.add_argument("--seed", type=int, default=0, help="Random seed.")
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--num_epochs", default=250, type=int)

parser.add_argument("--noise_dim", default=(16,), type=int_tuple)
parser.add_argument("--noise_type", default="gaussian")

parser.add_argument(
    "--traj_lstm_input_size", type=int, default=2, help="traj_lstm_input_size"
)
parser.add_argument("--traj_lstm_hidden_size", default=32, type=int)

parser.add_argument(
    "--heads", type=str, default="4,1", help="Heads in each layer, splitted with comma"
)
parser.add_argument(
    "--hidden-units",
    type=str,
    default="16",
    help="Hidden units in each hidden layer, splitted with comma",
)
parser.add_argument(
    "--graph_network_out_dims",
    type=int,
    default=32,
    help="dims of every node after through GAT module",
)
parser.add_argument("--graph_lstm_hidden_size", default=32, type=int)

parser.add_argument(
    "--dropout", type=float, default=0, help="Dropout rate (1 - keep probability)."
)
parser.add_argument(
    "--alpha", type=float, default=0.2, help="Alpha for the leaky_relu."
)


parser.add_argument(
    "--lr",
    default=1e-3,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)

parser.add_argument("--best_k", default=1, type=int)
parser.add_argument("--print_every", default=10, type=int)
parser.add_argument("--use_gpu", default=1, type=int)
parser.add_argument("--gpu_num", default="1", type=str)

parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)


best_ade = 100


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    train_path = get_dset_path(args.dataset_name, "train")
    val_path = get_dset_path(args.dataset_name, "test")

    logging.info("Initializing train dataset")
    train_dset, train_loader = data_loader(args, train_path)
    logging.info("Initializing val dataset")
    _, val_loader = data_loader(args, val_path)

    # writer = SummaryWriter()

    n_units = (
        [args.traj_lstm_hidden_size]
        + [int(x) for x in args.hidden_units.strip().split(",")]
        + [args.graph_lstm_hidden_size]
    )
    n_heads = [int(x) for x in args.heads.strip().split(",")]
    print(args.log_std)
    model =  TrajectoryGenerator(
        args,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        traj_lstm_input_size=args.traj_lstm_input_size,
        traj_lstm_hidden_size=args.traj_lstm_hidden_size,
        n_units=n_units,
        n_heads=n_heads,
        graph_network_out_dims=args.graph_network_out_dims,
        dropout=args.dropout,
        alpha=args.alpha,
        graph_lstm_hidden_size=args.graph_lstm_hidden_size,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        log_std=args.log_std,
        action_dim=2
        )
    model.cuda()
    optimizer = optim.Adam(
        [
            {"params": model.traj_lstm_model.parameters(), "lr": 1e-2},
            {"params": model.traj_hidden2pos.parameters()},
            {"params": model.gatencoder.parameters(), "lr": 3e-2},
            {"params": model.graph_lstm_model.parameters(), "lr": 1e-2},
            {"params": model.traj_gat_hidden2pos.parameters()},
            {"params": model.pred_lstm_model.parameters()},
            {"params": model.pred_hidden2pos.parameters()},
        ],
        lr=args.lr,
    )
    global best_ade
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("Restoring from checkpoint {}".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            logging.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    training_step = 1
    t0 = time.time()

    for epoch in range(args.start_epoch, args.num_epochs + 1):
        t1 = time.time()

        if epoch < 150:
            training_step = 1
        elif epoch < 250:
            training_step = 2
        else:
            if epoch == 250:
                # for param_group in optimizer.param_groups:
                #     param_group["lr"] = 5e-3
                print("working")
                save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_ade": best_ade,
                    "optimizer": optimizer.state_dict(),
                },
                epoch,
                args.pred_len,
                args,
                f"./pretrained_STGAT{args.dataset_name,args.best_k, epoch, args.pred_len, args.seed}.pth.tar")
            # training_step = 3
        train(args, model, train_loader, optimizer, epoch, training_step)
        # if training_step == 3:
        #     ade = validate(args, model, val_loader, epoch, writer)
        #     is_best = ade < best_ade
        #     best_ade = min(ade, best_ade)

        #     save_checkpoint(
        #         {
        #             "epoch": epoch + 1,
        #             "state_dict": model.state_dict(),
        #             "best_ade": best_ade,
        #             "optimizer": optimizer.state_dict(),
        #         },
        #         is_best,
        #         epoch,
        #         args.pred_len,
        #         f"./checkpoint/checkpoint{args.dataset_name,args.best_k, epoch, args.pred_len, args.seed}.pth.tar"
        #     )
        t2 = time.time()

        print_time(t0, t1, t2, epoch)
    # writer.close()


def train(args, model, train_loader, optimizer, epoch, training_step):
    losses = utils.AverageMeter("Loss", ":.6f")
    progress = utils.ProgressMeter(
        len(train_loader), [losses], prefix="Epoch: [{}]".format(epoch)
    )
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        batch = [tensor.cuda() for tensor in batch]
        (
            obs_traj,
            pred_traj_gt,
            obs_traj_rel,
            pred_traj_gt_rel,
            non_linear_ped,
            loss_mask,
            seq_start_end,
        ) = batch
        # print("seq_start_end",seq_start_end.shape, seq_start_end)

        # print("BATCH size", obs_traj_rel.shape[1])
        # print("obs_traj", obs_traj.shape)
        # print("pred_traj_gt", pred_traj_gt.shape)
        optimizer.zero_grad()
        loss = torch.zeros(1).to(pred_traj_gt)
        l2_loss_rel = []
        loss_mask = loss_mask[:, args.obs_len :]

        if training_step == 1 or training_step == 2:
            model_input = obs_traj_rel
            pred_traj_fake_rel, _, _ = model(
                model_input, obs_traj, seq_start_end, 1, training_step
            )
            # print("model_input", model_input.shape)
            # print("pred_traj_fake_rel", type(pred_traj_fake_rel))
            # model_input (8,b,2)
            # pred_traj_fake_rel (8,b,2)
            
            l2_loss_rel.append(
                l2_loss(pred_traj_fake_rel, model_input, loss_mask, mode="raw")
            )
        else:
            model_input = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)
            # model_input (20,b,2)
            # pred_traj_gt_rel (12,b,2)
            # pred_traj_fake_rel (12,b,2)            
            for _ in range(args.best_k):
                pred_traj_fake_rel = model(model_input, obs_traj, seq_start_end, 0)
                # print("model_input", model_input.shape)
                # print("seq_start_end", type(seq_start_end), seq_start_end)
                l2_loss_rel.append(
                    l2_loss(
                        pred_traj_fake_rel,
                        model_input[-args.pred_len :],
                        loss_mask,
                        mode="raw",
                    )
                )

        l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt) # list([(b)])
        # print("Original l2_loss_rel", len(l2_loss_rel), l2_loss_rel[0].shape)
        l2_loss_rel = torch.stack(l2_loss_rel, dim=1) # list b*[(1)]
        for start, end in seq_start_end.data:
            # print("l2_loss_rel", len(l2_loss_rel), l2_loss_rel[0].shape)
            _l2_loss_rel = torch.narrow(l2_loss_rel, 0, start, end - start)
            # print("_l2_loss_rel", _l2_loss_rel.shape) # [scene_trajectories,1]
            _l2_loss_rel = torch.sum(_l2_loss_rel, dim=0)  # [20]
            _l2_loss_rel = torch.min(_l2_loss_rel) / (
                (pred_traj_fake_rel.shape[0]) * (end - start)
            )                       # average per pedestrian per scene
            l2_loss_sum_rel += _l2_loss_rel

        loss += l2_loss_sum_rel
        losses.update(loss.item(), obs_traj.shape[1])
        # print("LOSS", loss)
        loss.backward()
        optimizer.step()
        if batch_idx % args.print_every == 0:
            progress.display(batch_idx)
    # writer.add_scalar("train_loss", losses.avg, epoch)


def validate(args, model, val_loader, epoch):
    ade = utils.AverageMeter("ADE", ":.6f")
    fde = utils.AverageMeter("FDE", ":.6f")
    progress = utils.ProgressMeter(len(val_loader), [ade, fde], prefix="Test: ")

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            batch = [tensor.cuda() for tensor in batch]
            (
                obs_traj,
                pred_traj_gt,
                obs_traj_rel,
                pred_traj_gt_rel,
                non_linear_ped,
                loss_mask,
                seq_start_end,
            ) = batch
  
            loss_mask = loss_mask[:, args.obs_len :]
            pred_traj_fake_rel = model(obs_traj_rel, obs_traj, seq_start_end)

            pred_traj_fake_rel_predpart = pred_traj_fake_rel[-args.pred_len :]
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel_predpart, obs_traj[-1])
            ade_, fde_ = cal_ade_fde(pred_traj_gt, pred_traj_fake)
            ade_ = ade_ / (obs_traj.shape[1] * args.pred_len)

            fde_ = fde_ / (obs_traj.shape[1])
            ade.update(ade_, obs_traj.shape[1])
            fde.update(fde_, obs_traj.shape[1])

            if i % args.print_every == 0:
                progress.display(i)

        logging.info(
            " * ADE  {ade.avg:.3f} FDE  {fde.avg:.3f}".format(ade=ade, fde=fde)
        )
        # writer.add_scalar("val_ade", ade.avg, epoch)
    return ade.avg


def cal_ade_fde(pred_traj_gt, pred_traj_fake):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    return ade, fde


def save_checkpoint(state, is_best,epoch,pred_len,  filename="checkpoint.pth.tar"):
    if is_best:
        torch.save(state, f"{args.checkpoint_dir}/model_best_{args.dataset_name, args.run}.pth.tar")
        logging.info("-------------- lower ade ----------------")
        # shutil.copyfile(filename, f"{args.checkpoint_dir}/model_best{args.dataset_name}.pth.tar")


if __name__ == "__main__":
    parser.add_argument('--runs', type=int, default=1, help='number of times the script runs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint_repetitions_1V_8steps', help='dir for checkpoint when improvement occurs')
    parser.add_argument('--run', type=int, default=1, help='run number')

    args = parser.parse_args()
    
    if args.all_datasets==False:
        
        # utils.set_logger(os.path.join(args.log_dir, "train.log"))
        checkpoint_dir = "./checkpoint_repetitions_1V_8steps"
        print("Dataset: " + args.dataset_name + ". Script execution number: {}".format(set, args.seed))
        if os.path.exists(checkpoint_dir) is False:
            os.mkdir(checkpoint_dir)
        main(args)
    else:
        for k in [ 0,3,4]:
            print("working")
            for i, set in enumerate([ 'zara1', 'univ', 'hotel']):
                args.dataset_name=set
                args.seed = k+5
                args.run=k
                print(args.seed)
                # utils.set_logger(os.path.join(args.log_dir, "train.log"))
                
                checkpoint_dir = "./pretrained_STGAT/kbest_1"
                print("path", f"{args.checkpoint_dir}/model_best_{args.dataset_name}_{args.run}.pth.tar")
                args.checkpoint_dir=checkpoint_dir
                print("Dataset: " + set + ". Script execution number: " + str(i))
                if os.path.exists(checkpoint_dir) is False:
                    os.mkdir(checkpoint_dir)
                main(args)
