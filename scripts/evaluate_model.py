import argparse
import os
import torch
import numpy as np
import random
import torch.nn.functional as F
import time
from collections import defaultdict
from attrdict import AttrDict
import os
metrics = defaultdict(list)
# Specify the directory you want to switch to
directory_path = '/home/ssukup/unified-pedestrian-path-prediction-framework/scripts'

# Change the current working directory
os.chdir(directory_path)

from irl.data.loader import data_loader
from irl.models import Policy
from irl.losses import displacement_error, final_displacement_error
from irl.utils import relative_to_abs, get_dset_path
from irl.model_stgat import TrajectoryGenerator
from irl.utils import (
    displacement_error,
    final_displacement_error,
    int_tuple,
    relative_to_abs,
    get_dset_path,
)



torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='eth', type=str)
parser.add_argument('--model_path', default="/home/ssukup/unified-pedestrian-path-prediction-framework/Results_SL_MSE_SPG/PPO_ONLY/superparam_search/lr_iters_10_or_15") # "../models/model_average" "/home/ssukup/unified-pedestrian-path-prediction-framework/Results_SL_MSE_SPG/with_pretrained_without_scene/1v1_12/saved_model_ADE_eth_run_0_Best_k_1_length_12.pt", type=str) #../models/model_average
parser.add_argument('--dset_type', default='test', type=str)
parser.add_argument('--model_average', default=True, type=bool)
parser.add_argument('--runs', type=int, default=1, help='number of models to compute average')
parser.add_argument('--prediction_steps', default=None, type=int)
parser.add_argument('--noise', default=False, type=bool)             # add noise to deterministic models to add stochasticity
parser.add_argument("--model", default="stgat", help="The learning model method. Current models: original or stgat")
parser.add_argument('--randomness_definition', default='stochastic',  type=str, help='either stochastic or deterministic')


# STGAT #######################################


parser.add_argument('--log-std', type=float, default=-2.99, metavar='G', help='log std for the policy (default=-0.0)')

parser.add_argument("--log_dir", default="./", help="Directory containing logging file")

parser.add_argument("--delim", default="\t")
parser.add_argument("--loader_num_workers", default=1, type=int)
parser.add_argument("--obs_len", default=8, type=int)
parser.add_argument("--pred_len", default=12, type=int)
parser.add_argument("--skip", default=1, type=int)

parser.add_argument("--seed", type=int, default=0, help="Random seed.")
parser.add_argument("--batch_size", default=64, type=int)

parser.add_argument("--noise_dim", default=(16,), type=int_tuple)
parser.add_argument("--noise_type", default="gaussian")
parser.add_argument("--noise_mix_type", default="global")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')

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


parser.add_argument("--num_samples", default=1, type=int)


parser.add_argument(
    "--dropout", type=float, default=0, help="Dropout rate (1 - keep probability)."
)
parser.add_argument(
    "--alpha", type=float, default=0.2, help="Alpha for the leaky_relu."
)

# parser.add_argument("--dset_type", default="test", type=str)


# parser.add_argument(
#     "--resume",
#     default= "/home/ssukup/checkpoint/checkpoint('zara1', 20, 266, 8).pth.tar", #"./model_best.pth.tar",
#     type=str,
#     metavar="PATH",
#     help="path to latest checkpoint (default: none)",
# )

################################################


# additional params
seeding = 73

if seeding is not None:
    torch.manual_seed(seeding)
    np.random.seed(seeding)
    random.seed(seeding)


def get_policy(args,model,checkpoint):
    if model=="original":
        _args = AttrDict(checkpoint['args'])
        policy_net = Policy(16, 2, log_std=_args.log_std)
        policy_net.load_state_dict(checkpoint['policy_net_state'])
        policy_net.cuda()
        policy_net.eval()
    elif model=="stgat":
        _args = AttrDict(checkpoint['args'])
        
        args.seed=_args.seed
        args.dataset_name=_args.dataset_name
        # print(args)
        n_units = (
        [_args.traj_lstm_hidden_size]
        + [int(x) for x in _args.hidden_units.strip().split(",")]
        + [_args.graph_lstm_hidden_size]
        )
        n_heads = [int(x) for x in _args.heads.strip().split(",")]
        policy_net = TrajectoryGenerator(
        _args,
        obs_len=_args.obs_len,
        pred_len=_args.pred_len,
        traj_lstm_input_size=_args.traj_lstm_input_size,
        traj_lstm_hidden_size=_args.traj_lstm_hidden_size,
        n_units=n_units,
        n_heads=n_heads,
        graph_network_out_dims=_args.graph_network_out_dims,
        dropout=_args.dropout,
        alpha=_args.alpha,
        graph_lstm_hidden_size=_args.graph_lstm_hidden_size,
        noise_dim=_args.noise_dim,
        noise_type=_args.noise_type,
        log_std=_args.log_std
        )
        
        policy_net.load_state_dict(checkpoint["policy_net_state"], strict=False
)

        policy_net.cuda()
        policy_net.eval()
    return policy_net


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def compute_av_std(ADE, FDE, minADE, minFDE):
    ADE = torch.stack(ADE)
    FDE = torch.stack(FDE)
    minADE = torch.stack(minADE)
    minFDE = torch.stack(minFDE)
    # print("ADEEEEEEEEEEEEEEE", ADE)
    av_ade = torch.mean(ADE)
    std_ade = torch.std(ADE, unbiased=False)
    low_ade = torch.min(ADE)

    av_fde = torch.mean(FDE)
    std_fde = torch.std(FDE, unbiased=False)
    low_fde = torch.min(FDE)
    print("ADE", ADE)
    print("minFDE", minFDE)
    av_min_ade = torch.mean(minADE)
    std_min_ade = torch.std(minADE, unbiased=False)
    low_min_ade = torch.min(minADE)

    av_min_fde = torch.mean(minFDE)
    std_min_fde = torch.std(minFDE, unbiased=False)
    low_min_fde = torch.min(minFDE)

    return av_ade, av_fde, av_min_ade, av_min_fde, std_ade, std_fde, std_min_ade, std_min_fde, low_ade, low_fde, low_min_ade, low_min_fde


def add_noise_state(args,state, device):
    # add noise from gaussian to state (b,16)
    noise_dim = 2
    noise_std = 0.05
    pad_dim = state.shape[1] - noise_dim
    if args.model=="stgat": 
        noise_shape = (state.shape[0], noise_dim)
        noise = torch.randn(noise_shape).to(device)
        # print(noise)
        noise = noise * noise_std
        pad = (pad_dim, 0, 0, 0)
        noise = F.pad(noise, pad, "constant", 0)  # effectively zero padding
    else:
        noise_shape = (state.shape[0], noise_dim)
        noise = torch.randn(noise_shape).to(device)
        noise = noise * noise_std

        pad = (pad_dim, 0, 0, 0)
        noise = F.pad(noise, pad, "constant", 0)  # effectively zero padding

    state = state + noise
    return state

def create_fake_trajectories(env,args,obs_traj_rel, pred_traj_gt_rel,seq_start_end,policy, pred_len, device, randomness_definition, mean_action, noise, seed):
    obs_len = obs_traj_rel.shape[0]
    state = obs_traj_rel.permute(1, 0, 2)
    state = torch.flatten(state, 1, 2)
    fake_traj = state                      # (b, 16) in x,y,x,y format
    if args.model=="original":
        for step in range(pred_len):
            if mean_action is False and randomness_definition == 'stochastic':
                action = policy.select_action(state)
            else:
                if noise:
                    state = add_noise_state(args, state, device)                                          # this is to add noise remove if no worky or make better if worky
                action, _, _ = policy(state)    #meanaction for deterministic
            fake_traj = torch.cat((fake_traj, action), dim=1)
            next_state = torch.cat((state, action), dim=1)[:, -obs_len * 2:]
            state = next_state

        fake_traj = fake_traj[:,-pred_len*2:]   # (b, 24)

        fake_traj = torch.reshape(fake_traj, (fake_traj.shape[0], pred_len, 2))  # (b, 12, 2)
        pred_traj_fake_rel = fake_traj.permute(1, 0, 2)                          # (12, b, 2)
    elif args.model=="stgat":
            # happens only in training_step==3 because only then we generate 12 trajectories
            
            # print(model_input.shape)
            # print("stohcasticity:", randomness_definition)
            if mean_action==False:
                model_input=torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)

                pred_traj_fake_rel  = policy.select_action(model_input, obs_traj_rel,seq_start_end, seed,3)
                # print("mistake")
            else:
                # obs_traj_rel = add_noise_state(args, obs_traj_rel, device) 
                model_input=torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)
                 
                pred_traj_fake_rel, _, _ = policy(model_input, obs_traj_rel,seq_start_end ,0, 3)  
                # print("output", pred_traj_fake_rel[0])

    return pred_traj_fake_rel   # (12, b, 2)


def evaluate_irl(env,args, loader, policy_net, num_samples, mean_action, noise, device):
    ade_outer, fde_outer = [], []
    total_traj = 0
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)
            # print(num_samples)
            for i in range(num_samples):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                # seed = time.time_ns() + i
                seed=args.seed
                # torch.manual_seed(seed)                
                pred_traj_fake_rel = create_fake_trajectories(env,args,obs_traj_rel, pred_traj_gt_rel, seq_start_end,policy_net, args.pred_len, device, args.randomness_definition, mean_action, noise, seed)
                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]
                )

                ade.append(displacement_error(
                    pred_traj_fake, pred_traj_gt, mode='raw'
                ))
                fde.append(final_displacement_error(
                    pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                ))

            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)
        # print(ade_outer)
        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / (total_traj)
        return ade, fde

class fake_env():
    def __init__(self, training_step=3):
        self.training_step=training_step
def main(args):
    if args.model_average:
        args.model_path ="/home/ssukup/unified-pedestrian-path-prediction-framework/Results_SL_MSE_SPG/PPO_ONLY/BEST_FINAL_RESULTS/saved_model_ADE_zara2_run_2_Policy_1_length_12.pt"# "../models/model_average" #"/home/ssukup/unified-pedestrian-path-prediction-framework/Results_SL_MSE_SPG/with_pretrained_without_scene/1v1_12/saved_model_ADE_eth_run_0_Best_k_1_length_12.pt" #"../models/model_average" # "/home/ssukup/unified-pedestrian-path-prediction-framework/Results_SL_MSE_SPG/with_pretrained_without_scene/1v1_12/saved_model_ADE_eth_run_0_Best_k_1_length_12.pt" # 
        ADE = []
        FDE = []
        minADE = []
        minFDE = []
    else:
        args.model_path = '/home/ssukup/unified-pedestrian-path-prediction-framework/Results_SL_MSE_SPG/PPO_ONLY/saved_model_ADE_zara1_run_0_Best_k_1_length_12.pt'#'/home/ssukup/unified-pedestrian-path-prediction-framework/Results_SL_MSE_SPG/Baseline_with_pretrained/1v1_12/saved_model_ADE_eth_run_0_Best_k_1_length_12.pt' #"/home/ssukup/unified-pedestrian-path-prediction-framework/Results_SL_MSE_SPG/with_pretrained_without_scene/1v1_12/saved_model_ADE_zara1_run_0_Best_k_1_length_12.pt" #"../models/irl-models" # "/home/ssukup/unified-pedestrian-path-prediction-framework/Results_SL_MSE_SPG/Baseline_with_pretrained/1v1_12/saved_model_ADE_eth_run_0_Best_k_1_length_12.pt"# #"/home/ssukup/unified-pedestrian-path-prediction-framework/Results_SL_MSE_SPG/with_pretrained_without_scene/1v1_12/saved_model_ADE_eth_run_0_Best_k_1_length_12.pt"  #"/home/ssukup/unified-pedestrian-path-prediction-framework/Results_SL_MSE_SPG/Baseline_with_pretrained/1v1_12/hyperparam_opt/saved_model_ADE_eth_trial_run_3_Best_k_0.0005709531017751183_length_0_bs_64.pt"# "/home/ssukup/unified-pedestrian-path-prediction-framework/Results_SL_MSE_SPG/Baseline_with_pretrained/1v1_12/saved_model_ADE_eth_run_0_Best_k_1_length_12.pt"#"../models/irl-models" #"/home/ssukup/unified-pedestrian-path-prediction-framework/Results_SL_MSE_SPG/with_pretrained_without_scene/1v1_12/saved_model_ADE_eth_run_0_Best_k_1_length_12.pt"  #
        list_to_sort = []

    # if seeding:
    #     torch.manual_seed(seeding)
    #     np.random.seed(seeding)


    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
            if os.path.isfile(os.path.join(args.model_path, file_)) and file_.endswith('.pt')
        ]
        # print(paths)
    else:
        if args.model_path.endswith('.pt'):
            paths = [args.model_path]
        else:
            paths = []
    env=fake_env()
    print("Evaluating IRL model \n")
    path_counter = 0
    for path in paths:
        path_counter = path_counter + 1
        # print("PATH", path)
        checkpoint = torch.load(path)
        # checkpoint["policy_net_state"].pop('action_log_std', None)
       
        policy_net = get_policy(args,args.model,checkpoint)
        # print("lr", checkpoint['lr'])
        _args = AttrDict(checkpoint['args'])
        print("LR", _args.lr)
        print("EPOCH", checkpoint['epoch'])
        _args.randomness_definition='deterministic'
        _args.pretraining=False
        # print("VALUE", _args.learning_rate)
        # print("ppo_clip", _args.ppo_clip)
        # print("optim_value_iternum", _args.optim_value_iternum)
        # print("l2_reg", _args.l2_reg)
        seeding = _args.seed
        # print("SEED", seeding)
        # args.seed= _args.seed
        # if seeding is not None:
        # torch.manual_seed(seeding)
        # np.random.seed(seeding)
        # random.seed(seeding)

        # print("_args.prediction_steps", _args.best_k)
        if args.prediction_steps is not None:
            _args.pred_len = args.prediction_steps
        # print("PRED ELN", _args.pred_len)
        if args.model_average is False:
            print("\n path: ", path)
        temp_path = path
        model_name = path.split("saved_model_")[1].split(".p")[0].ljust(30)
        path = get_dset_path(_args.dataset_name, args.dset_type)
        _, loader = data_loader(_args, path)
        print(_args.randomness_definition)
        ade, fde = evaluate_irl(env,_args, loader, policy_net, 1, mean_action=True, noise=False, device=device)
        metrics['ade'].append(ade)
        metrics['fde'].append(fde)


        if args.model_average is False:
            print('Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(
                _args.dataset_name, _args.pred_len, ade, fde))
        if args.model_average:
            ADE.append(ade)
            FDE.append(fde)

        ## best k predicting
        min_ade, min_fde = evaluate_irl(env,_args, loader, policy_net, 20, mean_action=False, noise=args.noise, device=device)
        metrics['min_ade'].append(min_ade)
        metrics['min_fde'].append(min_fde)
        if args.model_average is False:
            print('Dataset: {}, Pred Len: {}, k: {}, minADE: {:.2f}, minFDE: {:.2f}'.format(
                _args.dataset_name, _args.pred_len, args.num_samples, min_ade, min_fde))
        if args.model_average:
            minADE.append(min_ade)
            minFDE.append(min_fde)
        else:
            ade_2d = "{:.2f}".format(ade.item())
            fde_2d = "{:.2f}".format(fde.item())
            minade_2d = "{:.2f}".format(min_ade.item())
            minfde_2d = "{:.2f}".format(min_fde.item())
            list_entry = (model_name, ade_2d, fde_2d, minade_2d, minfde_2d)
            list_to_sort.append(list_entry)

        if args.model_average is True and path_counter == args.runs:
            print("\nAverage for path: ", temp_path)
            av_ade, av_fde, av_min_ade, av_min_fde, std_ade, std_fde, std_min_ade, std_min_fde, low_ade, low_fde, low_min_ade, low_min_fde = compute_av_std(ADE, FDE, minADE, minFDE)
            
            print('Dataset: {}, Pred Len: {}, mean ADE: {:.4f}, std ADE: {:.2f}, low ADE: {:.4f}, mean FDE: {:.4f}, std FDE: {:.2f}, low FDE: {:.2f}'.format(
                _args.dataset_name, _args.pred_len, av_ade, std_ade, low_ade, av_fde, std_fde, low_fde))
            print('Dataset: {}, Pred Len: {}, k: {}, mean minADE: {:.4f}, std minADE: {:.4f}, low minADE: {:.2f}, mean minFDE: {:.4f}, std minFDE: {:.2f}, low minFDE: {:.2f}'.format(
                _args.dataset_name, _args.pred_len, args.num_samples, av_min_ade, std_min_ade, low_min_ade, av_min_fde, std_min_fde, low_min_fde))

            path_counter = 0
            ADE = []
            FDE = []
            minADE = []
            minFDE = []
            if seeding:
                torch.manual_seed(seeding)
                np.random.seed(seeding)
            print("=================================================")

    values= [values for metric, values in metrics.items()]
    print(values)
    average_results = {metric: sum(values) / len(values) for metric, values in metrics.items()}
    print(f"Average ADE: {average_results['ade']:.4f}")
    print(f"Average minADE: {average_results['min_ade']:.4f}")
    print(f"Average FDE: {average_results['fde']:.4f}")
    print(f"Average minFDE: {average_results['min_fde']:.4f}")
if __name__ == '__main__':
    
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print("device: ", device)
    main(args)
