import argparse

import os
import random
from memory_profiler import profile
from functools import partial
import cProfile
import re
# from optuna.integration.tensorboard import TensorBoardCallback

import pstats
from cProfile import Profile
from pstats import SortKey, Stats
import logging
import gc
import sys
import optuna
class Profile:
    def __init__(self):
        self.prof = cProfile.Profile()

    def __enter__(self):
        self.prof.enable()
        return self.prof

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.prof.disable()
        self.prof.print_stats(sort='time')
    
    
def log_exception(exc_type, exc_value, exc_traceback):
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
# Set the global exception hook
# sys.excepthook = log_exception

# Configure logging
# logging.basicConfig(level=logging.ERROR, filename='errors.log')

from irl.utils import *
from irl.models import Policy, Discriminator, Value, TrajectoryDiscriminator
from irl.model_stgat import TrajectoryGenerator
from irl.replay_memory import Memory
from torch import nn
from irl.data.loader import data_loader
from irl.update_parameters import discriminator_step, reinforce_step, generator_step
from irl.accuracy import check_accuracy
from irl.agent import Agent
from irl.environment import Environment
from irl.discriminator_STGAT import STGAT_discriminator, Discriminator_LSTM
import torch.optim as optim
import gc
import logging
import psutil
from torch.utils.tensorboard import SummaryWriter
import psutil
 
torch.set_num_threads(32)
from scripts.evaluate_model import evaluate_irl


def initialize_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=0.001)  # gain=0.01 to make weights smaller
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
class DiscriminatorManager(nn.Module):
        def __init__(self, discriminator_early, discriminator_late, env):
            super(DiscriminatorManager, self).__init__()
            self.discriminator_early = discriminator_early
            self.discriminator_late = discriminator_late
            self.env = env
            self.current_discriminator = discriminator_early

        def forward(self, x):
            # Switch the model if necessary
            if self.env.training_step == 3:
                self.current_discriminator = self.discriminator_late
            return self.current_discriminator(x)

        def to(self, *args, **kwargs):
            super().to(*args, **kwargs)
            # Move both models to the specified device
            self.discriminator_early.to(*args, **kwargs)
            self.discriminator_late.to(*args, **kwargs)
            return self


"""arguments"""
parser = argparse.ArgumentParser(description='PyTorch Unified PPP framework')

parser.add_argument("--model", default="stgat", help="The learning model method. Current models: original or stgat")
parser.add_argument("--pretraining", default=False, help="pretraining in first 2 phases or not")
parser.add_argument("--l2_reg", default=0.0, help="PPO_regularization")

parser.add_argument("--reward", default="cumulative", help="type of reward/action definition, either cumulative or window (rolling window)")

parser.add_argument('--randomness_definition', default='stochastic',  type=str, help='either stochastic or deterministic')
parser.add_argument('--step_definition', default='single',  type=str, help='either single or multi')
parser.add_argument('--loss_definition', default='discriminator',  type=str, help='either discriminator or l2')
parser.add_argument('--disc_type', default='original', type=str, help='either stgat or original')
parser.add_argument('--discount_factor', type=float, default=1, help='discount factor gamma, value between 0.0 and 1.0')
parser.add_argument('--optim_value_iternum', type=int, default=1, help='minibatch size')

parser.add_argument('--training_algorithm', default='reinforce',  type=str, help='choose which RL updating algorithm, either "reinforce", "baseline" or "ppo" or "ppo_only"')
parser.add_argument('--trainable_noise', type=bool, default=False, help='add a noise to the input during training')
parser.add_argument('--ppo-iterations', type=int, default=10, help='number of ppo iterations (default=1)')
parser.add_argument('--ppo-clip', type=float, default=0.2, help='amount of ppo clipping (default=0.2)')
parser.add_argument('--learning-rate', type=float, default=0.001, metavar='G', help='learning rate (default: 1e-5)')
parser.add_argument('--batch_size', default=64, type=int, help='number of sequences in a batch (can be multiple paths)')
parser.add_argument('--log-std', type=float, default=-2.99, metavar='G', help='log std for the policy (default=-0.0)')
parser.add_argument('--num_epochs', default=330, type=int, help='number of times the model sees all data')

parser.add_argument('--seeding', type=bool, default=True, help='turn seeding on or off')
parser.add_argument('--seed', type=int, default=73, metavar='N', help='random seed (default: 0)')
parser.add_argument('--multiple_executions', type=bool, default=True, help='turn multiple runs on or off')
parser.add_argument('--runs', type=int, default=5, help='number of times the script runs')
parser.add_argument('--all_datasets', type=bool, default=True, help='run the script for all 5 datasets at once or not')
parser.add_argument('--dataset_name', default='hotel',  type=str, help='choose which dataset to train for')
parser.add_argument('--check_testset', type=bool, default=True, help='also evaluate on the testset, next to validation set')
parser.add_argument('--output_dir', default=os.getcwd(), help='path where models are saved (default=current directory)')
parser.add_argument('--save_model_name_ADE', default="saved_model_ADE", help='name of the saved model')
parser.add_argument('--save_model_name_FDE', default="saved_model_FDE", help='name of the saved model')
parser.add_argument('--num_samples_check', default=5000, type=int, help='limit the nr of samples during metric calculation')
parser.add_argument('--check_validation_every', default=1, type=int, help='check the metrics on the validation dataset every X epochs')

parser.add_argument('--obs_len', default=8, type=int, help='how many timesteps used for observation (default=8)')
parser.add_argument('--pred_len', default=12, type=int, help='how many timesteps used for prediction (default=12)')
parser.add_argument('--discriminator_steps', default=1, type=int, help='how many discriminator updates per iteration')
parser.add_argument('--policy_steps', default=1, type=int, help='how many policy updates per iteration')
parser.add_argument('--loader_num_workers', default=16, type=int, help='number cpu/gpu processes (default=0)')
parser.add_argument('--skip', default=1, type=int, help='used for skipping sequences (default=1)')
parser.add_argument('--delim', default='\t', help='how to read the data text file spacing')
parser.add_argument('--l2_loss_weight', default=1, type=float, help='l2 loss multiplier (default=0)')
parser.add_argument('--use_gpu', default=1, type=int)                   # use gpu, if 0, use cpu only
parser.add_argument('--gpu-index', type=int, default=1, metavar='N')
parser.add_argument('--load_saved_model', default=None, metavar='G', help='path of pre-trained model')
#STGAT ==========================================================

parser.add_argument('--scene_value_net', type=float, default=False, help='Decides whether the value network consideres whole scene or single pedestrian, boolean value')

parser.add_argument("--log_dir", default="./", help="Directory containing logging file")

parser.add_argument("--noise_dim", default=(16,), type=int_tuple)
parser.add_argument("--noise_type", default="gaussian")
parser.add_argument("--training_step", default=1)

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
    "--dropout", type=float, default=0.0, help="Dropout rate (1 - keep probability)."
)
parser.add_argument(
    "--alpha", type=float, default=0.2, help="Alpha for the leaky_relu."
)
parser.add_argument(
    "--lr",
    default=0.0005709531017751183, #=1e-3
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
parser.add_argument("--gpu_num", default="1", type=str)

parser.add_argument(
    "--resume",
    default="", #/home/ssukup/unified-pedestrian-path-prediction-framework/Results_GAN/1v1_12/saved_model_ADE_eth_run_1_Best_k_1_length_12.pt
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
best_ade = 100
# ================================================================


args = parser.parse_args()

def log_memory_usage(message):
    memory = psutil.Process().memory_info().rss / (1024 * 1024)  # Memory in MB
    # logging.info(f"{message}: Memory usage is {memory} MB")
    cuda_memory = torch.cuda.memory_allocated()
    print(f"Memory RAM usage is {memory} MB and CUDA {cuda_memory}")
def main_loop(args, writer, metric_dict, hparams, trial):
    """"definitions"""
    if args.randomness_definition == 'stochastic':
        mean_action = False
    elif args.randomness_definition == 'deterministic':
        mean_action = True
    else:
        print("Wrong definition for randomness, please choose either stochastic or deterministic")

    """seeding"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """dtype and device"""
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    print("device: ", device)
    global mid_pad
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    """models"""
    if args.model == "original":
        mid_pad = 0
        policy_net = Policy(16, 2, log_std=args.log_std)
    elif args.model == "stgat":
        mid_pad = 1
        n_units = (
            [args.traj_lstm_hidden_size]
            + [int(x) for x in args.hidden_units.strip().split(",")]
            + [args.graph_lstm_hidden_size]
        )
        n_heads = [int(x) for x in args.heads.strip().split(",")]
        policy_net = TrajectoryGenerator(
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
        policy_net.cuda()

        global best_ade
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("Restoring from checkpoint {}".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            policy_net.load_state_dict(checkpoint["state_dict"], strict=False)
            logging.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))
        writer.close()

    if args.step_definition == 'multi':
        disc_single = Discriminator(40)
        disc_multi = Discriminator(18)
        discriminator_net = disc_multi
    elif args.step_definition == 'single':
        if args.model == 'original':
            disc_single = Discriminator(40)
            disc_multi = Discriminator(18)
            discriminator_net = disc_single
        elif args.model == 'stgat':
            if args.disc_type == 'original':
                discriminator_net = Discriminator(40)
                num_inputs = 2  # traj_dim, typically 2 for (x, y) coordinates
                seq_len = 20  # Sequence length (assuming input is reshaped to have 20 time steps)
                hidden_dim = 128  # Hidden size of LSTM (smaller than before)

                # discriminator_net = TrajectoryDiscriminator(num_inputs, seq_len, hidden_dim)
                discriminator_net.apply(initialize_weights)
                print("Discriminator initialized")
            else:
                n_units = (
                    [4]
                    + [int(x) for x in args.hidden_units.strip().split(",")]
                    + [4]
                )
                n_heads = [int(x) for x in [1, 1]]
                discriminator_net = STGAT_discriminator(
                    args,
                    obs_len=16,
                    pred_len=1,
                    traj_lstm_input_size=2,
                    traj_lstm_hidden_size=4,
                    n_units=n_units,
                    n_heads=n_heads,
                    graph_network_out_dims=4,
                    dropout=args.dropout,
                    alpha=args.alpha,
                    graph_lstm_hidden_size=4,
                    noise_dim=args.noise_dim,
                    noise_type=args.noise_type,
                    log_std=args.log_std,
                    action_dim=2
                )

    """create environment"""
    env = Environment(args, device)

    policy_net.to(device)
    policy_net.type(dtype).train()
    if args.disc_type == 'original':
        discriminator_net.to(device)
        discriminator_net.cuda()
        discriminator_net.type(dtype).train()
    else:
        discriminator_net.to(device)
        discriminator_net.cuda()
        discriminator_net.type(dtype).train()

    print("Policy_net: ", policy_net)
    print("Discriminator_net: ", discriminator_net)
    if args.training_algorithm in ['baseline', 'ppo', 'ppo_only']:
        if args.scene_value_net:
            value_net = Value(64)
        else:
            value_net = Value(16)
        value_net.to(device)
        value_net.type(dtype).train()
        print("Value_net: ", value_net)
    else:
        value_net = None

    """optimizers"""
    if args.model == "original":
        policy_opt = torch.optim.Adam(policy_net.parameters(), lr=args.policy_lr)
        discriminator_opt = torch.optim.Adam(discriminator_net.parameters(), lr=args.discriminator_lr)
    elif args.model == "stgat":
        policy_opt = optim.Adam(
            [
                {"params": policy_net.traj_lstm_model.parameters(), "lr": args.policy_lr},
                {"params": policy_net.traj_hidden2pos.parameters(), "lr": args.policy_lr},
                {"params": policy_net.gatencoder.parameters(), "lr": args.policy_lr},
                {"params": policy_net.graph_lstm_model.parameters(), "lr": args.policy_lr},
                {"params": policy_net.traj_gat_hidden2pos.parameters(), "lr": args.policy_lr},
                {"params": policy_net.pred_lstm_model.parameters(), "lr": args.policy_lr},
                {"params": policy_net.pred_hidden2pos.parameters(), "lr": args.policy_lr},
            ],
            lr=args.policy_lr,
        )
        if args.disc_type == 'original':
            discriminator_opt = torch.optim.Adam(discriminator_net.parameters(), lr=args.discriminator_lr)
        else:
            discriminator_opt = optim.Adam(
                [
                    {"params": discriminator_net.traj_lstm_model.parameters(), "lr": args.discriminator_lr},
                    {"params": discriminator_net.traj_hidden2pos.parameters(), "lr": args.discriminator_lr},
                    {"params": discriminator_net.gatencoder.parameters(), "lr": args.discriminator_lr},
                    {"params": discriminator_net.graph_lstm_model.parameters(), "lr": args.discriminator_lr},
                    {"params": discriminator_net.traj_gat_hidden2pos.parameters(), "lr": args.discriminator_lr},
                    {"params": discriminator_net.pred_lstm_model.parameters(), "lr": args.discriminator_lr},
                    {"params": discriminator_net.pred_hidden2pos.parameters(), "lr": args.discriminator_lr},
                ],
                lr=args.discriminator_lr,
            )
    discriminator_crt = nn.BCELoss()
    custom_reward = nn.BCELoss(reduction='none')
    if args.training_algorithm in ['baseline', 'ppo', 'ppo_only']:
        value_opt = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)
        value_crt = nn.MSELoss()
    else:
        value_opt = None
        value_crt = None

    """datasets"""
    train_path = get_dset_path(args.dataset_name, 'train')
    val_path = get_dset_path(args.dataset_name, 'val')
    test_path = get_dset_path(args.dataset_name, 'test')
    print("Initializing train dataset")
    train_dset, train_loader = data_loader(args, train_path, mid_pad)
    train_loader_len = len(train_loader)
    print("Initializing val dataset")
    _, val_loader = data_loader(args, val_path, mid_pad)
    val_loader_len = len(val_loader)
    print("Initializing test dataset")
    _, test_loader = data_loader(args, test_path, mid_pad)
    test_loader_len = len(test_loader)

    """loading the model"""
    if args.load_saved_model is None:
        """we want to store a model optimized for ADE metric"""
        saved_model_ADE = {
            'args': args.__dict__,
            'epoch': 0,
            'ADE_train': [],
            'FDE_train': [],
            'ADE_val': [],
            'FDE_val': [],
            'policy-loss_train': [],
            'policy-loss_val': [],
            'policy_net_state': None,
            'policy_opt_state': None,
            'discriminator-loss_train': [],
            'discriminator-loss_val': [],
            'discriminator_net_state': None,
            'discriminator_opt_state': None,
            'value-loss_train': [],
            'value-loss_val': [],
        }
        epoch = saved_model_ADE['epoch']
        save_model_path_ADE = os.path.join(args.output_dir, '%s.pt' % args.save_model_name_ADE)

        """We also want a separate model that is optimized for FDE metric"""
        saved_model_FDE = {
            'args': args.__dict__,
            'epoch': 0,
            'ADE_train': [],
            'FDE_train': [],
            'ADE_val': [],
            'FDE_val': [],
            'policy-loss_train': [],
            'policy-loss_val': [],
            'policy_net_state': None,
            'policy_opt_state': None,
            'discriminator-loss_train': [],
            'discriminator-loss_val': [],
            'discriminator_net_state': None,
            'discriminator_opt_state': None,
            'value-loss_train': [],
            'value-loss_val': [],
        }
        save_model_path_FDE = os.path.join(args.output_dir, '%s.pt' % args.save_model_name_FDE)

    else:
        saved_model = torch.load(args.load_saved_model)
        policy_net.load_state_dict(saved_model['policy_net_state'])
        policy_opt.load_state_dict(saved_model['policy_opt_state'])
        discriminator_net.load_state_dict(saved_model['discriminator_net_state'])
        discriminator_opt.load_state_dict(saved_model['discriminator_opt_state'])

    """custom reward for policy"""
    def expert_reward(env, args, state, action, gt=0):
        state_action = torch.cat((state, action), dim=1)
        if args.loss_definition == 'discriminator':
            if args.model == 'original' or args.disc_type == 'original':
                disc_out = discriminator_net(state_action)
            elif args.model == 'stgat':
                if env.training_step == 1 or env.training_step == 2:
                    disc_out = discriminator_net(state_action, obs_traj_pos=None, seq_start_end=env.seq_start_end, teacher_forcing_ratio=1, training_step=env.training_step)
                else:
                    disc_out = discriminator_net(state_action, obs_traj_pos=None, seq_start_end=env.seq_start_end, teacher_forcing_ratio=0, training_step=env.training_step)
            labels = torch.ones_like(disc_out)
            expert_reward = -custom_reward(disc_out, labels)
        elif args.loss_definition == 'l2':
            if args.model == "original":
                l2 = (gt - state_action) ** 2
                l2 = l2[:, 16:]
                expert_reward = -l2.sum(dim=1, keepdim=True)
            elif args.model == "stgat":
                if training_step == 1 or training_step == 2:
                    l2 = (gt[:, :16] - state_action[:, 16:]) ** 2
                    expert_reward = -l2.sum(dim=1, keepdim=True)
                else:
                    l2 = (gt - state_action) ** 2
                    l2 = l2[:, 16:]
                    expert_reward = -l2.sum(dim=1, keepdim=True)
        else:
            print("Wrong definition for loss, please choose either discriminator or l2")
        return expert_reward

    """create agent"""
    agent = Agent(args, env, policy_net, device, custom_reward=expert_reward)

    """update parameters function"""
    def update_params(args, batch, expert, train, epoch):
        loss_policy = 0
        loss_discriminator = 0
        loss_value = 0

        states, actions, rewards, states_all, actions_all, rewards_all = batch

        if args.loss_definition == 'discriminator':
            """perform discriminator update"""
            for _ in range(args.discriminator_steps):
                if args.step_definition == 'single':
                    expert_state_actions = expert
                    pred_state_actions = torch.cat([states, actions], dim=1)
                elif args.step_definition == 'multi':
                    expert_state_actions = expert
                    pred_state_actions = torch.cat([states_all[0], actions_all[0]], dim=1)
                discriminator_loss = discriminator_step(args, env, discriminator_net, discriminator_opt, discriminator_crt, expert_state_actions, pred_state_actions, device, train, epoch, writer)
                loss_discriminator += discriminator_loss

        """perform policy (REINFORCE) update"""
        for _ in range(args.policy_steps):
            policy_loss, value_loss = reinforce_step(args, env, policy_net, policy_opt, expert_reward, states, states_all, actions_all,
                                                     rewards_all, rewards, expert, train, value_net, value_opt, value_crt, training_step=training_step, epoch=epoch, writer=writer)
            loss_policy += policy_loss
            loss_value += value_loss

        return loss_policy / args.policy_steps, loss_discriminator / args.discriminator_steps, loss_value / args.policy_steps

    def train_loop():
        # min_ade, min_fde = float('inf'), float('inf')  
        # epoch_ade, epoch_fde = -1, -1 
        global training_step
        t0 = time.time()
        training_step = 1

        for epoch in range(args.start_epoch, args.num_epochs):
            t1 = time.time()
            print("\nEpoch: ", epoch, "/", args.num_epochs)
            if args.model == "stgat":
                """perform conditioning for STGAT"""
                if epoch < 150:
                    training_step = 1
                    env.training_step = 1
                elif epoch < 250:
                    training_step = 2
                    env.training_step = 2
                else:
                    if epoch == 250:
                        for param_group in policy_opt.param_groups:
                            param_group["lr"] = 5e-3
                    training_step = 3
                    env.training_step = 3
            """perform training steps"""
            agent.training_step = training_step
            train = True
            loss_policy = torch.zeros(1, device=device)
            loss_discriminator = torch.zeros(1, device=device)
            loss_value = torch.zeros(1, device=device)

            for batch_input in train_loader:
                env.generate(batch_input)
                batch = agent.collect_samples(mean_action=mean_action)
                expert = env.collect_expert()

                policy_loss, discriminator_loss, value_loss = update_params(args, batch, expert, train, epoch)
                loss_policy += policy_loss
                del policy_loss
                loss_discriminator += discriminator_loss
                loss_value += value_loss

                writer.add_scalar('train_loss_policy', loss_policy.item(), epoch)
                writer.add_scalar('train_loss_discriminator', loss_discriminator.item(), epoch)
                writer.add_scalar('train_loss_value', loss_value.item(), epoch)

            if args.model in ["original", "stgat"]:
                metrics_train = check_accuracy(env, args, train_loader, policy_net, args.obs_len, args.pred_len, device, limit=False)
                loss_policy = loss_policy / train_loader_len
                loss_discriminator = loss_discriminator / train_loader_len
                loss_value = loss_value / train_loader_len

                writer.add_scalar('train_loss_policy', loss_policy.item(), epoch)
                writer.add_scalar('train_loss_discriminator', loss_discriminator.item(), epoch)
                writer.add_scalar('train_loss_value', loss_value.item(), epoch)
                writer.add_scalar('ADE_train', metrics_train['ade'], epoch)
                writer.add_scalar('FDE_train', metrics_train['fde'], epoch)

                print('train loss_policy: ', loss_policy.item())
                print('train loss_discriminator: ', loss_discriminator.item())
                print('train loss_value: ', loss_value.item())
                print('train ADE: ', metrics_train['ade'])
                print('train FDE: ', metrics_train['fde'])

                if epoch % args.check_validation_every == 0:
                    """perform validation steps"""
                    train = False
                    loss_policy_val = torch.zeros(1, device=device)
                    loss_discriminator_val = torch.zeros(1, device=device)
                    loss_value_val = torch.zeros(1, device=device)
                    for batch_input in val_loader:
                        env.generate(batch_input)
                        batch = agent.collect_samples(mean_action=mean_action)
                        expert = env.collect_expert()

                        policy_loss_val, discriminator_loss_val, value_loss_val = update_params(args, batch, expert, train, epoch)
                        loss_policy_val += policy_loss_val
                        loss_discriminator_val += discriminator_loss_val
                        loss_value_val += value_loss_val

                    if args.model == "stgat":
                        metrics_validation = check_accuracy(env, args, val_loader, policy_net, args.obs_len, args.pred_len, device, limit=False)
                    else:
                        metrics_validation = check_accuracy(env, args, val_loader, policy_net, args.obs_len, args.pred_len, device, limit=False)

                    """test set check"""
                    if args.check_testset:
                        # metrics_validation_test = check_accuracy(env, args, test_loader, policy_net, args.obs_len, args.pred_len, device, limit=False)
                        # test_ade, test_fde = metrics_validation_test['ade'], metrics_validation_test['fde']
                        test_ade, test_fde = evaluate_irl(env,args, test_loader, policy_net, num_samples=1, mean_action=True, noise=args.trainable_noise, device=device)
                        # test_minade, test_minfde = evaluate_irl(env,args, test_loader, policy_net, num_samples=20, mean_action=False, noise=args.trainable_noise, device=device)
                        writer.add_scalar('ADE_test', test_ade, epoch)
                        writer.add_scalar('FDE_test', test_fde, epoch)
                        print('ADE_test', test_ade)
                        print('FDE_test', test_fde)

                    loss_policy_val = loss_policy_val / val_loader_len
                    loss_discriminator_val = loss_discriminator_val / val_loader_len
                    loss_value_val = loss_value_val / val_loader_len

                    writer.add_scalar('validation_loss_policy', loss_policy_val.item(), epoch)
                    writer.add_scalar('validation_loss_discriminator', loss_discriminator_val.item(), epoch)
                    writer.add_scalar('validation_loss_value', loss_value_val.item(), epoch)
                    writer.add_scalar('ADE_val', metrics_validation['ade'], epoch)
                    writer.add_scalar('FDE_val', metrics_validation['fde'], epoch)


                    if saved_model_ADE['ADE_val']:
                        min_ade = saved_model_ADE['ADE_val']  # both linear and non-linear
                    
                    if saved_model_ADE['ADE_val']:
                        min_ade = saved_model_ADE['ADE_val']  # both linear and non-linear

                    print('validation loss_policy: ', loss_policy_val.item())
                    print('validation loss_discriminator: ', loss_discriminator_val.item())
                    print('validation loss_value: ', loss_value_val.item())
                    print('validation ADE: ', metrics_validation['ade'])
                    print('validation FDE: ', metrics_validation['fde'])

                    min_ade = saved_model_ADE['ADE_val'] if saved_model_ADE['ADE_val'] else metrics_validation['ade']
                    min_fde = saved_model_FDE['FDE_val'] if saved_model_FDE['FDE_val'] else metrics_validation['fde']

                    if metrics_validation['ade'] <= min_ade:
                        epoch_ade=epoch
                        print("New low for min ADE_val, model saved")
                        saved_model_ADE.update({
                            'epoch': epoch,
                            'policy_net_state': policy_net.state_dict(),
                            'policy_opt_state': policy_opt.state_dict(),
                            'discriminator_net_state': discriminator_net.state_dict(),
                            'discriminator_opt_state': discriminator_opt.state_dict(),
                            'ADE_val': metrics_validation['ade'],
                            'ADE_train': metrics_train['ade'],
                            'FDE_val': metrics_validation['fde'],
                            'FDE_train': metrics_train['fde'],
                            'policy-loss_val': loss_policy_val.item(),
                            'policy-loss_train': loss_policy.item(),
                            'discriminator-loss_val': loss_discriminator_val.item(),
                            'discriminator-loss_train': loss_discriminator.item(),
                            'value-loss_val': loss_value_val.item(),
                            'value-loss_train': loss_value.item(),
                        })
                        # torch.save(saved_model_ADE, save_model_path_ADE)
                        # torch.save(saved_model_ADE, save_model_path_ADE)
                        save_model_path_epoch = os.path.join(
                            args.output_dir,
                            f'model_ADE_{epoch}_trial_{trial.number}_policy_lr_{args.policy_lr}_disc_lr_{args.discriminator_lr}.pt'
                        )
                        torch.save(saved_model_ADE, save_model_path_epoch)
                    if metrics_validation['fde'] <= min_fde:
                        epoch_fde=epoch
                        print("New low for min FDE_val, model saved")
                        saved_model_FDE.update({
                            'epoch': epoch,
                            'policy_net_state': policy_net.state_dict(),
                            'policy_opt_state': policy_opt.state_dict(),
                            'discriminator_net_state': discriminator_net.state_dict(),
                            'discriminator_opt_state': discriminator_opt.state_dict(),
                            'ADE_val': metrics_validation['ade'],
                            'ADE_train': metrics_train['ade'],
                            'FDE_val': metrics_validation['fde'],
                            'FDE_train': metrics_train['fde'],
                            'policy-loss_val': loss_policy_val.item(),
                            'policy-loss_train': loss_policy.item(),
                            'discriminator-loss_val': loss_discriminator_val.item(),
                            'discriminator-loss_train': loss_discriminator.item(),
                            'value-loss_val': loss_value_val.item(),
                            'value-loss_train': loss_value.item(),
                        })
                        # torch.save(saved_model_FDE, save_model_path_FDE)
                        save_model_path_epoch = os.path.join(
                            args.output_dir,
                            f'model_FDE_{epoch}_trial_{trial.number}_policy_lr_{args.policy_lr}_disc_lr_{args.discriminator_lr}.pt'
                        )
                        torch.save(saved_model_FDE, save_model_path_epoch)
            t2 = time.time()
            print_time(t0, t1, t2, epoch)
            log_memory_usage(f"Epoch {epoch}")
        
        clear_memory()
        # final_metrics = {
        # 'hparam/ADE_train': metrics_train['ade'],
        # 'hparam/FDE_train': metrics_train['fde'],
        # 'hparam/ADE_val': metrics_validation['ade'],
        # 'hparam/FDE_val': metrics_validation['fde'],
        # }
    
        log_dir = f"runs/trial_{trial.number}"

        return min_ade, min_fde, epoch_ade, epoch_fde

    """execute train loop"""
    min_ade, min_fde, epoch_ade, epoch_fde = train_loop()
    
    print(f"Trial {trial.number} - ADE: {min_ade}, FDE: {min_fde}")
    print(f"Trial {trial.number} - Params: {hparams}")
    return min_ade, min_fde, epoch_ade, epoch_fde
    # del policy_net
    # del discriminator_net
    # del discriminator_opt
    # del policy_opt
 
#STGAT =========================================================



def validate(args, model, val_loader):
    ade = AverageMeter("ADE", ":.6f")
    fde = AverageMeter("FDE", ":.6f")
    progress = ProgressMeter(len(val_loader), [ade, fde], prefix="Test: ")
    metrics = {}
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
            pred_traj_fake_rel, _, _ = model(obs_traj_rel, obs_traj, seq_start_end)
            pred_traj_fake_rel=pred_traj_fake_rel.detach()
            _=_.detach()
            pred_traj_fake_rel_predpart = pred_traj_fake_rel[-args.pred_len :]
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel_predpart, obs_traj[-1])
            ade_, fde_ = cal_ade_fde(pred_traj_gt, pred_traj_fake)
            ade_ = ade_ / (obs_traj.shape[1] * args.pred_len)

            fde_ = fde_ / (obs_traj.shape[1])
            ade.update(ade_, obs_traj.shape[1])
            fde.update(fde_, obs_traj.shape[1])

            if i % args.print_every == 0:
                progress.display(i)

        # logging.info(
        #     " * ADE  {ade.avg:.3f} FDE  {fde.avg:.3f}".format(ade=ade, fde=fde)
        # )
        # writer.add_scalar("val_ade", ade.avg, epoch)
        metrics['ade'] =ade.avg
        metrics['fde'] = fde.avg
    return metrics

def print_best_trial_so_far(study, trial):
    with open('final_best_trial_params.txt', 'a') as f:
            # 
        f.write(f"Trial {trial.number} - ADE: {trial.value}\n")
        f.write(f" Params: {trial.params}\n")
        f.write("\nBest trial {} so far used the following hyper-parameters:".format(study.best_trial.number))
        for key, value in study.best_trial.params.items():
            f.write("{}: {}".format(key, value))
        f.write("Best achived ADE {}\n".format(study.best_trial.value))
        # Write final metrics to a file
    # 
    #     f.write(f"Trial {trial.number} - Best ADE: {ade}, Best FDE: {fde}\n")
    #     f.write(f"Best Params: {hparams}\n")
    #     f.write("\n")
def cal_ade_fde(pred_traj_gt, pred_traj_fake):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    return ade, fde

#===============================================================
if args.all_datasets:
    datasets = ['eth'] # ['eth', 'hotel', 'zara1', 'zara2', 'univ']
else:
    datasets = [args.dataset_name]

model_name_ADE_base = args.save_model_name_ADE
model_name_FDE_base = args.save_model_name_FDE
import webbrowser
def objective(args, trial):
    # (args, study) =input
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Suggest a learning rate for the policy network and the discriminator
    policy_lr = trial.suggest_loguniform('policy_lr', 1e-5, 1e-2) #0.0003331799195227339 #
    discriminator_lr = trial.suggest_loguniform('discriminator_lr', 1e-5, 1e-2) #0.0011562115934306177 #trial.suggest_loguniform('discriminator_lr', 1e-5, 1e-2)
    args.policy_lr = policy_lr
    args.discriminator_lr = discriminator_lr

    # args.ppo_iterations = trial.suggest_categorical('ppo_iterations', [8, 15])
    # args.l2_reg = 0.2
    # args.dropout = 0.0

    trial_dir = f"./optuna/trial_{trial.number}"
    os.makedirs(trial_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=trial_dir)

    hparams = {
        'learning_rate_policy': args.policy_lr,
        'learning_rate_discriminator': args.discriminator_lr,
        'ppo_iterations': args.ppo_iterations,
        'l2_reg': args.l2_reg,
        'dropout': args.dropout,
        'batch_size': args.batch_size,
        'ppo_clip': args.ppo_clip
    }

    metric_dict = {
        'ade': 0,
        'fde': 0
    }

    # Run main training loop with the current trial hyperparameters
    ade, fde, ade_epoch, fde_epoch = main_loop(args, writer, metric_dict, hparams, trial)
    # Save metrics in a dictionary
    metric_dict['ade'] = ade
    metric_dict['fde'] = fde
    writer.add_hparams(
            {
                "lr": args.policy_lr,
                "lr_disc": args.discriminator_lr,
            },  metric_dict=metric_dict)

    writer.close()
    trial.set_user_attr('best_ade', ade)
    trial.set_user_attr('best_fde', fde)
    trial.set_user_attr('best_params', hparams)

    print(f"Trial {trial.number} - Best ADE: {ade}, Best FDE: {fde}")
    print(f"Trial {trial.number} - Best Params: {hparams}")

    # Return a metric to minimize or maximize
    return ade  # Minimize the Average Displacement Error (ADE)
def load_or_create_study(study_name, storage_url):
    print("trying to load")
    return optuna.create_study(study_name=study_name, direction='minimize', storage=storage_url,  load_if_exists=True)

def run_study():
    gc.collect()
    
    study_name = 'example_study853.db'
    storage_url = 'sqlite:///example_study.db'
    
    # Load or create the study
    study = load_or_create_study(study_name, storage_url)
    
    # Add TensorBoardCallback to log results
    # tensorboard_callback = TensorBoardCallback("./optuna/", metric_name='ade')
    
    study.optimize(partial(objective, args), n_trials=35, n_jobs=1, callbacks=[print_best_trial_so_far])
    
    print("Best trial:")
    print(" Value: ", study.best_trial.value)
    print(" Params: ", study.best_trial.params)
    gc.collect()
for set in datasets:
    
#     trial_dir = f"./optuna/trial_bla"
#     model_name_ADE = model_name_ADE_base + '_' + set + '_trial_run_' + str( 1) +"_Best_k_"+  str(args.lr)  +"_length_"+ str(args.dropout)+ "_bs_"+ str(args.batch_size) + "_" + str(args.l2_reg)+"_" +str(args.ppo_clip)+"_" +str(args.ppo_iterations)
#     model_name_FDE = model_name_FDE_base + '_' + set + '_trial_run_' + str( 1) +"_Best_k_"+  str(args.lr)  +"_length_"+ str(args.dropout)+ "_bs_"+ str(args.batch_size) + "_" + str(args.l2_reg)+"_" +str(args.ppo_clip)+"_" +str(args.ppo_iterations)
    args.resume = f'/home/ssukup/unified-pedestrian-path-prediction-framework/scripts/pretrained_STGAT/kbest_{args.best_k}/model_best_{args.dataset_name}_{0}.pth.tar'
#     args.save_model_name_ADE = model_name_ADE
#     args.save_model_name_FDE = model_name_FDE
#     # os.makedirs(trial_dir, exist_ok=True)
#     writer = SummaryWriter(log_dir=trial_dir)
#     hparams = {
#         'learning_rate_value': args.learning_rate,
#         'learning_rate_policy': args.lr,
#         'ppo_iterations': args.ppo_iterations,
#         'l2_reg': args.l2_reg,
#         'dropout': args.dropout,
#         'batch_size': args.batch_size,  # Assuming you want to log this as well
#         'ppo_clip': args.ppo_clip
#     }

    
#     # Assign these suggested values to args
#     metric_dict = {
#     'ade': 0,  # You need to replace this with actual loss or another metric from your training loop
#     'fde': 0  # Placeholder, replace with your actual initial metric if applicable
# }
    
    gc.collect()
    args.dataset_name = set
    # ade, fde, ade_epoch, fde_epoch = main_loop(args, writer, metric_dict, hparams)
    
    if args.model=="original":
        print("Dataset: " + set + ". Script execution number: " + str(0))
    elif args.model=="stgat":
            print("Dataset: " + set + ". Script execution number: " + str(0)+ " Best k:"+ str(args.best_k))
    run_study()
    # torch.cuda.empty_cache()
    

