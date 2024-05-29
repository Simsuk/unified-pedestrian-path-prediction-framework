import argparse

import os
import random
from memory_profiler import profile
import cProfile
import re
import pstats
from cProfile import Profile
from pstats import SortKey, Stats
import logging
import gc
import sys
class Profile:
    def __init__(self):
        self.prof = cProfile.Profile()

    def __enter__(self):
        self.prof.enable()
        return self.prof

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.prof.disable()
        self.prof.print_stats(sort='time')
    
# def log_exception(exc_type, exc_value, exc_traceback):
#     logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

# Set the global exception hook
# sys.excepthook = log_exception

# Configure logging
# logging.basicConfig(level=logging.ERROR, filename='errors.log')

from irl.utils import *
from irl.models import Policy, Discriminator, Value
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
parser.add_argument("--l2_reg", default=0.1, help="PPO_regularization")

parser.add_argument('--randomness_definition', default='stochastic',  type=str, help='either stochastic or deterministic')
parser.add_argument('--step_definition', default='single',  type=str, help='either single or multi')
parser.add_argument('--loss_definition', default='l2',  type=str, help='either discriminator or l2')
parser.add_argument('--disc_type', default='original', type=str, help='either stgat or original')
parser.add_argument('--discount_factor', type=float, default=0.0, help='discount factor gamma, value between 0.0 and 1.0')
parser.add_argument('--optim_value_iternum', type=int, default=1, help='minibatch size')

parser.add_argument('--training_algorithm', default='reinforce',  type=str, help='choose which RL updating algorithm, either "reinforce", "baseline" or "ppo" or "ppo_only"')
parser.add_argument('--trainable_noise', type=bool, default=False, help='add a noise to the input during training')
parser.add_argument('--ppo-iterations', type=int, default=1, help='number of ppo iterations (default=1)')
parser.add_argument('--ppo-clip', type=float, default=0.2, help='amount of ppo clipping (default=0.2)')
parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G', help='learning rate (default: 1e-5)')
parser.add_argument('--batch_size', default=64, type=int, help='number of sequences in a batch (can be multiple paths)')
parser.add_argument('--log-std', type=float, default=-2.99, metavar='G', help='log std for the policy (default=-0.0)')
parser.add_argument('--num_epochs', default=400, type=int, help='number of times the model sees all data')

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
parser.add_argument('--use_gpu', default=0, type=int)                   # use gpu, if 0, use cpu only
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
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
    "--dropout", type=float, default=0, help="Dropout rate (1 - keep probability)."
)
parser.add_argument(
    "--alpha", type=float, default=0.2, help="Alpha for the leaky_relu."
)
parser.add_argument(
    "--lr",
    default=  8.623230816228654e-05, #=1e-3 #8.623230816228654e-05 #0.00024036092775471976
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--start-epoch",
    default=300,
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
def main_loop(writer):

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
    torch.backends.cudnn.deterministic = True           # what does this do again?
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
    if args.model=="original":
        mid_pad=0
        policy_net = Policy(16, 2, log_std=args.log_std)     # 16, 2
    elif args.model=="stgat":
        mid_pad=1
        n_units = (
        [args.traj_lstm_hidden_size]
        + [int(x) for x in args.hidden_units.strip().split(",")]
        + [args.graph_lstm_hidden_size]
        )
        n_heads = [int(x) for x in args.heads.strip().split(",")]
        policy_net =  TrajectoryGenerator(
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


            # train(args, model, train_loader, optimizer, epoch, training_step, writer)
            # if training_step == 3:
            #     ade = validate(args, model, val_loader, epoch, writer)
            #     is_best = ade < best_ade
            #     best_ade = min(ade, best_ade)

                # save_checkpoint(
                #     {
                #         "epoch": epoch + 1,
                #         "state_dict": model.state_dict(),
                #         "best_ade": best_ade,
                #         "optimizer": optimizer.state_dict(),
                #     },
                #     is_best,
                #     epoch,
                #     args.pred_len,
                #     f"./checkpoint/checkpoint{args.da1taset_name,args.best_k, epoch, args.pred_len}.pth.tar"
                # )
        writer.close()
    disc_single = Discriminator(40) #40
    disc_multi = Discriminator(18)
    
    if args.step_definition == 'multi':
        discriminator_net = disc_multi
    elif args.step_definition == 'single':
        if args.model=='original':
         discriminator_net = disc_single      #changed from single as experiment
        elif args.model=='stgat':
            if   args.disc_type=='original':
                # discriminator_net = Discriminator(32)
                discriminator_early = Discriminator(32)  # Assuming this uses input size 32
                discriminator_late = Discriminator(40)   # Modify according to your needs
                discriminator_net=Discriminator_LSTM()
       # You might want to move these definitions to a place where they can be initialized with appropriate device settings:

                print("bla")
            else:
                n_units = (
                [4]
                + [int(x) for x in args.hidden_units.strip().split(",")]
                + [4]
                )
                n_heads = [int(x) for x in [1,1]]
                discriminator_net =  STGAT_discriminator(
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
    if args.disc_type=='original':
                # discriminator_early.to(device)
                # discriminator_late.to(device)
                # discriminator_early.cuda()
                # discriminator_late.cuda()
                # discriminator_early.type(dtype).train()
                # discriminator_late.type(dtype).train()
                # discriminator_net=DiscriminatorManager(discriminator_early, discriminator_late, env)
                discriminator_net.to(device)
                discriminator_net.cuda()
                discriminator_net.type(dtype).train()

    else:
        discriminator_net.to(device)
        discriminator_net.cuda()
        discriminator_net.type(dtype).train()

    
    print("Policy_net: ", policy_net)
    print("Discriminator_net: ", discriminator_net)
    if (args.training_algorithm == 'baseline' or args.training_algorithm == 'ppo' or args.training_algorithm == 'ppo_only'):
        if args.scene_value_net==True:
            value_net = Value(64)
            # print("WORKING")
        else:
            value_net = Value(16)
        value_net.to(device)
        value_net.type(dtype).train()
        print("Value_net: ", value_net)
    else:
        value_net = None

    """optimizers"""
    if args.model=="original" :
        policy_opt = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
        disc_lr = args.learning_rate
        discriminator_opt = torch.optim.Adam(discriminator_net.parameters(), lr=disc_lr)
    elif args.model=="stgat":
        policy_opt = optim.Adam(
        [
            {"params": policy_net.traj_lstm_model.parameters(), "lr": args.lr},#1e-2
            {"params": policy_net.traj_hidden2pos.parameters()},
            {"params": policy_net.gatencoder.parameters(), "lr": args.lr}, #3e-2
            {"params": policy_net.graph_lstm_model.parameters(), "lr": args.lr}, #1e-2}
            {"params": policy_net.traj_gat_hidden2pos.parameters()},
            {"params": policy_net.pred_lstm_model.parameters()},
            {"params": policy_net.pred_hidden2pos.parameters()},
        ],
        lr= args.lr ,#args.lr,
        )
        if args.disc_type=='original':
            disc_lr = 1e-2 #args.learning_rate
            discriminator_opt = torch.optim.Adam(discriminator_net.parameters(), lr=disc_lr)
            # discriminator_early = torch.optim.Adam(discriminator_early.parameters(), lr=disc_lr)
            # discriminator_opt = torch.optim.Adam(discriminator_late.parameters(), lr=disc_lr)

        else:
            discriminator_opt = optim.Adam(
            [
                {"params": policy_net.traj_lstm_model.parameters(), "lr": 1e-2},
                {"params": policy_net.traj_hidden2pos.parameters()},
                {"params": policy_net.gatencoder.parameters(), "lr": 3e-2},
                {"params": policy_net.graph_lstm_model.parameters(), "lr": 1e-2},
                {"params": policy_net.traj_gat_hidden2pos.parameters()},
                {"params": policy_net.pred_lstm_model.parameters()},
                {"params": policy_net.pred_hidden2pos.parameters()},
            ],
            lr=1e-3,
            )




    discriminator_crt = nn.BCELoss()
    custom_reward = nn.BCELoss(reduction='none')
    if (args.training_algorithm == 'baseline' or args.training_algorithm == 'ppo' or args.training_algorithm == 'ppo_only'):
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
        print("output_dir", args.output_dir)
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
    def expert_reward(env, args, state, action, gt=0):     # probs separate function for discount
        # print(args)
        state_action = torch.cat((state, action), dim=1)  # (b, 16) + (b, 24) = (b, 40)
        if args.loss_definition == 'discriminator':
            if args.model=='original' or args.disc_type=='original':
                disc_out = discriminator_net(state_action)
            elif args.model=='stgat':
                if env.training_step == 1 or env.training_step == 2:
                    disc_out = discriminator_net(state_action, obs_traj_pos=None, seq_start_end=env.seq_start_end, teacher_forcing_ratio=1, training_step=env.training_step)
                else:
                    disc_out = discriminator_net(state_action, obs_traj_pos=None, seq_start_end=env.seq_start_end, teacher_forcing_ratio=0, training_step=env.training_step)
            labels = torch.ones_like(disc_out)
            expert_reward = -custom_reward(disc_out, labels)  # pytorch nn.BCELoss() already has a -

        elif args.loss_definition == 'l2':
            if args.model=="original":
                # print("state_action",state_action.shape)
                l2 = (gt - state_action)**2  # (b, 40) - (b, 40) ** 2 (for single, (b,18) for multi)
                l2 = l2[:, 16:]              # test to only include action difference instead of state-action
                expert_reward = -l2.sum(dim=1, keepdim=True)    #div dim action space
                
            elif args.model=="stgat":
                # for stgat the expected shape for trianing step==1 or 2: state =torch.Size([batch, 32])
                #                                 for trianing step==3:                     
                if training_step == 1 or training_step == 2:
                    # print("state_action",state_action.shape)
                    l2 = (gt[:, :16] - state_action[:, 16:] )**2  # (b, 16) - (b, 16) ** 2  here state_action is (b, 32)
                    #l2.shape= (b,16)
                    # print("gt", gt.shape, os.path.basename(__file__))
                   #  only includes action difference instead of state-action
                    expert_reward = -l2.sum(dim=1, keepdim=True)    #div dim action space  (b,1)
                    # print("expert_reward", expert_reward.shape, os.path.basename(__file__))
                    frame = inspect.currentframe()
                    # Get the caller's stack frame
                    caller_frame = frame.f_back
                    # Retrieve information about the caller
                    caller_info = inspect.getframeinfo(caller_frame)
                    
                    # Print the caller's file name and line number
                    # print(f"Called from {caller_info.filename} at line {caller_info.lineno}")
       
                else:
                    l2 = (gt - state_action)**2  # (b, 40) - (b, 40) ** 2 (for single, (b,18) for multi)
                    l2 = l2[:, 16:]              # test to only include action difference instead of state-action
                    expert_reward = -l2.sum(dim=1, keepdim=True)    #div dim action space
                    
        else:
            print("Wrong definition for loss, please choose either discriminator or l2")

        return expert_reward    # tensor(b,1)


    """create agent"""
    
    agent = Agent(args, env, policy_net, device, custom_reward=expert_reward)




    """update parameters function"""
    def update_params(args, batch, expert, train):

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
                    
                    # print(" expert_state_actions, pred_state_actions",  expert_state_actions, pred_state_actions)

                    # print(" expert_state_actions, pred_state_actions",  expert_state_actions.shape, pred_state_actions.shape)
                elif args.step_definition == 'multi':
                    expert_state_actions = expert   # (bx12, 18)
                   
                    pred_state_actions = torch.cat([states_all[0], actions_all[0]], dim=1)  #(bx12, 18)
                    #state_action, obs_traj_pos=None, seq_start_end=env.seq_start_end, teacher_forcing_ratio=1, training_step=env.training_step)
                discriminator_loss = discriminator_step(args,env, discriminator_net, discriminator_opt, discriminator_crt, expert_state_actions, pred_state_actions, device, train)
                loss_discriminator += discriminator_loss

        """perform policy (REINFORCE) update"""
        for _ in range(args.policy_steps):
            policy_loss, value_loss = reinforce_step(args, env, policy_net, policy_opt, expert_reward, states_all, actions_all,
                                         rewards_all, rewards, expert, train, value_net, value_opt, value_crt, training_step=training_step, epoch=epoch, writer=writer)

            loss_policy += policy_loss
            loss_value += value_loss

        return loss_policy/args.policy_steps, loss_discriminator/args.discriminator_steps, loss_value/args.policy_steps


    def train_loop():
        global training_step
        t0 = time.time()
        training_step = 1

        for epoch in range(args.start_epoch, args.num_epochs):     # epochs
            t1 = time.time()

            print("\nEpoch: ", epoch, "/", args.num_epochs)
            if args.model=="stgat":
                """perform conditioning ofr STGAT"""                
                if epoch < 150:
                    training_step = 1
                    env.training_step=1
                elif epoch < 250:
                    training_step = 2
                    env.training_step=2
                else:
                    if epoch == 250:
                        for param_group in policy_opt.param_groups:
                            param_group["lr"] = 5e-3
                    training_step = 3
                    env.training_step=3
            """perform training steps"""
            agent.training_step=training_step
            train = True
            loss_policy = torch.zeros(1, device=device)
            loss_discriminator = torch.zeros(1, device=device)
            loss_value = torch.zeros(1, device=device)

            for batch_input in train_loader:
                # with torch.autograd.set_detect_anomaly(True):
                    # print("BATCH START")
                    env.generate(batch_input)                                   # sets a batch of observed trajectories
                    # print("STEP ____",env.training_step)
                    batch = agent.collect_samples(mean_action=mean_action)      # batch contains a tensor of states (8 steps), a tensor of actions (12 steps) and a tensor of rewards (1 for the whole trajectory)
                    
                    expert = env.collect_expert()                               # the expert is a batch of full ground truth trajectories

                    policy_loss, discriminator_loss, value_loss = update_params(args, batch, expert, train)
                    loss_policy += policy_loss
                    del policy_loss
                    loss_discriminator += discriminator_loss
                    loss_value += value_loss
                    # print("BATCH END")
                    writer.add_scalar('train_loss_policy', loss_policy.item(), epoch)
                    writer.add_scalar('train_loss_discriminator', loss_discriminator.item(), epoch)
                    writer.add_scalar('train_loss_value', loss_value.item(), epoch)
            if args.model=="original" or args.model=="stgat" :
                metrics_train = check_accuracy(env,args, train_loader, policy_net, args.obs_len, args.pred_len, device, limit=False)       # limit=true causes sinusoidal train ADE

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

                        policy_loss_val, discriminator_loss_val, value_loss_val = update_params(args, batch, expert, train)
                        loss_policy_val += policy_loss_val
                        loss_discriminator_val += discriminator_loss_val
                        loss_value_val += value_loss_val
                    if args.model=="stgat":
                        # metrics_validation = validate(args, policy_net, val_loader)
                        metrics_validation = check_accuracy(env,args, val_loader, policy_net,args.obs_len, args.pred_len, device, limit=False)

                    else:
                        metrics_validation = check_accuracy(env,args, val_loader, policy_net,args.obs_len, args.pred_len, device, limit=False)

                    ### test set check
                    if args.check_testset is True:
                        metrics_validation_test = check_accuracy(env,args, test_loader, policy_net, args.obs_len, args.pred_len, device, limit=False) 
                        # test_ade, test_fde = metrics_validation_test['ade'], metrics_validation_test['fde']
                        test_ade, test_fde = evaluate_irl(env,args, test_loader, policy_net, num_samples=1, mean_action=True, noise=args.trainable_noise, device=device)
                        test_minade, test_minfde = evaluate_irl(env,args, test_loader, policy_net, num_samples=20, mean_action=False, noise=args.trainable_noise, device=device)
                        writer.add_scalar('ADE_test', test_ade, epoch)
                        writer.add_scalar('FDE_test', test_fde, epoch)
                        writer.add_scalar('minADE_test', test_minade, epoch)
                        writer.add_scalar('minFDE_test', test_minfde, epoch)
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


                    print('validation loss_policy: ', loss_policy_val.item())
                    print('validation loss_discriminator: ', loss_discriminator_val.item())
                    print('validation loss_value: ', loss_value_val.item())
                    print('validation ADE: ', metrics_validation['ade'])
                    print('validation FDE: ', metrics_validation['fde'])

                    if saved_model_ADE['ADE_val']:
                        min_ade = saved_model_ADE['ADE_val']  # both linear and non-linear

                    else:
                        min_ade = metrics_validation['ade']
                        
                    if saved_model_FDE['FDE_val']:
                        min_fde = saved_model_FDE['FDE_val']  # both linear and non-linear
                    else:
                        min_fde = metrics_validation['fde']
                    writer.add_scalar('min_ADE_VAL', min_ade, epoch)
                    writer.add_scalar('min_FDE_VAL', min_fde, epoch)
                    print('min_ADE_val: ', min_ade)
                    print('min_FDE_val: ', min_fde)
                    if metrics_validation['ade'] <= min_ade:
                        print("min_ade")
                        epoch_ade=epoch
                        print('New low for min ADE_val, model saved')
                        saved_model_ADE['epoch'] = epoch
                        saved_model_ADE['policy_net_state'] = policy_net.state_dict()
                        saved_model_ADE['policy_opt_state'] = policy_opt.state_dict()
                        saved_model_ADE['discriminator_net_state'] = discriminator_net.state_dict()
                        saved_model_ADE['discriminator_opt_state'] = discriminator_opt.state_dict()
                        saved_model_ADE['ADE_val'] = metrics_validation['ade']
                        saved_model_ADE['ADE_train'] = metrics_train['ade']
                        saved_model_ADE['FDE_val'] = metrics_validation['fde']
                        saved_model_ADE['FDE_train'] = metrics_train['fde']
                        saved_model_ADE['policy-loss_val'] = loss_policy_val.item()
                        saved_model_ADE['policy-loss_train'] = loss_policy.item()
                        saved_model_ADE['discriminator-loss_val'] = loss_discriminator_val.item()
                        saved_model_ADE['discriminator-loss_train'] = loss_discriminator.item()
                        saved_model_ADE['value-loss_val'] = loss_value_val.item()
                        saved_model_ADE['value-loss_train'] = loss_value.item()
                        torch.save(saved_model_ADE, save_model_path_ADE)
                        print("PATH", save_model_path_ADE)
                    if metrics_validation['fde'] <= min_fde:
                        epoch_fde=epoch
                        print('New low for min FDE_val, model saved')
                        saved_model_FDE['epoch'] = epoch
                        saved_model_FDE['policy_net_state'] = policy_net.state_dict()
                        saved_model_FDE['policy_opt_state'] = policy_opt.state_dict()
                        saved_model_FDE['discriminator_net_state'] = discriminator_net.state_dict()
                        saved_model_FDE['discriminator_opt_state'] = discriminator_opt.state_dict()
                        saved_model_FDE['ADE_val'] = metrics_validation['ade']
                        saved_model_FDE['ADE_train'] = metrics_train['ade']
                        saved_model_FDE['FDE_val'] = metrics_validation['fde']
                        saved_model_FDE['FDE_train'] = metrics_train['fde']
                        saved_model_FDE['policy-loss_val'] = loss_policy_val.item()
                        saved_model_FDE['policy-loss_train'] = loss_policy.item()
                        saved_model_FDE['discriminator-loss_val'] = loss_discriminator_val.item()
                        saved_model_FDE['discriminator-loss_train'] = loss_discriminator.item()
                        saved_model_FDE['value-loss_val'] = loss_value_val.item()
                        saved_model_FDE['value-loss_train'] = loss_value.item()
                        torch.save(saved_model_FDE, save_model_path_FDE)

            t2 = time.time()
            print_time(t0, t1, t2, epoch)
        return min_ade, min_fde, epoch_ade, epoch_fde

    """execute train loop"""
    min_ade, min_fde, epoch_ade, epoch_fde=train_loop()
    return min_ade, min_fde, epoch_ade, epoch_fde
    # del policy_net
    # del discriminator_net
    # del discriminator_opt
    # del policy_opt
print("HER5", os.getcwd())
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


def cal_ade_fde(pred_traj_gt, pred_traj_fake):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    return ade, fde

#===============================================================
if args.all_datasets:
    datasets = ['zara1']  # ['eth', 'hotel', 'zara1', 'zara2', 'univ']
else:
    datasets = [args.dataset_name]

model_name_ADE_base = args.save_model_name_ADE
model_name_FDE_base = args.save_model_name_FDE
import webbrowser
def start_tensorboard(logdir, port=6006):
    # Start TensorBoard as a subprocess and redirect its output to devnull
    tensorboard_url = f"http://localhost:{port}"
    command = f"tensorboard --logdir={logdir} --port={port}"
    with open('/dev/null', 'w') as devnull:
        subprocess.Popen(command, shell=True, stdout=devnull, stderr=devnull)
    
    # Wait for TensorBoard to initialize
    print(f"TensorBoard is starting at {tensorboard_url}")
    time.sleep(3)  # Wait a few seconds for TensorBoard to start

    # Open a web browser with the TensorBoard URL
    webbrowser.open_new(tensorboard_url)
def objective(args, trial):
    # Example: Suggest a learning rate and dropout rate
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-6)
    # dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.5)

    # Assign these suggested values to args
    args.learning_rate = learning_rate
    # args.dropout = dropout_rate

    # Call your main training function here
    ade, fde, ade_epoch, fde_epoch = main_loop(args)

    # Return a metric to minimize or maximize
    return ade  # Let's minimize the Average Displacement Error (ADE)


if args.multiple_executions:
    for i in [1]:
        for set in datasets:
            args.dataset_name = set
            # torch.cuda.empty_cache()
            # print("WORKING")
            gc.collect()
            if args.seeding:
                args.seed = i
            if args.model=="stgat":
                args.resume = f'/home/ssukup/unified-pedestrian-path-prediction-framework/scripts/pretrained_STGAT/kbest_{args.best_k}/model_best_{args.dataset_name}_{i}.pth.tar'
            print("RESUME PATH", args.resume )
            model_name_ADE = model_name_ADE_base + '_' + set + '_run_' + str(i) +"_Policy_"+  str(args.best_k)  +"_length_"+ str(args.pred_len)
            model_name_FDE = model_name_FDE_base + '_' + set + '_run_' + str(i) +"_Best_k_"+  str(args.best_k)  +"_length_"+ str(args.pred_len)
            # tensorboard_name =   tensorboard_name = os.path.join("/home/ssukup/unified-pedestrian-path-prediction-framework/tensorboard/bla", set + '_run_' + str(i)) #"../tensorboard/original_GAN" #'../tensorboard/' + set + 'run_' + str(i)
            tensorboard_name= f"./logging/kbest_{args.best_k}_{args.dataset_name}_{i}_"
            os.makedirs(tensorboard_name, exist_ok=True)
            args.save_model_name_ADE = model_name_ADE
            args.save_model_name_FDE = model_name_FDE
            writer = SummaryWriter(log_dir=tensorboard_name)
                    
            # Specify the log directory and port
            # log_directory = "./tensorboard_logs"
            # start_tensorboard(log_directory)

            for arg, value in vars(args).items():
                writer.add_text(arg, str(value))
            if args.model=="original":
                print("Dataset: " + set + ". Script execution number: " + str(i))
            elif args.model=="stgat":
                 print("Dataset: " + set + ". Script execution number: " + str(i)+ " Best k:"+ str(args.best_k))
            main_loop(writer)
            # torch.cuda.empty_cache()
            gc.collect()
else:
        # with Profile() as profile:
            gc.collect()
            # torch.cuda.empty_cache()
            model_name_ADE = model_name_ADE_base + '_' + set + '_run_' + str(0) +"_Best_k_"+  str(args.best_k)  +"_length_"+ str(args.pred_len)
            model_name_FDE = model_name_FDE_base + '_' + set + '_run_' + str(0) +"_Best_k_"+  str(args.best_k)  +"_length_"+ str(args.pred_len)
            args.resume = f'/home/ssukup/unified-pedestrian-path-prediction-framework/scripts/pretrained_STGAT/kbest_{args.best_k}/model_best_{args.dataset_name}_{0}.pth.tar'

            tensorboard_name = '../tensorboard/' + set
            args.save_model_name_ADE = model_name_ADE
            args.save_model_name_FDE = model_name_FDE
            writer = SummaryWriter(log_dir=tensorboard_name)
            
            if args.model=="original":
                print("Dataset: " + set + ". Script execution number: " + str(0))
            elif args.model=="stgat":
                 print("Dataset: " + set + ". Script execution number: " + str(0)+ " Best k:"+ str(args.best_k))
            main_loop()
            # torch.cuda.empty_cache()
            gc.collect()