
from irl.replay_memory import Memory
from irl.utils import *
import os
import torch.nn.functional as F
os.environ["OMP_NUM_THREADS"] = "1"

def add_noise_to_state(state):
    noise_dim = 2
    noise_std = 0.05
    pad_dim = state.shape[1] - noise_dim
    noise_shape = (state.shape[0], noise_dim)
    device = state.device
    noise = torch.randn(noise_shape).to(device)
    noise = noise * noise_std

    pad = (pad_dim, 0, 0, 0)
    noise = F.pad(noise, pad, "constant", 0)  # effectively zero padding

    noisy_state = state + noise
    return noisy_state


def collect_samples(args, env, policy, custom_reward, device, mean_action,  training_step):

    total_paths = env.total_paths
    memory = Memory()

    with torch.no_grad():
        state, seq_start_end,obs_traj_rel  = env.reset()             # take new initial state (observation) of shape (b, 16)
        state_0 = state.clone()
        obs_traj=None
        bs = state.shape[0]             # batch size
        # print("BATCH SIZE", bs)
        rewards = []
        states = []
        actions = []
        reward_full = torch.zeros(bs)   #(b,) of zeros
        ground_truth = env.collect_expert()   # (b,40) if single step, (bx12,18) if multistep
        ts = 0          # timestep for selecting correct ground truth

        done = False
        if args.model== "stgat":
            if training_step == 1 or training_step == 2:
                if mean_action:
                    action_all, _, _ = policy(state, obs_traj,seq_start_end ,1, training_step) 
                    
                else:
                    seed = time.time_ns()
                    action_all = policy.select_action(state, obs_traj, seq_start_end , seed, training_step)
                    
                    # print("action.shape", print_structure_and_dimensions(action_all))
            else: 
                if mean_action:
                    
                    model_input = torch.cat((env.obs_traj_rel, env.pred_traj_gt_rel), dim=0)
                    action_all, _, _ = policy(model_input, obs_traj,seq_start_end ,0, training_step)
                else:
                    model_input = torch.cat((env.obs_traj_rel, env.pred_traj_gt_rel), dim=0)
                    seed = time.time_ns()
                    action_all = policy.select_action(model_input, obs_traj, seq_start_end ,seed, training_step)
            iterator=iter(action_all)
            # print("action_all",action_all.shape)
        while not done:
            if mean_action:
                if args.trainable_noise == True:
                    state = add_noise_to_state(state)
                if args.model== "original":
                    action, _, _ = policy(state)                  # action is of shape (b, 2)
                elif args.model== "stgat":
                    action=next(iterator)
            else:
                if args.model== "original":
                    action = policy.select_action(state)          
                elif args.model== "stgat":
                    action =next(iterator)
                    # print("action.shape", action.shape)

                    
                    #I ENDED HERE
                
            # save action
            # print("ACTION", action.shape)
            # print("STATES", state.shape)
            actions.append(action)
            states.append(state)

            next_state, reward, done, = env.step(state, action)
            # print("Agent_reward", reward.shape) #should be torch.Size([b, 1])
            if custom_reward is not None:
                if args.step_definition == 'multi':
                    gt = ground_truth[ts * bs:(ts + 1) * bs, :] # take the ground truth of the correct timestep
                    reward = torch.squeeze(custom_reward(env,args, state, action, gt), dim=1)
                    rewards.append(reward)

            if done:
                action_full = torch.cat(actions, dim=1) # (b, 24) or (b,16) in case of phase 1 and 2
                
                if custom_reward is not None:
                    if args.step_definition == 'single':
                        gt = ground_truth
                        # print("GT_agent", gt.shape)
                        reward_full = torch.squeeze(custom_reward(env,args, state_0, action_full, gt), dim=1)

                if args.step_definition == 'multi':
                    rewards = torch.cat(rewards, dim=0)  # (bx12,)
                states = torch.cat(states, dim=0)        # (bx12, 16)
                # print_structure_and_dimensions(actions)
                # print("agent")
                actions = torch.cat(actions, dim=0)      # (bx12, 2)
                # print_structure_and_dimensions(actions)
                # print("agent")
                # print("REWARDS", len(rewards), len(reward_full))
                # print("memory", action_full.shape, actions.shape)
                memory.push(state_0, action_full, reward_full, states, actions, rewards)   # initial state, 12dim action, reward (single), all intermediate states, all intermediate actions, rewards (multi)
                break

            state = next_state
            ts = ts + 1
    return memory

def reshape_batch(batch):
    states = torch.stack(batch.state)
    actions = torch.stack(batch.action, dim=0) # should be Tuple(torch.Size([b]))
    # print("batch.reward")
    # print_structure_and_dimensions(batch.reward)
    rewards = torch.stack(batch.reward, dim=0) # should be torch.Size([1, b])
    # print("reward",rewards.shape)
    states = torch.flatten(states.permute(1,0,2), 0, 1)
    actions = torch.flatten(actions.permute(1,0,2), 0, 1)
    rewards = torch.flatten(rewards.permute(1, 0))

    states_all = batch.states_all
    actions_all = batch.actions_all
    rewards_all = batch.rewards_all

    batch = (states, actions, rewards, states_all, actions_all, rewards_all)
    return batch


class Agent:

    def __init__(self, args, env, policy, device, custom_reward=None, training_step=3):
        self.args = args
        self.env = env
        self.policy = policy
        self.device = device
        self.custom_reward = custom_reward
        self.trining_step=training_step

    def collect_samples(self, mean_action=False):
        # print("TRAINING STEP",self.training_step)
        memory = collect_samples(self.args, self.env, self.policy, self.custom_reward, self.device, mean_action,  self.training_step)        
        # print("memory", memory.memory[0][0])
        batch = memory.sample()
        
        # print("batch", batch[0][0])
        # print("equal", torch.equal(memory.memory[0][0], batch[0][0]))
        
        # print("batch_structure")
        # memory.print_structure_and_dimensions(batch)
        # print("memory_structure")
        # memory.print_structure_and_dimensions(memory.memory)
        batch = reshape_batch(batch)

        return batch

