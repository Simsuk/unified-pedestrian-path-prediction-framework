
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
        env.mask_disc=[]
        obs_traj=None
        bs = state.shape[0]             # batch size
        # print("BATCH SIZE", bs)
        rewards = []
        states = []
        actions = []
        env.pred_state_actions=[]
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
                    # seed = time.time_ns()
                    seed=args.seed
                    action_all = policy.select_action(model_input, obs_traj, seq_start_end ,seed, training_step)
                    actions_mean, _, _ = policy(model_input, obs_traj,env.seq_start_end ,0, training_step)
                    policy.mean_dist=actions_mean
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
            state_action =torch.cat((state, action), dim=1)
            padded_tensor = F.pad(state_action, (0, 40-state_action.shape[1])) 
            mask = padded_tensor == 0  
            env.mask_disc.append(mask)
            env.pred_state_actions.append(padded_tensor)
            next_state, reward, done, = env.step(state, action)
            # print("Agent_reward", reward.shape) #should be torch.Size([b, 1])
            if custom_reward is not None:
                if args.step_definition == 'multi':
                     # take the ground truth of the correct timestep
                    if args.model=='original':
                        gt = ground_truth[ts * bs:(ts + 1) * bs, :]
                        reward = torch.squeeze(custom_reward(env,args, state, action, gt), dim=1) #[b,]
                    elif args.model=='stgat' and args.loss_definition=='discriminator':
                        gt= ground_truth[:bs, :(args.obs_len+ts+1)*2]
                        reward = torch.squeeze(custom_reward(env,args, state, action, gt), dim=1)
                    elif args.model=='stgat' and args.loss_definition=='l2':
                        if args.reward=='cumulative':
                            state_action = torch.cat((state, action), dim=1)
                            gt=ground_truth[:bs, :(args.obs_len+ts+1)*2]
                            l2_loss_rel=(gt - state_action)**2 
                            reward=-l2_loss_rel.sum(dim=1, keepdim=True)
                        else:
                            # gt = ground_truth[ts * bs:(ts + 1) * bs, :]
                            #                         actions_mean, _, _ = policy_net(model_input, obs_traj,env.seq_start_end ,0, training_step)
                            # pred_traj_fake_rel = torch.reshape(actions_mean, (pred_len, batchsize, 2))
                            state_action = torch.cat((state, action), dim=1)  # (b, 16) appended action, discarded first time step
                            loss_mask=env.loss_mask[:, args.obs_len :]
                            gt = ground_truth[ts * bs:(ts + 1) * bs, :]
                            l2_loss_rel=(gt - state_action)**2 
                            reward=-l2_loss_rel.sum(dim=1, keepdim=True)
                            # gt = ground_truth[:, :state_action.shape[1]] # take the ground truth of the correct timestep
                            ##########################
                            # THIS REWARD IS USELESS Do NOT USE
                            # correct reward is calculated inside the update_parameters.py
                            
                            ###############
                            # original_shape = (states_part.shape[0],8, 2)  
                            # state_reversed = states_part.view(original_shape)
                            # state_reversed=state_reversed.permute(1,0,2)
                            # l2_loss_rel=[]
                            ##################
                            
                            # l2_loss_rel2=[]
                            # loss_mask=env.loss_mask[:, args.obs_len :]
                            # original_shape = (state_0.shape[0],8, 2)  
                            # state_reversed = state_0.view(original_shape)
                            # state_reversed=state_reversed.permute(1,0,2)
                            # actions_tensor=torch.stack(actions)
                            # l2_loss_rel2=l2_loss(actions_tensor,  env.pred_traj_gt_rel, loss_mask, mode="full") #[b,12] 
                            
                            
                            
                            ####################
                            # l2_loss_rel.append(
                            # l2_loss(state_action,  gt, loss_mask, mode="raw")
                            # )
                            # l2_loss_rel.shape= [b,18]
                            # gt [b,18]
                            # state_action [b,18]
                            
                            # policy_loss = torch.zeros(1).to(l2_loss_rel[0]) # list([(b)])
                            # l2_loss_rel = torch.stack(l2_loss_rel, dim=1) # list b*[(1)]
                            # l2_loss_rel=[]
                            # loss_mask=env.loss_mask[:, args.obs_len :]
                            # original_shape = (state_0.shape[0],8, 2)  
                            # state_reversed = state_0.view(original_shape)
                            # state_reversed=state_reversed.permute(1,0,2)
                            # actions_tensor=torch.stack(actions)
                            # l2_loss_rel.append(
                            # l2_loss(actions_tensor,  env.pred_traj_gt_rel, loss_mask, mode="raw")
                            # )
                            # reward_full = -torch.stack(l2_loss_rel, dim=1).squeeze() # list b*[(1)]
                            # # reward_full2 = torch.squeeze(custom_reward(env,args, state_0, action_full, gt), dim=1)
                            
                            #THIS ONE TO USE

                        
                        # policy_loss=l2_loss_rel.mean()   
                rewards.append(reward)

            if done:
                action_full = torch.cat(actions, dim=1) # (b, 24) or (b,16) in case of phase 1 and 2
                if custom_reward is not None:
                    if args.step_definition == 'single' or args.step_definition == 'multi':
                        
                        # print("GT_agent", gt.shape)
                        if args.model=='stgat':
                            gt = ground_truth
                            l2_loss_rel=[]
                            loss_mask=env.loss_mask[:, args.obs_len :]
                            original_shape = (state_0.shape[0],8, 2)  
                            state_reversed = state_0.view(original_shape)
                            state_reversed=state_reversed.permute(1,0,2)
                            actions_tensor=torch.stack(actions)
                            l2_loss_rel.append(
                            l2_loss(actions_tensor,  env.pred_traj_gt_rel, loss_mask, mode="raw")
                            )
                            reward_full = -torch.stack(l2_loss_rel, dim=1).squeeze() # list b*[(1)]
                            # reward_full2 = torch.squeeze(custom_reward(env,args, state_0, action_full, gt), dim=1)
                            # print("reward_full2")
                            # reward_full = torch.squeeze(custom_reward(env,args, state_0, action_full, gt), dim=1)
                    
                        elif args.model=='original':
                            if args.step_definition == 'single':
                                gt = ground_truth
                                reward_full = torch.squeeze(custom_reward(env,args, state_0, action_full, gt), dim=1)
                if args.loss_definition=='l2':             
                    if args.step_definition == 'multi':
                        rewards = torch.cat(rewards, dim=0)  # (bx12,)
                    if args.reward=='cumulative' and args.randomness_definition=='stochastic' and args.step_definition == 'multi':
                        states=state_0 
                    else:
                       states = torch.cat(states, dim=0)        # (bx12, 16)
                       
                    actions = torch.cat(actions, dim=0)      # (bx12, 2)
                else:
                    if args.step_definition == 'single':
                        rewards = torch.cat(rewards, dim=0)
                        states = torch.cat(states, dim=0) 
                    actions = torch.cat(actions, dim=0)  # (bx12, 2)
                    if args.step_definition == 'multi':
                        rewards = torch.cat(rewards, dim=0)  # (bx12,)
                        env.pred_state_actions= torch.cat(env.pred_state_actions, dim=0)
                        env.mask_disc=torch.cat(env.mask_disc, dim=0)
                # print_structure_and_dimensions(actions)
                # print("agent")
                
                # print_structure_and_dimensions(actions)
                # print("agent")
                # print("REWARDS", len(rewards), len(reward_full))
                # print("memory", action_full.shape, actions.shape)
                memory.push(state_0, action_full, reward_full, states, actions, rewards)   # initial state, 12dim action, reward (single), all intermediate states, all intermediate actions='actions_all, rewards (multi)=rewards_all
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

