import torch
import math
from torch.nn import BCELoss

BCE_loss = BCELoss()
import random
from irl.utils import *
from irl.advantages import calculate_return
import torch.nn.functional as F
from irl.utils import (
    l2_loss

)

def add_noise_to_states(states):
    noise_dim = 2
    noise_std = 0.05
    pad_dim = states.shape[1] - noise_dim
    noise_shape = (states.shape[0], noise_dim)
    device = states.device
    noise = torch.randn(noise_shape).to(device)
    noise = noise * noise_std

    pad = (pad_dim, 0, 0, 0)
    noise = F.pad(noise, pad, "constant", 0)  # effectively zero padding

    noisy_states = states + noise
    return noisy_states


def discriminator_step(args,env, discriminator_net, discriminator_opt, discriminator_crt, expert_state_actions, pred_state_actions, device, train):
    if args.model=='original':
        
        g_o = discriminator_net(pred_state_actions)
        e_o = discriminator_net(expert_state_actions)
    elif args.model=='stgat':
        if env.training_step == 1 or env.training_step == 2:
            first_16 = expert_state_actions[:, :16]
            expert_state_actions=torch.cat([first_16, first_16], dim=1) 
            g_o = discriminator_net(pred_state_actions,  obs_traj_pos=None,  seq_start_end=env.seq_start_end, teacher_forcing_ratio=1, training_step=env.training_step)  # generated/policy scores
            e_o = discriminator_net(expert_state_actions,obs_traj_pos=None,  seq_start_end=env.seq_start_end, teacher_forcing_ratio=1, training_step=env.training_step)  # expert scores
        else:
            g_o = discriminator_net(pred_state_actions,  obs_traj_pos=None,  seq_start_end=env.seq_start_end, teacher_forcing_ratio=0, training_step=env.training_step)  # generated/policy scores
            e_o = discriminator_net(expert_state_actions, obs_traj_pos=None, seq_start_end=env.seq_start_end, teacher_forcing_ratio=0, training_step=env.training_step)  # expert scores
    # g_o.shape=torch.Size([706, 1])
    

    g_o_l = torch.zeros((pred_state_actions.shape[0], 1), device=device)
    e_o_l = torch.ones((expert_state_actions.shape[0], 1), device=device)
    d_loss_policy = discriminator_crt(g_o, g_o_l)  # fake
    d_loss_expert = discriminator_crt(e_o, e_o_l)  # real
    discrim_loss = d_loss_policy + d_loss_expert

    discriminator_opt.zero_grad()

    if train:
        discrim_loss.backward()
        discriminator_opt.step()
    e_o=e_o.detach()
    g_o=g_o.detach()
    d_loss_policy=d_loss_policy.detach()
    d_loss_expert=d_loss_expert.detach()
    discrim_loss=discrim_loss.detach()
    return discrim_loss

def value_step(states, returns, value_net, value_crt):

    bl_pred = torch.squeeze(value_net(states))
    bl_trgt = returns
    value_loss = value_crt(bl_pred, bl_trgt)
    return value_loss


def ppo_step(states, states_v, actions, returns, advantages, fixed_log_probs, policy_net, value_net, optimizer_policy, optimizer_value, train, args):

    # for now assuming epochs and minibatches are 1
    optim_value_iternum = 1
    l2_reg = 0
    clip_epsilon = args.ppo_clip

    for _ in range(args.ppo_iterations):

        """update critic"""
        for _ in range(optim_value_iternum):
            values_pred = value_net(states_v).squeeze()
            value_loss = (values_pred - returns).pow(2).mean()                              # value loss
            # weight decay
            for param in value_net.parameters():
                value_loss += param.pow(2).sum() * l2_reg
            optimizer_value.zero_grad()

            if train:
                value_loss.backward()
                optimizer_value.step()

        """update policy"""
        log_probs = policy_net.get_log_prob(states, actions).squeeze()
        if args.step_definition == 'single':
            pred_len = args.pred_len
            batchsize = math.ceil(states.shape[0] / pred_len)
            log_probs_reshaped = torch.reshape(log_probs, (pred_len, batchsize)).T  # transpose / dim=0? same but doenst look logically
            log_probs_sum = log_probs_reshaped.sum(dim=1)  # (b, 12, 1)
            
            log_probs = log_probs_sum
        ratio = torch.exp(log_probs - fixed_log_probs)
        if args.training_algorithm == 'ppo_only':  ##### added this to check ppo without baseline
            advantages = returns
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        policy_surr = -torch.min(surr1, surr2).mean()                                       #ppo loss tensorboadrd
        optimizer_policy.zero_grad()

        if train:
            policy_surr.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 40)
            optimizer_policy.step()

    return policy_surr, value_loss


def reinforce_step(args, env, policy_net, optimizer_policy, custom_reward, states_all, actions_all, rewards_all, rewards, expert, train, value_net=None, optimizer_value=None, value_crt=None, training_step=3):    #probs rename to a more general name

    states = states_all[0]      # (bx12, 16) batch of combined 12step (intermediate black box) states
    pred_len = args.pred_len
    obs_len=args.obs_len
    
    if args.model=="original":
        batchsize = math.ceil(states.shape[0] / pred_len)  
    elif args.model=="stgat" and training_step==3:
        batchsize = math.ceil(states.shape[0] / pred_len)
    elif args.model=="stgat" and (training_step==1 or training_step==2):
        batchsize = math.ceil(states.shape[0] / obs_len)
    value_loss = 0

    if args.randomness_definition == 'stochastic':
        if args.model=="stgat":
            obs_traj=None
            # print("USED TRAINING STEP",training_step )
            ###################
            #Check why this is here, it should be only
            ################
            if training_step == 1 or training_step == 2:
                # print("states", states)
                # print("Weights of traj_hidden2pos layer:", policy_net.traj_hidden2pos.weight.data)
                # print("Biases of traj_hidden2pos layer:", policy_net.traj_hidden2pos.bias.data)
                actions_mean, _, _ = policy_net(states[0:batchsize], obs_traj,env.seq_start_end ,1, training_step)  
            else: 
                state_reversed=states[0:batchsize]
                # print("state_reversed",state_reversed.shape)
                original_shape = (state_reversed.shape[0],args.obs_len, 2)  
                # print("state_reversed",state_reversed.shape)
                state_reversed = state_reversed.view(original_shape)
                inter=state_reversed.permute(1,0,2)
                # print("shapes", inter.shape, env.pred_traj_gt_rel.shape)
                model_input = torch.cat((inter, env.pred_traj_gt_rel), dim=0)
                # print("MODEL INPUT", model_input.shape)
                actions_mean, _, _ = policy_net(model_input, obs_traj,env.seq_start_end ,0, training_step)

        if args.model=="original":
                    log_probs = policy_net.get_log_prob(states, actions_all[0])  # (bx12, 1) 

                    # print("actions_all", actions_all[0].shape)
        elif args.model=="stgat":
                    # print("actions_all", actions_all[0].shape) # ([1480, 2])
                    log_probs = policy_net.get_log_prob(states, actions_all[0], env,obs_traj_pos=None)  # (bx12, 1)

        # print("log_probs", log_probs.shape)
        if args.step_definition == 'single':
            #  do some batch summing to make it (610,1)
            # log_probs.shape= ([bX12, 1])
            if  args.model=='stgat' and training_step!=3:
                log_probs_reshaped = torch.reshape(log_probs, (obs_len, batchsize)).T      # transpose / dim=0? same but doenst look logically shape=([b,12])
            else:
                log_probs_reshaped = torch.reshape(log_probs, (pred_len, batchsize)).T    #(b,12)

            log_probs_sum = log_probs_reshaped.sum(dim=1)                               # (b)
            log_probs = log_probs_sum
            returns = rewards
            ###################
            # STGAT LOSS PER SCENE 
            ##################
            # state=all initial observed steps of all pedestrians in scene
            # actionsall next steps geenrated per scene of trajectoris of all pedestrians
            if args.model=="stgat" and training_step==3:
                states_part = states[0:batchsize] 
                # print("states_part", states_part.shape)

                pred_traj_fake_rel = torch.reshape(actions_mean, (pred_len, batchsize, 2))
                # actions_mean (12,b,2)
                # pred_traj_fake_rel (12,b,2)
                # states_part (b,16)
                # gt (b,40)
                # FORMERLY KNOWN AS actions_part               
                l2_loss_rel=[]

                loss_mask=env.loss_mask[:, args.obs_len :]
                original_shape = (states_part.shape[0],8, 2)  
                state_reversed = states_part.view(original_shape)
                state_reversed=state_reversed.permute(1,0,2)
                l2_loss_rel.append(
                l2_loss(actions_mean,  env.pred_traj_gt_rel, loss_mask, mode="raw")
                )
                scene_loss= [] #torch.empty(0).to(states_part)
                log_probs_scene= [] #torch.empty(0).to(states_part)
                policy_loss = torch.zeros(1).to(states_part) # list([(b)])
                l2_loss_rel = torch.stack(l2_loss_rel, dim=1) # list b*[(1)]
                for start, end in env.seq_start_end.data:
                    _l2_loss_rel = torch.narrow(l2_loss_rel, 0, start, end - start)
                    #_l2_loss_rel.shape[scene_trajectories,1]
                    _l2_loss_rel = torch.sum(_l2_loss_rel, dim=0)  # [20]
                    _l2_loss_rel = torch.min(_l2_loss_rel) / (
                        (pred_traj_fake_rel.shape[0]) * (end - start)
                    )                       # average per pedestrian per scene
                    # policy_loss += _l2_loss_rel# delete this
                    _log_probs_sum= torch.narrow(log_probs_sum, 0, start, end - start)
                    _log_probs_scene=torch.sum(_log_probs_sum, dim=0)
                    log_probs_scene.append(_log_probs_scene) #torch.cat((log_probs_scene, _log_probs_scene), 0)

                    scene_loss.append(_l2_loss_rel)#torch.cat((scene_loss, _l2_loss_rel), 0)
                    
                    
            elif args.model=="stgat" and (training_step==1 or training_step==2):
                states_part = states[0:batchsize] 
                pred_traj_fake_rel = torch.reshape(actions_mean, (obs_len, batchsize, 2))
                l2_loss_rel=[]
                loss_mask=env.loss_mask[:, args.obs_len :]
                # if training_step == 1 or training_step == 2:
                original_shape = (states_part.shape[0],8, 2)  
                state_reversed = states_part.view(original_shape)
                state_reversed=state_reversed.permute(1,0,2)
                l2_loss_rel.append(-l2_loss(state_reversed, actions_mean, loss_mask, mode="raw")
                )
                scene_loss= [] #torch.empty(0).to(states_part)
                log_probs_scene= [] #torch.empty(0).to(states_part)
                policy_loss = torch.zeros(1).to(states_part) # list([(b)])
                # print("Original l2_loss_rel", len(l2_loss_rel), l2_loss_rel[0].shape)
                # print("TYPES", l2_loss_rel.shape, log_probs_reshaped.shape)
                l2_loss_rel = torch.stack(l2_loss_rel, dim=1).to(states_part) # list b*[(1)]
                # log_probs_reshaped = torch.stack(log_probs_reshaped, dim=1)
                for start, end in env.seq_start_end.data:
                    
                    # print("l2_loss_rel", len(l2_loss_rel), l2_loss_rel[0].shape)
                    _l2_loss_rel = torch.narrow(l2_loss_rel, 0, start, end - start)
                    # print("_l2_loss_rel", _l2_loss_rel.shape) # [scene_trajectories,1]
                    _l2_loss_rel = torch.sum(_l2_loss_rel, dim=0)  # [20]
                    _l2_loss_rel = torch.min(_l2_loss_rel) / (
                        (pred_traj_fake_rel.shape[0]) * (end - start)
                    )                       # average per pedestrian per scene
                    # print("_l2_loss_rel", _l2_loss_rel)
                    policy_loss += _l2_loss_rel
                    _log_probs_sum= torch.narrow(log_probs_sum, 0, start, end - start)
                    _log_probs_scene=torch.sum(_log_probs_sum, dim=0)
                    log_probs_scene.append(_log_probs_scene)#torch.cat((log_probs_scene, _log_probs_scene), 0)

                    scene_loss.append(_l2_loss_rel)#torch.cat((scene_loss, _l2_loss_rel), 0)
                    # print("type", scene_loss, log_probs_scene)
            #########################
           
            if (args.training_algorithm == 'baseline' or args.training_algorithm == 'ppo' or args.training_algorithm == 'ppo_only'):       #added if statement
                states_v = states[0:batchsize]


        elif args.step_definition == 'multi':
            log_probs = log_probs.squeeze()
            returns = calculate_return(rewards_all[0], pred_len, batchsize, gamma=args.discount_factor)
            if (args.training_algorithm == 'baseline' or args.training_algorithm == 'ppo' or args.training_algorithm == 'ppo_only'):       #added if statement
                states_v = states

        if args.training_algorithm == 'baseline':
            # do reinforce with baseline stuff
            with torch.no_grad():
                values = value_net(states_v).squeeze()
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / advantages.std() # normalize advantages according to ppo
            policy_loss = -(advantages * log_probs).mean()
        elif (args.training_algorithm == 'ppo' or args.training_algorithm == 'ppo_only'):
            with torch.no_grad():
                values = value_net(states_v).squeeze()
            fixed_log_probs = log_probs.detach()
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / advantages.std() # normalize advantages according to ppo
            policy_loss, value_loss = ppo_step(states, states_v, actions_all[0], returns, advantages, fixed_log_probs, policy_net, value_net, optimizer_policy, optimizer_value, train, args)

        else:

            # print_structure_and_dimensions(log_probs)
            if args.model=='original':
                policy_loss = -(returns * log_probs).mean()
            elif args.model=="stgat":
                log_probs_scene=torch.stack(log_probs_scene)
                scene_loss=torch.stack(scene_loss).detach()
                policy_loss=-(scene_loss*log_probs_scene).mean()
        if args.model=="stgat":
            returns=returns.detach()
    elif args.randomness_definition == 'deterministic':
        gt = expert          
        # print("action_mean",actions_mean.shape) #Here it is already fine
        if args.step_definition == 'single':
            # compute trajactories
            obs_traj=None
            states_part = states[0:batchsize]  # only the first batch of initial states is kept (610, 16)
            if args.model=="original":
                actions_mean, _, _ = policy_net(states)  # (7320, 2)  # action_mean is the same as acions_all! (floating point drift difference?)
                actions_part = torch.reshape(actions_mean, (pred_len, batchsize, 2))  # reshape from (bx12, 2) to (12, b, 2)
                actions_part = actions_part.permute(1, 0, 2).flatten(1, 2)  # reorder from (12, b, 2) to (b, 24)
                # compute loss

                rewards = custom_reward(env,args, states_part, actions_part, gt)  # (610, 1) =(b,1)            
                policy_loss = -rewards.mean()  # tensor(float)
                # actions_mean (8,b,2)
                # actions_part (b,16)
                # states_part (b,16)
            elif args.model=="stgat" and training_step==3:
                state_reversed=states[0:batchsize]
                original_shape = (state_reversed.shape[0],args.obs_len, 2)  
                state_reversed = state_reversed.view(original_shape)
                inter=state_reversed.permute(1,0,2)
                model_input = torch.cat((inter, env.pred_traj_gt_rel), dim=0)
                # print("shapes", inter.shape, env.pred_traj_gt_rel.shape)
                l2_loss_rel=[]
                for _ in range(args.best_k):
                    
                    actions_mean, _, _ = policy_net(model_input, obs_traj,env.seq_start_end ,0, training_step)
                    pred_traj_fake_rel = torch.reshape(actions_mean, (pred_len, batchsize, 2))
                    # actions_mean (12,b,2)
                    # pred_traj_fake_rel (12,b,2)
                    # states_part (b,16)
                    # gt (b,40)
                    # FORMERLY KNOWN AS actions_part
                    # print("actions_part", actions_part.shape)
                    # print("states_part", states_part.shape)
                    # print("env.pred_traj_gt_rel", env.pred_traj_gt_rel.shape)
                    # print("gt", gt.shape)                
                    
                    # print("actions_part", actions_part.shape)
                    # print("states_part", states_part.shape)
                    loss_mask=env.loss_mask[:, args.obs_len :]
                    original_shape = (states_part.shape[0],8, 2)  
                    state_reversed = states_part.view(original_shape)
                    state_reversed=state_reversed.permute(1,0,2)
                    l2_loss_rel.append(
                    l2_loss(actions_mean,  env.pred_traj_gt_rel, loss_mask, mode="raw")
                    )
                policy_loss = torch.zeros(1).to(l2_loss_rel[0]) # list([(b)])
                # print("Original l2_loss_rel", len(l2_loss_rel), l2_loss_rel[0].shape)
                l2_loss_rel = torch.stack(l2_loss_rel, dim=1) # list b*[(1)]
                for start, end in env.seq_start_end.data:
                    # print("l2_loss_rel", len(l2_loss_rel), l2_loss_rel[0].shape)
                    _l2_loss_rel = torch.narrow(l2_loss_rel, 0, start, end - start)
                    # print("_l2_loss_rel", _l2_loss_rel.shape) # [scene_trajectories,1]
                    _l2_loss_rel = torch.sum(_l2_loss_rel, dim=0)  # [20]
                    _l2_loss_rel = torch.min(_l2_loss_rel) / (
                        (pred_traj_fake_rel.shape[0]) * (end - start)
                    )                       # average per pedestrian per scene
                    policy_loss += _l2_loss_rel
            elif args.model=="stgat" and (training_step==1 or training_step==2):
                l2_loss_rel=[]
                for _ in range(args.best_k):
                    print("MODEL INPUT")
                    actions_mean, _, _ = policy_net(states[0:batchsize], obs_traj,env.seq_start_end ,1, training_step)  
                    pred_traj_fake_rel = torch.reshape(actions_mean, (obs_len, batchsize, 2))
                    
                    loss_mask=env.loss_mask[:, args.obs_len :]
                    # if training_step == 1 or training_step == 2:
                    original_shape = (states_part.shape[0],8, 2)  
                    state_reversed = states_part.view(original_shape)
                    state_reversed=state_reversed.permute(1,0,2)
                    l2_loss_rel.append(
                    l2_loss(state_reversed, actions_mean, loss_mask, mode="raw")
                    )
                policy_loss = torch.zeros(1).to(l2_loss_rel[0]) # list([(b)])
                # print("Original l2_loss_rel", len(l2_loss_rel), l2_loss_rel[0].shape)
                l2_loss_rel = torch.stack(l2_loss_rel, dim=1) # list b*[(1)]
                for start, end in env.seq_start_end.data:
                    # print("l2_loss_rel", len(l2_loss_rel), l2_loss_rel[0].shape)
                    _l2_loss_rel = torch.narrow(l2_loss_rel, 0, start, end - start)
                    # print("_l2_loss_rel", _l2_loss_rel.shape) # [scene_trajectories,1]
                    _l2_loss_rel = torch.sum(_l2_loss_rel, dim=0)  # [20]
                    _l2_loss_rel = torch.min(_l2_loss_rel) / (
                        (pred_traj_fake_rel.shape[0]) * (end - start)
                    )                       # average per pedestrian per scene
                    
                    policy_loss += _l2_loss_rel
            
            # OLD CODE ################
            # NOTE: pred_traj_fake_rel is actions_part
            # actions_part = actions_part.permute(1, 0, 2).flatten(1, 2)  # reorder from (12, b, 2) to (b, 24)
            # # compute loss
            # rewards = custom_reward(args, states_part, actions_part, gt)  # (610, 1) =(b,1)            
            # policy_loss = -rewards.mean()  # tensor(float)
            ###########################
        elif args.step_definition == 'multi':
            rewards = custom_reward(args, states, actions_mean, gt)  # (bx12, 1)
            returns = calculate_return(rewards, pred_len, batchsize, gamma=args.discount_factor)
            policy_loss = -returns.mean()  # tensor(float)
            
    optimizer_policy.zero_grad()


    if train:
        if (args.training_algorithm == 'ppo' or args.training_algorithm == 'ppo_only'):
            return policy_loss, value_loss   # change to something usefull, but avoid policy_loss backward/update
        if args.training_algorithm == 'baseline':
            optimizer_value.zero_grad()
            value_loss = value_step(states_v, returns, value_net, value_crt)
            for param in value_net.parameters():
                value_loss += param.pow(2).sum() * 0
            value_loss.backward()
            optimizer_value.step()
        # policy_loss=-policy_loss
        print("policy_loss", policy_loss)
        policy_loss.backward()
        optimizer_policy.step()
    if args.model=="stgat":
        # returns=returns.detach()
        rewards=rewards.detach()
        l2_loss_rel=l2_loss_rel.detach()
        _l2_loss_rel=_l2_loss_rel.detach()
        policy_loss=policy_loss.detach()
        actions_mean=actions_mean.detach()
        pred_traj_fake_rel=pred_traj_fake_rel.detach()
    # print("acion_mean", actions_mean)
    return policy_loss, value_loss



def gan_g_loss(scores_fake, label_smoothening=False):
    """
    Input:
    - scores_fake: Tensor of shape (N,) containing scores for fake samples

    Output:
    - loss: Tensor of shape (,) giving GAN generator loss
    """
    if label_smoothening:
        y_fake = torch.ones_like(scores_fake) * random.uniform(0.7, 1.2)
    else:
        y_fake = torch.ones_like(scores_fake)

    return BCE_loss(scores_fake, y_fake)


"""Update generator step"""
def generator_step(inputs, generator, discriminator, optimizer_g, args, train):

    inputs = inputs[0]
    gen_out, _, _ = generator(inputs)

    # compute trajactories
    pred_len = args.pred_len
    batchsize = math.ceil(inputs.shape[0] / pred_len)
    observed_part = inputs[0:batchsize]  # only the first batch of initial states is kept
    predicted_part = torch.reshape(gen_out, (batchsize, pred_len * 2))

    full_trajectories = torch.cat((observed_part, predicted_part), dim=1)  # (b, 40) same as in discriminator update
    scores_gen = discriminator(full_trajectories)
    generator_loss = gan_g_loss(scores_gen)

    optimizer_g.zero_grad()
    if train:
        generator_loss.backward()
        optimizer_g.step()

    return generator_loss
