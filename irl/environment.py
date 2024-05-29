import torch
import os

class Environment:

    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.obs_traj = None
        self.pred_traj_gt = None
        self.obs_traj_rel = None
        self.pred_traj_gt_rel = None
        self.obs_len = None
        self.pred_len = None
        self.traj_len = None
        self.step_counter = 0
        self.path_counter = 0
        self.seq_start_end = None
        self.training_step=3
        self.loss_mask= None
    def generate(self, batch_input):

        batch_input = [tensor.to(self.device) for tensor in batch_input]

        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
         loss_mask, seq_start_end) = batch_input
        # print("BATCH_LEN", obs_traj.shape[1])
        # print("obs_traj", obs_traj.shape)
        # print("pred_traj_gt", pred_traj_gt.shape)
        self.obs_traj = obs_traj
        self.pred_traj_gt = pred_traj_gt
        self.obs_traj_rel = obs_traj_rel
        self.pred_traj_gt_rel = pred_traj_gt_rel
        self.seq_start_end=seq_start_end
        self.loss_mask=loss_mask
        self.obs_len = obs_traj.shape[0]
        self.pred_len = pred_traj_gt.shape[0]
        self.traj_len = self.obs_len + self.pred_len
        self.total_paths = obs_traj.shape[1]
        self.step_counter = 0
        self.path_counter = 0


    def reset(self):
        # Get a batch of paths and permute them from (8, b, 2) to (b, 8, 2)
        # use the flatten function (in the last two dimensions) to make it (b, 16) with alternating order
        # (x,y,x,y, etc.) this is easier to compute the next_state step later

        self.step_counter = 0
        # print("original state", self.obs_"batchtraj_rel.shape)
        self.obs_traj_rel[0]
        state = self.obs_traj_rel.permute(1,0,2)
        state_reshaped = torch.flatten(state, 1, 2)
        
        # original_shape = (state.shape[0],8, 2)  # This is the shape you want to go back to
        # state_reversed = state_reshaped.view(original_shape)
        # # state_reversed = state_reshaped.reshape(original_shape)
        # state_reversed=state_reversed.permute(1,0,2)
        # print("Equal", (self.obs_traj_rel==state_reversed), state_reversed.shape)
        return state_reshaped, self.seq_start_end, self.obs_traj_rel        # (b, 16) in x,y,x,y format checked

    def step(self, state, action):

        state = state.to(self.device)   #(b, 16)
        action = action.to(self.device) #(b, 2)
        if self.args.model=='stgat' and self.args.step_definition=='multi':
            next_state = torch.cat((state, action), dim=1)[:, -self.obs_len*2:]  # (b, 16) appended action, discarded first time step
        else:
            next_state = torch.cat((state, action), dim=1)[:, -self.obs_len*2:]  # (b, 16) appended action, discarded first time step
        reward = torch.zeros((state.shape[0], 1), device=self.device)
        self.step_counter = self.step_counter + 1
        if self.training_step==1 or self.training_step==2:
            if self.step_counter == self.obs_len:
                done = True
            else:
                done = False
        else:
            if self.step_counter == self.pred_len:
                done = True
            else:
                done = False

        return next_state, reward, done  # checked

    def collect_expert(self):
        # # we take the observed trajectory (initial state) and the ground truth 'predicted' trajectory for each
        # # path in the batch. We concatenate them to create the full path. Then to recreate as much path parts as
        # # we predict, we take a slice of the full path (of size obs_len + 1) for the amount of pred_len steps.
        # # the flatten function takes care of the xyxy ordering of the coordinates.
        state = self.obs_traj_rel.permute(1, 0, 2)  # (b, 8, 2)
        state = torch.flatten(state, 1, 2)  # (b, 16)
        gt = self.pred_traj_gt_rel.permute(1, 0, 2)  # (b, 12, 2)
        gt_sum = torch.sum(gt, dim=1)
        gt = torch.flatten(gt, 1, 2)  # (b,24)
        # print("gt_env", gt.shape, os.path.basename(__file__))
        if self.args.step_definition == 'single' or self.args.model=='stgat':
            expert_single = torch.concat((state, gt), dim=1)   #(b,40)
            expert = expert_single

        elif self.args.step_definition == 'multi' and self.args.model=='original':
            sa_len = state.shape[1] + 2         # 16 + 2 = 18
            expert_state_actions = []
            expert_full = torch.concat((state, gt), dim=1)  # (b,40)
            for i in range(self.args.pred_len):
                state_action = expert_full[:, i*2:(i*2) + sa_len]   # sliding window of (b,18) for 12 steps
                expert_state_actions.append(state_action)
            expert_multi = torch.cat(expert_state_actions, dim=0)     #(bx12, 18)
            # expert = expert_multi
            expert=expert_full
        return expert

