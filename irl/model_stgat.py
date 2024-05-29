import torch.nn as nn
from irl.utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

def get_noise(shape, noise_type):
    if noise_type == "gaussian":
        return torch.randn(*shape).cuda()
    elif noise_type == "uniform":
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


# this efficient implementation comes from https://github.com/xptree/DeepInf/
class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.w = nn.Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = nn.Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = nn.Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter("bias", None)

        nn.init.xavier_uniform_(self.w, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

    def forward(self, h):
        bs, n = h.size()[:2]
        h_prime = torch.matmul(h.unsqueeze(1), self.w)
        attn_src = torch.matmul(h_prime, self.a_src)
        attn_dst = torch.matmul(h_prime, self.a_dst)
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(
            0, 1, 3, 2
        )
        attn = self.leaky_relu(attn)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)
        if self.bias is not None:
            return output + self.bias, attn
        else:
            return output, attn

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.n_head)
            + " -> "
            + str(self.f_in)
            + " -> "
            + str(self.f_out)
            + ")"
        )


class GAT(nn.Module):
    def __init__(self, n_units, n_heads, dropout=0.2, alpha=0.2):
        super(GAT, self).__init__()
        self.n_layer = len(n_units) - 1
        self.dropout = dropout
        self.layer_stack = nn.ModuleList()

        for i in range(self.n_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(
                BatchMultiHeadGraphAttention(
                    n_heads[i], f_in=f_in, f_out=n_units[i + 1], attn_dropout=dropout
                )
            )

        self.norm_list = [
            torch.nn.InstanceNorm1d(32).cuda(),
            torch.nn.InstanceNorm1d(64).cuda(),
        ]

    def forward(self, x):
        bs, n = x.size()[:2]
        for i, gat_layer in enumerate(self.layer_stack):
            x = self.norm_list[i](x.permute(0, 2, 1)).permute(0, 2, 1)
            x, attn = gat_layer(x)
            if i + 1 == self.n_layer:
                x = x.squeeze(dim=1)
            else:
                x = F.elu(x.transpose(1, 2).contiguous().view(bs, n, -1))
                x = F.dropout(x, self.dropout, training=self.training)
        else:
            return x


class GATEncoder(nn.Module):
    def __init__(self, n_units, n_heads, dropout, alpha):
        super(GATEncoder, self).__init__()
        self.gat_net = GAT(n_units, n_heads, dropout, alpha)

    def forward(self, obs_traj_embedding, seq_start_end):
        graph_embeded_data = []
        for start, end in seq_start_end.data:
            curr_seq_embedding_traj = obs_traj_embedding[:, start:end, :]
            # print("curr_seq_embedding_traj", curr_seq_embedding_traj.shape)
            # print("seq_start_end", seq_start_end.shape)
            curr_seq_graph_embedding = self.gat_net(curr_seq_embedding_traj)
            graph_embeded_data.append(curr_seq_graph_embedding)
        graph_embeded_data = torch.cat(graph_embeded_data, dim=1)
        return graph_embeded_data


class TrajectoryGenerator(nn.Module):
    def __init__(
        self,
        args,
        obs_len,
        pred_len,
        traj_lstm_input_size,
        traj_lstm_hidden_size,
        n_units,
        n_heads,
        graph_network_out_dims,
        dropout,
        alpha,
        graph_lstm_hidden_size,
        noise_dim=(16,),
        noise_type="gaussian",
        log_std=None,
        action_dim=2
    ):
        super(TrajectoryGenerator, self).__init__()
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        # torch.manual_seed(72)
        # random.seed(72)
        # np.random.seed(72)
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed_all(72)

        self.args=args
        # torch.manual_seed(72)
        self.obs_len = obs_len
        self.pred_len = pred_len

        device = torch.device('cuda', index=self.args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
        # self.action_log_std = (torch.ones(1, action_dim) * log_std).to(device)
        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)
        # self.action_log_std = torch.Tensor(torch.ones(1, action_dim) * log_std).to(device)
        # self.action_log_std = torch.full((1, action_dim), fill_value=log_std, device=device)
             # existing initialization code
        self.noise_dim = noise_dim
        self.noise_type = noise_type
        self.inter_states=None
        # Initialize the fixed part of the noise
        if self.args.pretraining==True:
            self.fixed_noise = get_noise((1,) + self.noise_dim, self.noise_type)    
            print("AAAAAAAAAAAAAAAAAA")        
        else:
            self.register_buffer('fixed_noise', get_noise((1,) + self.noise_dim, self.noise_type))
        # self.pred_len = 1 if args.step_definition=="single" else pred_len

        self.gatencoder = GATEncoder(
            n_units=n_units, n_heads=n_heads, dropout=dropout, alpha=alpha
        )

        self.graph_lstm_hidden_size = graph_lstm_hidden_size
        self.traj_lstm_hidden_size = traj_lstm_hidden_size
        self.traj_lstm_input_size = traj_lstm_input_size

        self.pred_lstm_hidden_size = (
            self.traj_lstm_hidden_size + self.graph_lstm_hidden_size + noise_dim[0]
        )

        self.traj_lstm_model = nn.LSTMCell(traj_lstm_input_size, traj_lstm_hidden_size)
        self.graph_lstm_model = nn.LSTMCell(
            graph_network_out_dims, graph_lstm_hidden_size
        )
        self.traj_hidden2pos = nn.Linear(self.traj_lstm_hidden_size, 2)
        self.traj_gat_hidden2pos = nn.Linear(
            self.traj_lstm_hidden_size + self.graph_lstm_hidden_size, 2
        )
        self.pred_hidden2pos = nn.Linear(self.pred_lstm_hidden_size, 2)
        self.graph_lstm_layer_norm = nn.LayerNorm(graph_lstm_hidden_size)
        self.pred_lstm_layer_norm = nn.LayerNorm(self.pred_lstm_hidden_size)
        self.noise_dim = noise_dim
        self.noise_type = noise_type

        self.pred_lstm_model = nn.LSTMCell(
            traj_lstm_input_size, self.pred_lstm_hidden_size
        )

    def init_hidden_traj_lstm(self, batch):
        if self.args.randomness_definition=="stochastic":
            # torch.manual_seed(self.args.seed)
            pass
            
        return (
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
        )

    def init_hidden_graph_lstm(self, batch):
        if self.args.randomness_definition=="stochastic":
            # torch.manual_seed(self.args.seed)
            pass
        return (
            torch.randn(batch, self.graph_lstm_hidden_size).cuda(),
            torch.randn(batch, self.graph_lstm_hidden_size).cuda(),
        )

    def add_noise(self, _input, seq_start_end):
        noise_shape = (seq_start_end.size(0),) + self.noise_dim
        if self.args.randomness_definition=='deterministic':
            z_decoder = get_noise(noise_shape, self.noise_type)
            # print(z_decoder)
        elif self.args.randomness_definition=='stochastic':

                    dynamic_noise_shape = [seq_start_end.size(0)] + [1] * (len(self.fixed_noise.shape) - 1)
                    z_decoder = self.fixed_noise.repeat(*dynamic_noise_shape)
        # Repeat the fixed noise for each element in the batch
        
        _list = []
        for idx, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            _vec = z_decoder[idx].view(1, -1)
            _to_cat = _vec.repeat(end - start, 1)
            _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
        decoder_h = torch.cat(_list, dim=0)
        return decoder_h
    def get_log_prob(self, x, actions, env, obs_traj_pos):
        if env.training_step == 1 or env.training_step == 2:
            # x.shape ([bx12,16])
            # actions_mean ([8, 185, 2]

            # print("actions.shape")
            first_iter=int(x.shape[0]/env.obs_len)
            # print("first_iter", first_iter)
            model_input=x[:first_iter]
            # print("Weights of traj_hidden2pos layer:", self.traj_hidden2pos.weight.data)
            # print("Biases of traj_hidden2pos layer:", self.traj_hidden2pos.bias.data)
            action_mean, action_log_std, action_std = self.forward(model_input,  obs_traj_pos,  env.seq_start_end, 1, env.training_step)
            # print("actions", actions[0][0])
            # print("action_mean", action_mean[0][0])
            # print("action_log_std", action_log_std[0][0])
            # print("action_std",action_std[0][0])
            # print("env.seq_start_end", env.seq_start_end)
            
            # print("model_input", model_input[0])
            # print(action_std)
            action_mean, action_log_std, action_std = torch.flatten(action_mean, 0,1), torch.flatten(action_log_std, 0,1),torch.flatten(action_std, 0,1)
            
            # print("action_mean", action_mean[0])
            # print("action_log_std", action_log_std[0])
            # print("action_std",action_std[0])
        else:
            first_iter=int(x.shape[0]/env.pred_len)

            model_input=x[:first_iter]
            original_shape = (model_input.shape[0],env.obs_len, 2)  
            state_reversed = model_input.view(original_shape)
            inter=state_reversed.permute(1,0,2)
            model_input = torch.cat((inter, env.pred_traj_gt_rel), dim=0)
            action_mean, action_log_std, action_std = self.forward(model_input,  obs_traj_pos,  env.seq_start_end, 0, env.training_step)
            action_mean, action_log_std, action_std = torch.flatten(action_mean, 0,1), torch.flatten(action_log_std, 0,1),torch.flatten(action_std, 0,1)
            # print('as')
        return normal_log_density(actions, action_mean, action_log_std, action_std)
    def select_action(self, obs_traj_rel, obs_traj_pos,  seq_start_end,seed, training_step=3):
        if training_step == 1 or training_step == 2:
            action_mean, _, action_std = self.forward(obs_traj_rel, obs_traj_pos,  seq_start_end, 1,training_step)
            
        else:
            # print("obs_traj_rel",obs_traj_rel.shape)
            action_mean, _, action_std = self.forward(obs_traj_rel, obs_traj_pos,  seq_start_end,0, training_step)
            # print(action_std)
            # print(action_mean[0][0])
        # torch.manual_seed(seed)   
        # print(action_mean[0][0])   
        action = torch.normal(action_mean, 0.05)
        # torch.manual_seed(self.args.seed)   
        # print("action",action[0][0])

        return action
    def forward(
        self,
        obs_traj_rel,
        obs_traj_pos,
        seq_start_end,
        teacher_forcing_ratio=0.5,
        training_step=3,
    ):
        """_summary_

        Args:
            obs_traj_rel (_type_): 
                            Phase 1, 2:torch.Size([b, 2xobs_len])
                            Phase 3:   torch.Size([pred_len + obs_len, b, 2])
            
            obs_traj_pos (_type_): None
            seq_start_end (_type_): torch.Size([number of cene (64 becasue arg.batchsize=64), 2]) 
            teacher_forcing_ratio (float, optional): scalar. Defaults to 0.5.
            training_step (int, optional): scalar. Defaults to 3.

        Returns:
            _type_: _description_
        """
        # print("pretraining", self.args.pretraining)
        if (training_step==3 or self.args.pretraining==True):
            # print("forward obs_traj_rel", obs_traj_rel.shape)
            state_reversed=obs_traj_rel
        else:
            print("mistake")
            original_shape = (obs_traj_rel.shape[0],self.obs_len, 2)  
            state_reversed = obs_traj_rel.view(original_shape)
            state_reversed=state_reversed.permute(1,0,2) #added to match the framework input
        # state_reversed=obs_traj_rel
        
        batch = state_reversed.shape[1]
        
        # Causes stochasticity in output, not desired
        traj_lstm_h_t, traj_lstm_c_t = self.init_hidden_traj_lstm(batch)
        graph_lstm_h_t, graph_lstm_c_t = self.init_hidden_graph_lstm(batch)

        pred_traj_rel = []
        traj_lstm_hidden_states = []
        graph_lstm_hidden_states = []
        for i, input_t in enumerate(
            state_reversed[: self.obs_len].chunk(
                state_reversed[: self.obs_len].size(0), dim=0
            )
        ):          
            # print("input_t", i,input_t.shape)
            traj_lstm_h_t, traj_lstm_c_t = self.traj_lstm_model(
                input_t.squeeze(0), (traj_lstm_h_t, traj_lstm_c_t)
            )
            if training_step == 1:
                output = self.traj_hidden2pos(traj_lstm_h_t)
                pred_traj_rel += [output]
            else:
                traj_lstm_hidden_states += [traj_lstm_h_t]
        if training_step == 2:
            # for i, tensor in enumerate(traj_lstm_hidden_states):
                # print(f"Shape of tensor {i}:", tensor.cpu().detach().numpy().shape)
            graph_lstm_input = self.gatencoder(
                torch.stack(traj_lstm_hidden_states), seq_start_end
            )
            for i in range(self.obs_len):
                graph_lstm_h_t, graph_lstm_c_t = self.graph_lstm_model(
                    graph_lstm_input[i], (graph_lstm_h_t, graph_lstm_c_t)
                )
                encoded_before_noise_hidden = torch.cat(
                    (traj_lstm_hidden_states[i], graph_lstm_h_t), dim=1
                )
                output = self.traj_gat_hidden2pos(encoded_before_noise_hidden)
                pred_traj_rel += [output]
        if training_step == 3:
            graph_lstm_input = self.gatencoder(
                torch.stack(traj_lstm_hidden_states), seq_start_end
            )
            for i, input_t in enumerate(
                graph_lstm_input[: self.obs_len].chunk(
                    graph_lstm_input[: self.obs_len].size(0), dim=0
                )
            ):
                graph_lstm_h_t, graph_lstm_c_t = self.graph_lstm_model(
                    input_t.squeeze(0), (graph_lstm_h_t, graph_lstm_c_t)
                )
                # graph_lstm_h_t = self.graph_lstm_layer_norm(graph_lstm_h_t)
                graph_lstm_hidden_states += [graph_lstm_h_t]
        if training_step == 1 or training_step == 2:
            outputs= torch.stack(pred_traj_rel)
            action_log_std = self.action_log_std.expand_as(outputs)
            # print("self.fixed_noise", self.fixed_noise)
            
            action_std = torch.exp(action_log_std)
            return outputs,  action_log_std, action_std
        else:
            # print("traj_lstm_hidden_states",traj_lstm_hidden_states.shape, graph_lstm_hidden_states.shape)
            encoded_before_noise_hidden = torch.cat(
                (traj_lstm_hidden_states[-1], graph_lstm_hidden_states[-1]), dim=1
            )
            self.inter_states=encoded_before_noise_hidden
            pred_lstm_hidden = self.add_noise(
                encoded_before_noise_hidden, seq_start_end
            )
            # print("encoded_before_noise_hidden", encoded_before_noise_hidden[0][0])
            pred_lstm_c_t = torch.zeros_like(pred_lstm_hidden).cuda()
            output = state_reversed[self.obs_len-1]
            if self.training:
                # print("pred_lenself", self.pred_len)
                for i, input_t in enumerate(
                    state_reversed[-self.pred_len :].chunk(
                        state_reversed[-self.pred_len :].size(0), dim=0
                    )
                ):
                    teacher_force = random.random() < teacher_forcing_ratio
                    input_t = input_t if teacher_force else output.unsqueeze(0)
                    pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model(
                        input_t.squeeze(0), (pred_lstm_hidden, pred_lstm_c_t)
                    )
                    # pred_lstm_hidden = self.pred_lstm_layer_norm(pred_lstm_hidden)
                    output = self.pred_hidden2pos(pred_lstm_hidden)
                    pred_traj_rel += [output]
                outputs = torch.stack(pred_traj_rel)
            else:
                for i in range(self.pred_len):
                    pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model(
                        output, (pred_lstm_hidden, pred_lstm_c_t)
                    )
                    output = self.pred_hidden2pos(pred_lstm_hidden)
                    pred_traj_rel += [output]
                outputs = torch.stack(pred_traj_rel)
            # print("outputs",outputs.shape)

            action_log_std = self.action_log_std.expand_as(outputs)
            action_std = torch.exp(action_log_std)
            return outputs, action_log_std, action_std

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=(128, 128), activation='tanh', log_std=None):
        super().__init__()
        self.is_disc_action = False
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.action_mean = nn.Linear(last_dim, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        if log_std is not None:
            self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)
        else:
            self.action_log_std = nn.Linear(last_dim, 1)
            self.action_log_std.weight.data.mul_(0.1)
            self.action_log_std.bias.data.mul_(0.0)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        action_mean = self.action_mean(x)
        if not isinstance(self.action_log_std, nn.modules.linear.Linear):         # if log_std was not None
            action_log_std = self.action_log_std.expand_as(action_mean)
        else:                                                           # if trainable log_std
            action_log_std = self.action_log_std(x)
            action_log_std = torch.sigmoid(action_log_std) * -2.30      # should be between -0.0 and -2.30
            action_log_std = action_log_std.expand_as(action_mean)      # make two out of one
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

    def select_action(self, x):
        action_mean, _, action_std = self.forward(x)
        action = torch.normal(action_mean, action_std)
        return action

    def get_kl(self, x):
        mean1, log_std1, std1 = self.forward(x)

        mean0 = mean1.detach()
        log_std0 = log_std1.detach()
        std0 = std1.detach()
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x, actions):
        action_mean, action_log_std, action_std = self.forward(x)
        # print("x", x.shape)
        # print("\n action_mean: \n shape:", action_mean.shape, "\n values: ", action_mean)
        # print("\n action_log_std: \n shape:", action_log_std.shape, "\n values: ", action_log_std)
        # print("\n action_std: \n shape:", action_std.shape, "\n values: ", action_std)
        # print("\n normal_log_density(actions, action_mean, action_log_std, action_std): \n shape:", normal_log_density(actions, action_mean, action_log_std, action_std).shape, "\n values: ", normal_log_density(actions, action_mean, action_log_std, action_std))
        return normal_log_density(actions, action_mean, action_log_std, action_std)

    def get_fim(self, x):
        mean, _, _ = self.forward(x)
        cov_inv = self.action_log_std.exp().pow(-2).squeeze(0).repeat(x.size(0))
        param_count = 0
        std_index = 0
        id = 0
        for name, param in self.named_parameters():
            if name == "action_log_std":
                std_id = id
                std_index = param_count
            param_count += param.view(-1).shape[0]
            id += 1
        return cov_inv.detach(), mean, {'std_id': std_id, 'std_index': std_index}


class Discriminator(nn.Module):
    """_summary_

        Input:
            generated/policy trajectories or expert trajectorie:  tate_actions torch.Size([b, 40])
        Output:
            probabilities e_o or g_o with shape torch.Size([b, 1])
    """
    def __init__(self, num_inputs, hidden_size=(128, 128), activation='relu'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.nn.ReLU(inplace=False)
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = num_inputs
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        # last_dim = last_dim * 2                 # added for shape fix
        self.logic = nn.Linear(last_dim, 1)
        self.logic.weight.data.mul_(0.1)
        self.logic.bias.data.mul_(0.0)

    def forward(self, x):
        # print("Discriminator Input", x.shape)
        for affine in self.affine_layers:
            
            x = self.activation(affine(x))
        # x = torch.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]))   # added for shape fix
        preprob = self.logic(x)
        prob = torch.sigmoid(preprob)
        return prob


class Value(nn.Module):
    def __init__(self, state_dim, hidden_size=(128, 128), activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        # self.batch_norm = nn.BatchNorm1d(state_dim, affine=True)

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        # last_dim = last_dim * 2  # added for shape fix
        self.value_head = nn.Linear(last_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        # x = self.batch_norm(x)
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        value = self.value_head(x)
        return value