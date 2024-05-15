from irl.utils import *
from irl.losses import displacement_error, final_displacement_error, l2_loss



def create_fake_trajectories(env,args,obs_traj_rel,pred_traj_gt_rel, policy,obs_len, pred_len, seq_start_end,device):
    obs_len = obs_traj_rel.shape[0]

    state = obs_traj_rel.permute(1, 0, 2)
    state = torch.flatten(state, 1, 2)
    fake_traj = state                      # (b, 16) in x,y,x,y format
    # print("BATCH", fake_traj.shape[0])
    if args.model=="original":
        for step in range(pred_len):
            # state = state_filter(state, update=False)
            action, _, _ = policy(state)
            fake_traj = torch.cat((fake_traj, action), dim=1)
            next_state = torch.cat((state, action), dim=1)[:, -obs_len * 2:]
            state = next_state
        fake_traj = fake_traj[:,-pred_len*2:]   # (b, 24)

        fake_traj = torch.reshape(fake_traj, (fake_traj.shape[0], pred_len, 2))  # (b, 12, 2)
        pred_traj_fake_rel = fake_traj.permute(1, 0, 2)  # (12, b, 2)
    elif args.model=="stgat":
        if env.training_step==3:
            # happens only in training_step==3 because only then we generate 12 trajectories
            model_input=torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)
            pred_traj_fake_rel, _, _ = policy(model_input, obs_traj_rel,seq_start_end ,0, env.training_step)         # (12, b, 2)
        elif env.training_step==1 or env.training_step==2:
            pred_traj_fake_rel, _, _ = policy(state, obs_traj_rel,seq_start_end ,1, env.training_step)    
                        

    return pred_traj_fake_rel   # (12, b, 2) or (8,b,2) if stgat in phase 1 and 2


def check_accuracy(env,args, loader, policy_net,  obs_len,pred_len, device, limit=False):

    metrics = {}
    irl_losses_abs, irl_losses_rel = ([],) * 2
    disp_error, disp_error_l, disp_error_nl = ([],) * 3
    f_disp_error, f_disp_error_l, f_disp_error_nl = ([],) * 3
    total_traj, total_traj_l, total_traj_nl = 0, 0, 0
    loss_mask_sum = 0
    policy_net.eval()
    with torch.no_grad():
        for batch in loader:

            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch
            linear_ped = 1 - non_linear_ped
            loss_mask = loss_mask[:, args.obs_len:]

            pred_traj_fake_rel = create_fake_trajectories(env,args,obs_traj_rel, pred_traj_gt_rel,policy_net,obs_len, pred_len,seq_start_end, device)
            if args.model=="original":
                # print("pred_traj_fake_rel",pred_traj_fake_rel.shape, obs_traj[-1].shape)
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
            
                irl_loss_abs, irl_loss_rel = cal_l2_losses(
                    pred_traj_gt, pred_traj_gt_rel, pred_traj_fake,
                    pred_traj_fake_rel, loss_mask, mode="sum"
                ) # original stgat uses "raw" instead
                ade, ade_l, ade_nl = cal_ade(
                    pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
                )

                fde, fde_l, fde_nl = cal_fde(
                    pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
                )
            elif args.model=="stgat":
                if env.training_step==3:
                    # print("pred_traj_fake_rel",pred_traj_fake_rel.shape, obs_traj[-1].shape)
                    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

                    irl_loss_abs, irl_loss_rel = cal_l2_losses(
                        pred_traj_gt, pred_traj_gt_rel, pred_traj_fake,
                        pred_traj_fake_rel, loss_mask, mode="sum"
                    )
                    ade, ade_l, ade_nl = cal_ade(
                        pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
                    )

                    fde, fde_l, fde_nl = cal_fde(
                        pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
                    )
                elif env.training_step==1 or env.training_step==2:
                    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

                    irl_loss_abs, irl_loss_rel = cal_l2_losses(
                        obs_traj, obs_traj_rel, pred_traj_fake,
                        pred_traj_fake_rel, loss_mask=None, mode="sum"
                    )
                    ade, ade_l, ade_nl = cal_ade(
                        obs_traj, pred_traj_fake, linear_ped, non_linear_ped
                    )

                    fde, fde_l, fde_nl = cal_fde(
                        obs_traj, pred_traj_fake, linear_ped, non_linear_ped
                    )       
            # print("irl_loss_abs",irl_loss_abs.shape)
            irl_losses_abs.append(irl_loss_abs.item())
            irl_losses_rel.append(irl_loss_rel.item())
            disp_error.append(ade.item())
            disp_error_l.append(ade_l.item())
            disp_error_nl.append(ade_nl.item())
            f_disp_error.append(fde.item())
            f_disp_error_l.append(fde_l.item())
            f_disp_error_nl.append(fde_nl.item())

            loss_mask_sum += torch.numel(loss_mask.data)
            total_traj += pred_traj_gt.size(1)
            total_traj_l += torch.sum(linear_ped).item()
            total_traj_nl += torch.sum(non_linear_ped).item()
            if limit and total_traj >= args.num_samples_check:
                break

    metrics['irl_loss_abs'] = sum(irl_losses_abs) / loss_mask_sum
    metrics['irl_loss_rel'] = sum(irl_losses_rel) / loss_mask_sum

    metrics['ade'] = sum(disp_error) / (total_traj * args.pred_len)
    metrics['fde'] = sum(f_disp_error) / total_traj
    if total_traj_l != 0:
        metrics['ade_l'] = sum(disp_error_l) / (total_traj_l * args.pred_len)
        metrics['fde_l'] = sum(f_disp_error_l) / total_traj_l
    else:
        metrics['ade_l'] = 0
        metrics['fde_l'] = 0
    if total_traj_nl != 0:
        metrics['ade_nl'] = sum(disp_error_nl) / (
            total_traj_nl * args.pred_len)
        metrics['fde_nl'] = sum(f_disp_error_nl) / total_traj_nl
    else:
        metrics['ade_nl'] = 0
        metrics['fde_nl'] = 0

    policy_net.train()
    return metrics


def cal_l2_losses(
    pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel,
    loss_mask, mode
):
    g_l2_loss_abs = l2_loss(
        pred_traj_fake, pred_traj_gt, loss_mask, mode=mode
    )
    g_l2_loss_rel = l2_loss(
        pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode=mode
    )
    return g_l2_loss_abs, g_l2_loss_rel


def cal_ade(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    ade_l = displacement_error(pred_traj_fake, pred_traj_gt, linear_ped)
    ade_nl = displacement_error(pred_traj_fake, pred_traj_gt, non_linear_ped)
    return ade, ade_l, ade_nl


def cal_fde(
    pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
):
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    fde_l = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], linear_ped
    )
    fde_nl = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], non_linear_ped
    )
    return fde, fde_l, fde_nl