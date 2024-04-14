from torch.utils.data import DataLoader

from irl.data.trajectories import TrajectoryDataset, seq_collate


def data_loader(args, path, mid_pad=1):
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim,
        min_ped=mid_pad)                              # added this line to include frames with 1 ped as well (default >1)
    if args.model=="stgat":
        shuf=False
    else: shuf=True
    
    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=shuf,
        num_workers=args.loader_num_workers,
        # num_workers=10,
        collate_fn=seq_collate,
        pin_memory=True)
    # here the batch size merely denotes the number of scenes not the number fo pedestrians or trajcetories
    return dset, loader
# from torch.utils.data import DataLoader, Subset

# def data_loader(args, path, mid_pad=1, debug_mode=True, num_samples=10):
#     dset = TrajectoryDataset(
#         path,
#         obs_len=args.obs_len,
#         pred_len=args.pred_len,
#         skip=args.skip,
#         delim=args.delim,
#         min_ped=mid_pad)  # Frames with 1 ped as well now included

#     if args.model == "stgat":
#         shuf = False
#     else:
#         shuf = True

#     # If in debug mode, select a small subset of the dataset
#     if debug_mode:
#         indices = range(num_samples)  # Load the first `num_samples` samples
#         dset = Subset(dset, indices)

#     loader = DataLoader(
#         dset,
#         batch_size=args.batch_size,
#         shuffle=shuf,
#         num_workers=10,
#         collate_fn=seq_collate,
#         pin_memory=True)
#     return dset, loader