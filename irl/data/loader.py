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
        # num_workers=args.loader_num_workers,
        num_workers=10,
        collate_fn=seq_collate,
        pin_memory=True)
    return dset, loader
