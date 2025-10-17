from .focal_loss import FocalLoss


def get_criterion(args):
    criterion = FocalLoss(ignore_index=args.data_config["num_classes"], gamma=args.loss_config.get('focal_gamma', 0))

    return criterion
