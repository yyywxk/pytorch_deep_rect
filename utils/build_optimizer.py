import torch.optim as optim
from .optimization import BertAdam


def init_optimizer(net, max_epoch, batches_per_epoch, args):
    # Prepare optimizer
    # Each param is a tuple ( param name, Parameter(tensor(...)) )
    optimized_params = list(param for param in net.named_parameters() if param[1].requires_grad)

    low_decay = ['backbone']  # ['bias', 'LayerNorm.weight']
    no_decay = []
    high_lr = ['alphas']

    high_lr_params = []
    high_lr_names = []
    no_decay_params = []
    no_decay_names = []
    low_decay_params = []
    low_decay_names = []
    normal_params = []
    normal_names = []

    for n, p in optimized_params:
        if any(nd in n for nd in no_decay):
            no_decay_params.append(p)
            no_decay_names.append(n)
        elif any(nd in n for nd in low_decay):
            low_decay_params.append(p)
            low_decay_names.append(n)
        elif any(nd in n for nd in high_lr):
            high_lr_params.append(p)
            high_lr_names.append(n)
        else:
            normal_params.append(p)
            normal_names.append(n)

    optimizer_grouped_parameters = [
        {'params': normal_params, 'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': low_decay_params, 'weight_decay': args.weight_decay * 0.1, 'lr': args.lr},
        {'params': no_decay_params, 'weight_decay': 0.0, 'lr': args.lr},
        {'params': high_lr_params, 'weight_decay': 0.0, 'lr': args.lr * 100},
    ]

    for group_name, param_group in zip(('normal', 'low_decay', 'no_decay', 'high_lr'),
                                       (normal_params, low_decay_params, no_decay_params, high_lr_params)):
        print("{}: {} weights".format(group_name, len(param_group)))

    args.t_total = int(batches_per_epoch * max_epoch)
    print("Batches per epoch: %d" % batches_per_epoch)
    print("Total Iters: %d" % args.t_total)
    print("LR: %f" % args.lr)

    args.warmup = min(args.warmup, args.t_total // 2)
    args.lr_warmup_ratio = args.warmup / args.t_total
    print("LR Warm up: %.3f=%d iters" % (args.lr_warmup_ratio, args.warmup))

    # pytorch adamw performs much worse. Not sure about the reason.
    optimizer = BertAdam(optimizer_grouped_parameters,
                         warmup=args.lr_warmup_ratio, t_total=args.t_total,
                         weight_decay=args.weight_decay)

    return optimizer


def build_optimizer(args, model, batches_per_epoch):
    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay,
                              nesterov=args.nesterov)
    elif args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        # https://pytorch.org/docs/1.3.0/_modules/torch/optim/adamw.html
        # PyTorch > 1.2.0
        import torch
        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        if TORCH_MAJOR == 1 and TORCH_MINOR > 2:
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = init_optimizer(model, args.epochs, batches_per_epoch, args)
    else:
        raise AssertionError
    return optimizer
