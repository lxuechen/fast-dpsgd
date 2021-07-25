'''
Opacus experiments for all the models
'''
import time

import torch
from torch import nn, optim

import data
from experimental.privacy_utils import autograd_grad_sample
from experimental.privacy_utils.privacy_engine import EfficientPrivacyEngine
from pytorch import get_data, model_dict
import utils


def main(args):
    print(args)
    assert args.dpsgd
    torch.backends.cudnn.benchmark = True

    mdict = model_dict.copy()
    train_data, train_labels = get_data(args)
    model = mdict[args.experiment](vocab_size=args.max_features, batch_size=args.batch_size).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0)
    loss_function = nn.CrossEntropyLoss(reduction="none") if args.experiment != 'logreg' else nn.BCELoss(
        reduction="none")

    privacy_engine = EfficientPrivacyEngine(
        model,
        batch_size=args.batch_size,
        sample_size=len(train_data),
        alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
        noise_multiplier=args.sigma,
        max_grad_norm=args.max_per_sample_grad_norm,
    )
    privacy_engine.attach(optimizer)

    timings = []
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        dataloader = data.dataloader(train_data, train_labels, args.batch_size)
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            outputs = model(x)
            loss = loss_function(outputs, y)

            autograd_grad_sample.set_hooks_mode(mode="norm")
            first_loss = loss.mean(dim=0)
            first_loss.backward(retain_graph=True)

            autograd_grad_sample.set_hooks_mode(mode="grad")
            coef_sample = optimizer.privacy_engine.get_coef_sample()
            second_loss = (coef_sample * loss).sum(dim=0)
            second_loss.backward()

            optimizer.step()
            optimizer.zero_grad()
        torch.cuda.synchronize()
        duration = time.perf_counter() - start
        print("Time Taken for Epoch: ", duration)
        timings.append(duration)

        if args.dpsgd:
            epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(args.delta)
            print(f"Train Epoch: {epoch} \t"
                  f"(ε = {epsilon:.2f}, δ = {args.delta}) for α = {best_alpha}")
        else:
            print(f"Train Epoch: {epoch}")

    if not args.no_save:
        utils.save_runtimes(__file__.split('.')[0], args, timings)
    else:
        print('Not saving!')
    print('Done!')


if __name__ == '__main__':
    # python fast_torch_dp.py --batch_size 10
    parser = utils.get_parser(model_dict.keys())
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Noise multiplier (default 1.0)",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=1.0,
        help="Clip per-sample gradients to this norm (default 1.0)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        help="Target delta (default: 1e-5)",
    )
    args = parser.parse_args()
    main(args)
