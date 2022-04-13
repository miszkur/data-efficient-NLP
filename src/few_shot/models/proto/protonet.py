import json
import argparse
import sys
sys.path.append( '..' )
sys.path.append( '../..' )
from models.encoders.bert_encoder import BERTEncoder
from utils.data import FewShotDataLoader
from utils.python import now, set_seeds
import random
import collections
import os
from typing import List, Dict, Callable, Union
from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torch_functional
from torch.autograd import Variable
import warnings
import logging
from utils.math import euclidean_dist, cosine_similarity

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

warnings.simplefilter('ignore')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class ProtoNet(nn.Module):
    def __init__(self, encoder: BERTEncoder, metric="euclidean"):
        super(ProtoNet, self).__init__()

        self.encoder: BERTEncoder = encoder
        self.metric = metric
        assert self.metric in ('euclidean', 'cosine')

    def loss(self, sample, supervised_loss_share: float = 0):
        """
        :param supervised_loss_share: share of supervised loss in total loss
        :param sample: {
            "xs": [
                [support_A_1, support_A_2, ...],
                [support_B_1, support_B_2, ...],
                [support_C_1, support_C_2, ...],
                ...
            ],
            "xq": [
                [query_A_1, query_A_2, ...],
                [query_B_1, query_B_2, ...],
                [query_C_1, query_C_2, ...],
                ...
            ]
        }
        :return:
        """
        xs = sample['xs']  # support
        xq = sample['xq']  # query

        n_class = len(xs)
        assert len(xq) == n_class
        n_support = len(xs[0])
        n_query = len(xq[0])

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False).to(device)

        supports = [item for xs_ in xs for item in xs_]
        queries = [item for xq_ in xq for item in xq_]

        # Encode
        x = supports + queries
        z = self.encoder.embed_sentences(x)
        z_dim = z.size(-1)

        # Dispatch
        z_support = z[:len(supports)].view(n_class, n_support, z_dim).mean(dim=[1])
        z_query = z[len(supports):len(supports) + len(queries)]

        if self.metric == "euclidean":
            supervised_dists = euclidean_dist(z_query, z_support)

        elif self.metric == "cosine":
            supervised_dists = (-cosine_similarity(z_query, z_support) + 1) * 5
        else:
            raise NotImplementedError

        # Supervised loss
        # -- legacy
        # log_p_y = torch_functional.log_softmax(-supervised_dists, dim=1).view(n_class, n_query, -1)
        # dists.view(n_class, n_query, -1)
        # loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        # -- NEW
        from torch.nn import CrossEntropyLoss
        supervised_loss = CrossEntropyLoss()(-supervised_dists, target_inds.reshape(-1))
        _, y_hat_supervised = (-supervised_dists).max(1)
        acc_val_supervised = torch.eq(y_hat_supervised, target_inds.reshape(-1)).float().mean()

        return supervised_loss, {
            "metrics": {
                "acc": acc_val_supervised.item(),
                "loss": supervised_loss.item(),
            },
            "dists": supervised_dists,
            "target": target_inds
        }

    def train_step(self, optimizer, episode, supervised_loss_share: float, unlabeled: bool = False):
        self.train()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        loss, loss_dict = self.loss(episode, supervised_loss_share=supervised_loss_share)
        loss.backward()
        optimizer.step()

        return loss, loss_dict

    def test_step(self,
                  data_loader: FewShotDataLoader,
                  n_support: int,
                  n_query: int,
                  n_classes: int,
                  n_unlabeled: int = 0,
                  n_episodes: int = 1000):
        metrics = collections.defaultdict(list)

        self.eval()
        for i in range(n_episodes):
            episode = data_loader.create_episode(
                n_support=n_support,
                n_query=n_query,
                n_unlabeled=n_unlabeled,
                n_classes=n_classes,
                n_augment=0
            )

            with torch.no_grad():
                loss, loss_dict = self.loss(episode, supervised_loss_share=0)

            for k, v in loss_dict["metrics"].items():
                metrics[k].append(v)

        return {
            key: np.mean(value) for key, value in metrics.items()
        }

def run_proto(
        model_name_or_path: str,
        n_support: int,
        n_query: int,
        n_classes: int,
        unlabeled_path: str = None,
        n_unlabeled: int = 0,
        n_augment: int = 0,
        output_path: str = f'runs/{now()}',
        max_iter: int = 10000,
        evaluate_every: int = 100,
        early_stop: int = None,
        n_test_episodes: int = 1000,
        log_every: int = 10,
        metric: str = "euclidean",
        data_path: str = None,
        supervised_loss_share_fn: Callable[[int, int], float] = lambda x, y: 1 - (x / y)
):
    if output_path:
        if os.path.exists(output_path) and len(os.listdir(output_path)):
            raise FileExistsError(f"Output path {output_path} already exists. Exiting.")

    # --------------------
    # Creating Log Writers
    # --------------------
    os.makedirs(output_path)
    os.makedirs(os.path.join(output_path, "logs/train"))
    train_writer: SummaryWriter = SummaryWriter(logdir=os.path.join(output_path, "logs/train"), flush_secs=1, max_queue=1)
    valid_writer: SummaryWriter = None
    test_writer: SummaryWriter = None
    log_dict = dict(train=list())

    os.makedirs(os.path.join(output_path, "logs/valid"))
    valid_writer = SummaryWriter(logdir=os.path.join(output_path, "logs/valid"), flush_secs=1, max_queue=1)
    log_dict["valid"] = list()

    os.makedirs(os.path.join(output_path, "logs/test"))
    test_writer = SummaryWriter(logdir=os.path.join(output_path, "logs/test"), flush_secs=1, max_queue=1)
    log_dict["test"] = list()

    # Load model
    bert = BERTEncoder(model_name_or_path).to(device)
    protonet = ProtoNet(encoder=bert, metric=metric)
    optimizer = torch.optim.Adam(protonet.parameters(), lr=2e-5)

    # Load data
    train_data_loader = FewShotDataLoader('train')
    valid_data_loader = FewShotDataLoader('valid')
    test_data_loader = FewShotDataLoader('test')

    logger.info(f"train labels: {train_data_loader.data_dict.keys()}")
    logger.info(f"valid labels: {valid_data_loader.data_dict.keys()}")

    train_metrics = collections.defaultdict(list)
    n_eval_since_last_best = 0
    best_valid_acc = 0.0

    for step in range(max_iter):
        episode = train_data_loader.create_episode(
            n_support=n_support,
            n_query=n_query,
            n_classes=n_classes,
            n_unlabeled=n_unlabeled,
            n_augment=n_augment
        )

        supervised_loss_share = supervised_loss_share_fn(step, max_iter)
        loss, loss_dict = protonet.train_step(optimizer=optimizer, episode=episode, unlabeled=(n_unlabeled > 0), supervised_loss_share=supervised_loss_share)

        for key, value in loss_dict["metrics"].items():
            train_metrics[key].append(value)

        # Logging
        if (step + 1) % log_every == 0:
            for key, value in train_metrics.items():
                train_writer.add_scalar(tag=key, scalar_value=np.mean(value), global_step=step)
            logger.info(f"train | " + " | ".join([f"{key}:{np.mean(value):.4f}" for key, value in train_metrics.items()]))
            log_dict["train"].append({
                "metrics": [
                    {
                        "tag": key,
                        "value": np.mean(value)
                    }
                    for key, value in train_metrics.items()
                ],
                "global_step": step
            })

            train_metrics = collections.defaultdict(list)

        valid_path = 'valid_path'
        test_path = 'test_path'
        if valid_path or test_path:
            if (step + 1) % evaluate_every == 0:
                for path, writer, set_type, set_data_loader in zip(
                        [valid_path, test_path],
                        [valid_writer, test_writer],
                        ["valid", "test"],
                        [valid_data_loader, valid_data_loader]
                ):
                    if path:
                        set_results = protonet.test_step(
                            data_loader=set_data_loader,
                            n_unlabeled=n_unlabeled,
                            n_support=n_support,
                            n_query=n_query,
                            n_classes=n_classes,
                            n_episodes=n_test_episodes
                        )


                        for key, val in set_results.items():
                            writer.add_scalar(tag=key, scalar_value=val, global_step=step)
                        log_dict[set_type].append({
                            "metrics": [
                                {
                                    "tag": key,
                                    "value": val
                                }
                                for key, val in set_results.items()
                            ],
                            "global_step": step
                        })

                        logger.info(f"{set_type} | " + " | ".join([f"{key}:{np.mean(value):.4f}" for key, value in set_results.items()]))
                        if set_type == "valid":
                            if set_results["acc"] > best_valid_acc:
                                best_valid_acc = set_results["acc"]
                                n_eval_since_last_best = 0
                                logger.info(f"Better eval results!")
                            else:
                                n_eval_since_last_best += 1
                                logger.info(f"Worse eval results ({n_eval_since_last_best}/{early_stop})")

                if early_stop and n_eval_since_last_best >= early_stop:
                    logger.warning(f"Early-stopping.")
                    break
    with open(os.path.join(output_path, 'metrics.json'), "w") as file:
        json.dump(log_dict, file, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=None, help="Path to data (ARSC only)")

    parser.add_argument("--output-path", type=str, default=f'runs/{now()}')
    parser.add_argument("--model-name-or-path", type=str, default='bert-base-cased', help="Transformer model to use")
    parser.add_argument("--max-iter", type=int, default=10000, help="Max number of training episodes")
    parser.add_argument("--evaluate-every", type=int, default=100, help="Number of training episodes between each evaluation (on both valid, test)")
    parser.add_argument("--log-every", type=int, default=10, help="Number of training episodes between each logging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed to set")
    parser.add_argument("--early-stop", type=int, default=0, help="Number of worse evaluation steps before stopping. 0=disabled")

    # Few-Shot related stuff
    parser.add_argument("--n-support", type=int, default=5, help="Number of support points for each class")
    parser.add_argument("--n-query", type=int, default=5, help="Number of query points for each class")
    parser.add_argument("--n-classes", type=int, default=5, help="Number of classes per episode")
    parser.add_argument("--n-augment", type=int, default=0, help="Number of augmented samples to take")
    parser.add_argument("--n-test-episodes", type=int, default=1000, help="Number of episodes during evaluation (valid, test)")

    # Metric to use in proto distance calculation
    parser.add_argument("--metric", type=str, default="euclidean", help="Metric to use", choices=("euclidean", "cosine"))

    # Supervised loss share
    parser.add_argument("--supervised-loss-share-power", default=1.0, type=float, help="supervised_loss_share = 1 - (x/y) ** <param>")

    args = parser.parse_args()

    # Set random seed
    set_seeds(args.seed)

    # Create supervised_loss_share_fn
    def get_supervised_loss_share_fn(supervised_loss_share_power: Union[int, float]) -> Callable[[int, int], float]:
        def _supervised_loss_share_fn(current_step: int, max_steps: int) -> float:
            assert current_step <= max_steps
            return 1 - (current_step / max_steps) ** supervised_loss_share_power

        return _supervised_loss_share_fn

    supervised_loss_share_fn = get_supervised_loss_share_fn(args.supervised_loss_share_power)

    # Run
    run_proto(
        output_path=args.output_path,
        unlabeled_path=None,

        model_name_or_path=args.model_name_or_path,
        n_unlabeled=0,

        n_support=args.n_support,
        n_query=args.n_query,
        n_classes=args.n_classes,
        n_test_episodes=args.n_test_episodes,
        n_augment=args.n_augment,

        max_iter=args.max_iter,
        evaluate_every=args.evaluate_every,

        metric=args.metric,
        early_stop=args.early_stop,
        data_path=args.data_path,
        log_every=args.log_every,

        supervised_loss_share_fn=supervised_loss_share_fn
    )

    # Save config
    with open(os.path.join(args.output_path, "config.json"), "w") as file:
        json.dump(vars(args), file, ensure_ascii=False)


if __name__ == '__main__':
    main()
