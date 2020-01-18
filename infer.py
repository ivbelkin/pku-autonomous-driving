from train_config import load_config
import torch
import argparse
import os
from tqdm import tqdm


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    return parser


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cfg = load_config(args.config)

    cfg.model.load_state_dict(torch.load(os.path.join(cfg.workdir, 'checkpoints', args.checkpoint)))
    cfg.model.eval()

    results = []
    for batch in tqdm(cfg.test_dl):
        x = cfg.prepare_test_batch(batch)
        with torch.no_grad():
            y_pred = cfg.model(**x)
        y_pred_decoded = cfg.decode_y_pred(y_pred)
        batch_decoded = cfg.decode_batch(batch)
        for sample, y_pred in zip(batch_decoded, y_pred_decoded):
            results.append(cfg.decode_test_sample(sample, y_pred))

    print(results)

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
