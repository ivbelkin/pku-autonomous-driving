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
    parser.add_argument("--submission", type=str, required=True)
    return parser


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cfg = load_config(args.config)

    cfg.model.load_state_dict(torch.load(os.path.join(cfg.workdir, 'checkpoints', args.checkpoint)))
    cfg.model.eval()

    image_id_to_ID = {image['id']: image['file_name'].split('.')[0] for image in cfg.train_ds.gt['images']}

    results = []
    for batch in tqdm(cfg.train_dl):
        x = cfg.prepare_test_batch(batch)
        with torch.no_grad():
            y_pred = cfg.model(**x)
        y_pred_decoded = cfg.decode_y_pred(y_pred)
        batch_decoded = cfg.decode_batch(batch)
        for sample, y_pred in zip(batch_decoded, y_pred_decoded):
            results.append(cfg.decode_test_sample(sample, y_pred))
    
    ID_to_PredictionString = {}
    for r in results:
        image_id = int(r['image_id'])
        ID = image_id_to_ID[image_id]
        if ID not in ID_to_PredictionString:
            ID_to_PredictionString[ID] = ""
        ID_to_PredictionString[ID] += "{} {} {} {} {} {} {} ".format(
            r['rotation'][1], r['rotation'][0], r['rotation'][2],
            r['translation'][0], r['translation'][1], r['translation'][2],
            r['score']
        )
    
    with open(args.submission, "w") as f:
        f.write("ImageId,PredictionString\n")
        for ID, PredictionString in ID_to_PredictionString.items():
            f.write("{},{}\n".format(ID, PredictionString))

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
