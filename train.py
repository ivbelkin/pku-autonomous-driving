import argparse
from ignite.engine import Engine, Events
from train_config import load_config
import os
import config as C
import logging
import sys
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import *

logger = logging.getLogger('logger')


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--config", type=str, required=True)
    return parser


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cfg = load_config(args.config)

    workdir = os.path.join(C.WORKDIR, cfg.workdir)
    os.makedirs(workdir, exist_ok=True)

    pbar = ProgressBar()
    tb_logger = TensorboardLogger(log_dir=os.path.join(workdir, "tb_logs"))

    def _update(engine, batch):
        cfg.model.train()
        cfg.optimizer.zero_grad()
        x, y = cfg.prepare_batch(batch)
        y_pred = cfg.model(**x)
        loss = cfg.loss_fn(y_pred, y)
        loss['loss'].backward()
        cfg.optimizer.step()
        for k in loss:
            loss[k] = loss[k].item()
        return loss

    trainer = Engine(_update)
    pbar.attach(trainer, output_transform=lambda x: {k: "{:.5f}".format(v) for k, v in x.items()})
    trainer.add_event_handler(Events.ITERATION_STARTED, cfg.scheduler)
    tb_logger.attach(trainer,
                     log_handler=OutputHandler(tag="training", output_transform=lambda x: x),
                     event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach(trainer,
                     log_handler=OptimizerParamsHandler(cfg.optimizer),
                     event_name=Events.ITERATION_STARTED)
    tb_logger.attach(trainer,
                     log_handler=WeightsScalarHandler(cfg.model),
                     event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach(trainer,
                     log_handler=WeightsHistHandler(cfg.model),
                     event_name=Events.EPOCH_COMPLETED)
    tb_logger.attach(trainer,
                     log_handler=GradsScalarHandler(cfg.model),
                     event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach(trainer,
                     log_handler=GradsHistHandler(cfg.model),
                     event_name=Events.EPOCH_COMPLETED)

    trainer.run(cfg.train_dl, cfg.n_epochs)
    tb_logger.close()


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
