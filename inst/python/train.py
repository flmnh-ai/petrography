#!/usr/bin/env python3
import warnings
import logging

# Suppress warnings before any other imports
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

# Suppress torch meshgrid indexing warnings
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release...")

# Suppress detectron2/fvcore log noise
logging.getLogger("fvcore").setLevel(logging.ERROR)
logging.getLogger("detectron2").setLevel(logging.ERROR)

import os, argparse
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator

def setup_cfg(args):
    cfg = get_cfg()
    model_zoo_key = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(model_zoo_key))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_zoo_key)

    cfg.OUTPUT_DIR = args.output_dir
    cfg.MODEL.DEVICE = args.device

    cfg.DATASETS.TRAIN = (args.dataset_name,)
    cfg.DATASETS.TEST = (args.val_dataset_name,)
    cfg.TEST.EVAL_PERIOD = args.eval_period or 1000

    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.DATALOADER.PIN_MEMORY  = True

    cfg.SOLVER.IMS_PER_BATCH      = 4
    cfg.SOLVER.BASE_LR            = args.learning_rate
    cfg.SOLVER.MAX_ITER           = args.max_iter
    cfg.SOLVER.STEPS              = (int(args.max_iter * 0.75), int(args.max_iter * 0.9))
    cfg.SOLVER.GAMMA              = 0.1
    cfg.SOLVER.WARMUP_ITERS       = 500
    cfg.SOLVER.WARMUP_FACTOR      = 1.0/1000
    checkpoint_period = args.checkpoint_period if args.checkpoint_period > 0 else 999999
    cfg.SOLVER.CHECKPOINT_PERIOD  = checkpoint_period
    cfg.SOLVER.AMP.ENABLED = args.device == "cuda"

    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST  = 800
    cfg.INPUT.MAX_SIZE_TEST  = 1333
    cfg.INPUT.RANDOM_FLIP    = "horizontal"

    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [256], [512]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES          = args.num_classes
    cfg.MODEL.BACKBONE.FREEZE_AT = 2
    cfg.TEST.DETECTIONS_PER_IMAGE = 500

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()
    return cfg

def load_and_filter_dataset(annotation_json, image_root):
    dataset_dicts = load_coco_json(annotation_json, image_root)
    filtered = [d for d in dataset_dicts if len(d["annotations"]) > 0]
    return filtered

def main(args):
    args.val_dataset_name = args.dataset_name.replace("_train", "_val")

    # Clear prior registrations if they exist
    for name in [args.dataset_name, args.val_dataset_name]:
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)

    # Register training dataset
    DatasetCatalog.register(args.dataset_name, lambda: load_and_filter_dataset(args.annotation_json, args.image_root))
    MetadataCatalog.get(args.dataset_name).set(json_file=args.annotation_json, image_root=args.image_root)

    # Register validation dataset
    DatasetCatalog.register(args.val_dataset_name, lambda: load_and_filter_dataset(args.val_annotation_json, args.val_image_root))
    MetadataCatalog.get(args.val_dataset_name).set(json_file=args.val_annotation_json, image_root=args.val_image_root)

    cfg = setup_cfg(args)
    default_setup(cfg, args)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
        f.write(cfg.dump())

    class CocoTrainer(DefaultTrainer):
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            if output_folder is None:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            return COCOEvaluator(dataset_name, output_dir=output_folder)

    trainer = CocoTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--dataset-name", default="shell_train")
    parser.add_argument("--annotation-json", default="data/shell_mixed/train/_annotations.coco.json")
    parser.add_argument("--image-root", default="data/shell_mixed/train")
    parser.add_argument("--val-annotation-json", default="data/shell_mixed/val/_annotations.coco.json")
    parser.add_argument("--val-image-root", default="data/shell_mixed/val")
    parser.add_argument("--output-dir", default="Detectron2_Models")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--eval-period", type=int, default=500)
    parser.add_argument("--checkpoint-period", type=int, default=0)
    parser.add_argument("--opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    launch(main, args.num_gpus, num_machines=1, machine_rank=0, dist_url="auto", args=(args,))
