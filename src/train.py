#!/usr/bin/env python3
import os, argparse, yaml
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.data import MetadataCatalog, DatasetCatalog

import warnings
import logging

# Suppress pkg_resources deprecation warning
warnings.filterwarnings("ignore", category=UserWarning, module=".*pkg_resources.*")

# Suppress torch.meshgrid indexing warning
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*indexing argument.*")

# Suppress fvcore and detectron2 log spam
logging.getLogger("fvcore.common.checkpoint").setLevel(logging.ERROR)
logging.getLogger("detectron2").setLevel(logging.ERROR)

def setup_cfg(args):
    cfg = get_cfg()
    base_cfg = args.config_file or "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(base_cfg))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(base_cfg) 
    # (optionally propagate back so downstream printouts show it)
    args.config_file = base_cfg

    cfg.OUTPUT_DIR = args.output_dir
    cfg.MODEL.DEVICE = args.device

    cfg.DATASETS.TRAIN = (args.dataset_name,)
    cfg.DATASETS.TEST  = ("shell_val",)        # <-- change
    cfg.TEST.EVAL_PERIOD = 1000                # <-- change

    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.DATALOADER.PIN_MEMORY  = True

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.config_file)

    cfg.SOLVER.IMS_PER_BATCH      = 4
    cfg.SOLVER.BASE_LR            = 0.00025
    cfg.SOLVER.MAX_ITER           = args.max_iter
    cfg.SOLVER.STEPS              = [3000, 4500]
    cfg.SOLVER.GAMMA              = 0.1
    cfg.SOLVER.WARMUP_ITERS       = 500
    cfg.SOLVER.WARMUP_FACTOR      = 1.0/1000
    cfg.SOLVER.CHECKPOINT_PERIOD  = 1000
    cfg.SOLVER.AMP.ENABLED        = False # should only turn on with gpu

    cfg.INPUT.MIN_SIZE_TRAIN = (640,)
    cfg.INPUT.MAX_SIZE_TRAIN = 640
    cfg.INPUT.MIN_SIZE_TEST  = 640
    cfg.INPUT.MAX_SIZE_TEST  = 640
    cfg.INPUT.RANDOM_FLIP    = "horizontal"

    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16], [32], [64], [128], [256]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES          = args.num_classes
    cfg.MODEL.BACKBONE.FREEZE_AT = 0

    # Set higher detection limit for dense object images
    cfg.TEST.DETECTIONS_PER_IMAGE = 500

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()
    return cfg

def load_and_filter_dataset(annotation_json, image_root):
    """Load COCO dataset and filter out images without annotations"""
    dataset_dicts = load_coco_json(annotation_json, image_root)
    # Filter out entries that don't have annotations
    filtered_dataset_dicts = [d for d in dataset_dicts if len(d["annotations"]) > 0]
    print(f"Filtered dataset: {len(filtered_dataset_dicts)}/{len(dataset_dicts)} images have annotations")
    return filtered_dataset_dicts

def main(args):
    # 1. Register datasets with filtering
    # Clean up any existing registrations
    if args.dataset_name in DatasetCatalog.list():
        DatasetCatalog.remove(args.dataset_name)
    if "shell_val" in DatasetCatalog.list():
        DatasetCatalog.remove("shell_val")
    
    # Register filtered training dataset
    DatasetCatalog.register(
        args.dataset_name, 
        lambda: load_and_filter_dataset(args.annotation_json, args.image_root)
    )
    MetadataCatalog.get(args.dataset_name).set(
        json_file=args.annotation_json,
        image_root=args.image_root
    )
    
    # Register filtered validation dataset
    DatasetCatalog.register(
        "shell_val",
        lambda: load_and_filter_dataset(args.val_annotation_json, args.val_image_root)
    )
    MetadataCatalog.get("shell_val").set(
        json_file=args.val_annotation_json,
        image_root=args.val_image_root
    )

    # 2. Build config
    cfg = setup_cfg(args)

    # --- FIX: give default_setup a real YAML path --------------
    if not args.config_file or not os.path.isfile(args.config_file):
        args.config_file = model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    # -----------------------------------------------------------
    default_setup(cfg, args)

    # Ensure output dir exists
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    # 3. Train
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()



if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--dataset-name",    default="shell_train")
    parser.add_argument("--annotation-json",
        default="data/shell_mixed/train/_annotations.coco.json")
    parser.add_argument("--image-root",
        default="data/shell_mixed/train")
    parser.add_argument("--val-annotation-json",
        default="data/shell_mixed/val/_annotations.coco.json")
    parser.add_argument("--val-image-root",
        default="data/shell_mixed/val")
    parser.add_argument("--output-dir",
        default="Detectron2_Models")
    parser.add_argument("--num-workers",     type=int, default=4)
    parser.add_argument("--num-classes",     type=int, default=5)
    parser.add_argument("--device",          default="cpu", choices=["cpu", "cuda", "mps"], help="Device to use for training")
    parser.add_argument("--max-iter",        type=int, default=5000, help="Maximum training iterations")
    parser.add_argument("--opts",            nargs=argparse.REMAINDER)
    args = parser.parse_args()
    
    launch(main, 0, num_machines=1, machine_rank=0, dist_url="auto", args=(args,))
