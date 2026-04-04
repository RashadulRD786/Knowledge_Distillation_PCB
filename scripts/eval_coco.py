from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

gt = COCO("coco_annotations.json")
dt = gt.loadRes("predictions_baseline.json")

eval = COCOeval(gt, dt, "bbox")
eval.evaluate()
eval.accumulate()
eval.summarize()