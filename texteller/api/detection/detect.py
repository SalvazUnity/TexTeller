from typing import List

from onnxruntime import InferenceSession

from texteller.types import Bbox

from .preprocess import Compose

_config = {
    "mode": "paddle",
    "draw_threshold": 0.5,
    "metric": "COCO",
    "use_dynamic_shape": False,
    "arch": "DETR",
    "min_subgraph_size": 3,
    "preprocess": [
        {"interp": 2, "keep_ratio": False, "target_size": [1600, 1600], "type": "Resize"},
        {
            "mean": [0.0, 0.0, 0.0],
            "norm_type": "none",
            "std": [1.0, 1.0, 1.0],
            "type": "NormalizeImage",
        },
        {"type": "Permute"},
    ],
    "label_list": ["isolated", "embedding"],
}


def latex_detect(img_path: str, predictor: InferenceSession) -> List[Bbox]:
    transforms = Compose(_config["preprocess"])
    inputs = transforms(img_path)
    inputs_name = [var.name for var in predictor.get_inputs()]
    inputs = {k: inputs[k][None,] for k in inputs_name}

    outputs = predictor.run(output_names=None, input_feed=inputs)[0]
    res = []
    for output in outputs:
        cls_name = _config["label_list"][int(output[0])]
        score = output[1]
        xmin = int(max(output[2], 0))
        ymin = int(max(output[3], 0))
        xmax = int(output[4])
        ymax = int(output[5])
        if score > 0.5:
            res.append(Bbox(xmin, ymin, ymax - ymin, xmax - xmin, cls_name, score))

    return res
