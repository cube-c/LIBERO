import torch
import numpy as np
from third_party.sam2.sam2.sam2_image_predictor import SAM2ImagePredictor
from third_party.sam2.sam2.build_sam import build_sam2
from PIL import Image

checkpoint = "./third_party/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    image = Image.open("outputs/img_agentview.png")
    predictor.set_image(image)
    masks, _, _ = predictor.predict(
        point_coords=np.array([[338, 492]]), point_labels=np.array([1])
    )

    # save the masks
    for i in range(masks.shape[0]):
        mask = masks[i]
        mask = mask.astype(np.uint8)
        mask = mask * 255
        mask = Image.fromarray(mask)
        mask.save(f"outputs/mask_{i}.png")
