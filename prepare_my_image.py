import os
import argparse
import numpy as np
import torch
import cv2
import imageio.v3 as iio

from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, build_sam
from transformers import AutoModelForImageSegmentation

from modules.label_2d_mask.label_parts import (
    prepare_image,
    get_sam_mask,
    clean_segment_edges,
    resize_and_pad_to_square,
    size_th as DEFAULT_SIZE_TH,
)
from modules.label_2d_mask.visualizer import Visualizer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--outdir", default="assets/my_images/my_object")
    parser.add_argument(
        "--sam_ckpt",
        default="ckpt/sam_vit_h_4b8939.pth"
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    image_path = os.path.abspath(args.image)
    img_name = os.path.splitext(os.path.basename(image_path))[0]

    print("=" * 80)
    print("Input:", image_path)
    print("Output:", os.path.abspath(args.outdir))
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load models
    # ------------------------------------------------------------------
    print("\n[1/4] Loading SAM...")
    sam_model = build_sam(checkpoint=args.sam_ckpt).to(device=DEVICE)
    sam_mask_generator = SamAutomaticMaskGenerator(sam_model)

    print("[1/4] Loading Bria RMBG-2.0...")
    rmbg_model = AutoModelForImageSegmentation.from_pretrained(
        "briaai/RMBG-2.0",
        trust_remote_code=True,
    )
    rmbg_model.to(DEVICE)
    rmbg_model.eval()

    # ------------------------------------------------------------------
    # 2. RMBG preprocessing
    # ------------------------------------------------------------------
    print("\n[2/4] Removing background with RMBG...")

    img = Image.open(image_path).convert("RGB")

    processed_image = prepare_image(
        img,
        rmbg_net=rmbg_model.to(DEVICE),
    )

    processed_image = resize_and_pad_to_square(processed_image)

    # White background for SAM, exactly as PAct does.
    white_bg = Image.new(
        "RGBA",
        processed_image.size,
        (255, 255, 255, 255),
    )

    white_bg_img = Image.alpha_composite(
        white_bg,
        processed_image.convert("RGBA"),
    )

    image = np.array(white_bg_img.convert("RGB"))

    processed_path = os.path.join(
        args.outdir,
        f"{img_name}_processed.png",
    )

    processed_image.save(processed_path)

    print("Processed image:", processed_path)
    print("Resolution:", image.shape[:2])

    # ------------------------------------------------------------------
    # 3. SAM + PAct segmentation
    # ------------------------------------------------------------------
    print("\n[3/4] Running PAct's SAM segmentation...")

    visual = Visualizer(image)

    group_ids, _ = get_sam_mask(
        image,
        sam_mask_generator,
        visual,
        merge_groups=None,
        rgba_image=processed_image,
        img_name=img_name,
        save_dir=args.outdir,
        size_threshold=DEFAULT_SIZE_TH,
    )

    print(
        "Initial segments:",
        len(np.unique(group_ids[group_ids >= 0])),
    )

    # PAct's final edge cleaning, same operation used by apply_merge().
    print("[3/4] Cleaning segmentation edges...")

    new_group_ids = clean_segment_edges(group_ids)

    # ------------------------------------------------------------------
    # 4. Save PAct-compatible EXR
    # ------------------------------------------------------------------
    print("\n[4/4] Writing PAct EXR mask...")

    # PAct uses background=-1 internally.
    # It shifts everything by +1 before saving:
    #
    #   background -1 -> 0
    #   part 0        -> 1
    #   part 1        -> 2
    #   ...
    save_mask = new_group_ids + 1

    save_mask = save_mask.reshape(
        518,
        518,
        1,
    ).repeat(
        3,
        axis=-1,
    )

    mask_path = os.path.join(
        args.outdir,
        f"{img_name}_mask.exr",
    )

    iio.imwrite(
        mask_path,
        save_mask.astype(np.float32),
    )

    print("EXR mask:", mask_path)

    # Also save a simple visualization so we can inspect
    # whether SAM produced sensible parts.
    vis = np.ones(
        (*new_group_ids.shape, 3),
        dtype=np.uint8,
    ) * 255

    ids = np.unique(new_group_ids)
    ids = ids[ids >= 0]

    for i, uid in enumerate(ids):
        color = np.array(
            [
                (i * 50 + 80) % 256,
                (i * 120 + 40) % 256,
                (i * 180 + 20) % 256,
            ],
            dtype=np.uint8,
        )
        vis[new_group_ids == uid] = color

    vis_path = os.path.join(
        args.outdir,
        f"{img_name}_mask_preview.png",
    )

    Image.fromarray(vis).save(vis_path)

    print("Preview:", vis_path)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print("Processed PNG:", processed_path)
    print("EXR mask:     ", mask_path)
    print("Preview:      ", vis_path)
    print("Number of parts:", len(ids))


if __name__ == "__main__":
    main()
