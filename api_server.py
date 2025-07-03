'''
curl -X POST "http://localhost:8000/track/video" \
     -F "video=@./demo_images/3.mp4" \
     -F "prompt=ship" \

curl -X POST "http://localhost:8000/segment/image" \
     -F "image=@./demo_images/100000002.bmp" \
     -F "prompt=ship" 
'''
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import JSONResponse
import os
import cv2
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from torchvision.ops import box_convert
import json
import uuid

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images

app = FastAPI()

# Constants
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
PROMPT_TYPE_FOR_VIDEO = "box"
RESULT_DIR = Path("result/tmp")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Model Initialization
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
grounding_model = load_model(
    model_config_path="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    model_checkpoint_path="gdino_checkpoints/groundingdino_swint_ogc.pth",
    device=DEVICE
)
video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)

@app.post("/track/video")
async def track_video(
    video: UploadFile = File(...),
    prompt: str = Form(...)
):
    task_id = uuid.uuid4().hex
    video_path = RESULT_DIR / f"{task_id}_{video.filename}"
    with open(video_path, "wb") as f:
        f.write(await video.read())

    frame_dir = RESULT_DIR / f"frames_{task_id}"
    result_dir = RESULT_DIR / f"tracking_{task_id}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    frame_generator = sv.get_video_frames_generator(str(video_path), stride=1)
    with sv.ImageSink(frame_dir, overwrite=True, image_name_pattern="{:05d}.jpg") as sink:
        for frame in tqdm(frame_generator, desc="Extracting Frames"):
            sink.save_image(frame)

    frame_names = sorted([f for f in os.listdir(frame_dir) if f.endswith(".jpg")], key=lambda p: int(Path(p).stem))
    inference_state = video_predictor.init_state(video_path=str(frame_dir))

    ann_frame_idx = 0
    img_path = frame_dir / frame_names[ann_frame_idx]
    image_source, image = load_image(str(img_path))
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    confidences = confidences.numpy().tolist()
    class_names = labels

    image_predictor.set_image(image_source)
    torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    for object_id, box in enumerate(input_boxes, 1):
        video_predictor.add_new_points_or_box(inference_state, ann_frame_idx, object_id, box=box)

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    ID_TO_OBJECTS = {i: obj for i, obj in enumerate(class_names, start=1)}

    for frame_idx, segments in video_segments.items():
        img = cv2.imread(str(frame_dir / frame_names[frame_idx]))
        object_ids = list(segments.keys())
        masks = list(segments.values())
        masks = np.concatenate(masks, axis=0)
        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks),
            mask=masks,
            class_id=np.array(object_ids, dtype=np.int32),
        )
        annotated = sv.BoxAnnotator().annotate(scene=img.copy(), detections=detections)
        annotated = sv.LabelAnnotator().annotate(annotated, detections=detections,
                                                 labels=[ID_TO_OBJECTS[i] for i in object_ids])
        annotated = sv.MaskAnnotator().annotate(scene=annotated, detections=detections)
        cv2.imwrite(str(result_dir / f"annotated_frame_{frame_idx:05d}.jpg"), annotated)

    output_video_path = RESULT_DIR / f"{task_id}_tracking_result.mp4"
    create_video_from_images(str(result_dir), str(output_video_path))
    return {"result_path": str(output_video_path)}

@app.post("/segment/image")
async def segment_image(
    image: UploadFile = File(...),
    prompt: str = Form(...)
):
    task_id = uuid.uuid4().hex
    img_path = RESULT_DIR / f"{task_id}_{image.filename}"
    with open(img_path, "wb") as f:
        f.write(await image.read())

    image_source, image_tensor = load_image(str(img_path))
    sam2_predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam2_checkpoint, device=DEVICE))
    sam2_predictor.set_image(image_source)

    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image_tensor,
        caption=prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=DEVICE
    )

    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    confidences = confidences.numpy().tolist()
    class_names = labels

    torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    img = cv2.imread(str(img_path))
    class_ids = np.array(list(range(len(class_names))))
    labels_text = [f"{cls} {conf:.2f}" for cls, conf in zip(class_names, confidences)]

    detections = sv.Detections(
        xyxy=input_boxes,
        mask=masks.astype(bool),
        class_id=class_ids
    )
    annotated = sv.BoxAnnotator().annotate(scene=img.copy(), detections=detections)
    annotated = sv.LabelAnnotator().annotate(scene=annotated, detections=detections, labels=labels_text)
    annotated = sv.MaskAnnotator().annotate(scene=annotated, detections=detections)

    result_path = RESULT_DIR / f"{task_id}_annotated.jpg"
    cv2.imwrite(str(result_path), annotated)

    return {"result_path": str(result_path)}
