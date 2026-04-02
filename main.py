import argparse
import base64
import json
import os
import re
import shutil
from datetime import datetime
from math import atan2

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from nuscenes import NuScenes
from openai import OpenAI
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
)

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.model.quantization import set_quant_act_mode
from llava.utils import disable_torch_init
from openemma.YOLO3D.inference import yolo3d_nuScenes
from utils import (
    EstimateCurvatureFromTrajectory,
    IntegrateCurvatureForPoints,
    OverlayTrajectory,
    WriteImageSequenceToVideo,
)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "[your-openai-api-key]"))

OBS_LEN = 10
FUT_LEN = 10
TTL_LEN = OBS_LEN + FUT_LEN

QAPRUNER_TEXT_SCENE = "Driving scene: lights, vehicles, pedestrians, lane markings."
QAPRUNER_TEXT_OBJECTS = "Important road users and their image locations."
QAPRUNER_TEXT_INTENT = "Ego driving intent from lanes and traffic."
QAPRUNER_TEXT_MOTION = "Scene and ego motion for next driving action."


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got `{value}`.")

def get_model_device(model):
    if hasattr(model, "device") and model.device is not None:
        return model.device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer_llava_conv_mode(model_name):
    model_name = (model_name or "").lower()
    if "llama-2" in model_name:
        return "llava_llama_2"
    if "mistral" in model_name:
        return "mistral_instruct"
    if "v1.6-34b" in model_name:
        return "chatml_direct"
    if "v1" in model_name:
        return "llava_v1"
    if "mpt" in model_name:
        return "mpt"
    return "llava_v0"


def format_speed_curvature_history(obs_velocities, obs_curvatures):
    obs_velocities_norm = np.linalg.norm(obs_velocities, axis=1)
    scaled_curvatures = obs_curvatures * 100
    obs_speed_curvature_str = [f"[{speed:.1f},{curvature:.1f}]" for speed, curvature in zip(obs_velocities_norm, scaled_curvatures)]
    return ", ".join(obs_speed_curvature_str)


def build_calibration_prompts(obs_velocities, obs_curvatures):
    history_str = format_speed_curvature_history(obs_velocities, obs_curvatures)
    return [
        (
            "You are an autonomous driving labeller. You are given a front-view camera image from the ego vehicle. "
            "Provide a concise description of the driving scene, focusing on traffic lights, nearby vehicles, pedestrians, and lane markings."
        ),
        (
            "You are an autonomous driving labeller. You are given a front-view camera image from the ego vehicle. "
            "List two or three road users the ego vehicle should pay attention to, and briefly mention where each one appears in the image."
        ),
        (
            "You are an autonomous driving labeller. You are given a front-view camera image from the ego vehicle. "
            "Based on the lane markings and traffic participants, provide a concise description of the ego vehicle's likely driving intent."
        ),
        (
            "You are an autonomous driving labeller. You are given a front-view camera image from the ego vehicle. "
            f"The 5 second historical velocities and curvatures of the ego car are {history_str}. "
            "Infer the association between the image and this motion history, then briefly describe the likely next driving action."
        ),
    ]


def load_scene_sequence(nusc, scene, args):
    token = scene["token"]
    first_sample_token = scene["first_sample_token"]
    last_sample_token = scene["last_sample_token"]
    name = scene["name"]

    front_camera_images = []
    ego_poses = []
    camera_params = []
    curr_sample_token = first_sample_token
    while True:
        sample = nusc.get("sample", curr_sample_token)
        cam_front_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        image_path = os.path.join(nusc.dataroot, cam_front_data["filename"])

        if "gpt" in args.model_path:
            with open(image_path, "rb") as image_file:
                front_camera_images.append(base64.b64encode(image_file.read()).decode("utf-8"))
        else:
            front_camera_images.append(image_path)

        ego_poses.append(nusc.get("ego_pose", cam_front_data["ego_pose_token"]))
        camera_params.append(nusc.get("calibrated_sensor", cam_front_data["calibrated_sensor_token"]))

        if curr_sample_token == last_sample_token:
            break
        curr_sample_token = sample["next"]

    return token, name, front_camera_images, ego_poses, camera_params


def compute_scene_motion_features(ego_poses):
    scene_length = len(ego_poses)
    ego_poses_world = np.array([ego_pose["translation"][:3] for ego_pose in ego_poses])

    ego_velocities = np.zeros_like(ego_poses_world)
    if scene_length > 1:
        ego_velocities[1:] = ego_poses_world[1:] - ego_poses_world[:-1]
        ego_velocities[0] = ego_velocities[1]

    ego_curvatures = EstimateCurvatureFromTrajectory(ego_poses_world)
    ego_velocities_norm = np.linalg.norm(ego_velocities, axis=1)
    initial_heading = atan2(ego_velocities[0][1], ego_velocities[0][0]) if scene_length > 0 else 0.0
    estimated_points = IntegrateCurvatureForPoints(
        ego_curvatures,
        ego_velocities_norm,
        ego_poses_world[0],
        initial_heading,
        scene_length,
    )
    ego_traj_world = [ego_pose["translation"][:3] for ego_pose in ego_poses]

    return scene_length, ego_poses_world, ego_velocities, ego_curvatures, estimated_points, ego_traj_world


def build_scene_output_dir(output_dir, scene_idx, scene_name):
    scene_dir = os.path.join(output_dir, f"scene_{scene_idx:05d}_{scene_name}")
    os.makedirs(scene_dir, exist_ok=True)
    return scene_dir


def load_completed_scene_indices(results_path):
    completed_scene_indices = set()
    if not os.path.exists(results_path):
        return completed_scene_indices

    with open(results_path, "r") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"Skipping malformed JSONL record at {results_path}:{line_number}; "
                    "resume will ignore this partial line."
                )
                continue

            scene_index = record.get("scene_index")
            if scene_index is None:
                continue
            try:
                completed_scene_indices.add(int(scene_index))
            except (TypeError, ValueError):
                print(
                    f"Skipping invalid scene_index `{scene_index}` at {results_path}:{line_number}."
                )
    return completed_scene_indices


def reset_output_dir(output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)


def getMessage(prompt, image=None, args=None):
    if "llama" in args.model_path or "Llama" in args.model_path:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    if "qwen" in args.model_path or "Qwen" in args.model_path:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    return []


def prepare_llava_inputs(text, image_path, processor, tokenizer, model, args):
    pruning_text = text.replace(IMAGE_PLACEHOLDER, "").strip()
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

    if IMAGE_PLACEHOLDER in text:
        if model.config.mm_use_im_start_end:
            text = re.sub(IMAGE_PLACEHOLDER, image_token_se, text)
        else:
            text = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, text)
    else:
        if model.config.mm_use_im_start_end:
            text = image_token_se + "\n" + text
        else:
            text = DEFAULT_IMAGE_TOKEN + "\n" + text

    conv_mode = infer_llava_conv_mode(getattr(args, "llava_model_name", None))
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    model_device = get_model_device(model)
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model_device)

    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [tensor.to(model_device, dtype=torch.float16) for tensor in image_tensor]
    else:
        image_tensor = image_tensor.to(model_device, dtype=torch.float16)

    return input_ids, image_tensor, image.size, pruning_text


def build_llava_generate_kwargs(pruning_text, tokenizer, args, qapruner_text=None, enable_qapruner=True, **overrides):
    generate_kwargs = {
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if getattr(args, "visual_token_num", None) and enable_qapruner:
        generate_kwargs.update(
            texts=qapruner_text or pruning_text,
            add_quant=args.add_quant,
            alpha=args.alpha,
            dynamic_alpha=args.dynamic_alpha,
            quant_method=args.quant_method,
            pruning_method=args.pruning_method,
        )
    generate_kwargs.update(overrides)
    return generate_kwargs


def generate_llava_text(
    text,
    image_path,
    processor,
    model,
    tokenizer,
    args,
    *,
    max_new_tokens=2048,
    do_sample=True,
    temperature=0.2,
    top_p=None,
    num_beams=1,
    qapruner_text=None,
    enable_qapruner=True,
):
    input_ids, image_tensor, image_size, pruning_text = prepare_llava_inputs(
        text,
        image_path,
        processor,
        tokenizer,
        model,
        args,
    )
    generate_kwargs = build_llava_generate_kwargs(
        pruning_text,
        tokenizer,
        args,
        qapruner_text=qapruner_text,
        enable_qapruner=enable_qapruner,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            **generate_kwargs,
        )

    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def save_calibration_summary(state):
    if not state or not state.get("summary_path"):
        return

    summary = {
        "status": state["status"],
        "completed": state["completed"],
        "module_count": state["module_count"],
        "phase": state["phase"],
        "calibrate_target": state["calibrate_target"],
        "search_target": state["search_target"],
        "calibrate_done": state["calibrate_done"],
        "search_done": state["search_done"],
        "started_at": state["started_at"],
        "ended_at": state.get("ended_at"),
        "error": state.get("error"),
    }
    with open(state["summary_path"], "w") as file:
        json.dump(summary, file, indent=2)


def finish_quant_calibration(model, state, status):
    if state is None or state["completed"]:
        return
    set_quant_act_mode(model, calibrate=False, search=False)
    state["completed"] = True
    state["status"] = status
    state["ended_at"] = datetime.now().isoformat()
    save_calibration_summary(state)


def initialize_quant_calibration(model, args, summary_path):
    if model is None or "llava" not in args.model_path or not args.run_calibration:
        return None

    state = {
        "status": "initializing",
        "completed": False,
        "module_count": 0,
        "phase": "calibrate" if args.calibration_samples > 0 else "search",
        "calibrate_target": max(0, args.calibration_samples),
        "search_target": max(0, args.calibration_search_samples),
        "calibrate_done": 0,
        "search_done": 0,
        "prompt_index": 0,
        "summary_path": summary_path,
        "started_at": datetime.now().isoformat(),
    }

    if state["calibrate_target"] == 0 and state["search_target"] == 0:
        state["completed"] = True
        state["status"] = "skipped_no_targets"
        state["ended_at"] = datetime.now().isoformat()
        save_calibration_summary(state)
        return state

    module_count = set_quant_act_mode(
        model,
        calibrate=True,
        search=(state["phase"] == "search"),
    )
    state["module_count"] = module_count

    if module_count == 0:
        state["completed"] = True
        state["status"] = "skipped_no_quant_act_modules"
        state["ended_at"] = datetime.now().isoformat()
        save_calibration_summary(state)
        print("QuantAct calibration skipped: no QuantAct modules were found in the loaded model.")
        return state

    state["status"] = "running"
    save_calibration_summary(state)
    print(
        "Starting QuantAct calibration with "
        f"{module_count} modules, {state['calibrate_target']} calibration samples, "
        f"and {state['search_target']} search samples."
    )
    return state


def run_scene_quant_calibration(
    scene_name,
    front_camera_images,
    ego_velocities,
    ego_curvatures,
    processor,
    model,
    tokenizer,
    args,
    state,
):
    if state is None or state["completed"]:
        return

    total_windows = max(0, len(front_camera_images) - OBS_LEN + 1)
    for window_idx in range(total_windows):
        if state["completed"]:
            break

        current_image = front_camera_images[window_idx + OBS_LEN - 1]
        obs_ego_velocities = ego_velocities[window_idx:window_idx + OBS_LEN]
        obs_ego_curvatures = ego_curvatures[window_idx:window_idx + OBS_LEN]
        prompts = build_calibration_prompts(obs_ego_velocities, obs_ego_curvatures)
        prompt = prompts[state["prompt_index"] % len(prompts)]

        generate_llava_text(
            prompt,
            current_image,
            processor,
            model,
            tokenizer,
            args,
            max_new_tokens=args.calibration_max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=None,
            num_beams=1,
            enable_qapruner=False,
        )

        state["prompt_index"] += 1
        if state["phase"] == "calibrate":
            state["calibrate_done"] += 1
            print(
                f"Calibration phase: {state['calibrate_done']}/{state['calibrate_target']} "
                f"(scene={scene_name}, window={window_idx})"
            )
            if state["calibrate_done"] >= state["calibrate_target"]:
                if state["search_target"] > 0:
                    state["phase"] = "search"
                    set_quant_act_mode(model, calibrate=True, search=True)
                    print("Switching QuantAct calibration to search mode.")
                else:
                    finish_quant_calibration(model, state, status="completed")
        else:
            state["search_done"] += 1
            print(
                f"Search phase: {state['search_done']}/{state['search_target']} "
                f"(scene={scene_name}, window={window_idx})"
            )
            if state["search_done"] >= state["search_target"]:
                finish_quant_calibration(model, state, status="completed")

        save_calibration_summary(state)


def vlm_inference(text=None, images=None, sys_message=None, processor=None, model=None, tokenizer=None, args=None, qapruner_text=None):
    del sys_message

    if "llama" in args.model_path or "Llama" in args.model_path:
        image = Image.open(images).convert("RGB")
        message = getMessage(text, args=args)
        input_text = processor.apply_chat_template(message, add_generation_prompt=True)
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(model.device)

        output = model.generate(**inputs, max_new_tokens=2048)
        output_text = processor.decode(output[0])
        output_text = re.findall(
            r"<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>",
            output_text,
            re.DOTALL,
        )[0].strip()
        return output_text

    if "qwen" in args.model_path or "Qwen" in args.model_path:
        message = getMessage(text, image=images, args=args)
        prompt_text = processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(message)
        inputs = processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]

    if "llava" in args.model_path:
        return generate_llava_text(
            text,
            images,
            processor,
            model,
            tokenizer,
            args,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.2,
            top_p=None,
            num_beams=1,
            qapruner_text=qapruner_text,
        )

    if "gpt" in args.model_path:
        prompt_messages = [
            {
                "role": "user",
                "content": [
                    *map(lambda item: {"image": item, "resize": 768}, images),
                    text,
                ],
            },
        ]
        params = {
            "model": "gpt-4o-2024-11-20",
            "messages": prompt_messages,
            "max_tokens": 400,
        }
        result = client.chat.completions.create(**params)
        return result.choices[0].message.content

    raise ValueError(f"Unsupported model path: {args.model_path}")


def SceneDescription(obs_images, processor=None, model=None, tokenizer=None, args=None):
    prompt = (
        "You are an autonomous driving labeller. You have access to these front-view camera images "
        "of a car taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. "
        "Provide a concise description of the driving scene according to traffic lights, movements of other "
        "cars or pedestrians and lane markings."
    )
    return vlm_inference(
        text=prompt,
        images=obs_images,
        processor=processor,
        model=model,
        tokenizer=tokenizer,
        args=args,
        qapruner_text=QAPRUNER_TEXT_SCENE,
    )


def DescribeObjects(obs_images, processor=None, model=None, tokenizer=None, args=None):
    prompt = (
        "You are an autonomous driving labeller. You have access to a front-view camera image of a vehicle "
        "taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. "
        "What other road users should you pay attention to in the driving scene? List two or three of them, "
        "specifying the location within the image and a short description of what each road user is doing."
    )
    return vlm_inference(
        text=prompt,
        images=obs_images,
        processor=processor,
        model=model,
        tokenizer=tokenizer,
        args=args,
        qapruner_text=QAPRUNER_TEXT_OBJECTS,
    )


def DescribeOrUpdateIntent(obs_images, prev_intent=None, processor=None, model=None, tokenizer=None, args=None):
    if prev_intent is None:
        prompt = (
            "You are an autonomous driving labeller. You have access to a front-view camera image of a vehicle "
            "taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. "
            "Based on the lane markings and the movement of other cars and pedestrians, provide a concise "
            "description of the desired intent of the ego car."
        )
    else:
        prompt = (
            "You are an autonomous driving labeller. You have access to a front-view camera image of a vehicle "
            "taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. "
            f"Half a second ago your intent was to {prev_intent}. Based on the updated lane markings and "
            "the updated movement of other cars and pedestrians, provide a concise explanation of your current intent."
        )
    return vlm_inference(
        text=prompt,
        images=obs_images,
        processor=processor,
        model=model,
        tokenizer=tokenizer,
        args=args,
        qapruner_text=QAPRUNER_TEXT_INTENT,
    )


def GenerateMotion(obs_images, obs_waypoints, obs_velocities, obs_curvatures, given_intent, processor=None, model=None, tokenizer=None, args=None):
    scene_description, object_description, intent_description = None, None, None

    if args.method == "openemma":
        scene_description = SceneDescription(obs_images, processor=processor, model=model, tokenizer=tokenizer, args=args)
        object_description = DescribeObjects(obs_images, processor=processor, model=model, tokenizer=tokenizer, args=args)
        intent_description = DescribeOrUpdateIntent(
            obs_images,
            prev_intent=given_intent,
            processor=processor,
            model=model,
            tokenizer=tokenizer,
            args=args,
        )
        print(f"Scene Description: {scene_description}")
        print(f"Object Description: {object_description}")
        print(f"Intent Description: {intent_description}")

    obs_waypoints_str = [f"[{waypoint[0]:.2f},{waypoint[1]:.2f}]" for waypoint in obs_waypoints]
    obs_waypoints_str = ", ".join(obs_waypoints_str)
    del obs_waypoints_str

    obs_speed_curvature_str = format_speed_curvature_history(obs_velocities, obs_curvatures)
    print(f"Observed Speed and Curvature: {obs_speed_curvature_str}")

    if args.method == "openemma":
        prompt = (
            "These are frames from a video taken by a camera mounted in the front of a car. "
            f"The scene is described as follows: {scene_description}. "
            f"The identified critical objects are {object_description}. "
            f"The car's intent is {intent_description}. "
            f"The 5 second historical velocities and curvatures of the ego car are {obs_speed_curvature_str}. "
            "Infer the association between these numbers and the image sequence. Generate the predicted future speeds "
            "and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. "
            "Write the raw text not markdown or latex. Future speeds and curvatures:"
        )
    else:
        prompt = (
            "These are frames from a video taken by a camera mounted in the front of a car. "
            f"The 5 second historical velocities and curvatures of the ego car are {obs_speed_curvature_str}. "
            "Infer the association between these numbers and the image sequence. Generate the predicted future speeds "
            "and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. "
            "Write the raw text not markdown or latex. Future speeds and curvatures:"
        )

    for _ in range(3):
        result = vlm_inference(
            text=prompt,
            images=obs_images,
            processor=processor,
            model=model,
            tokenizer=tokenizer,
            args=args,
            qapruner_text=QAPRUNER_TEXT_MOTION,
        )
        if "unable" not in result and "sorry" not in result and "[" in result:
            break

    return result, scene_description, object_description, intent_description


def load_requested_model(args):
    model = None
    processor = None
    tokenizer = None

    if "qwen" in args.model_path or "Qwen" in args.model_path:
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "/root/OpenEMMA/models/Qwen2.5-VL-3B-Instruct",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
            processor = AutoProcessor.from_pretrained("/root/OpenEMMA/models/Qwen2.5-VL-3B-Instruct")
            print("已本地加载 Qwen2.5-VL-3B-Instruct 并启用 flash attention。")
        except Exception as exc:
            print("Qwen2.5-VL-3B-Instruct 加载失败，尝试加载 Qwen2-VL-7B-Instruct。")
            print(exc)
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct",
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
            print("已加载 Qwen2-VL-7B-Instruct。")
    elif "llama" in args.model_path or "Llama" in args.model_path:
        model = MllamaForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(args.model_path)
        print(f"已加载 Llama 视觉模型：{args.model_path}")
    elif args.model_path == "llava":
        disable_torch_init()
        args.llava_model_name = "llava-v1.6-mistral-7b"
        tokenizer, model, processor, _ = load_pretrained_model(
            "liuhaotian/llava-v1.6-mistral-7b",
            None,
            args.llava_model_name,
            load_8bit=args.load_8bit,
            load_4bit=args.load_4bit,
            use_flash_attn=args.use_flash_attn,
            visual_token_num=args.visual_token_num,
        )
    elif "llava" in args.model_path:
        disable_torch_init()
        args.llava_model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, processor, _ = load_pretrained_model(
            args.model_path,
            None,
            args.llava_model_name,
            load_8bit=args.load_8bit,
            load_4bit=args.load_4bit,
            use_flash_attn=args.use_flash_attn,
            visual_token_num=args.visual_token_num,
        )
    else:
        print(f"未加载本地模型，推理将依赖外部服务或后续逻辑：{args.model_path}")


    return tokenizer, model, processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="gpt")
    parser.add_argument("--plot", type=str2bool, default=True)
    parser.add_argument("--dataroot", type=str, default="datasets/NuScenes")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--method", type=str, default="openemma")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument("--visual-token-num", type=int, default=None)
    parser.add_argument("--add-quant", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--dynamic-alpha", action="store_true", default=False)
    parser.add_argument("--quant-method", type=str, default="l2_norm")
    parser.add_argument("--pruning-method", type=str, default="cdpruner", choices=["cdpruner", "visionzip"])
    parser.add_argument("--run-calibration", action="store_true")
    parser.add_argument("--calibration-samples", type=int, default=8)
    parser.add_argument("--calibration-search-samples", type=int, default=2)
    parser.add_argument("--calibration-max-new-tokens", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    print(args.model_path)

    tokenizer, model, processor = load_requested_model(args)

    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = args.model_path + f"_results/{args.method}/" + timestamp
    if args.resume:
        os.makedirs(output_dir, exist_ok=True)
    else:
        reset_output_dir(output_dir)

    results_path = os.path.join(output_dir, "ade_results.jsonl")
    completed_scene_indices = load_completed_scene_indices(results_path) if args.resume else set()
    if args.resume:
        print(f"Resume enabled: found {len(completed_scene_indices)} completed scenes in {results_path}.")

    calibration_state = initialize_quant_calibration(
        model,
        args,
        summary_path=os.path.join(output_dir, "quant_calibration.json"),
    )

    nusc = NuScenes(version=args.version, dataroot=args.dataroot)
    scenes = nusc.scene
    print(f"Number of scenes: {len(scenes)}")

    if calibration_state and not calibration_state["completed"]:
        print("Running offline QuantAct calibration pass before full evaluation.")
        for scene in scenes:
            _, name, front_camera_images, ego_poses, _ = load_scene_sequence(nusc, scene, args)
            scene_length = len(front_camera_images)

            print(f"Calibration scene {name} has {scene_length} frames")
            if scene_length < TTL_LEN:
                print(f"Calibration scene {name} has less than {TTL_LEN} frames, skipping...")
                continue

            _, _, ego_velocities, ego_curvatures, _, _ = compute_scene_motion_features(ego_poses)

            run_scene_quant_calibration(
                name,
                front_camera_images,
                ego_velocities,
                ego_curvatures,
                processor,
                model,
                tokenizer,
                args,
                calibration_state,
            )
            if calibration_state["completed"]:
                break

        if not calibration_state["completed"]:
            calibration_state["error"] = (
                "Calibration dataset was exhausted before reaching the requested "
                "number of calibration/search samples."
            )
            finish_quant_calibration(model, calibration_state, status="partial")

    for scene_idx, scene in enumerate(scenes):
        if scene_idx in completed_scene_indices:
            print(f"Scene {scene['name']} (index={scene_idx}) already exists in ade_results.jsonl, skipping.")
            continue

        token, name, front_camera_images, ego_poses, camera_params = load_scene_sequence(nusc, scene, args)
        scene_length = len(front_camera_images)

        print(f"Scene {name} (index={scene_idx}) has {scene_length} frames")
        if scene_length < TTL_LEN:
            print(f"Scene {name} has less than {TTL_LEN} frames, skipping...")
            continue

        scene_output_dir = build_scene_output_dir(output_dir, scene_idx, name)
        scene_length, ego_poses_world, ego_velocities, ego_curvatures, estimated_points, ego_traj_world = compute_scene_motion_features(ego_poses)

        plt.plot(ego_poses_world[:, 0], ego_poses_world[:, 1], "r-", label="GT")

        if args.plot:
            plt.quiver(
                ego_poses_world[:, 0],
                ego_poses_world[:, 1],
                ego_velocities[:, 0],
                ego_velocities[:, 1],
                color="b",
            )
            plt.plot(estimated_points[:, 0], estimated_points[:, 1], "g-", label="Reconstruction")
            plt.legend()
            plt.savefig(f"{scene_output_dir}/{name}_interpolation.jpg")
            plt.close()

        prev_intent = None
        cam_images_sequence = []
        ade1s_list = []
        ade2s_list = []
        ade3s_list = []

        for i in range(scene_length - TTL_LEN):
            obs_images = front_camera_images[i:i + OBS_LEN]
            obs_ego_poses = ego_poses[i:i + OBS_LEN]
            obs_camera_params = camera_params[i:i + OBS_LEN]
            obs_ego_traj_world = ego_traj_world[i:i + OBS_LEN]
            fut_ego_traj_world = ego_traj_world[i + OBS_LEN:i + TTL_LEN]
            obs_ego_velocities = ego_velocities[i:i + OBS_LEN]
            obs_ego_curvatures = ego_curvatures[i:i + OBS_LEN]
            fut_start_world = obs_ego_traj_world[-1]
            curr_image = obs_images[-1]

            if "gpt" in args.model_path:
                img = cv2.imdecode(np.frombuffer(base64.b64decode(curr_image), dtype=np.uint8), cv2.IMREAD_COLOR)
                img = yolo3d_nuScenes(img, calib=obs_camera_params[-1])[0]
            else:
                with open(curr_image, "rb") as image_file:
                    img = cv2.imdecode(np.frombuffer(image_file.read(), dtype=np.uint8), cv2.IMREAD_COLOR)

            coordinates = []
            for _ in range(3):
                prompt_images = curr_image if "gpt" not in args.model_path else obs_images
                prediction, scene_description, object_description, updated_intent = GenerateMotion(
                    prompt_images,
                    obs_ego_traj_world,
                    obs_ego_velocities,
                    obs_ego_curvatures,
                    prev_intent,
                    processor=processor,
                    model=model,
                    tokenizer=tokenizer,
                    args=args,
                )

                prev_intent = updated_intent
                pred_waypoints = prediction.replace("Future speeds and curvatures:", "").strip()
                coordinates = re.findall(r"\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]", pred_waypoints)
                if coordinates:
                    break

            if not coordinates:
                continue

            speed_curvature_pred = [[float(v), float(k)] for v, k in coordinates][:10]
            print(f"Got {len(speed_curvature_pred)} future actions: {speed_curvature_pred}")

            pred_len = min(FUT_LEN, len(speed_curvature_pred))
            pred_curvatures = np.array(speed_curvature_pred)[:, 1] / 100
            pred_speeds = np.array(speed_curvature_pred)[:, 0]
            pred_traj = np.zeros((pred_len, 3))
            pred_traj[:pred_len, :2] = IntegrateCurvatureForPoints(
                pred_curvatures,
                pred_speeds,
                fut_start_world,
                atan2(obs_ego_velocities[-1][1], obs_ego_velocities[-1][0]),
                pred_len,
            )

            OverlayTrajectory(
                img,
                pred_traj.tolist(),
                obs_camera_params[-1],
                obs_ego_poses[-1],
                color=(255, 0, 0),
                args=args,
            )

            fut_ego_traj_world = np.array(fut_ego_traj_world)
            pred1_len = min(pred_len, 2)
            pred2_len = min(pred_len, 4)
            pred3_len = min(pred_len, 6)

            ade = np.mean(np.linalg.norm(fut_ego_traj_world[:pred_len] - pred_traj, axis=1))
            ade1s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred1_len] - pred_traj[:pred1_len], axis=1))
            ade2s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred2_len] - pred_traj[:pred2_len], axis=1))
            ade3s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred3_len] - pred_traj[:pred3_len], axis=1))

            ade1s_list.append(ade1s)
            ade2s_list.append(ade2s)
            ade3s_list.append(ade3s)

            if args.plot:
                cam_images_sequence.append(img.copy())
                cv2.imwrite(f"{scene_output_dir}/{name}_{i}_front_cam.jpg", img)

                plt.plot(fut_ego_traj_world[:, 0], fut_ego_traj_world[:, 1], "r-", label="GT")
                plt.plot(pred_traj[:, 0], pred_traj[:, 1], "b-", label="Pred")
                plt.legend()
                plt.title(f"Scene: {name}, Frame: {i}, ADE: {ade}")
                plt.savefig(f"{scene_output_dir}/{name}_{i}_traj.jpg")
                plt.close()

                np.save(f"{scene_output_dir}/{name}_{i}_pred_traj.npy", pred_traj)
                np.save(f"{scene_output_dir}/{name}_{i}_pred_curvatures.npy", pred_curvatures)
                np.save(f"{scene_output_dir}/{name}_{i}_pred_speeds.npy", pred_speeds)

                with open(f"{scene_output_dir}/{name}_{i}_logs.txt", "w") as file:
                    file.write(f"Scene Description: {scene_description}\n")
                    file.write(f"Object Description: {object_description}\n")
                    file.write(f"Intent Description: {updated_intent}\n")
                    file.write(f"Average Displacement Error: {ade}\n")

        if not ade1s_list:
            print(f"No valid predictions were generated for scene {name}.")
            continue

        mean_ade1s = np.mean(ade1s_list)
        mean_ade2s = np.mean(ade2s_list)
        mean_ade3s = np.mean(ade3s_list)
        aveg_ade = np.mean([mean_ade1s, mean_ade2s, mean_ade3s])

        result = {
            "name": name,
            "scene_index": scene_idx,
            "token": token,
            "ade1s": mean_ade1s,
            "ade2s": mean_ade2s,
            "ade3s": mean_ade3s,
            "avgade": aveg_ade,
        }
        with open(results_path, "a") as file:
            file.write(json.dumps(result))
            file.write("\n")

        if args.plot and cam_images_sequence:
            WriteImageSequenceToVideo(cam_images_sequence, f"{scene_output_dir}/{name}")


if __name__ == "__main__":
    main()
