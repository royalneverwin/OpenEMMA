import argparse
import base64
import json
import os
import re
from datetime import datetime
from math import atan2

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from nuscenes import NuScenes
from transformers import AutoProcessor, MllamaForConditionalGeneration, Qwen2VLForConditionalGeneration

import main as single_main
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None


def env_or_arg_int(env_name, arg_value):
    if env_name in os.environ:
        return int(os.environ[env_name])
    return arg_value


def resolve_runtime_context(args):
    rank = env_or_arg_int("RANK", args.rank)
    world_size = env_or_arg_int("WORLD_SIZE", args.world_size)
    local_rank = env_or_arg_int("LOCAL_RANK", args.local_rank)

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if local_rank < 0 or local_rank >= device_count:
            raise ValueError(
                f"Invalid local rank {local_rank} for {device_count} visible CUDA devices."
            )
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        backend = "nccl"
    else:
        device = "cpu"
        backend = "gloo"

    distributed = False
    initialized_dist = False
    if world_size > 1 and "MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)
            initialized_dist = True
        distributed = True

    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "device": device,
        "distributed": distributed,
        "initialized_dist": initialized_dist,
    }


def log(context, message):
    print(f"[rank {context['rank']}] {message}", flush=True)


def barrier(context):
    if context["distributed"] and dist.is_initialized():
        dist.barrier()


def cleanup_distributed(context):
    if context["initialized_dist"] and dist.is_initialized():
        dist.destroy_process_group()


def resolve_run_name(args, context):
    if args.run_name:
        run_name = args.run_name
    elif context["distributed"] and dist.is_initialized():
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S") if context["rank"] == 0 else None
        run_name_list = [run_name]
        dist.broadcast_object_list(run_name_list, src=0)
        run_name = run_name_list[0]
    elif context["world_size"] > 1:
        run_name = f"manual_multi_ws{context['world_size']}"
    else:
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")

    return run_name


def build_output_dirs(args, context):
    run_name = resolve_run_name(args, context)
    if args.output_dir:
        output_root = args.output_dir
    else:
        output_root = os.path.join(f"{args.model_path}_results", args.method, f"multi_{run_name}")

    rank_output_dir = os.path.join(output_root, f"rank{context['rank']:02d}")
    os.makedirs(rank_output_dir, exist_ok=True)
    return output_root, rank_output_dir


def load_requested_model_on_device(args, device):
    model = None
    processor = None
    tokenizer = None
    device_map = {"": device}

    if "qwen" in args.model_path or "Qwen" in args.model_path:
        preferred_qwen_path = (
            args.model_path if os.path.exists(args.model_path) else "/root/OpenEMMA/models/Qwen2.5-VL-3B-Instruct"
        )
        if Qwen2_5_VLForConditionalGeneration is not None:
            try:
                qwen_kwargs = {
                    "torch_dtype": torch.bfloat16,
                    "device_map": device_map,
                }
                if args.use_flash_attn:
                    qwen_kwargs["attn_implementation"] = "flash_attention_2"
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    preferred_qwen_path,
                    **qwen_kwargs,
                )
                processor = AutoProcessor.from_pretrained(preferred_qwen_path)
                print(f"Loaded Qwen2.5-VL model on {device}: {preferred_qwen_path}")
            except Exception as exc:
                print("Qwen2.5-VL load failed, falling back to Qwen2-VL-7B-Instruct.")
                print(exc)
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen2-VL-7B-Instruct",
                    torch_dtype=torch.bfloat16,
                    device_map=device_map,
                )
                processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
            )
            processor = AutoProcessor.from_pretrained(args.model_path)
    elif "llama" in args.model_path or "Llama" in args.model_path:
        model = MllamaForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(args.model_path)
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
            use_qvlm_custom_bnb=args.use_qvlm_custom_bnb,
            custom_bnb_path=args.custom_bnb_path,
            device=device,
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
            use_qvlm_custom_bnb=args.use_qvlm_custom_bnb,
            custom_bnb_path=args.custom_bnb_path,
            device=device,
        )
    else:
        print(f"No local model loaded for `{args.model_path}`; external-service logic may be used instead.")

    if model is not None and hasattr(model, "eval"):
        model.eval()
    return tokenizer, model, processor


def run_shared_offline_calibration(nusc, scenes, processor, model, tokenizer, args, calibration_state, context):
    if calibration_state is None or calibration_state["completed"]:
        return

    log(context, "Running offline QuantAct calibration pass on the shared prefix of the dataset.")
    for scene in scenes:
        _, name, front_camera_images, ego_poses, _ = single_main.load_scene_sequence(nusc, scene, args)
        scene_length = len(front_camera_images)

        log(context, f"Calibration scene {name} has {scene_length} frames")
        if scene_length < single_main.TTL_LEN:
            log(context, f"Calibration scene {name} has less than {single_main.TTL_LEN} frames, skipping.")
            continue

        _, _, ego_velocities, ego_curvatures, _, _ = single_main.compute_scene_motion_features(ego_poses)

        single_main.run_scene_quant_calibration(
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
        single_main.finish_quant_calibration(model, calibration_state, status="partial")


def evaluate_shard(nusc, scenes, processor, model, tokenizer, args, output_dir, context):
    results_path = os.path.join(output_dir, "ade_results.jsonl")
    processed_scene_count = 0

    for scene in scenes:
        token, name, front_camera_images, ego_poses, camera_params = single_main.load_scene_sequence(nusc, scene, args)
        scene_length = len(front_camera_images)

        log(context, f"Scene {name} has {scene_length} frames")
        if scene_length < single_main.TTL_LEN:
            log(context, f"Scene {name} has less than {single_main.TTL_LEN} frames, skipping.")
            continue

        scene_length, ego_poses_world, ego_velocities, ego_curvatures, estimated_points, ego_traj_world = (
            single_main.compute_scene_motion_features(ego_poses)
        )

        if args.plot:
            plt.plot(ego_poses_world[:, 0], ego_poses_world[:, 1], "r-", label="GT")
            plt.quiver(
                ego_poses_world[:, 0],
                ego_poses_world[:, 1],
                ego_velocities[:, 0],
                ego_velocities[:, 1],
                color="b",
            )
            plt.plot(estimated_points[:, 0], estimated_points[:, 1], "g-", label="Reconstruction")
            plt.legend()
            plt.savefig(os.path.join(output_dir, f"{name}_interpolation.jpg"))
            plt.close()

        prev_intent = None
        cam_images_sequence = []
        ade1s_list = []
        ade2s_list = []
        ade3s_list = []

        for i in range(scene_length - single_main.TTL_LEN):
            obs_images = front_camera_images[i:i + single_main.OBS_LEN]
            obs_ego_poses = ego_poses[i:i + single_main.OBS_LEN]
            obs_camera_params = camera_params[i:i + single_main.OBS_LEN]
            obs_ego_traj_world = ego_traj_world[i:i + single_main.OBS_LEN]
            fut_ego_traj_world = ego_traj_world[i + single_main.OBS_LEN:i + single_main.TTL_LEN]
            obs_ego_velocities = ego_velocities[i:i + single_main.OBS_LEN]
            obs_ego_curvatures = ego_curvatures[i:i + single_main.OBS_LEN]
            fut_start_world = obs_ego_traj_world[-1]
            curr_image = obs_images[-1]

            if "gpt" in args.model_path:
                img = cv2.imdecode(np.frombuffer(base64.b64decode(curr_image), dtype=np.uint8), cv2.IMREAD_COLOR)
                img = single_main.yolo3d_nuScenes(img, calib=obs_camera_params[-1])[0]
            else:
                with open(curr_image, "rb") as image_file:
                    img = cv2.imdecode(np.frombuffer(image_file.read(), dtype=np.uint8), cv2.IMREAD_COLOR)

            coordinates = []
            for _ in range(3):
                prompt_images = curr_image if "gpt" not in args.model_path else obs_images
                prediction, scene_description, object_description, updated_intent = single_main.GenerateMotion(
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
            log(context, f"Scene {name} frame {i}: got {len(speed_curvature_pred)} future actions.")

            pred_len = min(single_main.FUT_LEN, len(speed_curvature_pred))
            pred_curvatures = np.array(speed_curvature_pred)[:, 1] / 100
            pred_speeds = np.array(speed_curvature_pred)[:, 0]
            pred_traj = np.zeros((pred_len, 3))
            pred_traj[:pred_len, :2] = single_main.IntegrateCurvatureForPoints(
                pred_curvatures,
                pred_speeds,
                fut_start_world,
                atan2(obs_ego_velocities[-1][1], obs_ego_velocities[-1][0]),
                pred_len,
            )

            single_main.OverlayTrajectory(
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
                cv2.imwrite(os.path.join(output_dir, f"{name}_{i}_front_cam.jpg"), img)

                plt.plot(fut_ego_traj_world[:, 0], fut_ego_traj_world[:, 1], "r-", label="GT")
                plt.plot(pred_traj[:, 0], pred_traj[:, 1], "b-", label="Pred")
                plt.legend()
                plt.title(f"Scene: {name}, Frame: {i}, ADE: {ade}")
                plt.savefig(os.path.join(output_dir, f"{name}_{i}_traj.jpg"))
                plt.close()

                np.save(os.path.join(output_dir, f"{name}_{i}_pred_traj.npy"), pred_traj)
                np.save(os.path.join(output_dir, f"{name}_{i}_pred_curvatures.npy"), pred_curvatures)
                np.save(os.path.join(output_dir, f"{name}_{i}_pred_speeds.npy"), pred_speeds)

                with open(os.path.join(output_dir, f"{name}_{i}_logs.txt"), "w") as file:
                    file.write(f"Scene Description: {scene_description}\n")
                    file.write(f"Object Description: {object_description}\n")
                    file.write(f"Intent Description: {updated_intent}\n")
                    file.write(f"Average Displacement Error: {ade}\n")

        if not ade1s_list:
            log(context, f"No valid predictions were generated for scene {name}.")
            continue

        mean_ade1s = np.mean(ade1s_list)
        mean_ade2s = np.mean(ade2s_list)
        mean_ade3s = np.mean(ade3s_list)
        aveg_ade = np.mean([mean_ade1s, mean_ade2s, mean_ade3s])

        result = {
            "name": name,
            "token": token,
            "rank": context["rank"],
            "ade1s": mean_ade1s,
            "ade2s": mean_ade2s,
            "ade3s": mean_ade3s,
            "avgade": aveg_ade,
        }
        with open(results_path, "a") as file:
            file.write(json.dumps(result))
            file.write("\n")

        processed_scene_count += 1

        if args.plot and cam_images_sequence:
            single_main.WriteImageSequenceToVideo(cam_images_sequence, os.path.join(output_dir, name))

    return processed_scene_count


def merge_rank_results(output_root, world_size):
    merged_path = os.path.join(output_root, "ade_results.jsonl")
    with open(merged_path, "w") as merged_file:
        for rank in range(world_size):
            shard_file = os.path.join(output_root, f"rank{rank:02d}", "ade_results.jsonl")
            if not os.path.exists(shard_file):
                continue
            with open(shard_file, "r") as shard_reader:
                for line in shard_reader:
                    merged_file.write(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="gpt")
    parser.add_argument("--plot", type=single_main.str2bool, default=True)
    parser.add_argument("--dataroot", type=str, default="datasets/NuScenes")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--method", type=str, default="openemma")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--use-qvlm-custom-bnb", action="store_true")
    parser.add_argument("--custom-bnb-path", type=str, default=None)
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
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=0)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    context = resolve_runtime_context(args)
    output_root = None
    try:
        log(context, f"Loading model `{args.model_path}` on {context['device']}.")
        tokenizer, model, processor = load_requested_model_on_device(args, context["device"])

        output_root, rank_output_dir = build_output_dirs(args, context)
        with open(os.path.join(rank_output_dir, "runtime.json"), "w") as file:
            json.dump(
                {
                    "rank": context["rank"],
                    "world_size": context["world_size"],
                    "local_rank": context["local_rank"],
                    "device": context["device"],
                    "model_path": args.model_path,
                },
                file,
                indent=2,
            )

        calibration_state = single_main.initialize_quant_calibration(
            model,
            args,
            summary_path=os.path.join(rank_output_dir, "quant_calibration.json"),
        )

        nusc = NuScenes(version=args.version, dataroot=args.dataroot)
        scenes = nusc.scene
        shard_scenes = scenes[context["rank"]::context["world_size"]]
        log(context, f"Number of scenes: total={len(scenes)}, shard={len(shard_scenes)}")

        run_shared_offline_calibration(
            nusc,
            scenes,
            processor,
            model,
            tokenizer,
            args,
            calibration_state,
            context,
        )

        processed_scene_count = evaluate_shard(
            nusc,
            shard_scenes,
            processor,
            model,
            tokenizer,
            args,
            rank_output_dir,
            context,
        )
        log(context, f"Finished evaluation for {processed_scene_count} scenes.")

        barrier(context)
        if context["rank"] == 0 and context["world_size"] > 1:
            merge_rank_results(output_root, context["world_size"])
            log(context, f"Merged shard results into {os.path.join(output_root, 'ade_results.jsonl')}")
    finally:
        cleanup_distributed(context)


if __name__ == "__main__":
    main()
