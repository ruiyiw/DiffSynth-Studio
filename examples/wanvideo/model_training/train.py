import torch, os, argparse, accelerate, warnings
try:
    import hf_transfer
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
except ImportError:
    hf_transfer = None

try:
    import wandb
except ImportError:
    wandb = None
try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WandbLogger(ModelLogger):
    def __init__(self, accelerator, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x, args=None):
        super().__init__(output_path, remove_prefix_in_ckpt, state_dict_converter)
        self.use_wandb = False
        self.log_steps = getattr(args, "log_steps", 1)
        if args is not None and args.use_wandb:
            if wandb is None:
                warnings.warn("wandb is not installed. Please install it using `pip install wandb`.")
            elif accelerator.is_main_process:
                wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))
                self.use_wandb = True

        self.push_to_hub = getattr(args, "push_to_hub", False)
        self.hub_model_id = getattr(args, "hub_model_id", None)
        self.hub_token = getattr(args, "hub_token", None)
        if self.hub_token == "":
            self.hub_token = None
        if self.push_to_hub:
            if HfApi is None:
                warnings.warn("huggingface_hub is not installed. Please install it using `pip install huggingface_hub`.")
                self.push_to_hub = False
            elif accelerator.is_main_process:
                self.api = HfApi(token=self.hub_token)
                
                if hf_transfer is None:
                    print("Warning: `hf_transfer` not installed. Recommended for faster uploads: `pip install hf_transfer`")
                else:
                    print(f"hf_transfer enabled: {os.environ.get('HF_HUB_ENABLE_HF_TRANSFER')}")

                if self.hub_model_id:
                    if "/" not in self.hub_model_id:
                        try:
                            username = self.api.whoami()["name"]
                            self.hub_model_id = f"{username}/{self.hub_model_id}"
                        except Exception as e:
                            print(f"Failed to auto-detect username: {e}")

                    try:
                        self.api.create_repo(repo_id=self.hub_model_id, exist_ok=True, repo_type="model")
                    except Exception as e:
                        print(f"Warning: Failed to create URL: {e}")

    def on_step_end(self, accelerator, model, save_steps=None, **kwargs):
        if self.use_wandb and self.num_steps % self.log_steps == 0:
            logs = {}
            if "loss" in kwargs:
                logs["loss"] = kwargs["loss"].item()
            if "learning_rate" in kwargs:
                logs["learning_rate"] = kwargs["learning_rate"]
            if "epoch" in kwargs:
                logs["epoch"] = kwargs["epoch"]
            if logs:
                wandb.log(logs, step=self.num_steps)
        super().on_step_end(accelerator, model, save_steps, **kwargs)

    def on_epoch_end(self, accelerator, model, epoch_id, **kwargs):
        super().on_epoch_end(accelerator, model, epoch_id)
        if self.push_to_hub and accelerator.is_main_process and self.hub_model_id:
            filename = f"epoch-{epoch_id}.safetensors"
            file_path = os.path.join(self.output_path, filename)
            if os.path.exists(file_path):
                self.api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=filename,
                    repo_id=self.hub_model_id,
                    repo_type="model"
                )


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    ):
        super().__init__()
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True
        
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        audio_processor_config = ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/") if audio_processor_path is None else ModelConfig(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, audio_processor_config=audio_processor_config)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        
    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging.")
    parser.add_argument("--wandb_project", type=str, default="wan-video", help="Wandb project name.")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name.")
    parser.add_argument("--log_steps", type=int, default=1, help="Frequency of logging loss.")
    parser.add_argument("--push_to_hub", action="store_true", help="Push checkpoints to Hugging Face Hub.")
    parser.add_argument("--hub_model_id", type=str, default=None, help="The name of the repository to keep in sync with the local `output_dir`.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
        }
    )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )
    model_logger = WandbLogger(
        accelerator,
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        args=args
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
