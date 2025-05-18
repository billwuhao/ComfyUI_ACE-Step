import torchaudio
import tempfile
from typing import Optional
import torch
import os
import ast
import sys
import librosa

from transformers import UMT5EncoderModel, AutoTokenizer

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from ace_step.pipeline_ace_step import ACEStepPipeline as AP
from ace_step.music_dcae.music_dcae_pipeline import MusicDCAE
from ace_step.ace_models.ace_step_transformer import ACEStepTransformer2DModel

import folder_paths
cache_dir = folder_paths.get_temp_directory()
models_dir = folder_paths.models_dir
model_path = os.path.join(models_dir, "TTS", "ACE-Step-v1-3.5B")

torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def cache_audio_tensor(
    cache_dir,
    audio_tensor: torch.Tensor,
    sample_rate: int,
    filename_prefix: str = "cached_audio_",
    audio_format: Optional[str] = ".wav"
) -> str:
    try:
        with tempfile.NamedTemporaryFile(
            prefix=filename_prefix,
            suffix=audio_format,
            dir=cache_dir,
            delete=False 
        ) as tmp_file:
            temp_filepath = tmp_file.name
        
        torchaudio.save(temp_filepath, audio_tensor, sample_rate)

        return temp_filepath
    except Exception as e:
        raise Exception(f"Error caching audio tensor: {e}")


def set_all_seeds(seed):
    # import random
    # import numpy as np
    # # 1. Python 内置随机模块
    # random.seed(seed)
    # # 2. NumPy 随机数生成器
    # np.random.seed(seed)
    # 3. PyTorch CPU 和 GPU 种子
    torch.manual_seed(seed)
    # 4. 如果使用 CUDA（GPU）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多 GPU 情况
        # torch.backends.cudnn.deterministic = True  # 确保卷积结果确定
        # torch.backends.cudnn.benchmark = False     # 关闭优化（牺牲速度换取确定性）


from ace_step.data_sampler import DataSampler

def sample_data(json_data):
    return {
            "audio_duration" : json_data["audio_duration"],
            "infer_step": json_data["infer_step"],
            "guidance_scale": json_data["guidance_scale"],
            "scheduler_type": json_data["scheduler_type"],
            "cfg_type": json_data["cfg_type"],
            "omega_scale": json_data["omega_scale"],
            "seed": int(json_data["actual_seeds"][0]),
            "guidance_interval": json_data["guidance_interval"],
            "guidance_interval_decay": json_data["guidance_interval_decay"],
            "min_guidance_scale": json_data["min_guidance_scale"],
            "use_erg_tag": json_data["use_erg_tag"],
            "use_erg_lyric": json_data["use_erg_lyric"],
            "use_erg_diffusion": json_data["use_erg_diffusion"],
            "oss_steps": ", ".join(map(str, json_data["oss_steps"])),
            "guidance_scale_text": json_data["guidance_scale_text"] if "guidance_scale_text" in json_data else 0.0,
            "guidance_scale_lyric": json_data["guidance_scale_lyric"] if "guidance_scale_lyric" in json_data else 0.0,
    }

data_sampler = DataSampler()
json_data = data_sampler.sample()
jd= sample_data(json_data)

device = torch.device("cpu")
dtype = torch.float32
if torch.cuda.is_available():
    device = torch.device("cuda")
    dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    dtype = torch.float16


class GenerationParameters:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    { "audio_duration": ("FLOAT", {"default": jd["audio_duration"], "min": 0.0, "max": 240.0, "step": 1.0, "tooltip": "0 is a random length"}),
                      "infer_step": ("INT", {"default": jd["infer_step"], "min": 1, "max": 200, "step": 1}),
                      "guidance_scale": ("FLOAT", {"default": jd["guidance_scale"], "min": 0.0, "max": 200.0, "step": 0.1, "tooltip": "When guidance_scale_lyric > 1 and guidance_scale_text > 1, the guidance scale will not be applied."}),
                      "scheduler_type": (["euler", "heun", "pingpong"], {"default": jd["scheduler_type"], "tooltip": "euler is recommended. heun will take more time."}),
                      "cfg_type": (["cfg", "apg", "cfg_star"], {"default": jd["cfg_type"], "tooltip": "apg is recommended. cfg and cfg_star are almost the same."}),
                      "omega_scale": ("FLOAT", {"default": jd["omega_scale"], "min": -100.0, "max": 100.0, "step": 0.1, "tooltip": "Higher values can reduce artifacts"}),
                      "seed": ("INT", {"default": jd["seed"], "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
                      "guidance_interval": ("FLOAT", {"default": jd["guidance_interval"], "min": 0, "max": 1, "step": 0.01, "tooltip": "0.5 means only apply guidance in the middle steps"}),
                      "guidance_interval_decay": ("FLOAT", {"default": jd["guidance_interval_decay"], "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Guidance scale will decay from guidance_scale to min_guidance_scale in the interval. 0.0 means no decay."}),
                      "min_guidance_scale": ("INT", {"default": jd["min_guidance_scale"], "min": 0, "max": 200, "step": 1}),
                      "use_erg_tag": ("BOOLEAN", {"default": jd["use_erg_tag"]}),
                      "use_erg_lyric": ("BOOLEAN", {"default": jd["use_erg_lyric"]}),
                      "use_erg_diffusion": ("BOOLEAN", {"default": jd["use_erg_diffusion"]}),
                      "oss_steps": ("STRING", {"default": jd["oss_steps"]}),
                      "guidance_scale_text": ("FLOAT", {"default": jd["guidance_scale_text"], "min": 0.0, "max": 10.0, "step": 0.1}),
                      "guidance_scale_lyric": ("FLOAT", {"default": jd["guidance_scale_lyric"], "min": 0.0, "max": 10.0, "step": 0.1}),
                    },
                    "optional": {
                    }
                }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("parameters",)
    FUNCTION = "generate"
    CATEGORY = "🎤MW/MW-ACE-Step"

    def generate(self, **kwargs):
        if kwargs["seed"] != 0:
            kwargs["manual_seeds"] = set_all_seeds(kwargs.pop("seed")) 
        return (str(kwargs),)


class MultiLinePromptACES:
    @classmethod
    def INPUT_TYPES(cls):
               
        return {
            "required": {
                "multi_line_prompt": ("STRING", {
                    "multiline": True, 
                    "default": json_data["prompt"]}),
                },
        }

    CATEGORY = "🎤MW/MW-ACE-Step"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "promptgen"
    
    def promptgen(self, multi_line_prompt: str):
        return (multi_line_prompt.strip(),)


class MultiLineLyrics:
    @classmethod
    def INPUT_TYPES(cls):
               
        return {
            "required": {
                "multi_line_prompt": ("STRING", {
                    "multiline": True, 
                    "default": json_data["lyrics"]}),
                },
        }

    CATEGORY = "🎤MW/MW-ACE-Step"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "lyricsgen"
    
    def lyricsgen(self, multi_line_prompt: str):
        return (multi_line_prompt.strip(),)


class ACEModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        models = [name for name in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, name)) and not name.startswith(".")]
        return {
            "required": {
                "dcae_checkpoint": (models, {"default": "music_dcae_f8c8"}),
                "vocoder_checkpoint": (models, {"default": "music_vocoder"}),
                "ace_step_checkpoint": (models, {"default": "ace_step_transformer"}),
                "text_encoder_checkpoint": (models, {"default": "umt5-base"}),
                # "quantized": ("BOOLEAN", {"default": False}),
                "cpu_offload": ("BOOLEAN", {"default": False}),
                "torch_compile": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("ACE_MODELS",)
    RETURN_NAMES = ("models",)
    FUNCTION = "load"
    CATEGORY = "🎤MW/MW-ACE-Step"

    def load(self, dcae_checkpoint, vocoder_checkpoint, ace_step_checkpoint, text_encoder_checkpoint, quantized=False, cpu_offload=False, torch_compile=False):
        dcae_checkpoint = os.path.join(model_path, "music_dcae_f8c8")
        vocoder_checkpoint = os.path.join(model_path, "music_vocoder")
        ace_step_checkpoint = os.path.join(model_path, "ace_step_transformer")
        text_encoder_checkpoint = os.path.join(model_path, "umt5-base")

        for path in [dcae_checkpoint, vocoder_checkpoint, ace_step_checkpoint, text_encoder_checkpoint]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Checkpoint not found: {path}")

        music_dcae = MusicDCAE(
            dcae_checkpoint_path=dcae_checkpoint,
            vocoder_checkpoint_path=vocoder_checkpoint
        )
        if cpu_offload:  # might be redundant
            music_dcae = music_dcae.to("cpu").eval().to(dtype)
        else:
            music_dcae = music_dcae.to(device).eval().to(dtype)
            
        ace_step_transformer = ACEStepTransformer2DModel.from_pretrained(ace_step_checkpoint, torch_dtype=dtype)
        if cpu_offload:
            ace_step_transformer = (
                ace_step_transformer.to("cpu").eval().to(dtype)
            )
        else:
            ace_step_transformer = (
                ace_step_transformer.to(device).eval().to(dtype)
            )

        text_encoder_model = UMT5EncoderModel.from_pretrained(text_encoder_checkpoint, torch_dtype=dtype)
        if cpu_offload:
            text_encoder_model = text_encoder_model.to("cpu").eval().to(dtype)
        else:
            text_encoder_model = text_encoder_model.to(device).eval().to(dtype)
            
        text_encoder_model.requires_grad_(False)
        text_tokenizer = AutoTokenizer.from_pretrained(text_encoder_checkpoint)

        if torch_compile:
            music_dcae = torch.compile(music_dcae)
            ace_step_transformer = torch.compile(ace_step_transformer)
            text_encoder_model = torch.compile(text_encoder_model)

        elif quantized:
            from torchao.quantization import (
                quantize_,
                Int4WeightOnlyConfig,
            )

            group_size = 128
            use_hqq = True

            music_dcae = torch.compile(music_dcae)
            ace_step_transformer = torch.compile(ace_step_transformer)
            text_encoder_model = torch.compile(text_encoder_model)
            
            quant_ace_path = os.path.join(ace_step_checkpoint, "diffusion_pytorch_model_int4wo.bin")
            if not os.path.exists(quant_ace_path):
                quantize_(
                    ace_step_transformer,
                    Int4WeightOnlyConfig(group_size=group_size, use_hqq=use_hqq),
                )
            # save quantized weights
            torch.save(
                ace_step_transformer.state_dict(),
                os.path.join(
                    ace_step_checkpoint, "diffusion_pytorch_model_int4wo.bin"
                ),
            )
            print("Quantized Weights Saved to: ", quant_ace_path,)

            ace_step_transformer.load_state_dict(
                torch.load(quant_ace_path, map_location=device,),
                assign=True
            )
            ace_step_transformer.torchao_quantized = True

            quant_encoder_path = os.path.join(text_encoder_checkpoint, "pytorch_model_int4wo.bin")
            if not os.path.exists(quant_encoder_path):
                quantize_(
                    text_encoder_model,
                    Int4WeightOnlyConfig(group_size=group_size, use_hqq=use_hqq),
                )

                torch.save(
                    text_encoder_model.state_dict(),
                    quant_encoder_path
                )
                print("Quantized Weights Saved to: ", quant_encoder_path)

            text_encoder_model.load_state_dict(
                torch.load(quant_encoder_path, map_location=device,),
                assign=True
            )
            text_encoder_model.torchao_quantized = True

            text_tokenizer = AutoTokenizer.from_pretrained(
                text_encoder_checkpoint
            )

        models = (
            music_dcae,
            ace_step_transformer,
            text_encoder_model,
            text_tokenizer,
            device,
            dtype
        )
        return (models,)


class ACELoRALoader:
    @classmethod
    def INPUT_TYPES(cls):
        loras_path = os.path.join(model_path, "loras")
        models = [name for name in os.listdir(loras_path) if not name.startswith(".")]
        return {
            "required": {
                "models": ("ACE_MODELS",),
                "lora_name": (models, {"default": "ACE-Step-v1-chinese-rap-LoRA"}),
                },
        }

    RETURN_TYPES = ("ACE_MODELS",)
    RETURN_NAMES = ("models",)
    FUNCTION = "load"
    CATEGORY = "🎤MW/MW-ACE-Step"

    def load(self, models, lora_name):
        lora_path = os.path.join(model_path, "loras", lora_name)
        models[1].load_lora_adapter(
            os.path.join(lora_path, "pytorch_lora_weights.safetensors"),
            adapter_name="zh_rap_lora",
            with_alpha=True,
        )
        return (models,)


ap = None
class ACEStepGen:
    files = DataSampler().input_params_files
    songs = {os.path.basename(file): file for file in files}

    @classmethod
    def INPUT_TYPES(cls):
               
        return {
            "required": {
                "models": ("ACE_MODELS",),
                # "unload_model": ("BOOLEAN", {"default": True}),
                },
            "optional": {
                "prompt": ("STRING", {"forceInput": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True,}),
                "lyrics": ("STRING", {"forceInput": True}),
                "parameters": ("STRING", {"forceInput": True}),
                "ref_audio": ("AUDIO",),
                "ref_audio_strength": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01}),
                "overlapped_decode": ("BOOLEAN", {"default": False}),
                "delicious_song": (list(cls.songs.keys()) + ["None"],{"default": "None"}),
                },
        }

    CATEGORY = "🎤MW/MW-ACE-Step"
    RETURN_TYPES = ("AUDIO", "STRING", "STRING",)
    RETURN_NAMES = ("music", "delicious_song_prompt", "delicious_song_lyrics",)
    FUNCTION = "acestepgen"
    
    def acestepgen(self, 
        models, 
        parameters: str="", 
        prompt: str="", 
        negative_prompt: str="",
        lyrics: str="", 
        ref_audio=None, 
        ref_audio_strength=None, 
        overlapped_decode=False, 
        delicious_song="None",
        # unload_model=True
        ):
        
        if delicious_song != "None":
            json_data = data_sampler.load_json(ACEStepGen.songs[delicious_song])
            prompt = json_data["prompt"]
            lyrics = json_data["lyrics"]
            parameters = sample_data(json_data)
            parameters["manual_seeds"] = parameters.pop("seed")
        else:
            assert parameters and prompt and lyrics, "parameters, prompt and lyrics are required"
            parameters = ast.literal_eval(parameters)

        global ap
        if ap is None:
            ap = AP(*models, overlapped_decode=overlapped_decode)

        audio2audio_enable = False
        ref_audio_input = None

        if ref_audio is not None:
            ref_audio_path = cache_audio_tensor(cache_dir, ref_audio["waveform"].squeeze(0), ref_audio["sample_rate"], filename_prefix="ref_audio_")
            audio2audio_enable = True
            ref_audio_strength = ref_audio_strength
            ref_audio_input = ref_audio_path
    
        audio_output = ap(
            prompt=prompt, 
            negative_prompt=negative_prompt.strip(),
            lyrics=lyrics, 
            task="audio2audio", 
            audio2audio_enable=audio2audio_enable, 
            ref_audio_strength=ref_audio_strength, 
            ref_audio_input=ref_audio_input, 
            **parameters
            )
        audio, sr = audio_output[0][0].unsqueeze(0), audio_output[0][1]

        # if unload_model:
        #     ap.cleanup()
        #     ap = None
        
        return ({"waveform": audio, "sample_rate": sr}, prompt, lyrics)


class ACEStepRepainting:
    @classmethod
    def INPUT_TYPES(cls):
               
        return {
            "required": {
                "models": ("ACE_MODELS",),
                "src_audio": ("AUDIO",),
                "prompt": ("STRING", {"forceInput": True}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "lyrics": ("STRING", {"forceInput": True}),
                "parameters": ("STRING", {"forceInput": True}),
                "repaint_start": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "repaint_end": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "repaint_variance": ("FLOAT", {"default": 0.01, "min": 0.01, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default":0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
                # "unload_model": ("BOOLEAN", {"default": True}),
                "overlapped_decode": ("BOOLEAN", {"default": False}),
                },
        }

    CATEGORY = "🎤MW/MW-ACE-Step"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("music",)
    FUNCTION = "acesteprepainting"
    
    def acesteprepainting(self, 
        models, 
        src_audio, 
        prompt: str, 
        lyrics: str, 
        parameters: str, 
        repaint_start, 
        repaint_end, 
        repaint_variance, 
        seed, 
        # unload_model=True, 
        negative_prompt: str="",
        overlapped_decode=False
        ):
        if seed != 0:
            set_all_seeds(seed)
        retake_seeds = [str(seed)]

        src_audio_path = cache_audio_tensor(cache_dir, src_audio["waveform"].squeeze(0), src_audio["sample_rate"], filename_prefix="src_audio_")
        
        audio_duration = librosa.get_duration(filename=src_audio_path)
        if repaint_end > audio_duration:
            repaint_end = audio_duration

        parameters = ast.literal_eval(parameters)
        parameters["audio_duration"] = audio_duration
        global ap
        if ap is None:
            ap = AP(*models, overlapped_decode=overlapped_decode)

        audio_output = ap(
            prompt=prompt, 
            negative_prompt=negative_prompt.strip(),
            lyrics=lyrics, 
            task="repaint", 
            retake_seeds=retake_seeds, 
            src_audio_path=src_audio_path, 
            repaint_start=repaint_start, 
            repaint_end=repaint_end, 
            retake_variance=repaint_variance, 
            **parameters)
            
        audio, sr = audio_output[0][0].unsqueeze(0), audio_output[0][1]

        # if unload_model:
        #     ap.cleanup()
        #     ap = None
        
        return ({"waveform": audio, "sample_rate": sr},)


class ACEStepEdit:
    @classmethod
    def INPUT_TYPES(cls):
               
        return {
            "required": {
                "models": ("ACE_MODELS",),
                "src_audio": ("AUDIO",),
                "prompt": ("STRING", {"forceInput": True}),
                "lyrics": ("STRING", {"forceInput": True}),
                "parameters": ("STRING", {"forceInput": True}),
                "edit_prompt": ("STRING", {"forceInput": True}),
                "edit_lyrics": ("STRING", {"forceInput": True}),
                "edit_n_min": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "edit_n_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default":0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
                # "unload_model": ("BOOLEAN", {"default": True}),
                "overlapped_decode": ("BOOLEAN", {"default": False}),
                },
        }

    CATEGORY = "🎤MW/MW-ACE-Step"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("music",)
    FUNCTION = "acestepedit"
    
    def acestepedit(self, 
        models, 
        src_audio, 
        prompt: str, 
        lyrics: str, 
        parameters: str, 
        edit_prompt, 
        edit_lyrics, 
        edit_n_min, 
        edit_n_max, 
        seed, 
        # unload_model=True, 
        overlapped_decode=False
        ):
        if seed!= 0:
            set_all_seeds(seed)
        retake_seeds = [str(seed)]

        src_audio_path = cache_audio_tensor(cache_dir, src_audio["waveform"].squeeze(0), src_audio["sample_rate"], filename_prefix="src_audio_")
        
        audio_duration = librosa.get_duration(filename=src_audio_path)
        parameters = ast.literal_eval(parameters)
        parameters["audio_duration"] = audio_duration
        global ap
        if ap is None:
            ap = AP(*models, overlapped_decode=overlapped_decode)

        audio_output = ap(
            prompt=prompt, 
            lyrics=lyrics, 
            task="edit", 
            retake_seeds=retake_seeds, 
            src_audio_path=src_audio_path, 
            edit_target_prompt = edit_prompt,
            edit_target_lyrics = edit_lyrics,
            edit_n_min = edit_n_min,
            edit_n_max = edit_n_max,
            **parameters)
            
        audio, sr = audio_output[0][0].unsqueeze(0), audio_output[0][1]

        # if unload_model:
        #     ap.cleanup()
        #     ap = None
        
        return ({"waveform": audio, "sample_rate": sr},)


class ACEStepExtend:
    @classmethod
    def INPUT_TYPES(cls):
               
        return {
            "required": {
                "models": ("ACE_MODELS",),
                "src_audio": ("AUDIO",),
                "prompt": ("STRING", {"forceInput": True}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "lyrics": ("STRING", {"forceInput": True}),
                "parameters": ("STRING", {"forceInput": True}),
                "left_extend_length": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "right_extend_length": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                # "repaint_variance": ("FLOAT", {"default": 0.01, "min": 0.01, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default":0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
                # "unload_model": ("BOOLEAN", {"default": True}),
                "overlapped_decode": ("BOOLEAN", {"default": False}),
                },
        }

    CATEGORY = "🎤MW/MW-ACE-Step"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("music",)
    FUNCTION = "acestepextend"
    
    def acestepextend(self, 
        models, 
        src_audio, 
        prompt: str, 
        lyrics: str, 
        parameters: str, 
        left_extend_length, 
        right_extend_length, 
        seed, 
        # unload_model=True, 
        negative_prompt: str="",
        overlapped_decode=False
        ):
        if seed!= 0:
            set_all_seeds(seed)
        retake_seeds = [str(seed)]

        src_audio_path = cache_audio_tensor(cache_dir, src_audio["waveform"].squeeze(0), src_audio["sample_rate"], filename_prefix="src_audio_")
        
        audio_duration = librosa.get_duration(filename=src_audio_path)
        repaint_start = -left_extend_length
        repaint_end = audio_duration + right_extend_length

        parameters = ast.literal_eval(parameters)
        parameters["audio_duration"] = audio_duration
        global ap
        if ap is None:
            ap = AP(*models, overlapped_decode=overlapped_decode)

        audio_output = ap(
            prompt=prompt, 
            negative_prompt=negative_prompt.strip(),
            lyrics=lyrics, 
            task="extend", 
            retake_seeds=retake_seeds, 
            src_audio_path=src_audio_path, 
            repaint_start=repaint_start, 
            repaint_end=repaint_end, 
            retake_variance=1.0,
            **parameters)
            
        audio, sr = audio_output[0][0].unsqueeze(0), audio_output[0][1]

        # if unload_model:
        #     ap.cleanup()
        #     ap = None
        
        return ({"waveform": audio, "sample_rate": sr},)


from .text2lyric import LyricsLangSwitch

NODE_CLASS_MAPPINGS = {
    "ACELoRALoader": ACELoRALoader,
    "ACEModelLoader": ACEModelLoader,
    "LyricsLangSwitch": LyricsLangSwitch,
    "ACEStepGen": ACEStepGen,
    "GenerationParameters": GenerationParameters,
    "MultiLinePromptACES": MultiLinePromptACES,
    "MultiLineLyrics": MultiLineLyrics,
    "ACEStepRepainting": ACEStepRepainting,
    "ACEStepEdit": ACEStepEdit,
    "ACEStepExtend": ACEStepExtend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ACELoRALoader": "ACE-Step LoRA Loader",
    "ACEModelLoader": "ACE-Step Model Loader",
    "LyricsLangSwitch": "ACE-Step Lyrics Language Switch",
    "ACEStepGen": "ACE-Step",
    "GenerationParameters": "ACE-Step Parameters",
    "MultiLinePromptACES": "ACE-Step Prompt",
    "MultiLineLyrics": "ACE-Step Lyrics",
    "ACEStepRepainting": "ACE-Step Repainting",
    "ACEStepEdit": "ACE-Step Edit",
    "ACEStepExtend": "ACE-Step Extend",
}
