import torchaudio
import tempfile
from typing import Optional, List
import torch
import os
import ast
import sys
import librosa

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from pipeline_ace_step import ACEStepPipeline as AP

import folder_paths
cache_dir = folder_paths.get_temp_directory()
models_dir = folder_paths.models_dir
model_path = os.path.join(models_dir, "TTS", "ACE-Step-v1-3.5B")


class AudioCacher:
    """
    一个用于缓存音频张量到临时文件，并在之后清理这些文件的类。
    支持作为上下文管理器使用，以便自动清理。
    """
    def __init__(self, cache_dir: Optional[str] = None, default_format: str = "wav"):
        """
        初始化 AudioCacher。

        Args:
            cache_dir (Optional[str]): 缓存文件存放的目录。
                                       如果为 None，则使用系统默认的临时目录。
            default_format (str): 默认的音频文件格式后缀 (例如 "wav", "mp3", "flac")。
        """
        if cache_dir is None:
            self.cache_dir = tempfile.gettempdir()
        else:
            self.cache_dir = cache_dir
        # 确保缓存目录存在
        if not os.path.exists(self.cache_dir):
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
            except OSError as e:
                raise  # 重新抛出异常，因为这是一个关键的初始化步骤
        self.default_format = default_format.lstrip('.') # 确保没有前导点
        self._files_to_cleanup_in_context: List[str] = [] # 用于上下文管理器

    def cache_audio_tensor(
        self,
        audio_tensor: torch.Tensor,
        sample_rate: int,
        filename_prefix: str = "cached_audio_",
        audio_format: Optional[str] = None
    ) -> str:
        """
        将音频张量保存到缓存文件，并返回文件路径。

        Args:
            audio_tensor (torch.Tensor): 要保存的音频张量。
            sample_rate (int): 音频的采样率。
            filename_prefix (str): 缓存文件名的前缀。
            audio_format (Optional[str]): 要使用的音频格式 (例如 "wav", "mp3")。
                                       如果为 None，则使用初始化时设置的 default_format。

        Returns:
            str: 保存的缓存文件的绝对路径。

        Raises:
            RuntimeError: 如果保存音频失败。
        """
        current_format = (audio_format or self.default_format).lstrip('.')
        # 创建一个带特定后缀的临时文件，但不立即删除
        # NamedTemporaryFile 会在创建时打开文件，我们需要先关闭它才能让 torchaudio.save 使用
        try:
            with tempfile.NamedTemporaryFile(
                prefix=filename_prefix,
                suffix=f".{current_format}",
                dir=self.cache_dir,
                delete=False  # 这是关键，我们手动管理删除
            ) as tmp_file:
                temp_filepath = tmp_file.name
            # 此时 tmp_file 已经关闭，但文件因 delete=False 而保留
            torchaudio.save(temp_filepath, audio_tensor, sample_rate)
            # 如果在上下文管理器中使用，则记录此文件以备自动清理
            self._files_to_cleanup_in_context.append(temp_filepath)
            return temp_filepath
        except Exception as e:
            # 如果 temp_filepath 已定义且文件存在，尝试删除，因为它可能不完整或损坏
            if 'temp_filepath' in locals() and os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except OSError as e_clean:
                    logger.warning(f"Error cleaning temporary file {temp_filepath}: {e_clean}")
            raise RuntimeError(f"Failed to save audio: {e}") from e

    def cleanup_file(self, filepath: str) -> bool:
        """
        清理指定的缓存文件。

        Args:
            filepath (str): 要删除的文件的路径。

        Returns:
            bool: 如果文件成功删除或文件不存在，则返回 True；如果删除失败，则返回 False。
        """
        if not filepath:
            return True # 没有文件可以删除，所以认为是“成功”
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                # 如果文件在上下文中被跟踪，也从中移除
                if filepath in self._files_to_cleanup_in_context:
                    self._files_to_cleanup_in_context.remove(filepath)
                return True
            except OSError as e:
                return False
        else:
            # 如果文件在上下文中被跟踪，也从中移除
            if filepath in self._files_to_cleanup_in_context:
                self._files_to_cleanup_in_context.remove(filepath)
            return True # 文件不存在，也视为清理“成功”

    def cleanup_all_tracked_files(self) -> None:
        """
        清理所有由当前上下文管理器实例跟踪的缓存文件。
        """
        # 迭代列表的副本，因为 cleanup_file 可能会修改列表
        for f_path in list(self._files_to_cleanup_in_context):
            self.cleanup_file(f_path)
        self._files_to_cleanup_in_context.clear() # 确保列表被清空

    def __enter__(self):
        """进入上下文管理器时调用。"""
        # 重置跟踪文件列表，以防同一个实例被多次用于 'with' 语句
        self._files_to_cleanup_in_context = []
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器时调用，负责清理。"""
        self.cleanup_all_tracked_files()
        # 返回 False 以便在发生异常时重新抛出异常
        return False


from data_sampler import DataSampler

def sample_data(json_data):
    return (
            json_data["audio_duration"],
            json_data["prompt"],
            json_data["lyrics"],
            json_data["infer_step"],
            json_data["guidance_scale"],
            json_data["scheduler_type"],
            json_data["cfg_type"],
            json_data["omega_scale"],
            json_data["actual_seeds"][0],
            json_data["guidance_interval"],
            json_data["guidance_interval_decay"],
            json_data["min_guidance_scale"],
            json_data["use_erg_tag"],
            json_data["use_erg_lyric"],
            json_data["use_erg_diffusion"],
            ", ".join(map(str, json_data["oss_steps"])),
            json_data["guidance_scale_text"] if "guidance_scale_text" in json_data else 0.0,
            json_data["guidance_scale_lyric"] if "guidance_scale_lyric" in json_data else 0.0,
            )

data_sampler = DataSampler()

json_data = data_sampler.sample()
json_data = sample_data(json_data)

audio_duration,\
prompt, \
lyrics,\
infer_step, \
guidance_scale,\
scheduler_type, \
cfg_type, \
omega_scale, \
manual_seeds, \
guidance_interval, \
guidance_interval_decay, \
min_guidance_scale, \
use_erg_tag, \
use_erg_lyric, \
use_erg_diffusion, \
oss_steps, \
guidance_scale_text, \
guidance_scale_lyric = json_data


class GenerationParameters:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    { "audio_duration": ("FLOAT", {"default": audio_duration, "min": 0.0, "max": 240.0, "step": 1.0, "tooltip": "0 is a random length"}),
                      "infer_step": ("INT", {"default": infer_step, "min": 1, "max": 60, "step": 1}),
                      "guidance_scale": ("FLOAT", {"default": guidance_scale, "min": 0.0, "max": 200.0, "step": 0.1, "tooltip": "When guidance_scale_lyric > 1 and guidance_scale_text > 1, the guidance scale will not be applied."}),
                      "scheduler_type": (["euler", "heun"], {"default": scheduler_type, "tooltip": "euler is recommended. heun will take more time."}),
                      "cfg_type": (["cfg", "apg", "cfg_star"], {"default": cfg_type, "tooltip": "apg is recommended. cfg and cfg_star are almost the same."}),
                      "omega_scale": ("FLOAT", {"default": omega_scale, "min": -100.0, "max": 100.0, "step": 0.1, "tooltip": "Higher values can reduce artifacts"}),
                      "seed": ("INT", {"default":manual_seeds, "min": 0, "max": 4294967295, "step": 1}),
                      "guidance_interval": ("FLOAT", {"default": guidance_interval, "min": 0, "max": 1, "step": 0.01, "tooltip": "0.5 means only apply guidance in the middle steps"}),
                      "guidance_interval_decay": ("FLOAT", {"default": guidance_interval_decay, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Guidance scale will decay from guidance_scale to min_guidance_scale in the interval. 0.0 means no decay."}),
                      "min_guidance_scale": ("INT", {"default": min_guidance_scale, "min": 0, "max": 200, "step": 1}),
                      "use_erg_tag": ("BOOLEAN", {"default": use_erg_tag}),
                      "use_erg_lyric": ("BOOLEAN", {"default": use_erg_lyric}),
                      "use_erg_diffusion": ("BOOLEAN", {"default": use_erg_diffusion}),
                      "oss_steps": ("STRING", {"default": oss_steps}),
                      "guidance_scale_text": ("FLOAT", {"default": guidance_scale_text, "min": 0.0, "max": 10.0, "step": 0.1}),
                      "guidance_scale_lyric": ("FLOAT", {"default": guidance_scale_lyric, "min": 0.0, "max": 10.0, "step": 0.1}),
                    },
                }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("parameters",)
    FUNCTION = "generate"
    CATEGORY = "🎤MW/MW-ACE-Step"

    def generate(self, **kwargs):
        kwargs["manual_seeds"] = kwargs.pop("seed")
        return (str(kwargs),)


class MultiLinePromptACES:
    @classmethod
    def INPUT_TYPES(cls):
               
        return {
            "required": {
                "multi_line_prompt": ("STRING", {
                    "multiline": True, 
                    "default": prompt}),
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
                    "default": lyrics}),
                },
        }

    CATEGORY = "🎤MW/MW-ACE-Step"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "lyricsgen"
    
    def lyricsgen(self, multi_line_prompt: str):
        return (multi_line_prompt.strip(),)


class ACEStepGen:
    @classmethod
    def INPUT_TYPES(cls):
               
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True}),
                "lyrics": ("STRING", {"forceInput": True}),
                "parameters": ("STRING", {"forceInput": True}),
                # "unload_model": ("BOOLEAN", {"default": False}),
                },
        }

    CATEGORY = "🎤MW/MW-ACE-Step"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("music",)
    FUNCTION = "acestepgen"
    
    def acestepgen(self, prompt: str, lyrics: str, parameters: str, unload_model=True):
        
        parameters = ast.literal_eval(parameters)
        ap = AP(model_path)
        audio_output = ap(prompt=prompt, lyrics=lyrics, task="text2music", **parameters)
        audio, sr = audio_output[0][0].unsqueeze(0), audio_output[0][1]
        if unload_model:
            ap.cleanup()
        
        return ({"waveform": audio, "sample_rate": sr},)


class ACEStepRepainting:
    @classmethod
    def INPUT_TYPES(cls):
               
        return {
            "required": {
                "src_audio": ("AUDIO",),
                "prompt": ("STRING", {"forceInput": True}),
                "lyrics": ("STRING", {"forceInput": True}),
                "parameters": ("STRING", {"forceInput": True}),
                "repaint_start": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "repaint_end": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "repaint_variance": ("FLOAT", {"default": 0.01, "min": 0.01, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default":0, "min": 0, "max": 4294967295, "step": 1}),
                # "unload_model": ("BOOLEAN", {"default": False}),
                },
        }

    CATEGORY = "🎤MW/MW-ACE-Step"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("music",)
    FUNCTION = "acesteprepainting"
    
    def acesteprepainting(self, src_audio, prompt: str, lyrics: str, parameters: str, repaint_start, repaint_end, repaint_variance, seed, unload_model=True):
        retake_seeds = [str(seed)]
        ac = AudioCacher(cache_dir=cache_dir)
        src_audio_path = ac.cache_audio_tensor(src_audio["waveform"].squeeze(0), src_audio["sample_rate"], filename_prefix="src_audio_")
        
        audio_duration = librosa.get_duration(filename=src_audio_path)
        if repaint_end > audio_duration:
            repaint_end = audio_duration

        parameters = ast.literal_eval(parameters)
        parameters["audio_duration"] = audio_duration

        ap = AP(model_path)
        audio_output = ap(
            prompt=prompt, 
            lyrics=lyrics, 
            task="repaint", 
            retake_seeds=retake_seeds, 
            src_audio_path=src_audio_path, 
            repaint_start=repaint_start, 
            repaint_end=repaint_end, 
            retake_variance=repaint_variance, 
            **parameters)
            
        audio, sr = audio_output[0][0].unsqueeze(0), audio_output[0][1]

        ac.cleanup_file(src_audio_path)
        if unload_model:
            ap.cleanup()
        
        return ({"waveform": audio, "sample_rate": sr},)


class ACEStepEdit:
    @classmethod
    def INPUT_TYPES(cls):
               
        return {
            "required": {
                "src_audio": ("AUDIO",),
                "prompt": ("STRING", {"forceInput": True}),
                "lyrics": ("STRING", {"forceInput": True}),
                "parameters": ("STRING", {"forceInput": True}),
                "edit_prompt": ("STRING", {"forceInput": True}),
                "edit_lyrics": ("STRING", {"forceInput": True}),
                "edit_n_min": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "edit_n_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default":0, "min": 0, "max": 4294967295, "step": 1}),
                # "unload_model": ("BOOLEAN", {"default": False}),
                },
        }

    CATEGORY = "🎤MW/MW-ACE-Step"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("music",)
    FUNCTION = "acestepedit"
    
    def acestepedit(self, src_audio, prompt: str, lyrics: str, parameters: str, edit_prompt, edit_lyrics, edit_n_min, edit_n_max, seed, unload_model=True):
        retake_seeds = [str(seed)]
        ac = AudioCacher(cache_dir=cache_dir)
        src_audio_path = ac.cache_audio_tensor(src_audio["waveform"].squeeze(0), src_audio["sample_rate"], filename_prefix="src_audio_")
        
        audio_duration = librosa.get_duration(filename=src_audio_path)
        parameters = ast.literal_eval(parameters)
        parameters["audio_duration"] = audio_duration

        ap = AP(model_path)
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

        ac.cleanup_file(src_audio_path)
        if unload_model:
            ap.cleanup()
        
        return ({"waveform": audio, "sample_rate": sr},)


class ACEStepExtend:
    @classmethod
    def INPUT_TYPES(cls):
               
        return {
            "required": {
                "src_audio": ("AUDIO",),
                "prompt": ("STRING", {"forceInput": True}),
                "lyrics": ("STRING", {"forceInput": True}),
                "parameters": ("STRING", {"forceInput": True}),
                "left_extend_length": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "right_extend_length": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                # "repaint_variance": ("FLOAT", {"default": 0.01, "min": 0.01, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default":0, "min": 0, "max": 4294967295, "step": 1}),
                # "unload_model": ("BOOLEAN", {"default": False}),
                },
        }

    CATEGORY = "🎤MW/MW-ACE-Step"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("music",)
    FUNCTION = "acestepextend"
    
    def acestepextend(self, src_audio, prompt: str, lyrics: str, parameters: str, left_extend_length, right_extend_length, seed, unload_model=True):
        retake_seeds = [str(seed)]
        ac = AudioCacher(cache_dir=cache_dir)
        src_audio_path = ac.cache_audio_tensor(src_audio["waveform"].squeeze(0), src_audio["sample_rate"], filename_prefix="src_audio_")
        
        audio_duration = librosa.get_duration(filename=src_audio_path)
        repaint_start = -left_extend_length
        repaint_end = audio_duration + right_extend_length

        parameters = ast.literal_eval(parameters)
        parameters["audio_duration"] = audio_duration

        ap = AP(model_path)
        audio_output = ap(
            prompt=prompt, 
            lyrics=lyrics, 
            task="extend", 
            retake_seeds=retake_seeds, 
            src_audio_path=src_audio_path, 
            repaint_start=repaint_start, 
            repaint_end=repaint_end, 
            retake_variance=1.0,
            **parameters)
            
        audio, sr = audio_output[0][0].unsqueeze(0), audio_output[0][1]

        ac.cleanup_file(src_audio_path)
        if unload_model:
            ap.cleanup()
        
        return ({"waveform": audio, "sample_rate": sr},)


NODE_CLASS_MAPPINGS = {
    "ACEStepGen": ACEStepGen,
    "GenerationParameters": GenerationParameters,
    "MultiLinePromptACES": MultiLinePromptACES,
    "MultiLineLyrics": MultiLineLyrics,
    "ACEStepRepainting": ACEStepRepainting,
    "ACEStepEdit": ACEStepEdit,
    "ACEStepExtend": ACEStepExtend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ACEStepGen": "ACE-Step",
    "GenerationParameters": "ACE-Step Parameters",
    "MultiLinePromptACES": "ACE-Step Prompt",
    "MultiLineLyrics": "ACE-Step Lyrics",
    "ACEStepRepainting": "ACE-Step Repainting",
    "ACEStepEdit": "ACE-Step Edit",
    "ACEStepExtend": "ACE-Step Extend",
}