import random
import time
import os
import re
import torch
from loguru import logger
from tqdm import tqdm
import math
from huggingface_hub import hf_hub_download

# from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from schedulers.scheduling_flow_match_heun_discrete import FlowMatchHeunDiscreteScheduler
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor
from transformers import UMT5EncoderModel, AutoTokenizer

from language_segmentation import LangSegment
from music_dcae.music_dcae_pipeline import MusicDCAE
from ace_models.ace_step_transformer import ACEStepTransformer2DModel
from ace_models.lyrics_utils.lyric_tokenizer import VoiceBpeTokenizer
from apg_guidance import apg_forward, MomentumBuffer, cfg_forward, cfg_zero_star, cfg_double_condition_forward

torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"


SUPPORT_LANGUAGES = {
    "en": 259, "de": 260, "fr": 262, "es": 284, "it": 285, 
    "pt": 286, "pl": 294, "tr": 295, "ru": 267, "cs": 293, 
    "nl": 297, "ar": 5022, "zh": 5023, "ja": 5412, "hu": 5753,
    "ko": 6152, "hi": 6680
}

structure_pattern = re.compile(r"\[.*?\]")


def ensure_directory_exists(directory):
    directory = str(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

REPO_ID = "ACE-Step/ACE-Step-v1-3.5B"

# class ACEStepPipeline(DiffusionPipeline):
class ACEStepPipeline:

    def __init__(self, checkpoint_dir, device_id=0, dtype="bfloat16", torch_compile=False, **kwargs):
        ensure_directory_exists(checkpoint_dir)
        self.checkpoint_dir = checkpoint_dir

        device = torch.device(f"cuda:{device_id}") if torch.cuda.is_available() else torch.device("cpu")
        if device.type == "cpu" and torch.backends.mps.is_available():
            device = torch.device("mps")

        self.dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32

        if device.type == "mps" and self.dtype == torch.bfloat16:
            self.dtype = torch.float16

        self.device = device
        
        import platform
        system = platform.system()
        if system == "Linux":
            torch_compile = True
        self.torch_compile = torch_compile

        start_time = time.time()
        print("Checkpoint not loaded, loading checkpoint...")
        self.load_checkpoint()
        load_model_cost = time.time() - start_time
        print(f"Model loaded in {load_model_cost:.2f} seconds.")
        
    def load_checkpoint(self):
        device = self.device

        dcae_model_path = os.path.join(self.checkpoint_dir, "music_dcae_f8c8")
        vocoder_model_path = os.path.join(self.checkpoint_dir, "music_vocoder")
        ace_step_model_path = os.path.join(self.checkpoint_dir, "ace_step_transformer")
        text_encoder_model_path = os.path.join(self.checkpoint_dir, "umt5-base")

        files_exist = (
            os.path.exists(os.path.join(dcae_model_path, "config.json")) and
            os.path.exists(os.path.join(dcae_model_path, "diffusion_pytorch_model.safetensors")) and
            os.path.exists(os.path.join(vocoder_model_path, "config.json")) and
            os.path.exists(os.path.join(vocoder_model_path, "diffusion_pytorch_model.safetensors")) and
            os.path.exists(os.path.join(ace_step_model_path, "config.json")) and
            os.path.exists(os.path.join(ace_step_model_path, "diffusion_pytorch_model.safetensors")) and
            os.path.exists(os.path.join(text_encoder_model_path, "config.json")) and
            os.path.exists(os.path.join(text_encoder_model_path, "model.safetensors")) and
            os.path.exists(os.path.join(text_encoder_model_path, "special_tokens_map.json")) and
            os.path.exists(os.path.join(text_encoder_model_path, "tokenizer_config.json")) and
            os.path.exists(os.path.join(text_encoder_model_path, "tokenizer.json"))
        )

        if not files_exist:
            logger.info(f"Checkpoint directory {self.checkpoint_dir} is not complete, downloading from Hugging Face Hub")

            # download music dcae model
            os.makedirs(dcae_model_path, exist_ok=True)
            hf_hub_download(repo_id=REPO_ID, subfolder="music_dcae_f8c8",
                            filename="config.json", local_dir=self.checkpoint_dir, local_dir_use_symlinks=False)
            hf_hub_download(repo_id=REPO_ID, subfolder="music_dcae_f8c8",
                            filename="diffusion_pytorch_model.safetensors", local_dir=self.checkpoint_dir, local_dir_use_symlinks=False)

            # download vocoder model
            os.makedirs(vocoder_model_path, exist_ok=True)
            hf_hub_download(repo_id=REPO_ID, subfolder="music_vocoder",
                            filename="config.json", local_dir=self.checkpoint_dir, local_dir_use_symlinks=False)
            hf_hub_download(repo_id=REPO_ID, subfolder="music_vocoder",
                            filename="diffusion_pytorch_model.safetensors", local_dir=self.checkpoint_dir, local_dir_use_symlinks=False)

            # download ace_step transformer model
            os.makedirs(ace_step_model_path, exist_ok=True)
            hf_hub_download(repo_id=REPO_ID, subfolder="ace_step_transformer",
                            filename="config.json", local_dir=self.checkpoint_dir, local_dir_use_symlinks=False)
            hf_hub_download(repo_id=REPO_ID, subfolder="ace_step_transformer",
                            filename="diffusion_pytorch_model.safetensors", local_dir=self.checkpoint_dir, local_dir_use_symlinks=False)

            # download text encoder model
            os.makedirs(text_encoder_model_path, exist_ok=True)
            hf_hub_download(repo_id=REPO_ID, subfolder="umt5-base",
                            filename="config.json", local_dir=self.checkpoint_dir, local_dir_use_symlinks=False)
            hf_hub_download(repo_id=REPO_ID, subfolder="umt5-base",
                            filename="model.safetensors", local_dir=self.checkpoint_dir, local_dir_use_symlinks=False)
            hf_hub_download(repo_id=REPO_ID, subfolder="umt5-base",
                            filename="special_tokens_map.json", local_dir=self.checkpoint_dir, local_dir_use_symlinks=False)
            hf_hub_download(repo_id=REPO_ID, subfolder="umt5-base",
                            filename="tokenizer_config.json", local_dir=self.checkpoint_dir, local_dir_use_symlinks=False)
            hf_hub_download(repo_id=REPO_ID, subfolder="umt5-base",
                            filename="tokenizer.json", local_dir=self.checkpoint_dir, local_dir_use_symlinks=False)

            logger.info("Models downloaded")

        dcae_checkpoint_path = dcae_model_path
        vocoder_checkpoint_path = vocoder_model_path
        ace_step_checkpoint_path = ace_step_model_path
        text_encoder_checkpoint_path = text_encoder_model_path

        self.music_dcae = MusicDCAE(dcae_checkpoint_path=dcae_checkpoint_path, vocoder_checkpoint_path=vocoder_checkpoint_path)
        self.music_dcae.to(device).eval().to(self.dtype)

        self.ace_step_transformer = ACEStepTransformer2DModel.from_pretrained(ace_step_checkpoint_path, torch_dtype=self.dtype)
        self.ace_step_transformer.to(device).eval().to(self.dtype)

        lang_segment = LangSegment()

        lang_segment.setfilters([
            'af', 'am', 'an', 'ar', 'as', 'az', 'be', 'bg', 'bn', 'br', 'bs', 'ca', 'cs', 'cy', 'da', 'de', 'dz', 'el',
            'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fo', 'fr', 'ga', 'gl', 'gu', 'he', 'hi', 'hr', 'ht', 'hu', 'hy',
            'id', 'is', 'it', 'ja', 'jv', 'ka', 'kk', 'km', 'kn', 'ko', 'ku', 'ky', 'la', 'lb', 'lo', 'lt', 'lv', 'mg',
            'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'nb', 'ne', 'nl', 'nn', 'no', 'oc', 'or', 'pa', 'pl', 'ps', 'pt', 'qu',
            'ro', 'ru', 'rw', 'se', 'si', 'sk', 'sl', 'sq', 'sr', 'sv', 'sw', 'ta', 'te', 'th', 'tl', 'tr', 'ug', 'uk',
            'ur', 'vi', 'vo', 'wa', 'xh', 'zh', 'zu'
        ])
        self.lang_segment = lang_segment
        self.lyric_tokenizer = VoiceBpeTokenizer()
        text_encoder_model = UMT5EncoderModel.from_pretrained(text_encoder_checkpoint_path, torch_dtype=self.dtype).eval()
        text_encoder_model = text_encoder_model.to(device).to(self.dtype)
        text_encoder_model.requires_grad_(False)
        self.text_encoder_model = text_encoder_model
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_encoder_checkpoint_path)

        # compile
        if self.torch_compile:
            self.music_dcae = torch.compile(self.music_dcae)
            self.ace_step_transformer = torch.compile(self.ace_step_transformer)
            self.text_encoder_model = torch.compile(self.text_encoder_model)

    def cleanup(self):
        import gc
        self.music_dcae = None
        self.ace_step_transformer = None
        self.lang_segment = None
        self.lyric_tokenizer = None
        self.text_encoder_model = None
        self.text_tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()

    def get_text_embeddings(self, texts, device, text_max_length=256):
        inputs = self.text_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=text_max_length)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        if self.text_encoder_model.device != device:
            self.text_encoder_model.to(device)
        with torch.no_grad():
            outputs = self.text_encoder_model(**inputs)
            last_hidden_states = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        return last_hidden_states, attention_mask
    
    def get_text_embeddings_null(self, texts, device, text_max_length=256, tau=0.01, l_min=8, l_max=10):
        inputs = self.text_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=text_max_length)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        if self.text_encoder_model.device != device:
            self.text_encoder_model.to(device)
        
        def forward_with_temperature(inputs, tau=0.01, l_min=8, l_max=10):
            handlers = []
            
            def hook(module, input, output):
                output[:] *= tau
                return output
        
            for i in range(l_min, l_max):
                handler = self.text_encoder_model.encoder.block[i].layer[0].SelfAttention.q.register_forward_hook(hook)
                handlers.append(handler)
        
            with torch.no_grad():
                outputs = self.text_encoder_model(**inputs)
                last_hidden_states = outputs.last_hidden_state
        
            for hook in handlers:
                hook.remove()
        
            return last_hidden_states
    
        last_hidden_states = forward_with_temperature(inputs, tau, l_min, l_max)
        return last_hidden_states

    def set_seeds(self, batch_size, manual_seeds=None):
        processed_input_seeds = None
        if manual_seeds is not None:
            if isinstance(manual_seeds, str):
                if "," in manual_seeds:
                    processed_input_seeds = list(map(int, manual_seeds.split(",")))
                elif manual_seeds.isdigit():
                    processed_input_seeds = int(manual_seeds)
            elif isinstance(manual_seeds, list) and all(isinstance(s, int) for s in manual_seeds):
                if len(manual_seeds) > 0:
                    processed_input_seeds = list(manual_seeds)
            elif isinstance(manual_seeds, int):
                processed_input_seeds = manual_seeds
        random_generators = [
            torch.Generator(device=self.device) for _ in range(batch_size)
        ]
        actual_seeds = []
        for i in range(batch_size):
            current_seed_for_generator = None
            if processed_input_seeds is None:
                current_seed_for_generator = torch.randint(0, 2**32, (1,)).item()
            elif isinstance(processed_input_seeds, int):
                current_seed_for_generator = processed_input_seeds
            elif isinstance(processed_input_seeds, list):
                if i < len(processed_input_seeds):
                    current_seed_for_generator = processed_input_seeds[i]
                else:
                    current_seed_for_generator = processed_input_seeds[-1]
            if current_seed_for_generator is None:
                 current_seed_for_generator = torch.randint(0, 2**32, (1,)).item()
            random_generators[i].manual_seed(current_seed_for_generator)
            actual_seeds.append(current_seed_for_generator)
        return random_generators, actual_seeds

    def get_lang(self, text):
        language = "en"
        try:    
            _ = self.lang_segment.getTexts(text)
            langCounts = self.lang_segment.getCounts()
            language = langCounts[0][0]
            if len(langCounts) > 1 and language == "en":
                language = langCounts[1][0]
        except Exception as err:
            language = "en"
        return language

    def tokenize_lyrics(self, lyrics, debug=False):
        lines = lyrics.split("\n")
        lyric_token_idx = [261]
        for line in lines:
            line = line.strip()
            if not line:
                lyric_token_idx += [2]
                continue

            lang = self.get_lang(line)

            if lang not in SUPPORT_LANGUAGES:
                lang = "en"
            if "zh" in lang:
                lang = "zh"
            if "spa" in lang:
                lang = "es"

            try:
                if structure_pattern.match(line):
                    token_idx = self.lyric_tokenizer.encode(line, "en")
                else:
                    token_idx = self.lyric_tokenizer.encode(line, lang)
                if debug:
                    toks = self.lyric_tokenizer.batch_decode([[tok_id] for tok_id in token_idx])
                    logger.info(f"debbug {line} --> {lang} --> {toks}")
                lyric_token_idx = lyric_token_idx + token_idx + [2]
            except Exception as e:
                print("tokenize error", e, "for line", line, "major_language", lang)
        return lyric_token_idx

    def calc_v(
        self,
        zt_src,
        zt_tar,
        t,
        encoder_text_hidden_states,
        text_attention_mask,
        target_encoder_text_hidden_states,
        target_text_attention_mask,
        speaker_embds,
        target_speaker_embeds,
        lyric_token_ids,
        lyric_mask,
        target_lyric_token_ids,
        target_lyric_mask,
        do_classifier_free_guidance=False,
        guidance_scale=1.0,
        target_guidance_scale=1.0,
        cfg_type="apg",
        attention_mask=None,
        momentum_buffer=None,
        momentum_buffer_tar=None,
        return_src_pred=True
    ):
        noise_pred_src = None
        if return_src_pred:
            src_latent_model_input = torch.cat([zt_src, zt_src]) if do_classifier_free_guidance else zt_src
            timestep = t.expand(src_latent_model_input.shape[0])
            # source
            noise_pred_src = self.ace_step_transformer(
                hidden_states=src_latent_model_input,
                attention_mask=attention_mask,
                encoder_text_hidden_states=encoder_text_hidden_states,
                text_attention_mask=text_attention_mask,
                speaker_embeds=speaker_embds,
                lyric_token_idx=lyric_token_ids,
                lyric_mask=lyric_mask,
                timestep=timestep,
            ).sample

            if do_classifier_free_guidance:
                noise_pred_with_cond_src, noise_pred_uncond_src = noise_pred_src.chunk(2)
                if cfg_type == "apg":
                    noise_pred_src = apg_forward(
                        pred_cond=noise_pred_with_cond_src,
                        pred_uncond=noise_pred_uncond_src,
                        guidance_scale=guidance_scale,
                        momentum_buffer=momentum_buffer,
                    )
                elif cfg_type == "cfg":
                    noise_pred_src = cfg_forward(
                        cond_output=noise_pred_with_cond_src,
                        uncond_output=noise_pred_uncond_src,
                        cfg_strength=guidance_scale,
                    )

        tar_latent_model_input = torch.cat([zt_tar, zt_tar]) if do_classifier_free_guidance else zt_tar
        timestep = t.expand(tar_latent_model_input.shape[0])
        # target
        noise_pred_tar = self.ace_step_transformer(
            hidden_states=tar_latent_model_input,
            attention_mask=attention_mask,
            encoder_text_hidden_states=target_encoder_text_hidden_states,
            text_attention_mask=target_text_attention_mask,
            speaker_embeds=target_speaker_embeds,
            lyric_token_idx=target_lyric_token_ids,
            lyric_mask=target_lyric_mask,
            timestep=timestep,
        ).sample

        if do_classifier_free_guidance:
            noise_pred_with_cond_tar, noise_pred_uncond_tar = noise_pred_tar.chunk(2)
            if cfg_type == "apg":
                noise_pred_tar = apg_forward(
                    pred_cond=noise_pred_with_cond_tar,
                    pred_uncond=noise_pred_uncond_tar,
                    guidance_scale=target_guidance_scale,
                    momentum_buffer=momentum_buffer_tar,
                )
            elif cfg_type == "cfg":
                noise_pred_tar = cfg_forward(
                    cond_output=noise_pred_with_cond_tar,
                    uncond_output=noise_pred_uncond_tar,
                    cfg_strength=target_guidance_scale,
                )
        return noise_pred_src, noise_pred_tar

    @torch.no_grad()
    def flowedit_diffusion_process(
        self,
        encoder_text_hidden_states,
        text_attention_mask,
        speaker_embds,
        lyric_token_ids,
        lyric_mask,
        target_encoder_text_hidden_states,
        target_text_attention_mask,
        target_speaker_embeds,
        target_lyric_token_ids,
        target_lyric_mask,
        src_latents,
        random_generators=None,
        infer_steps=60,
        guidance_scale=15.0,
        n_min=0,
        n_max=1.0,
        n_avg=1,
    ):

        do_classifier_free_guidance = True
        if guidance_scale == 0.0 or guidance_scale == 1.0:
            do_classifier_free_guidance = False

        target_guidance_scale = guidance_scale
        device = encoder_text_hidden_states.device
        dtype = encoder_text_hidden_states.dtype
        bsz = encoder_text_hidden_states.shape[0]

        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=3.0,
        )

        T_steps = infer_steps
        frame_length = src_latents.shape[-1]
        attention_mask = torch.ones(bsz, frame_length, device=device, dtype=dtype)
        
        timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None)

        if do_classifier_free_guidance:
            attention_mask = torch.cat([attention_mask] * 2, dim=0)
            
            encoder_text_hidden_states = torch.cat([encoder_text_hidden_states, torch.zeros_like(encoder_text_hidden_states)], 0)
            text_attention_mask = torch.cat([text_attention_mask] * 2, dim=0)

            target_encoder_text_hidden_states = torch.cat([target_encoder_text_hidden_states, torch.zeros_like(target_encoder_text_hidden_states)], 0)
            target_text_attention_mask = torch.cat([target_text_attention_mask] * 2, dim=0)

            speaker_embds = torch.cat([speaker_embds, torch.zeros_like(speaker_embds)], 0)
            target_speaker_embeds = torch.cat([target_speaker_embeds, torch.zeros_like(target_speaker_embeds)], 0)

            lyric_token_ids = torch.cat([lyric_token_ids, torch.zeros_like(lyric_token_ids)], 0)
            lyric_mask = torch.cat([lyric_mask, torch.zeros_like(lyric_mask)], 0)

            target_lyric_token_ids = torch.cat([target_lyric_token_ids, torch.zeros_like(target_lyric_token_ids)], 0)
            target_lyric_mask = torch.cat([target_lyric_mask, torch.zeros_like(target_lyric_mask)], 0)

        momentum_buffer = MomentumBuffer()
        momentum_buffer_tar = MomentumBuffer()
        x_src = src_latents
        zt_edit = x_src.clone()
        xt_tar = None
        n_min = int(infer_steps * n_min)
        n_max = int(infer_steps * n_max)

        logger.info("flowedit start from {} to {}".format(n_min, n_max))

        for i, t in tqdm(enumerate(timesteps), total=T_steps):

            if i < n_min:
                continue

            t_i = t/1000

            if i+1 < len(timesteps): 
                t_im1 = (timesteps[i+1])/1000
            else:
                t_im1 = torch.zeros_like(t_i).to(t_i.device)

            if i < n_max:
                # Calculate the average of the V predictions
                V_delta_avg = torch.zeros_like(x_src)
                for k in range(n_avg):
                    fwd_noise = randn_tensor(shape=x_src.shape, generator=random_generators, device=device, dtype=dtype)

                    zt_src = (1 - t_i) * x_src + (t_i) * fwd_noise

                    zt_tar = zt_edit + zt_src - x_src

                    Vt_src, Vt_tar = self.calc_v(
                        zt_src=zt_src,
                        zt_tar=zt_tar,
                        t=t,
                        encoder_text_hidden_states=encoder_text_hidden_states,
                        text_attention_mask=text_attention_mask,
                        target_encoder_text_hidden_states=target_encoder_text_hidden_states,
                        target_text_attention_mask=target_text_attention_mask,
                        speaker_embds=speaker_embds,
                        target_speaker_embeds=target_speaker_embeds,
                        lyric_token_ids=lyric_token_ids,
                        lyric_mask=lyric_mask,
                        target_lyric_token_ids=target_lyric_token_ids,
                        target_lyric_mask=target_lyric_mask,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        guidance_scale=guidance_scale,
                        target_guidance_scale=target_guidance_scale,
                        attention_mask=attention_mask,
                        momentum_buffer=momentum_buffer
                    )
                    V_delta_avg += (1 / n_avg) * (Vt_tar - Vt_src) # - (hfg-1)*( x_src))

                # propagate direct ODE
                zt_edit = zt_edit.to(torch.float32)
                zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg
                zt_edit = zt_edit.to(V_delta_avg.dtype)
            else: # i >= T_steps-n_min # regular sampling for last n_min steps
                if i == n_max:
                    fwd_noise = randn_tensor(shape=x_src.shape, generator=random_generators, device=device, dtype=dtype)
                    scheduler._init_step_index(t)
                    sigma = scheduler.sigmas[scheduler.step_index]
                    xt_src = sigma * fwd_noise + (1.0 - sigma) * x_src
                    xt_tar = zt_edit + xt_src - x_src
                
                _, Vt_tar = self.calc_v(
                    zt_src=None,
                    zt_tar=xt_tar,
                    t=t,
                    encoder_text_hidden_states=encoder_text_hidden_states,
                    text_attention_mask=text_attention_mask,
                    target_encoder_text_hidden_states=target_encoder_text_hidden_states,
                    target_text_attention_mask=target_text_attention_mask,
                    speaker_embds=speaker_embds,
                    target_speaker_embeds=target_speaker_embeds,
                    lyric_token_ids=lyric_token_ids,
                    lyric_mask=lyric_mask,
                    target_lyric_token_ids=target_lyric_token_ids,
                    target_lyric_mask=target_lyric_mask,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guidance_scale=guidance_scale,
                    target_guidance_scale=target_guidance_scale,
                    attention_mask=attention_mask,
                    momentum_buffer_tar=momentum_buffer_tar,
                    return_src_pred=False,
                )
                
                dtype = Vt_tar.dtype
                xt_tar = xt_tar.to(torch.float32)
                prev_sample = xt_tar + (t_im1 - t_i) * Vt_tar
                prev_sample = prev_sample.to(dtype) 
                xt_tar = prev_sample
        
        target_latents = zt_edit if xt_tar is None else xt_tar
        return target_latents

    @torch.no_grad()
    def text2music_diffusion_process(
        self,
        duration,
        encoder_text_hidden_states,
        text_attention_mask,
        speaker_embds,
        lyric_token_ids,
        lyric_mask,
        random_generators=None,
        infer_steps=60,
        guidance_scale=15.0,
        omega_scale=10.0,
        scheduler_type="euler",
        cfg_type="apg",
        zero_steps=1,
        use_zero_init=True,
        guidance_interval=0.5,
        guidance_interval_decay=1.0,
        min_guidance_scale=3.0,
        oss_steps=[],
        encoder_text_hidden_states_null=None,
        use_erg_lyric=False,
        use_erg_diffusion=False,
        retake_random_generators=None,
        retake_variance=0.5,
        add_retake_noise=False,
        guidance_scale_text=0.0,
        guidance_scale_lyric=0.0,
        repaint_start=0,
        repaint_end=0,
        src_latents=None,
    ):

        logger.info("cfg_type: {}, guidance_scale: {}, omega_scale: {}".format(cfg_type, guidance_scale, omega_scale))
        do_classifier_free_guidance = True
        if guidance_scale == 0.0 or guidance_scale == 1.0:
            do_classifier_free_guidance = False
        
        do_double_condition_guidance = False
        if guidance_scale_text is not None and guidance_scale_text > 1.0 and guidance_scale_lyric is not None and guidance_scale_lyric > 1.0:
            do_double_condition_guidance = True
            logger.info("do_double_condition_guidance: {}, guidance_scale_text: {}, guidance_scale_lyric: {}".format(do_double_condition_guidance, guidance_scale_text, guidance_scale_lyric))

        device = encoder_text_hidden_states.device
        dtype = encoder_text_hidden_states.dtype
        bsz = encoder_text_hidden_states.shape[0]

        if scheduler_type == "euler":
            scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=1000,
                shift=3.0,
            )
        elif scheduler_type == "heun":
            scheduler = FlowMatchHeunDiscreteScheduler(
                num_train_timesteps=1000,
                shift=3.0,
            )
        
        frame_length = int(duration * 44100 / 512 / 8)
        if src_latents is not None:
            frame_length = src_latents.shape[-1]

        if len(oss_steps) > 0:
            infer_steps = max(oss_steps)
            scheduler.set_timesteps
            timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps=infer_steps, device=device, timesteps=None)
            new_timesteps = torch.zeros(len(oss_steps), dtype=dtype, device=device)
            for idx in range(len(oss_steps)):
                new_timesteps[idx] = timesteps[oss_steps[idx]-1]
            num_inference_steps = len(oss_steps)
            sigmas = (new_timesteps / 1000).float().cpu().numpy()
            timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps=num_inference_steps, device=device, sigmas=sigmas)
            logger.info(f"oss_steps: {oss_steps}, num_inference_steps: {num_inference_steps} after remapping to timesteps {timesteps}")
        else:
            timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps=infer_steps, device=device, timesteps=None)
        
        target_latents = randn_tensor(shape=(bsz, 8, 16, frame_length), generator=random_generators, device=device, dtype=dtype)
        
        is_repaint = False
        is_extend  = False
        if add_retake_noise:
            n_min = int(infer_steps * (1 - retake_variance))
            retake_variance = torch.tensor(retake_variance * math.pi/2).to(device).to(dtype)
            retake_latents = randn_tensor(shape=(bsz, 8, 16, frame_length), generator=retake_random_generators, device=device, dtype=dtype)
            repaint_start_frame = int(repaint_start * 44100 / 512 / 8)
            repaint_end_frame = int(repaint_end * 44100 / 512 / 8)
            x0 = src_latents
            # retake
            is_repaint = (repaint_end_frame - repaint_start_frame != frame_length) 
            
            is_extend = (repaint_start_frame < 0) or (repaint_end_frame > frame_length)
            if is_extend:
                is_repaint = True

            # TODO: train a mask aware repainting controlnet
            # to make sure mean = 0, std = 1
            if not is_repaint:
                target_latents = torch.cos(retake_variance) * target_latents + torch.sin(retake_variance) * retake_latents
            elif not is_extend:
                # if repaint_end_frame 
                repaint_mask = torch.zeros((bsz, 8, 16, frame_length), device=device, dtype=dtype)
                repaint_mask[:, :, :, repaint_start_frame:repaint_end_frame] = 1.0
                repaint_noise = torch.cos(retake_variance) * target_latents + torch.sin(retake_variance) * retake_latents
                repaint_noise = torch.where(repaint_mask == 1.0, repaint_noise, target_latents)
                zt_edit = x0.clone()
                z0 = repaint_noise
            elif is_extend:
                to_right_pad_gt_latents = None
                to_left_pad_gt_latents = None
                gt_latents = src_latents
                src_latents_length = gt_latents.shape[-1]
                max_infer_fame_length = int(240 * 44100 / 512 / 8)
                left_pad_frame_length = 0
                right_pad_frame_length = 0
                right_trim_length = 0
                left_trim_length = 0
                if repaint_start_frame < 0:
                    left_pad_frame_length = abs(repaint_start_frame)
                    frame_length = left_pad_frame_length + gt_latents.shape[-1]
                    extend_gt_latents = torch.nn.functional.pad(gt_latents, (left_pad_frame_length, 0), "constant", 0)
                    if frame_length > max_infer_fame_length:
                        right_trim_length = frame_length - max_infer_fame_length
                        extend_gt_latents = extend_gt_latents[:,:,:,:max_infer_fame_length]
                        to_right_pad_gt_latents = extend_gt_latents[:,:,:,-right_trim_length:]
                        frame_length = max_infer_fame_length
                    repaint_start_frame = 0
                    gt_latents = extend_gt_latents
                
                if repaint_end_frame > src_latents_length:
                    right_pad_frame_length = repaint_end_frame - gt_latents.shape[-1]
                    frame_length = gt_latents.shape[-1] + right_pad_frame_length
                    extend_gt_latents = torch.nn.functional.pad(gt_latents, (0, right_pad_frame_length), "constant", 0)
                    if frame_length > max_infer_fame_length:
                        left_trim_length = frame_length - max_infer_fame_length
                        extend_gt_latents = extend_gt_latents[:,:,:,-max_infer_fame_length:]
                        to_left_pad_gt_latents = extend_gt_latents[:,:,:,:left_trim_length]
                        frame_length = max_infer_fame_length
                    repaint_end_frame = frame_length
                    gt_latents = extend_gt_latents

                repaint_mask = torch.zeros((bsz, 8, 16, frame_length), device=device, dtype=dtype)
                if left_pad_frame_length > 0:
                    repaint_mask[:,:,:,:left_pad_frame_length] = 1.0
                if right_pad_frame_length > 0:
                    repaint_mask[:,:,:,-right_pad_frame_length:] = 1.0
                x0 = gt_latents
                padd_list = []
                if left_pad_frame_length > 0:
                    padd_list.append(retake_latents[:, :, :, :left_pad_frame_length])
                padd_list.append(target_latents[:,:,:,left_trim_length:target_latents.shape[-1]-right_trim_length])
                if right_pad_frame_length > 0:
                    padd_list.append(retake_latents[:, :, :, -right_pad_frame_length:])
                target_latents = torch.cat(padd_list, dim=-1)
                assert target_latents.shape[-1] == x0.shape[-1], f"{target_latents.shape=} {x0.shape=}"
                zt_edit = x0.clone()
                z0 = target_latents

        attention_mask = torch.ones(bsz, frame_length, device=device, dtype=dtype)
        
        # guidance interval
        start_idx = int(num_inference_steps * ((1 - guidance_interval) / 2))
        end_idx = int(num_inference_steps * (guidance_interval / 2 + 0.5))
        logger.info(f"start_idx: {start_idx}, end_idx: {end_idx}, num_inference_steps: {num_inference_steps}")

        momentum_buffer = MomentumBuffer()

        def forward_encoder_with_temperature(self, inputs, tau=0.01, l_min=4, l_max=6):
            handlers = []
            
            def hook(module, input, output):
                output[:] *= tau
                return output
            
            for i in range(l_min, l_max):
                handler = self.ace_step_transformer.lyric_encoder.encoders[i].self_attn.linear_q.register_forward_hook(hook)
                handlers.append(handler)
        
            encoder_hidden_states, encoder_hidden_mask = self.ace_step_transformer.encode(**inputs)
            
            for hook in handlers:
                hook.remove()
            
            return encoder_hidden_states

        # P(speaker, text, lyric)
        encoder_hidden_states, encoder_hidden_mask = self.ace_step_transformer.encode(
            encoder_text_hidden_states,
            text_attention_mask,
            speaker_embds,
            lyric_token_ids,
            lyric_mask,
        )

        if use_erg_lyric:
            # P(null_speaker, text_weaker, lyric_weaker)
            encoder_hidden_states_null = forward_encoder_with_temperature(
                self,
                inputs={
                    "encoder_text_hidden_states": encoder_text_hidden_states_null if encoder_text_hidden_states_null is not None else torch.zeros_like(encoder_text_hidden_states),
                    "text_attention_mask": text_attention_mask,
                    "speaker_embeds": torch.zeros_like(speaker_embds),
                    "lyric_token_idx": lyric_token_ids,
                    "lyric_mask": lyric_mask,
                }
            )
        else:
            # P(null_speaker, null_text, null_lyric)
            encoder_hidden_states_null, _ = self.ace_step_transformer.encode(
                torch.zeros_like(encoder_text_hidden_states),
                text_attention_mask,
                torch.zeros_like(speaker_embds),
                torch.zeros_like(lyric_token_ids),
                lyric_mask,
            )
        
        encoder_hidden_states_no_lyric = None
        if do_double_condition_guidance:
            # P(null_speaker, text, lyric_weaker)
            if use_erg_lyric:
                encoder_hidden_states_no_lyric = forward_encoder_with_temperature(
                    self,
                    inputs={
                        "encoder_text_hidden_states": encoder_text_hidden_states,
                        "text_attention_mask": text_attention_mask,
                        "speaker_embeds": torch.zeros_like(speaker_embds),
                        "lyric_token_idx": lyric_token_ids,
                        "lyric_mask": lyric_mask,
                    }
                )
            # P(null_speaker, text, no_lyric)
            else:
                encoder_hidden_states_no_lyric, _ = self.ace_step_transformer.encode(
                    encoder_text_hidden_states,
                    text_attention_mask,
                    torch.zeros_like(speaker_embds),
                    torch.zeros_like(lyric_token_ids),
                    lyric_mask,
                )

        def forward_diffusion_with_temperature(self, hidden_states, timestep, inputs, tau=0.01, l_min=15, l_max=20):
            handlers = []
            
            def hook(module, input, output):
                output[:] *= tau
                return output
            
            for i in range(l_min, l_max):
                handler = self.ace_step_transformer.transformer_blocks[i].attn.to_q.register_forward_hook(hook)
                handlers.append(handler)
                handler = self.ace_step_transformer.transformer_blocks[i].cross_attn.to_q.register_forward_hook(hook)
                handlers.append(handler)

            sample = self.ace_step_transformer.decode(hidden_states=hidden_states, timestep=timestep, **inputs).sample
            
            for hook in handlers:
                hook.remove()
            
            return sample
    
        for i, t in tqdm(enumerate(timesteps), total=num_inference_steps):
            
            if is_repaint:
                if i < n_min:
                    continue
                elif i == n_min:
                    t_i = t / 1000
                    zt_src = (1 - t_i) * x0 + (t_i) * z0
                    target_latents = zt_edit + zt_src - x0
                    logger.info(f"repaint start from {n_min} add {t_i} level of noise")

            # expand the latents if we are doing classifier free guidance
            latents = target_latents

            is_in_guidance_interval = start_idx <= i < end_idx
            if is_in_guidance_interval and do_classifier_free_guidance:
                # compute current guidance scale
                if guidance_interval_decay > 0:
                    # Linearly interpolate to calculate the current guidance scale
                    progress = (i - start_idx) / (end_idx - start_idx - 1)  # 归一化到[0,1]
                    current_guidance_scale = guidance_scale - (guidance_scale - min_guidance_scale) * progress * guidance_interval_decay
                else:
                    current_guidance_scale = guidance_scale

                latent_model_input = latents
                timestep = t.expand(latent_model_input.shape[0])
                output_length = latent_model_input.shape[-1]
                # P(x|speaker, text, lyric)
                noise_pred_with_cond = self.ace_step_transformer.decode(
                    hidden_states=latent_model_input,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_mask=encoder_hidden_mask,
                    output_length=output_length,
                    timestep=timestep,
                ).sample

                noise_pred_with_only_text_cond = None
                if do_double_condition_guidance and encoder_hidden_states_no_lyric is not None:
                    noise_pred_with_only_text_cond = self.ace_step_transformer.decode(
                        hidden_states=latent_model_input,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states_no_lyric,
                        encoder_hidden_mask=encoder_hidden_mask,
                        output_length=output_length,
                        timestep=timestep,
                    ).sample

                if use_erg_diffusion:
                    noise_pred_uncond = forward_diffusion_with_temperature(
                        self,
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        inputs={
                            "encoder_hidden_states": encoder_hidden_states_null,
                            "encoder_hidden_mask": encoder_hidden_mask,
                            "output_length": output_length,
                            "attention_mask": attention_mask,
                        },
                    )
                else:
                    noise_pred_uncond = self.ace_step_transformer.decode(
                        hidden_states=latent_model_input,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states_null,
                        encoder_hidden_mask=encoder_hidden_mask,
                        output_length=output_length,
                        timestep=timestep,
                    ).sample

                if do_double_condition_guidance and noise_pred_with_only_text_cond is not None:
                    noise_pred = cfg_double_condition_forward(
                        cond_output=noise_pred_with_cond,
                        uncond_output=noise_pred_uncond,
                        only_text_cond_output=noise_pred_with_only_text_cond,
                        guidance_scale_text=guidance_scale_text,
                        guidance_scale_lyric=guidance_scale_lyric,
                    )

                elif cfg_type == "apg":
                    noise_pred = apg_forward(
                        pred_cond=noise_pred_with_cond,
                        pred_uncond=noise_pred_uncond,
                        guidance_scale=current_guidance_scale,
                        momentum_buffer=momentum_buffer,
                    )
                elif cfg_type == "cfg":
                    noise_pred = cfg_forward(
                        cond_output=noise_pred_with_cond,
                        uncond_output=noise_pred_uncond,
                        cfg_strength=current_guidance_scale,
                    )
                elif cfg_type == "cfg_star":
                    noise_pred = cfg_zero_star(
                        noise_pred_with_cond=noise_pred_with_cond,
                        noise_pred_uncond=noise_pred_uncond,
                        guidance_scale=current_guidance_scale,
                        i=i,
                        zero_steps=zero_steps,
                        use_zero_init=use_zero_init
                    )
            else:
                latent_model_input = latents
                timestep = t.expand(latent_model_input.shape[0])
                noise_pred = self.ace_step_transformer.decode(
                    hidden_states=latent_model_input,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_mask=encoder_hidden_mask,
                    output_length=latent_model_input.shape[-1],
                    timestep=timestep,
                ).sample

            if is_repaint and i >= n_min:
                t_i = t/1000
                if i+1 < len(timesteps): 
                    t_im1 = (timesteps[i+1])/1000
                else:
                    t_im1 = torch.zeros_like(t_i).to(t_i.device)
                dtype = noise_pred.dtype
                target_latents = target_latents.to(torch.float32)
                prev_sample = target_latents + (t_im1 - t_i) * noise_pred
                prev_sample = prev_sample.to(dtype)
                target_latents = prev_sample
                zt_src = (1 - t_im1) * x0 + (t_im1) * z0
                target_latents = torch.where(repaint_mask == 1.0, target_latents, zt_src)
            else:
                target_latents = scheduler.step(model_output=noise_pred, timestep=t, sample=target_latents, return_dict=False, omega=omega_scale)[0]

        if is_extend:
            if to_right_pad_gt_latents is not None:
                target_latents = torch.cat([target_latents, to_right_pad_gt_latents], dim=-1)
            if to_left_pad_gt_latents is not None:
                target_latents = torch.cat([to_right_pad_gt_latents, target_latents], dim=0)
        return target_latents

    def latents2audio(self, latents, target_wav_duration_second=30.0, sample_rate=48000):
        output_audios = []
        bs = latents.shape[0]
        audio_lengths = [target_wav_duration_second * sample_rate] * bs
        pred_latents = latents
        with torch.no_grad():
            _, pred_wavs = self.music_dcae.decode(pred_latents, sr=sample_rate)
        pred_wavs = [pred_wav.cpu().float() for pred_wav in pred_wavs]
        for i in tqdm(range(bs)):
            output_audio = (pred_wavs[i], sample_rate)
            output_audios.append(output_audio)
        return output_audios

    def infer_latents(self, input_audio_path):
        if input_audio_path is None:
            return None
        input_audio, sr = self.music_dcae.load_audio(input_audio_path)
        input_audio = input_audio.unsqueeze(0)
        device, dtype = self.device, self.dtype
        input_audio = input_audio.to(device=device, dtype=dtype)
        latents, _ = self.music_dcae.encode(input_audio, sr=sr)
        return latents

    def __call__(
        self,
        audio_duration: float = 60.0,
        prompt: str = None,
        lyrics: str = None,
        infer_step: int = 60,
        guidance_scale: float = 15.0,
        scheduler_type: str = "euler",
        cfg_type: str = "apg",
        omega_scale: int = 10.0,
        manual_seeds: list = None,
        guidance_interval: float = 0.5,
        guidance_interval_decay: float = 0.,
        min_guidance_scale: float = 3.0,
        use_erg_tag: bool = True,
        use_erg_lyric: bool = True,
        use_erg_diffusion: bool = True,
        oss_steps: str = None,
        guidance_scale_text: float = 0.0,
        guidance_scale_lyric: float = 0.0,
        retake_seeds: list = None,
        retake_variance: float = 0.5,
        task: str = "text2music",
        repaint_start: int = 0,
        repaint_end: int = 0,
        src_audio_path: str = None,
        edit_target_prompt: str = None,
        edit_target_lyrics: str = None,
        edit_n_min: float = 0.0,
        edit_n_max: float = 1.0,
        edit_n_avg: int = 1,
        # save_path: str = None,
        # format: str = "flac",
        batch_size: int = 1,
        debug: bool = False,
    ):

        start_time = time.time()

        random_generators, actual_seeds = self.set_seeds(batch_size, manual_seeds)
        retake_random_generators, actual_retake_seeds = self.set_seeds(batch_size, retake_seeds)

        if isinstance(oss_steps, str) and len(oss_steps) > 0:
            oss_steps = list(map(int, oss_steps.split(",")))
        else:
            oss_steps = []
        
        texts = [prompt]
        encoder_text_hidden_states, text_attention_mask = self.get_text_embeddings(texts, self.device)
        encoder_text_hidden_states = encoder_text_hidden_states.repeat(batch_size, 1, 1)
        text_attention_mask = text_attention_mask.repeat(batch_size, 1)

        encoder_text_hidden_states_null = None
        if use_erg_tag:
            encoder_text_hidden_states_null = self.get_text_embeddings_null(texts, self.device)
            encoder_text_hidden_states_null = encoder_text_hidden_states_null.repeat(batch_size, 1, 1)

        # not support for released checkpoint
        speaker_embeds = torch.zeros(batch_size, 512).to(self.device).to(self.dtype)

        # 6 lyric
        lyric_token_idx = torch.tensor([0]).repeat(batch_size, 1).to(self.device).long()
        lyric_mask = torch.tensor([0]).repeat(batch_size, 1).to(self.device).long()
        if len(lyrics) > 0:
            lyric_token_idx = self.tokenize_lyrics(lyrics, debug=debug)
            lyric_mask = [1] * len(lyric_token_idx)
            lyric_token_idx = torch.tensor(lyric_token_idx).unsqueeze(0).to(self.device).repeat(batch_size, 1)
            lyric_mask = torch.tensor(lyric_mask).unsqueeze(0).to(self.device).repeat(batch_size, 1)

        if audio_duration <= 0:
            audio_duration = random.uniform(30.0, 240.0)
            logger.info(f"random audio duration: {audio_duration}")

        end_time = time.time()
        preprocess_time_cost = end_time - start_time
        print(f"preprocess time cost: {preprocess_time_cost:.2f} seconds.")
        start_time = end_time

        add_retake_noise = task in ("retake", "repaint", "extend")
        # retake equal to repaint
        if task == "retake":
            repaint_start = 0
            repaint_end = audio_duration
        
        src_latents = None
        if src_audio_path is not None:
            assert src_audio_path is not None and task in ("repaint", "edit", "extend"), "src_audio_path is required for retake/repaint/extend task"
            assert os.path.exists(src_audio_path), f"src_audio_path {src_audio_path} does not exist"
            src_latents = self.infer_latents(src_audio_path)

        if task == "edit":
            texts = [edit_target_prompt]
            target_encoder_text_hidden_states, target_text_attention_mask = self.get_text_embeddings(texts, self.device)
            target_encoder_text_hidden_states = target_encoder_text_hidden_states.repeat(batch_size, 1, 1)
            target_text_attention_mask = target_text_attention_mask.repeat(batch_size, 1)

            target_lyric_token_idx = torch.tensor([0]).repeat(batch_size, 1).to(self.device).long()
            target_lyric_mask = torch.tensor([0]).repeat(batch_size, 1).to(self.device).long()
            if len(edit_target_lyrics) > 0:
                target_lyric_token_idx = self.tokenize_lyrics(edit_target_lyrics, debug=True)
                target_lyric_mask = [1] * len(target_lyric_token_idx)
                target_lyric_token_idx = torch.tensor(target_lyric_token_idx).unsqueeze(0).to(self.device).repeat(batch_size, 1)
                target_lyric_mask = torch.tensor(target_lyric_mask).unsqueeze(0).to(self.device).repeat(batch_size, 1)

            target_speaker_embeds = speaker_embeds.clone()

            target_latents = self.flowedit_diffusion_process(
                encoder_text_hidden_states=encoder_text_hidden_states,
                text_attention_mask=text_attention_mask,
                speaker_embds=speaker_embeds,
                lyric_token_ids=lyric_token_idx,
                lyric_mask=lyric_mask,
                target_encoder_text_hidden_states=target_encoder_text_hidden_states,
                target_text_attention_mask=target_text_attention_mask,
                target_speaker_embeds=target_speaker_embeds,
                target_lyric_token_ids=target_lyric_token_idx,
                target_lyric_mask=target_lyric_mask,
                src_latents=src_latents,
                random_generators=retake_random_generators, # more diversity
                infer_steps=infer_step,
                guidance_scale=guidance_scale,
                n_min=edit_n_min,
                n_max=edit_n_max,
                n_avg=edit_n_avg,
            )
        else:
            target_latents = self.text2music_diffusion_process(
                duration=audio_duration,
                encoder_text_hidden_states=encoder_text_hidden_states,
                text_attention_mask=text_attention_mask,
                speaker_embds=speaker_embeds,
                lyric_token_ids=lyric_token_idx,
                lyric_mask=lyric_mask,
                guidance_scale=guidance_scale,
                omega_scale=omega_scale,
                infer_steps=infer_step,
                random_generators=random_generators,
                scheduler_type=scheduler_type,
                cfg_type=cfg_type,
                guidance_interval=guidance_interval,
                guidance_interval_decay=guidance_interval_decay,
                min_guidance_scale=min_guidance_scale,
                oss_steps=oss_steps,
                encoder_text_hidden_states_null=encoder_text_hidden_states_null,
                use_erg_lyric=use_erg_lyric,
                use_erg_diffusion=use_erg_diffusion,
                retake_random_generators=retake_random_generators,
                retake_variance=retake_variance,
                add_retake_noise=add_retake_noise,
                guidance_scale_text=guidance_scale_text,
                guidance_scale_lyric=guidance_scale_lyric,
                repaint_start=repaint_start,
                repaint_end=repaint_end,
                src_latents=src_latents,
            )

        end_time = time.time()
        diffusion_time_cost = end_time - start_time
        print(f"diffusion time cost: {diffusion_time_cost}")
        start_time = end_time

        output_audios = self.latents2audio(
            latents=target_latents,
            target_wav_duration_second=audio_duration,
        )

        end_time = time.time()
        latent2audio_time_cost = end_time - start_time
        print(f"latent2audio time cost: {latent2audio_time_cost}")

        return output_audios
