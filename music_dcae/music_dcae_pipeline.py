import os
import torch
from diffusers import AutoencoderDC
import torchaudio
import torchvision.transforms as transforms
from diffusers.models.modeling_utils import ModelMixin
from diffusers.loaders import FromOriginalModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config


try:
    from .music_vocoder import ADaMoSHiFiGANV1
except ImportError:
    from music_vocoder import ADaMoSHiFiGANV1


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PRETRAINED_PATH = os.path.join(root_dir, "checkpoints", "music_dcae_f8c8")
VOCODER_PRETRAINED_PATH = os.path.join(root_dir, "checkpoints", "music_vocoder")


class MusicDCAE(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    @register_to_config
    def __init__(self, source_sample_rate=None, dcae_checkpoint_path=DEFAULT_PRETRAINED_PATH, vocoder_checkpoint_path=VOCODER_PRETRAINED_PATH):
        super(MusicDCAE, self).__init__()

        self.dcae = AutoencoderDC.from_pretrained(dcae_checkpoint_path)
        self.vocoder = ADaMoSHiFiGANV1.from_pretrained(vocoder_checkpoint_path)

        if source_sample_rate is None:
            source_sample_rate = 48000

        self.resampler = torchaudio.transforms.Resample(source_sample_rate, 44100)

        self.transform = transforms.Compose([
            transforms.Normalize(0.5, 0.5),
        ])
        self.min_mel_value = -11.0
        self.max_mel_value = 3.0
        self.audio_chunk_size = int(round((1024 * 512 / 44100 * 48000)))
        self.mel_chunk_size = 1024
        self.time_dimention_multiple = 8
        self.latent_chunk_size = self.mel_chunk_size // self.time_dimention_multiple
        self.scale_factor = 0.1786
        self.shift_factor = -1.9091

    def load_audio(self, audio_path):
        audio, sr = torchaudio.load(audio_path)
        return audio, sr

    def forward_mel(self, audios):
        mels = []
        for i in range(len(audios)):
            image = self.vocoder.mel_transform(audios[i])
            mels.append(image)
        mels = torch.stack(mels)
        return mels

    @torch.no_grad()
    def encode(self, audios, audio_lengths=None, sr=None):
        if audio_lengths is None:
            audio_lengths = torch.tensor([audios.shape[2]] * audios.shape[0])
            audio_lengths = audio_lengths.to(audios.device)

        # audios: N x 2 x T, 48kHz
        device = audios.device
        dtype = audios.dtype

        if sr is None:
            sr = 48000
            resampler = self.resampler
        else:
            resampler = torchaudio.transforms.Resample(sr, 44100).to(device).to(dtype)

        audio = resampler(audios)

        max_audio_len = audio.shape[-1]
        if max_audio_len % (8 * 512) != 0:
            audio = torch.nn.functional.pad(audio, (0, 8 * 512 - max_audio_len % (8 * 512)))

        mels = self.forward_mel(audio)
        mels = (mels - self.min_mel_value) / (self.max_mel_value - self.min_mel_value)
        mels = self.transform(mels)
        latents = []
        for mel in mels:
            latent = self.dcae.encoder(mel.unsqueeze(0))
            latents.append(latent)
        latents = torch.cat(latents, dim=0)
        latent_lengths = (audio_lengths / sr * 44100 / 512 / self.time_dimention_multiple).long()
        latents = (latents - self.shift_factor) * self.scale_factor
        return latents, latent_lengths

    @torch.no_grad()
    def decode(self, latents, audio_lengths=None, sr=None):
        latents = latents / self.scale_factor + self.shift_factor

        pred_wavs = []

        for latent in latents:
            mels = self.dcae.decoder(latent.unsqueeze(0))
            mels = mels * 0.5 + 0.5
            mels = mels * (self.max_mel_value - self.min_mel_value) + self.min_mel_value
            wav = self.vocoder.decode(mels[0]).squeeze(1)

            if sr is not None:
                resampler = torchaudio.transforms.Resample(44100, sr).to(latents.device).to(latents.dtype)
                wav = resampler(wav)
            else:
                sr = 44100
            pred_wavs.append(wav)

        if audio_lengths is not None:
            pred_wavs = [wav[:, :length].cpu() for wav, length in zip(pred_wavs, audio_lengths)]
        return sr, pred_wavs

    def forward(self, audios, audio_lengths=None, sr=None):
        latents, latent_lengths = self.encode(audios=audios, audio_lengths=audio_lengths, sr=sr)
        sr, pred_wavs = self.decode(latents=latents, audio_lengths=audio_lengths, sr=sr)
        return sr, pred_wavs, latents, latent_lengths


if __name__ == "__main__":

    audio, sr = torchaudio.load("test.wav")
    audio_lengths = torch.tensor([audio.shape[1]])
    audios = audio.unsqueeze(0)
    
    # test encode only
    model = MusicDCAE()
    # latents, latent_lengths = model.encode(audios, audio_lengths)
    # print("latents shape: ", latents.shape)
    # print("latent_lengths: ", latent_lengths)

    # test encode and decode
    sr, pred_wavs, latents, latent_lengths = model(audios, audio_lengths, sr)
    print("reconstructed wavs: ", pred_wavs[0].shape)
    print("latents shape: ", latents.shape)
    print("latent_lengths: ", latent_lengths)
    print("sr: ", sr)
    torchaudio.save("test_reconstructed.flac", pred_wavs[0], sr)
    print("test_reconstructed.flac")
