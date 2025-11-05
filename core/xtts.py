import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import librosa
import torch
import torch.nn.functional as F
import torchaudio
from coqpit import Coqpit
from .generic_utils import load_fsspec
from scipy.signal import butter, lfilter
import noisereduce as nr
import numpy as np

from .shared_configs import BaseTTSConfig
from .gpt import GPT
from .hifigan_decoder import HifiDecoder
from .stream_generator import init_stream_support

@dataclass
class StreamingMetrics:
    time_to_first_token: float | None
    time_to_first_audio: float | None
    real_time_factor: float | None
    latency: float  # average chunk generation time

from .tokenizer import VoiceBpeTokenizer, split_sentence
from .xtts_manager import LanguageManager, SpeakerManager
from .base_tts import BaseTTS
from .generic_utils import (
    is_pytorch_at_least_2_4,
    warn_synthesize_config_deprecated,
    warn_synthesize_speaker_id_deprecated,
)

logger = logging.getLogger(__name__)

init_stream_support()

def wav_to_mel_cloning(
    wav,
    mel_norms_file: str,
    mel_norms=None,
    device=torch.device("cpu"),
    n_fft=4096,
    hop_length=1024,
    win_length=4096,
    power=2,
    normalized=False,
    sample_rate=22050,
    f_min=0,
    f_max=8000,
    n_mels=80,
):
    """
    Convert waveform to mel-spectrogram with hard-coded parameters for cloning.

    Args:
        wav (torch.Tensor): Input waveform tensor.
        mel_norms_file (str): Path to mel-spectrogram normalization file.
        mel_norms (torch.Tensor): Mel-spectrogram normalization tensor.
        device (torch.device): Device to use for computation.

    Returns:
        torch.Tensor: Mel-spectrogram tensor.
    """
    mel_stft = torchaudio.transforms.MelSpectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=power,
        normalized=normalized,
        sample_rate=sample_rate,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        norm="slaney",
    ).to(device)
    wav = wav.to(device)
    mel = mel_stft(wav)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    if mel_norms is None:
        mel_norms = torch.load(mel_norms_file, map_location=device, weights_only=is_pytorch_at_least_2_4())
    mel = mel / mel_norms.unsqueeze(0).unsqueeze(-1)
    return mel


def load_audio(audiopath, sampling_rate):
    # better load setting following: https://github.com/faroit/python_audio_loading_benchmark

    # torchaudio should chose proper backend to load audio depending on platform
    audio, lsr = torchaudio.load(audiopath)

    # stereo to mono if needed
    if audio.size(0) != 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    if lsr != sampling_rate:
        audio = torchaudio.functional.resample(audio, lsr, sampling_rate)

    # Check some assumptions about audio range. This should be automatically fixed in load_wav_to_torch, but might not be in some edge cases, where we should squawk.
    # '10' is arbitrarily chosen since it seems like audio will often "overdrive" the [-1,1] bounds.
    if torch.any(audio > 10) or not torch.any(audio < 0):
        logger.error("Error with %s. Max=%.2f min=%.2f", audiopath, audio.max(), audio.min())
    # clip audio invalid values
    audio.clip_(-1, 1)
    return audio


@dataclass
class XttsAudioConfig(Coqpit):
    """
    Configuration class for audio-related parameters in the XTTS model.

    Args:
        sample_rate (int): The sample rate in which the GPT operates.
        output_sample_rate (int): The sample rate of the output audio waveform.
        dvae_sample_rate (int): The sample rate of the DVAE
    """

    sample_rate: int = 22050
    output_sample_rate: int = 24000
    dvae_sample_rate: int = 22050


@dataclass
class XttsArgs(Coqpit):
    """A dataclass to represent XTTS model arguments that define the model structure.

    Args:
        gpt_batch_size (int): The size of the auto-regressive batch.
        enable_redaction (bool, optional): Whether to enable redaction. Defaults to True.
        kv_cache (bool, optional): Whether to use the kv_cache. Defaults to True.
        num_chars (int, optional): The maximum number of characters to generate. Defaults to 255.

        For GPT model:
        gpt_max_audio_tokens (int, optional): The maximum mel tokens for the autoregressive model. Defaults to 604.
        gpt_max_text_tokens (int, optional): The maximum text tokens for the autoregressive model. Defaults to 402.
        gpt_max_prompt_tokens (int, optional): The maximum prompt tokens or the autoregressive model. Defaults to 70.
        gpt_layers (int, optional): The number of layers for the autoregressive model. Defaults to 30.
        gpt_n_model_channels (int, optional): The model dimension for the autoregressive model. Defaults to 1024.
        gpt_n_heads (int, optional): The number of heads for the autoregressive model. Defaults to 16.
        gpt_number_text_tokens (int, optional): The number of text tokens for the autoregressive model. Defaults to 255.
        gpt_start_text_token (int, optional): The start text token for the autoregressive model. Defaults to 255.
        gpt_checkpointing (bool, optional): Whether to use checkpointing for the autoregressive model. Defaults to False.
        gpt_train_solo_embeddings (bool, optional): Whether to train embeddings for the autoregressive model. Defaults to False.
        gpt_code_stride_len (int, optional): The hop_size of dvae and consequently of the gpt output. Defaults to 1024.
        gpt_use_masking_gt_prompt_approach (bool, optional):  If True, it will use ground truth as prompt and it will mask the loss to avoid repetition. Defaults to True.
        gpt_use_perceiver_resampler (bool, optional):  If True, it will use perceiver resampler from flamingo paper - https://arxiv.org/abs/2204.14198. Defaults to False.
    """

    # General
    gpt_batch_size: int = 1
    enable_redaction: bool = False
    kv_cache: bool = True
    num_chars: int = 255

    # XTTS GPT Encoder params
    tokenizer_file: str = ""
    gpt_max_audio_tokens: int = 605
    gpt_max_text_tokens: int = 402
    gpt_max_prompt_tokens: int = 70
    gpt_layers: int = 30
    gpt_n_model_channels: int = 1024
    gpt_n_heads: int = 16
    gpt_number_text_tokens: int = None  # type: ignore
    gpt_start_text_token: int = None    # type: ignore
    gpt_stop_text_token: int = None     # type: ignore
    gpt_num_audio_tokens: int = 8194
    gpt_start_audio_token: int = 8192
    gpt_stop_audio_token: int = 8193
    gpt_code_stride_len: int = 1024
    gpt_use_masking_gt_prompt_approach: bool = True
    gpt_use_perceiver_resampler: bool = False

    # HifiGAN Decoder params
    input_sample_rate: int = 22050
    output_sample_rate: int = 24000
    output_hop_length: int = 256
    decoder_input_dim: int = 1024
    d_vector_dim: int = 512
    cond_d_vector_in_each_upsampling_layer: bool = True

    # Constants / limits
    duration_const: int = 102400

    # === Added from JSON ===
    # Conditioning window limits
    min_conditioning_length: int = 66150
    max_conditioning_length: int = 132300

    # Loss weights (present in JSON under model_args)
    gpt_loss_text_ce_weight: float = 0.01
    gpt_loss_mel_ce_weight: float = 1.0

    # Debug / misc
    debug_loading_failures: bool = False

    # Sequence limits (dataset-related)
    max_wav_length: int = 255995
    max_text_length: int = 250

    # Checkpoints and normalization assets
    mel_norm_file: str = "mel_stats.pth"
    dvae_checkpoint: str = "dvae.pth"
    xtts_checkpoint: str = "model.pth"
    vocoder: str = ""

    # Additional checkpoints from JSON
    gpt_checkpoint: str = ""
    clvp_checkpoint: Optional[str] = None
    decoder_checkpoint: Optional[str] = None


class Xtts(BaseTTS):
    """XTTS model implementation.

    ❗ Currently it only supports inference.

    Examples:
        >>> from TTS.tts.configs.xtts_config import XttsConfig
        >>> from TTS.tts.models.xtts import Xtts
        >>> config = XttsConfig()
        >>> model = Xtts.init_from_config(config)
        >>> model.load_checkpoint(config, checkpoint_dir="paths/to/models_dir/", eval=True)
    """

    def __init__(self, config: Coqpit):
        super().__init__(config)
        self.mel_stats_path = None
        self.config = config
        self.models_dir = config.model_dir
        self.gpt_batch_size = self.args.gpt_batch_size

        self.tokenizer = VoiceBpeTokenizer()
        self.gpt = None
        self.init_models()
        self.register_buffer("mel_stats", torch.ones(80))

    def init_models(self):
        """Initialize the models. We do it here since we need to load the tokenizer first."""
        if self.tokenizer.tokenizer is not None:
            self.args.gpt_number_text_tokens = self.tokenizer.get_number_tokens() # type: ignore
            self.args.gpt_start_text_token = self.tokenizer.tokenizer.token_to_id("[START]") # type: ignore
            self.args.gpt_stop_text_token = self.tokenizer.tokenizer.token_to_id("[STOP]") # type: ignore

        if self.args.gpt_number_text_tokens:
            self.gpt = GPT(
                layers=self.args.gpt_layers,
                model_dim=self.args.gpt_n_model_channels,
                start_text_token=self.args.gpt_start_text_token,
                stop_text_token=self.args.gpt_stop_text_token,
                heads=self.args.gpt_n_heads,
                max_text_tokens=self.args.gpt_max_text_tokens,
                max_mel_tokens=self.args.gpt_max_audio_tokens,
                max_prompt_tokens=self.args.gpt_max_prompt_tokens,
                number_text_tokens=self.args.gpt_number_text_tokens,
                num_audio_tokens=self.args.gpt_num_audio_tokens,
                start_audio_token=self.args.gpt_start_audio_token,
                stop_audio_token=self.args.gpt_stop_audio_token,
                use_perceiver_resampler=self.args.gpt_use_perceiver_resampler,
                code_stride_len=self.args.gpt_code_stride_len,
            )

        self.hifigan_decoder = HifiDecoder(
            input_sample_rate=self.args.input_sample_rate,
            output_sample_rate=self.args.output_sample_rate,
            output_hop_length=self.args.output_hop_length,
            ar_mel_length_compression=self.args.gpt_code_stride_len,
            decoder_input_dim=self.args.decoder_input_dim,
            d_vector_dim=self.args.d_vector_dim,
            cond_d_vector_in_each_upsampling_layer=self.args.cond_d_vector_in_each_upsampling_layer,
        )

    def _apply_high_pass_filter(self, wav_chunk_numpy, cutoff=75, order=5):
        sample_rate = 24_000
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False) # type: ignore
        
        # lfilter применяет фильтр. Возвращает отфильтрованный массив.
        filtered_chunk = lfilter(b, a, wav_chunk_numpy)
        return filtered_chunk

    def _apply_noise_reduction(self, wav_chunk_numpy):
        """Применяет простое шумоподавление."""
        # reduce_noise уменьшает шум в аудио
        try:
            reduced_noise_chunk = nr.reduce_noise(
                y=wav_chunk_numpy, 
                sr=self.args.output_sample_rate, 
                stationary=True, 
                prop_decrease=0.75
            )
        except:
            reduced_noise_chunk = wav_chunk_numpy
        
        return reduced_noise_chunk
    
    @torch.inference_mode()
    def get_gpt_cond_latents(self, audio, sr, length: int = 30, chunk_length: int = 6):
        """Compute the conditioning latents for the GPT model from the given audio.

        Args:
            audio (tensor): audio tensor.
            sr (int): Sample rate of the audio.
            length (int): Length of the audio in seconds. If < 0, use the whole audio. Defaults to 30.
            chunk_length (int): Length of the audio chunks in seconds. When `length == chunk_length`, the whole audio
                is being used without chunking. It must be < `length`. Defaults to 6.
        """
        MIN_AUDIO_SECONDS = 0.33
        if sr != 22050:
            audio = torchaudio.functional.resample(audio, sr, 22050)
        if length > 0:
            audio = audio[:, : 22050 * length]
        if self.args.gpt_use_perceiver_resampler:
            style_embs = []
            for i in range(0, audio.shape[1], 22050 * chunk_length):
                audio_chunk = audio[:, i : i + 22050 * chunk_length]

                # if the chunk is too short ignore it
                if audio_chunk.size(-1) < 22050 * MIN_AUDIO_SECONDS:
                    continue

                mel_chunk = wav_to_mel_cloning(
                    audio_chunk,
                    self._mel_norms_file(),
                    mel_norms=self.mel_stats.cpu(),
                    n_fft=2048,
                    hop_length=256,
                    win_length=1024,
                    power=2,
                    normalized=False,
                    sample_rate=22050,
                    f_min=0,
                    f_max=8000,
                    n_mels=80,
                ) # type: ignore

                style_emb = self.gpt.get_style_emb(mel_chunk.to(self.device), None) # type: ignore
                style_embs.append(style_emb)

            # mean style embedding
            if len(style_embs) == 0:
                msg = f"Provided reference audio too short (minimum length: {MIN_AUDIO_SECONDS:.2f} seconds)."
                raise RuntimeError(msg)
            
            cond_latent = torch.stack(style_embs).mean(dim=0)

        else:
            mel = wav_to_mel_cloning(
                audio,
                self._mel_norms_file(),
                mel_norms=self.mel_stats.cpu(),
                n_fft=4096,
                hop_length=1024,
                win_length=4096,
                power=2,
                normalized=False,
                sample_rate=22050,
                f_min=0,
                f_max=8000,
                n_mels=80,
            ) # type: ignore

            cond_latent = self.gpt.get_style_emb(mel.to(self.device)) # type: ignore

        return cond_latent.transpose(1, 2)

    @torch.inference_mode()
    def get_speaker_embedding(self, audio, sr):
        audio_16k = torchaudio.functional.resample(audio, sr, 16000)
        return (
            self.hifigan_decoder.speaker_encoder.forward(audio_16k.to(self.device), l2_norm=True)
            .unsqueeze(-1)
            .to(self.device)
        )

    def _clone_voice(
        self,
        speaker_wav: str | os.PathLike[Any] | list[str | os.PathLike[Any]],
        speaker: str | None = None,
        voice_dir: str | os.PathLike[Any] | None = None,
        **generate_kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        cache_path: Path | None = None

        if speaker and voice_dir:
            cache_path = Path(voice_dir) / f"{speaker}.pt"

            if cache_path.is_file():
                payload = torch.load(cache_path, map_location=self.device)
                voice = payload.get("voice", payload)
                voice = {
                    key: value.to(self.device) if torch.is_tensor(value) else value
                    for key, value in voice.items()
                }

                metadata = payload.get("metadata", {"name": self.config["model"]})

                if self.speaker_manager is not None:
                    self.speaker_manager.speakers[speaker] = voice

                return voice, metadata

        gpt_conditioning_latents, speaker_embedding = self.get_conditioning_latents(
            audio_path=speaker_wav,
            **generate_kwargs,
        )

        voice = {"gpt_conditioning_latents": gpt_conditioning_latents, "speaker_embedding": speaker_embedding}
        metadata = {"name": self.config["model"]}

        if speaker and self.speaker_manager is not None:
            self.speaker_manager.speakers[speaker] = voice

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            serializable_voice = {
                key: value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in voice.items()
            }

            torch.save({"voice": serializable_voice, "metadata": metadata}, cache_path)

        return voice, metadata

    @torch.inference_mode()
    def get_conditioning_latents(
        self,
        audio_path: str | os.PathLike[Any] | list[str | os.PathLike[Any]],
        max_ref_length: int = 30,
        gpt_cond_len: int = 6,
        gpt_cond_chunk_len: int = 6,
        librosa_trim_db: int | None = None,
        sound_norm_refs: bool = False,
        load_sr: int = 22050,
    ):
        """Get the conditioning latents for the GPT model from the given audio.

        Args:
            audio_path (str or List[str]): Path to reference audio file(s).
            max_ref_length (int): Maximum length of each reference audio in seconds. Defaults to 30.
            gpt_cond_len (int): Length of the audio used for gpt latents. Defaults to 6.
            gpt_cond_chunk_len (int): Chunk length used for gpt latents. It must be <= gpt_conf_len. Defaults to 6.
            librosa_trim_db (int, optional): Trim the audio using this value. If None, not trimming. Defaults to None.
            sound_norm_refs (bool, optional): Whether to normalize the audio. Defaults to False.
            load_sr (int, optional): Sample rate to load the audio. Defaults to 22050.
        """
        # deal with multiples references
        if not isinstance(audio_path, list):
            audio_paths = [audio_path]
        else:
            audio_paths = audio_path

        speaker_embeddings = []
        audios = []
        speaker_embedding = None

        for file_path in audio_paths:
            audio = load_audio(file_path, load_sr)
            audio = audio[:, : load_sr * max_ref_length].to(self.device)

            if sound_norm_refs:
                audio = (audio / torch.abs(audio).max()) * 0.75
            if librosa_trim_db is not None:
                audio = librosa.effects.trim(audio, top_db=librosa_trim_db)[0] # type: ignore

            # compute latents for the decoder
            speaker_embedding = self.get_speaker_embedding(audio, load_sr)
            speaker_embeddings.append(speaker_embedding)

            audios.append(audio)

        # merge all the audios and compute the latents for the gpt
        full_audio = torch.cat(audios, dim=-1)
        gpt_cond_latents = self.get_gpt_cond_latents(
            full_audio, load_sr, length=gpt_cond_len, chunk_length=gpt_cond_chunk_len
        )  # [1, 1024, T]

        if speaker_embeddings:
            speaker_embedding = torch.stack(speaker_embeddings)
            speaker_embedding = speaker_embedding.mean(dim=0)

        return gpt_cond_latents, speaker_embedding

    def synthesize(
        self,
        text: str,
        config: BaseTTSConfig | None = None,
        *,
        speaker: str | None = None,
        speaker_wav: str | os.PathLike[Any] | list[str | os.PathLike[Any]] | None = None,
        voice_dir: str | os.PathLike[Any] | None = None,
        language: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Synthesize speech with the given input text.

        Args:
            text (str): Input text.
            config: DEPRECATED. Not used.
            speaker: Custom speaker ID to cache or retrieve a voice.
            speaker_wav: Path(s) to reference audio, should be >3 seconds long.
            voice_dir: Folder for cached voices.
            language (str): Language of the input text.
            **kwargs: Inference settings. See `inference()`.

        Returns:
            A dictionary of the output values with `wav` as output waveform, `deterministic_seed` as seed used at inference,
            `text_input` as text token IDs after tokenizer, `voice_samples` as samples used for cloning, `conditioning_latents`
            as latents used at inference.

        """
        if config is not None:
            warn_synthesize_config_deprecated()
        if (speaker_id := kwargs.pop("speaker_id", None)) is not None:
            speaker = speaker_id
            warn_synthesize_speaker_id_deprecated()
        for key in ("use_griffin_lim", "do_trim_silence", "extra_aux_input"):
            kwargs.pop(key, None)
        assert "zh-cn" if language == "zh" else language in self.config.languages, (
            f" ❗ Language {language} is not supported. Supported languages are {self.config.languages}"
        )

        # Use generally found best tuning knobs for generation.
        voice_settings = {
            key: kwargs.pop(key, self.config[key])
            for key in ["gpt_cond_len", "gpt_cond_chunk_len", "max_ref_len", "sound_norm_refs"]
        }
        voice_settings["max_ref_length"] = voice_settings.pop("max_ref_len")

        inference_settings = {
            "temperature": self.config.temperature,
            "length_penalty": self.config.length_penalty,
            "repetition_penalty": self.config.repetition_penalty,
            "top_k": self.config.top_k,
            "top_p": self.config.top_p,
        }
        inference_settings.update(kwargs)  # allow overriding of preset settings with kwargs

        if speaker is not None and speaker in self.speaker_manager.speakers: # type: ignore
            gpt_cond_latent, speaker_embedding = self.speaker_manager.speakers[speaker].values() # type: ignore

        else:
            voice = self.clone_voice(speaker_wav, speaker, voice_dir, **voice_settings)
            gpt_cond_latent = voice["gpt_conditioning_latents"]
            speaker_embedding = voice["speaker_embedding"]

        return self.inference(text, language, gpt_cond_latent, speaker_embedding, **inference_settings)

    @torch.inference_mode()
    def inference(
        self,
        text,
        language,
        gpt_cond_latent,
        speaker_embedding,
        # GPT inference
        temperature: float = 0.75,
        length_penalty: float = 1.0,
        repetition_penalty: float = 10.0,
        top_k: int = 50,
        top_p: float = 0.85,
        do_sample: bool = True,
        num_beams: int = 1,
        speed: float = 1.0,
        enable_text_splitting: bool = False,
        **hf_generate_kwargs: Any,
    ):
        """
        This function produces an audio clip of the given text being spoken with the given reference voice.

        Args:
            text: (str) Text to be spoken.

            gpt_cond_latent: GPT conditioning latents.

            speaker_embedding: Target speaker embedding.

            language: (str) Language of the voice to be generated.

            temperature: (float) The softmax temperature of the autoregressive model. Defaults to 0.65.

            length_penalty: (float) A length penalty applied to the autoregressive decoder. Higher settings causes the
                model to produce more terse outputs. Defaults to 1.0.

            repetition_penalty: (float) A penalty that prevents the autoregressive decoder from repeating itself during
                decoding. Can be used to reduce the incidence of long silences or "uhhhhhhs", etc. Defaults to 2.0.

            top_k: (int) K value used in top-k sampling. [0,inf]. Lower values mean the decoder produces more "likely"
                (aka boring) outputs. Defaults to 50.

            top_p: (float) P value used in nucleus sampling. (0,1]. Lower values mean the decoder produces more "likely"
                (aka boring) outputs. Defaults to 0.8.

            hf_generate_kwargs: (`**kwargs`) The huggingface Transformers generate API is used for the autoregressive
                transformer. Extra keyword args fed to this function get forwarded directly to that API. Documentation
                here: https://huggingface.co/docs/transformers/internal/generation_utils

        Returns:
            Generated audio clip(s) as a torch tensor. Shape 1,S if k=1 else, (k,1,S) where S is the sample length.
            Sample rate is 24kHz.
        """
        language = language.split("-")[0]  # remove the country code
        length_scale = 1.0 / max(speed, 0.05)
        gpt_cond_latent = gpt_cond_latent.to(self.device)
        speaker_embedding = speaker_embedding.to(self.device)

        if enable_text_splitting:
            text_list = split_sentence(text, language, self.tokenizer.char_limits[language])
        else:
            text_list = [text]

        wavs = []
        gpt_latents_list = []

        for sent in text_list:
            sent = sent.strip().lower()
            text_tokens = torch.IntTensor(self.tokenizer.encode(sent, lang=language)).unsqueeze(0).to(self.device)

            assert text_tokens.shape[-1] < self.args.gpt_max_text_tokens, (
                " ❗ XTTS can only generate text with a maximum of 400 tokens."
            )

            with torch.no_grad():
                gpt_codes = self.gpt.generate( # type: ignore
                    cond_latents=gpt_cond_latent,
                    text_inputs=text_tokens,
                    do_sample=do_sample,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    num_return_sequences=self.gpt_batch_size,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    repetition_penalty=repetition_penalty,
                    output_attentions=False,
                    **hf_generate_kwargs,
                ) 

                expected_output_len = torch.tensor(
                    [gpt_codes.shape[-1] * self.gpt.code_stride_len], device=text_tokens.device # type: ignore
                ) 

                text_len = torch.tensor([text_tokens.shape[-1]], device=self.device)
                gpt_latents = self.gpt(
                    text_tokens,
                    text_len,
                    gpt_codes,
                    expected_output_len,
                    cond_latents=gpt_cond_latent,
                    return_attentions=False,
                    return_latent=True,
                ) # type: ignore

                if length_scale != 1.0:
                    gpt_latents = F.interpolate(
                        gpt_latents.transpose(1, 2), scale_factor=length_scale, mode="linear"
                    ).transpose(1, 2)

                gpt_latents_list.append(gpt_latents.cpu())
                wavs.append(self.hifigan_decoder(gpt_latents, g=speaker_embedding).cpu().squeeze())

        final_wav = torch.cat(wavs, dim=0).cpu().detach().numpy()

        return {
            "wav": final_wav,
            "gpt_latents": torch.cat(gpt_latents_list, dim=1).numpy(),
            "speaker_embedding": speaker_embedding,
        }
    
    def handle_chunks(self, wav_gen, wav_gen_prev, wav_overlap, fade_in, fade_out):
        """Handle chunk formatting in streaming mode (cumulative mode)"""
        overlap_len = fade_in.numel()
        wav_chunk = wav_gen[:-overlap_len]
        if wav_gen_prev is not None:
            wav_chunk = wav_gen[(wav_gen_prev.shape[0] - overlap_len) : -overlap_len]

        if wav_overlap is not None:
            if overlap_len > wav_chunk.numel():
                if wav_gen_prev is not None:
                    wav_chunk = wav_gen[(wav_gen_prev.shape[0] - overlap_len):]
                else:
                    wav_chunk = wav_gen[-overlap_len:]
                return wav_chunk, wav_gen, None

            # было: wav_chunk[:overlap_len].mul_(fade_out).add_(crossfade_wav)  ← это неправильно
            head = wav_chunk[:overlap_len]
            head.mul_(fade_in).add_(wav_overlap * fade_out)   # старый*fade_out + новый*fade_in

        wav_overlap = wav_gen[-overlap_len:]
        return wav_chunk, wav_gen, wav_overlap


    @torch.inference_mode()
    def inference_stream(
        self,
        text,
        language,
        gpt_cond_latent,
        speaker_embedding,
        apply_denoise: bool = True,
        # Streaming
        stream_chunk_size=20,
        overlap_wav_len=1024,
        # Новый опциональный режим усечения истории:
        left_context_tokens: int | None = None,
        left_context_seconds: float | None = None,
        # GPT inference
        temperature=0.75,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=50,
        top_p=0.85,
        do_sample=True,
        speed=1.0,
        enable_text_splitting=False,
        **hf_generate_kwargs,
    ):
        language = language.split("-")[0]
        length_scale = 1.0 / max(speed, 0.05)
        gpt_cond_latent = gpt_cond_latent.to(self.device)
        speaker_embedding = speaker_embedding.to(self.device)

        start_time = time.perf_counter()
        last_chunk_time = start_time
        chunk_generation_time_total = 0.0
        chunk_count = 0
        time_to_first_token: float | None = None
        time_to_first_audio: float | None = None
        generated_audio_samples = 0

        # частоты и перевод «токены -> сэмплы»
        sr_in = 22050
        sr_out = getattr(self.args, "output_sample_rate", 24000)
        tokens_per_second = sr_in / float(self.args.gpt_code_stride_len)
        samples_per_token_out = sr_out * (self.args.gpt_code_stride_len / sr_in) * length_scale

        # если секунды заданы, переводим в токены
        if left_context_seconds is not None and (left_context_tokens is None or left_context_tokens <= 0):
            left_context_tokens = max(0, int(round(left_context_seconds * tokens_per_second)))
        trim_to_context = bool(left_context_tokens and left_context_tokens > 0)

        if enable_text_splitting:
            text_list = split_sentence(text, language, self.tokenizer.char_limits[language])
        else:
            text_list = [text]

        for sent in text_list:
            sent = sent.strip().lower()
            text_tokens = torch.IntTensor(self.tokenizer.encode(sent, lang=language)).unsqueeze(0).to(self.device)
            assert text_tokens.shape[-1] < self.args.gpt_max_text_tokens, " ❗ XTTS can only generate text with a maximum of 400 tokens."

            fake_inputs = self.gpt.compute_embeddings(  # type: ignore
                gpt_cond_latent.to(self.device),
                text_tokens,
            )
            gpt_generator = self.gpt.get_generator(  # type: ignore
                fake_inputs=fake_inputs,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                do_sample=do_sample,
                num_beams=1,
                num_return_sequences=1,
                length_penalty=float(length_penalty),
                repetition_penalty=float(repetition_penalty),
                output_attentions=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
                **hf_generate_kwargs,
            )

            last_tokens: list = []
            all_latents: list[torch.Tensor] = []
            wav_gen_prev = None
            wav_overlap = None
            is_end = False

            # Фейдеры
            if not trim_to_context:
                win = torch.hann_window(2 * overlap_wav_len, periodic=False, device=self.device, dtype=torch.float32)
                fade_in, fade_out = win[:overlap_wav_len], win[overlap_wav_len:]
                overlap_len = overlap_wav_len
            else:
                hop = getattr(self.args, "output_hop_length", 256)
                overlap_len = max(hop, (overlap_wav_len // hop) * hop)
                win = torch.hann_window(2 * overlap_len, periodic=False, device=self.device, dtype=torch.float32)
                fade_in, fade_out = win[:overlap_len], win[overlap_len:]

            while not is_end:
                try:
                    x, latent = next(gpt_generator)
                    if time_to_first_token is None:
                        time_to_first_token = time.perf_counter() - start_time
                    last_tokens.append(x)
                    all_latents.append(latent)
                except StopIteration:
                    is_end = True

                if is_end or (stream_chunk_size > 0 and len(last_tokens) >= stream_chunk_size):
                    if not trim_to_context:
                        # ---- Кумулятивный путь ----
                        gpt_latents = torch.cat(all_latents, dim=0)[None, :]
                        if length_scale != 1.0:
                            gpt_latents = F.interpolate(gpt_latents.transpose(1, 2), scale_factor=length_scale, mode="linear").transpose(1, 2)

                        wav_gen = self.hifigan_decoder(gpt_latents, g=speaker_embedding)  # [1, S]
                        wav_chunk, wav_gen_prev, wav_overlap = self.handle_chunks(
                            wav_gen.squeeze(), wav_gen_prev, wav_overlap, fade_in, fade_out
                        )

                        # шумодав только на отдаваемом куске
                        if wav_chunk is not None and wav_chunk.numel() > 0 and apply_denoise:
                            wav_np = wav_chunk.detach().cpu().numpy()
                            wav_np = self._apply_noise_reduction(wav_np)
                            processed_wav_chunk = torch.from_numpy(wav_np.copy()).to(self.device).float()
                        else:
                            processed_wav_chunk = wav_chunk

                        # метрика TTFA
                        if processed_wav_chunk is not None and processed_wav_chunk.numel() > 0 and time_to_first_audio is None:
                            time_to_first_audio = time.perf_counter() - start_time

                        # Учёт метрик
                        if processed_wav_chunk is not None and processed_wav_chunk.numel() > 0:
                            now = time.perf_counter()
                            chunk_generation_time_total += now - last_chunk_time
                            last_chunk_time = now
                            chunk_count += 1
                            generated_audio_samples += processed_wav_chunk.numel()

                        # Отдаём наружу
                        if processed_wav_chunk is not None and processed_wav_chunk.numel() > 0:
                            yield processed_wav_chunk

                        last_tokens = []
                        continue

                    # ---- Путь усечения истории до небольшого контекста ----
                    L = len(all_latents)
                    new_cnt = len(last_tokens)
                    ctx_tok = min(int(left_context_tokens), max(0, L - new_cnt))  # type: ignore

                    decode_start = L - (ctx_tok + new_cnt)
                    if decode_start < 0:
                        decode_start = 0

                    window = all_latents[decode_start:]
                    if len(window) == 0:
                        # Нечего декодировать.
                        if is_end:
                            # Финальный хвост: отдай overlap и завершай.
                            if wav_overlap is not None and wav_overlap.numel() > 0:
                                out_np = wav_overlap.detach().cpu().numpy()
                                out_np = self._apply_noise_reduction(out_np)
                                yield torch.from_numpy(out_np.copy()).to(self.device).float()
                            break
                        else:
                            last_tokens = []
                            continue

                    gpt_latents = torch.cat(window, dim=0)[None, :]
                    if length_scale != 1.0:
                        gpt_latents = F.interpolate(gpt_latents.transpose(1, 2), scale_factor=length_scale, mode="linear").transpose(1, 2)

                    wav_partial = self.hifigan_decoder(gpt_latents, g=speaker_embedding).squeeze().contiguous()  # [S_part]

                    # отбрасываем аудио-часть, соответствующую контекстным токенам (выровнено по hop)
                    hop = getattr(self.args, "output_hop_length", 256)
                    ctx_samples = int(round(ctx_tok * samples_per_token_out))
                    ctx_samples = (ctx_samples // hop) * hop

                    if ctx_samples >= wav_partial.shape[-1]:
                        new_wav = wav_partial
                    else:
                        new_wav = wav_partial[ctx_samples:]

                    # единообразный денойз до разбиения
                    if new_wav.numel() > 0 and apply_denoise:
                        _cpu = new_wav.detach().cpu().numpy()
                        _cpu = self._apply_noise_reduction(_cpu)
                        new_wav = torch.from_numpy(_cpu.copy()).to(self.device).float()

                    # кроссфейд
                    if wav_overlap is not None and new_wav.numel() >= overlap_len:
                        head = new_wav[:overlap_len]
                        head.mul_(fade_in).add_(wav_overlap * fade_out)
                        new_wav[:overlap_len] = head

                    if new_wav.numel() > overlap_len:
                        out_chunk = new_wav[:-overlap_len]
                        wav_overlap = new_wav[-overlap_len:]
                    else:
                        out_chunk = torch.empty(0, device=new_wav.device, dtype=new_wav.dtype)
                        wav_overlap = new_wav.clone()

                    processed_wav_chunk = out_chunk

                    if processed_wav_chunk.numel() > 0 and time_to_first_audio is None:
                        time_to_first_audio = time.perf_counter() - start_time
                    if processed_wav_chunk.numel() > 0:
                        generated_audio_samples += processed_wav_chunk.numel()
                        now = time.perf_counter()
                        chunk_generation_time_total += now - last_chunk_time
                        last_chunk_time = now
                        chunk_count += 1

                    # чистим историю, оставляя левый контекст
                    if ctx_tok > 0:
                        all_latents = all_latents[-ctx_tok:]
                    else:
                        all_latents = []
                    last_tokens = []

                    if processed_wav_chunk.numel() > 0:
                        yield processed_wav_chunk

        total_time = time.perf_counter() - start_time
        generated_audio_seconds = generated_audio_samples / sr_out if sr_out else 0.0
        real_time_factor = (total_time / generated_audio_seconds) if generated_audio_seconds > 0 else None
        average_chunk_generation_time = (chunk_generation_time_total / chunk_count) if chunk_count > 0 else 0.0

        metrics = StreamingMetrics(
            time_to_first_token=time_to_first_token,
            time_to_first_audio=time_to_first_audio,
            real_time_factor=real_time_factor,
            latency=average_chunk_generation_time,
        )
        return metrics


    def forward(self):
        raise NotImplementedError(
            "XTTS has a dedicated trainer, please check the XTTS docs: https://coqui-tts.readthedocs.io/en/latest/models/xtts.html#training"
        )

    def eval_step(self):
        raise NotImplementedError(
            "XTTS has a dedicated trainer, please check the XTTS docs: https://coqui-tts.readthedocs.io/en/latest/models/xtts.html#training"
        )

    @staticmethod
    def init_from_config(config: "XttsConfig", **kwargs):  # pylint: disable=unused-argument # type: ignore
        return Xtts(config)

    def eval(self):  # pylint: disable=redefined-builtin
        """Sets the model to evaluation mode. Overrides the default eval() method to also set the GPT model to eval mode."""
        self.gpt.init_gpt_for_inference() # type: ignore
        super().eval()

    def get_compatible_checkpoint_state_dict(self, model_path):
        _checkpoint = load_fsspec(model_path, map_location=torch.device("cpu"))

        if _checkpoint.get("model"):
            checkpoint = _checkpoint["model"]
        else:
            checkpoint = _checkpoint
        
        # remove xtts gpt trainer extra keys
        ignore_keys = ["torch_mel_spectrogram_style_encoder", "torch_mel_spectrogram_dvae", "dvae"]
        for key in list(checkpoint.keys()):
            # check if it is from the coqui Trainer if so convert it
            if key.startswith("xtts."):
                new_key = key.replace("xtts.", "")
                checkpoint[new_key] = checkpoint[key]
                del checkpoint[key]
                key = new_key

            # remove unused keys
            if key.split(".")[0] in ignore_keys:
                del checkpoint[key]

        # Rename legacy speaker encoder torch spectrogram buffers
        legacy_torch_spec_keys = {
            "hifigan_decoder.speaker_encoder.torch_spec.0.filter": "hifigan_decoder.speaker_encoder.torch_spec.kernel",
            "hifigan_decoder.speaker_encoder.torch_spec.1.spectrogram.window": "hifigan_decoder.speaker_encoder.torch_spec.base.spectrogram.window",
            "hifigan_decoder.speaker_encoder.torch_spec.1.mel_scale.fb": "hifigan_decoder.speaker_encoder.torch_spec.base.mel_scale.fb",
        }
        for old_key, new_key in legacy_torch_spec_keys.items():
            if old_key in checkpoint and new_key not in checkpoint:
                checkpoint[new_key] = checkpoint.pop(old_key)

        # Legacy checkpoints may miss the nested `gpt` prefix that mirrors the module hierarchy
        if any(key.startswith("gpt.transformer.") for key in checkpoint):
            for key in list(checkpoint.keys()):
                if key.startswith("gpt.transformer."):
                    new_key = key.replace("gpt.transformer.", "gpt.gpt.transformer.", 1)
                    checkpoint[new_key] = checkpoint.pop(key)
                elif key.startswith("gpt.ln_f."):
                    new_key = key.replace("gpt.ln_f.", "gpt.gpt.ln_f.", 1)
                    checkpoint[new_key] = checkpoint.pop(key)
                elif key.startswith("gpt.wte."):
                    new_key = key.replace("gpt.wte.", "gpt.gpt.wte.", 1)
                    checkpoint[new_key] = checkpoint.pop(key)

        # Ensure inference module aliases exist so `strict=True` loads succeed
        def ensure_alias(src_prefix: str, dst_prefix: str):
            for key in list(checkpoint.keys()):
                if key.startswith(src_prefix):
                    alias_key = key.replace(src_prefix, dst_prefix, 1)
                    if alias_key not in checkpoint:
                        checkpoint[alias_key] = checkpoint[key]

        ensure_alias("gpt.gpt.", "gpt.gpt_inference.transformer.")

        alias_pairs = [
            ("gpt.mel_embedding.weight", [
                "gpt.gpt.wte.weight",
                "gpt.gpt_inference.embeddings.weight",
                "gpt.gpt_inference.transformer.wte.weight",
            ]),
            ("gpt.mel_pos_embedding.emb.weight", ["gpt.gpt_inference.pos_embedding.emb.weight"]),
            ("gpt.final_norm.weight", [
                "gpt.gpt_inference.final_norm.weight",
                "gpt.gpt_inference.lm_head.0.weight",
            ]),
            ("gpt.final_norm.bias", [
                "gpt.gpt_inference.final_norm.bias",
                "gpt.gpt_inference.lm_head.0.bias",
            ]),
            ("gpt.mel_head.weight", ["gpt.gpt_inference.lm_head.1.weight"]),
            ("gpt.mel_head.bias", ["gpt.gpt_inference.lm_head.1.bias"]),
        ]

        for source_key, targets in alias_pairs:
            if source_key in checkpoint:
                for target_key in targets:
                    if target_key not in checkpoint:
                        checkpoint[target_key] = checkpoint[source_key]

        return checkpoint

    def load_checkpoint(
        self,
        config: "XttsConfig", # type: ignore
        checkpoint_dir: str | None = None,
        checkpoint_path: str | None = None,
        vocab_path: str | None = None,
        eval: bool = True,
        strict: bool = True,
        use_deepspeed: bool = False,
        speaker_file_path: str | None = None,
    ):
        """
        Loads a checkpoint from disk and initializes the model's state and tokenizer.

        Args:
            config (dict): The configuration dictionary for the model.
            checkpoint_dir (str, optional): The directory where the checkpoint is stored. Defaults to None.
            checkpoint_path (str, optional): The path to the checkpoint file. Defaults to None.
            vocab_path (str, optional): The path to the vocabulary file. Defaults to None.
            eval (bool, optional): Whether to set the model to evaluation mode. Defaults to True.
            strict (bool, optional): Whether to strictly enforce that the keys in the checkpoint match the keys in the model. Defaults to True.

        Returns:
            None
        """
        if checkpoint_dir is not None and Path(checkpoint_dir).is_file():
            msg = f"You passed a file to `checkpoint_dir=`. Use `checkpoint_path={checkpoint_dir}` instead."
            raise ValueError(msg)
        
        model_path = checkpoint_path or os.path.join(checkpoint_dir, "model.pth") # type: ignore

        if vocab_path is None:
            if checkpoint_dir is not None and (Path(checkpoint_dir) / "vocab.json").is_file():
                vocab_path = str(Path(checkpoint_dir) / "vocab.json")
            else:
                vocab_path = config.model_args.tokenizer_file

        if speaker_file_path is None and checkpoint_dir is not None:
            speaker_file_path = os.path.join(checkpoint_dir, "speakers_xtts.pth")

        self.language_manager = LanguageManager(config)
        self.speaker_manager = None
        if speaker_file_path is not None and os.path.exists(speaker_file_path):
            self.speaker_manager = SpeakerManager(speaker_file_path)

        if os.path.exists(vocab_path): # type: ignore
            self.tokenizer = VoiceBpeTokenizer(vocab_file=vocab_path)

        else:
            msg = (
                f"`vocab.json` file not found in `{checkpoint_dir}`. Move the file there or "
                "specify alternative path in `model_args.tokenizer_file` in `config.json`"
            )
            raise FileNotFoundError(msg)

        self.init_models()

        checkpoint = self.get_compatible_checkpoint_state_dict(model_path)

        # deal with v1 and v1.1. V1 has the init_gpt_for_inference keys, v1.1 do not
        try:
            self.load_state_dict(checkpoint, strict=strict)

        except:
            if eval:
                self.gpt.init_gpt_for_inference(kv_cache=self.args.kv_cache) # type: ignore

            self.load_state_dict(checkpoint, strict=strict)

        if eval:
            self.hifigan_decoder.eval()
            self.gpt.init_gpt_for_inference(kv_cache=self.args.kv_cache, use_deepspeed=use_deepspeed) # type: ignore
            self.gpt.eval() # type: ignore

        self._load_mel_stats(checkpoint_dir=checkpoint_dir)

    def _mel_norms_file(self) -> str:
        if not self.mel_stats_path:
            msg = "Mel-spectrogram normalization file not loaded. Ensure `mel_stats.pth` is present in the model directory."
            raise RuntimeError(msg)
        return self.mel_stats_path

    def _load_mel_stats(self, *, checkpoint_dir: str | None) -> None:
        mel_norm_file = getattr(self.args, "mel_norm_file", None)
        if not mel_norm_file:
            logger.warning("`mel_norm_file` was not provided in the model configuration.")
            return

        candidate_paths: list[str] = []
        if os.path.isabs(mel_norm_file):
            candidate_paths.append(mel_norm_file)
        else:
            if checkpoint_dir:
                candidate_paths.append(os.path.join(os.fspath(checkpoint_dir), mel_norm_file))
            if self.config.model_dir:
                candidate_paths.append(os.path.join(os.fspath(self.config.model_dir), mel_norm_file))
            candidate_paths.append(mel_norm_file)

        for candidate in candidate_paths:
            if os.path.exists(candidate):
                self.mel_stats_path = os.fspath(candidate)
                break

        if not self.mel_stats_path:
            logger.warning(
                "Mel-spectrogram normalization file `%s` could not be found. Checked: %s",
                mel_norm_file,
                ", ".join(candidate_paths) if candidate_paths else "<none>",
            )
            return

        stats = torch.load(
            self.mel_stats_path,
            map_location=torch.device("cpu"),
            weights_only=is_pytorch_at_least_2_4(),
        )
        if isinstance(stats, dict):
            for key in ("mel", "mel_stats", "stats", "mean", "std"):
                if key in stats:
                    stats = stats[key]
                    break
            else:
                stats = next(iter(stats.values()))

        if not isinstance(stats, torch.Tensor):
            stats = torch.as_tensor(stats, dtype=self.mel_stats.dtype)
        else:
            stats = stats.to(dtype=self.mel_stats.dtype)
        self.mel_stats.copy_(stats.to(self.mel_stats.device))

    def train_step(self):
        raise NotImplementedError(
            "XTTS has a dedicated trainer, please check the XTTS docs: https://coqui-tts.readthedocs.io/en/latest/models/xtts.html#training"
        )
