import re
import unicodedata

import torch
import numpy as np

TONE_SR = 8000  # StreamingCTCModel.SAMPLE_RATE
PCM16_MAX = 32767
PCM16_MIN = -32768

def _to_mono_np(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w)
    if w.ndim == 1:
        return w
    # [C, T] или [T, C]
    if w.ndim == 2:
        if w.shape[0] in (1, 2):
            return w[0] if w.shape[0] == 1 else w.mean(axis=0)
        if w.shape[1] in (1, 2):
            return w[:, 0] if w.shape[1] == 1 else w.mean(axis=1)
    # на всякий — сплющим
    return w.reshape(-1)

def _resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return x
    n_out = int(round(len(x) * dst_sr / src_sr))
    if n_out <= 0 or len(x) == 0:
        return np.zeros((0,), dtype=x.dtype)
    # линейная интерполяция без внешних зависимостей
    src_idx = np.arange(len(x), dtype=np.float64)
    dst_idx = np.linspace(0.0, len(x) - 1, num=n_out, endpoint=True, dtype=np.float64)
    return np.interp(dst_idx, src_idx, x).astype(x.dtype, copy=False)

def _resample(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    # попробуем torchaudio/scipy, иначе — наш линейный
    try:
        import torch
        import torchaudio.functional as AF
        xt = torch.from_numpy(x).unsqueeze(0)  # [1, T]
        yt = AF.resample(xt, src_sr, dst_sr)
        return yt.squeeze(0).cpu().numpy()
    except Exception:
        try:
            from scipy.signal import resample_poly
            # resample_poly качественнее для нецелых отношений
            g = np.gcd(src_sr, dst_sr)
            up, down = dst_sr // g, src_sr // g
            return resample_poly(x, up, down).astype(x.dtype, copy=False)
        except Exception:
            return _resample_linear(x, src_sr, dst_sr)

def _float_to_pcm16_as_int32(x: np.ndarray) -> np.ndarray:
    # ожидаем float в [-1, 1]; если не так — зажмём
    y = np.clip(x, -1.0, 1.0)
    y = np.rint(y * PCM16_MAX).astype(np.int32)
    # safety
    np.clip(y, PCM16_MIN, PCM16_MAX, out=y)
    return y

def prepare_for_tone(audio, sr: int) -> np.ndarray:
    """
    -> np.ndarray[int32] формы (L,), 8 kHz, моно, PCM16 в контейнере int32.
    Подходит для StreamingCTCPipeline.forward_offline.
    """
    # torch.Tensor -> numpy
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    x = np.asarray(audio)

    # к float32
    if np.issubdtype(x.dtype, np.integer):
        # приведём к float в [-1,1] если это уже int16/int32 PCM
        # эвристика: нормализуем по максимальному |value|
        denom = float(np.max(np.abs(x))) or 1.0
        x = (x.astype(np.float32) / denom) if denom > 1.0 else x.astype(np.float32)
    else:
        x = x.astype(np.float32, copy=False)

    x = _to_mono_np(x)
    x = _resample(x, sr, TONE_SR)
    pcm_int32 = _float_to_pcm16_as_int32(x)
    return pcm_int32

def trim_by_seconds(wav: np.ndarray, sr: int, t_end: float) -> np.ndarray:
    n = max(0, min(len(wav), int(np.floor(t_end * sr))))  # floor — чтобы не «залезть» дальше
    return wav[:n]

def trim_by_seconds_torch(wav: torch.Tensor, sr: int, t_end: float) -> torch.Tensor:
    n = int(torch.floor(torch.tensor(t_end * sr)).item())
    n = max(0, min(wav.shape[-1], n))
    return wav[..., :n]

def normalize_text(text: str) -> str:
    # Приводим к нижнему регистру
    lower_text = text.lower()
    
    # Убираем все символы, кроме кириллических букв (включая ё)
    cyrillic_only = re.sub(r'[^а-яё]', '', lower_text)
    
    return cyrillic_only