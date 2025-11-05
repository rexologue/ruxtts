import logging
import random
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from core.xtts import load_audio
from core.xtts_config import XttsConfig
from core.tokenizer import VoiceBpeTokenizer

logger = logging.getLogger(__name__)


class AudioDataset(Dataset):
    """
    Датасет для GPT-части XTTS.

    Семантика как в оригинале:
    - Если gpt_use_masking_gt_prompt_approach=True:
        используем cond_idxs=[start, end] (mask GT prompt), а cond_lens -> NaN (сентинел).
    - Иначе:
        используем cond_len (длина conditioning), а cond_idxs -> NaN (сентинел).

    Возвращаем и "твои" ключи, и "оригинальные" (на этапе collate всё приведём к нужным формам).
    """

    def __init__(
        self,
        metadata_csv: Path,
        text_column: str,
        tokenizer: VoiceBpeTokenizer,
        config: XttsConfig,
        max_retries: int = 20,
    ) -> None:
        super().__init__()

        # Важно: low_memory=False, чтобы не словить DtypeWarning и не дробить типы столбцов
        self.df = pd.read_csv(metadata_csv, sep="|", low_memory=False)

        required = {"audio_path", "speaker_name", "language", text_column}
        if not required.issubset(self.df.columns):
            missing = required.difference(self.df.columns)
            raise ValueError(f"Missing required columns in {metadata_csv}: {sorted(missing)}")

        self.text_column = text_column
        self.tokenizer = tokenizer

        args = config.model_args
        self.sr: int = int(args.input_sample_rate)
        self.max_wav_len: int = int(args.max_wav_length)
        self.max_text_len: int = int(args.max_text_length)
        self.max_cond_len: int = int(args.max_conditioning_length)
        self.min_cond_len: int = int(args.min_conditioning_length)
        self.use_masking_gt_prompt_approach: bool = bool(args.gpt_use_masking_gt_prompt_approach)

        if not (0 < self.min_cond_len <= self.max_cond_len):
            raise ValueError("min_conditioning_length must be > 0 and <= max_conditioning_length")

        self.num_rows = int(self.df.shape[0])
        self.max_retries = int(max_retries)

    @staticmethod
    def _build_cond_slice(
        wav: torch.Tensor, max_len: int, min_len: int
    ) -> Tuple[torch.Tensor, int, list[int]]:
        """
        Возвращает:
          cond      : FloatTensor[1, max_len] (правый нулевой паддинг)
          cond_len  : int (реальная длина до паддинга)
          cond_idxs : [start, end] (индексы в исходном wav до паддинга)
        """
        T = int(wav.shape[-1])

        if T <= min_len:
            seg_len = T
            start = 0
        else:
            seg_len = min(random.randint(min_len, max_len), T)
            max_start = max(T - seg_len, 0)
            start = random.randint(0, max_start) if max_start > 0 else 0

        end = start + seg_len
        cond = wav[:, start:end]  # [1, seg_len]
        cond_len = int(cond.shape[-1])

        if cond_len < max_len:
            cond = F.pad(cond, (0, max_len - cond_len))  # -> [1, max_len]

        return cond, cond_len, [start, end]

    def _encode_text(self, text: str, lang: str) -> torch.Tensor:
        tokens = self.tokenizer.encode(text, lang)
        t = torch.as_tensor(tokens, dtype=torch.long)
        # Строгие проверки как в оригинале:
        assert not torch.any(t == 1), f"UNK token found in: {text} -> {self.tokenizer.decode(t.tolist())}"
        assert not torch.any(t == 0), "Stop token (id=0) found in text"
        return t

    def _load_item(self, row: pd.Series):
        # text/lang
        raw_text = str(row[self.text_column])
        lang = str(row["language"])
        if not raw_text or not raw_text.strip():
            raise ValueError("Empty text")

        tseq = self._encode_text(raw_text, lang)  # [L_text]

        # audio
        audiopath = str(row["audio_path"])
        wav = load_audio(audiopath, self.sr)  # FloatTensor [1, T]
        if wav is None or wav.ndim != 2 or wav.shape[0] != 1:
            raise ValueError(f"Bad audio tensor for: {audiopath}")
        if wav.shape[-1] < int(0.5 * self.sr):
            raise ValueError(f"Too short audio (<0.5s): {audiopath}")

        # conditioning согласно флагу
        if self.use_masking_gt_prompt_approach:
            # Маскирование GT prompt → используем cond_idxs, cond_len отключаем (NaN-сентинел)
            cond, _, cond_idxs = self._build_cond_slice(wav, self.max_cond_len, self.min_cond_len)
            cond_len = float("nan")
        else:
            # Не маскируем GT prompt → используем cond_len, cond_idxs отключаем
            cond, cond_len, _ = self._build_cond_slice(wav, self.max_cond_len, self.min_cond_len)
            cond_idxs = float("nan")

        return tseq, audiopath, wav, cond, cond_len, cond_idxs

    def __len__(self) -> int:
        return self.num_rows

    def __getitem__(self, index: int):
        """
        Безопасные ретраи: если пример невалиден — пробуем другой случайный.
        """
        for attempt in range(self.max_retries):
            j = index if attempt == 0 else random.randrange(self.num_rows)
            row = self.df.iloc[j]
            try:
                tseq, audiopath, wav, cond, cond_len, cond_idxs = self._load_item(row)
            except Exception as e:
                if attempt == 0:
                    logger.debug("Skip row %d due to error: %s", j, e)
                continue

            # Лимиты длин
            if (self.max_wav_len is not None and wav.shape[-1] > self.max_wav_len) or (
                self.max_text_len is not None and tseq.shape[0] > self.max_text_len
            ):
                continue

            # Готовим «твои» ключи — и дубликаты «оригинальных»
            item = {
                # твои ключи
                "text": tseq,                                   # [L_text]
                "text_len": torch.tensor(tseq.shape[0]).long(), # []
                "wav": wav,                                     # [1, T]
                "wav_len": torch.tensor(wav.shape[-1]).long(),  # []
                "cond": cond,                                   # [1, L_cond_max] (уже запэддено)
                # оригинальные ключи (совместимость с format_batch_on_device оригинала)
                "padded_text": tseq,                            # паддинг сделаем в collate
                "text_lengths": torch.tensor(tseq.shape[0]).long(),
                "wav_lengths": torch.tensor(wav.shape[-1]).long(),
                "filenames": audiopath,
                "conditioning": cond.unsqueeze(0),  # -> [num_cond=1, 1, L]
            }

            # Сентинелы как в оригинале: если «не используется» — кладём NaN-тензор совместимой формы
            if isinstance(cond_len, float) and (cond_len != cond_len):  # NaN
                item["cond_lens"] = torch.tensor([float("nan")])  # shape [1], float
            else:
                item["cond_lens"] = torch.tensor(cond_len, dtype=torch.long)  # shape [], long

            if isinstance(cond_idxs, float) and (cond_idxs != cond_idxs):  # NaN
                item["cond_idxs"] = torch.tensor([float("nan")])  # shape [1], float
            else:
                item["cond_idxs"] = torch.tensor(cond_idxs, dtype=torch.long)  # shape [2], long

            return item

        raise RuntimeError("Failed to fetch a valid sample after max_retries")


def collate_fn(batch: list[dict]) -> dict:
    """
    Совмещает твою и оригинальную схему:
    - Паддим текст/аудио.
    - Стеким 'conditioning' → [B, num_cond(=1), 1, T_cond_max].
    - Если в батче cond_idxs или cond_lens имеют NaN-сентинел → выставляем их в None.
    - Дублируем ключи: text/text_len и padded_text/text_lengths; wav_len/wav_lengths.
    """
    B = len(batch)

    # dict of lists
    bd = {k: [d[k] for d in batch] for k in batch[0].keys()}

    # длины
    text_lengths = torch.stack(bd["text_len"]).long()   # [B]
    wav_lengths  = torch.stack(bd["wav_len"]).long()    # [B]

    # макс. длины
    max_text = int(text_lengths.max().item())
    max_wav  = int(wav_lengths.max().item())

    # паддинг текста
    text_padded = torch.zeros((B, max_text), dtype=torch.long)
    for i, t in enumerate(bd["text"]):
        L = int(bd["text_len"][i])
        text_padded[i, :L] = t.to(dtype=torch.long)

    # паддинг аудио
    wav_padded = torch.zeros((B, 1, max_wav), dtype=torch.float32)
    for i, w in enumerate(bd["wav"]):
        L = int(bd["wav_len"][i])
        wav_padded[i, 0, :L] = w.to(dtype=torch.float32)

    # conditioning уже запэдден до max_conditioning_length в датасете.
    # Каждая запись: [num_cond=1, 1, L_cond_max] → stack даст [B, 1, 1, L_cond_max]
    conditioning = torch.stack(bd["conditioning"])  # [B, 1, 1, Lc]

    # cond_lens / cond_idxs
    cond_lens = torch.stack(bd["cond_lens"])
    cond_idxs = torch.stack(bd["cond_idxs"])

    # Если в любом элементе есть NaN → это «выключенный» канал (как в оригинале)
    if torch.isnan(cond_lens).any():
        cond_lens_out = None
    else:
        # привели к shape [B] long
        cond_lens_out = cond_lens.view(B).to(dtype=torch.long)

    if torch.isnan(cond_idxs).any():
        cond_idxs_out = None
    else:
        # shape [B, 2] long
        cond_idxs_out = cond_idxs.to(dtype=torch.long)

    # Собираем выход:
    out = {
        # массивы
        "wav": wav_padded,                   # [B, 1, Tw]
        "conditioning": conditioning,        # [B, 1, 1, Tc]
        "padded_text": text_padded,          # [B, Ttext]
        # длины (двойной набор ключей для совместимости)
        "text_lengths": text_lengths,        # [B]
        "wav_lengths": wav_lengths,          # [B]
        "text_len": text_lengths,            # [B] (твои)
        "wav_len": wav_lengths,              # [B] (твои)
        # cond-метаданные
        "cond_lens": cond_lens_out,          # либо Tensor[B] long, либо None
        "cond_idxs": cond_idxs_out,          # либо Tensor[B,2] long, либо None
        # продублируем «твои» ключи (если кто-то снаружи ждёт их)
        "text": text_padded,                 # [B, Ttext]
    }
    return out
