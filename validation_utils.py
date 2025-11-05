from pathlib import Path
from typing import Any

import gigaam
from nisqa.NISQA_model import nisqaModel
from wespeaker.cli.speaker import Speaker

from jiwer import cer

class GigaCER:
    """
    Обёртка над GigaAM ASR.
    model_type: "ctc", "v2_ctc", "rnnt", "v2_rnnt" — нам нужен CTC.
    """
    def __init__(self, model_type: str = "v2_ctc") -> None:
        self.model = gigaam.load_model(model_type)

    def transcribe(self, wav_path: Path) -> str:
        text = self.model.transcribe(str(wav_path)) # type: ignore
        return (text or "").strip()
    
    def cer(self, base_text: str, gen_wav: Path) -> float:
        return cer(base_text, self.transcribe(gen_wav))
    
class Cossim:
    def __init__(self, model_dir: Path, ref_wav: Path, device):
        self.model = self.load_model_from_dir(model_dir, device)
        self.ref_embed = self.model.extract_embedding(str(ref_wav))

    def load_model_from_dir(self, model_dir: Path, device) -> Speaker:
        avg = model_dir / "avg_model.pt"
        cfg = model_dir / "config.yaml"
        if not avg.exists() or not cfg.exists():
            raise SystemExit(f"В {model_dir} нет avg_model.pt и/или config.yaml")

        model = Speaker(str(model_dir))
        model.set_device(device)

        return model
    
    def __call__(self, pcm_list: list) -> float:
        cs = 0

        for pcm in pcm_list:
            embed = self.model.extract_embedding_from_pcm(pcm, sample_rate=24000)
            cs += self.model.cosine_similarity(self.ref_embed, embed)

        return cs / len(pcm_list) if len(pcm_list) != 0 else 0
        
    
def mos(data_dir: Path) -> float:
    args = {
        "pretrained_model": "nisqa/nisqa_mos_only.tar",
        "data_dir": data_dir,
        "mode": "predict_dir",
        "ms_channel": 1
    }
    nisqa = nisqaModel(args)
    return nisqa.predict()["mos_pred"].mean()

