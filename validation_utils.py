from pathlib import Path
from typing import Any

import gigaam
from nisqa.NISQA_model import nisqaModel
from wespeaker.cli.speaker import Speaker

from jiwer import cer

class CER:
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
    
    def cer_list(self, base_texts: list[str], samples_list: list[Path]) -> float:
        c = 0

        for t, s in zip(base_texts, samples_list):
            c += self.cer(t, s)

        return c / len(base_texts) if len(base_texts) != 0 else 0
    
class COSSIM:
    def __init__(self, ref_wav: Path, device):
        self.model = self.load_model_from_dir(device)
        self.ref_embed = self.model.extract_embedding(str(ref_wav))

    def load_model_from_dir(self, device) -> Speaker:
        model_dir = Path("wespeak")

        avg = model_dir / "avg_model.pt"
        cfg = model_dir / "config.yaml"

        if not avg.exists() or not cfg.exists():
            raise SystemExit(f"В {model_dir} нет avg_model.pt и/или config.yaml")

        model = Speaker(str(model_dir))
        model.set_device(device)

        return model
    
    def __call__(self, samples_list: list) -> float:
        cs = 0

        for sample in samples_list:
            embed = self.model.extract_embedding(str(sample))
            cs += self.model.cosine_similarity(self.ref_embed, embed)

        return cs / len(samples_list) if len(samples_list) != 0 else 0
        
    
def mos(data_dir: Path) -> float:
    args = {
        "pretrained_model": "nisqa/nisqa_mos_only.tar",
        "data_dir": data_dir,
        "mode": "predict_dir",
        "ms_channel": 1
    }
    nisqa = nisqaModel(args)
    return nisqa.predict()["mos_pred"].mean()

class SpeechEval:
    def __init__(self, ref_wav: Path, device):
        self.cer = CER()
        self.cossim = COSSIM(ref_wav, device)

    def __call__(
        self,
        base_texts: list,
        samples_list: list[Path]
    ) -> dict[str, float]:
        
        _mos = mos(samples_list[0].parent)
        _cossim = self.cossim(samples_list)
        _cer = self.cer.cer_list(base_texts, samples_list)

        return {"mos": _mos, "cossim": _cossim, "cer": _cer}



    