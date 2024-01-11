import os
import argparse
import pathlib
import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import Resample
from hf_ecapa_tdnn import Encoder
from tqdm import tqdm

classifier = Encoder.from_hparams(
    source="yangwang825/ecapa-tdnn-vox2"
)

def verification(wav1, wav2):
    signal1, sr1 = torchaudio.load(wav1)
    signal2, sr2 = torchaudio.load(wav2)

    resample1 = Resample(orig_freq=sr1, new_freq=16000)
    resample2 = Resample(orig_freq=sr2, new_freq=16000)

    signal1 = resample1(signal1)
    signal2 = resample2(signal2)
    
    emb1 = classifier.encode_batch(signal1).squeeze(0)
    emb2 = classifier.encode_batch(signal2).squeeze(0)
    
    sim = F.cosine_similarity(emb1, emb2)
    
    return sim[0].item()
    
def main(args):
    groudtruth_dir = pathlib.Path(args.groundtruth_dir)
    
    groudtruth_wavs = [file for file in groudtruth_dir.glob("*.wav")]
    
    speaker_similarity = 0
    for label in tqdm(groudtruth_wavs):
        filename = str(label).split("/")[-1]
        pred = os.path.abspath(os.path.join(args.generated_dir, filename))
        speaker_similarity += verification(label, pred)
    
    print("Avg speaker similarity: ", speaker_similarity/len(groudtruth_wavs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_dir", type=str, required=True)
    parser.add_argument("--groundtruth_dir", type=str, required=True)
    
    args = parser.parse_args()
    main(args)
