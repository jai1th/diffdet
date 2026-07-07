# AI-Generated Video Detection — Mixed-Domain Supervised Contrastive Adaptation

When a video forgery detector is trained on one family of generators and then
tested on a different one, accuracy usually **collapses** — much of what the model
learned turns out to be a dataset shortcut (frame-count and resolution artifacts
that inflate in-distribution scores and disappear under transfer) rather than a
generator-agnostic notion of "synthetic." This project asks whether a model can
be **adapted to a new generator family (AEGIS) while retaining** its performance
on the families it already knew (DVF, GenVideo). The answer here is yes: a
**mixed-domain supervised-contrastive** stage, with generator-aware batch
sampling, adapts an R(2+1)D backbone to the new target *and* holds onto the
source domains — instead of trading one for the other.

> 🚀 **Live demo:** https://huggingface.co/spaces/jai1th/ai-generated-video-detection
> &nbsp;·&nbsp; 🧠 **Model & weights:** https://huggingface.co/jai1th/dvf_trained_transferred_aegis
> Thesis Publication: https://doi.org/10.7302/dspace/29187

> **What this repo is.** The faithful implementation and headline results from the
> thesis — *not* a turnkey reproduction package. Datasets are not redistributed
> and exact splits/caches are not shipped (see [Scope & limitations](#scope--limitations)).

---

## Method

Three stages, all on pixels in `[0, 1]` with **no** Kinetics mean/std normalisation:

1. **Backbone (from scratch).** `torchvision` `r2plus1d_18` trained from random
   init — *not* Kinetics-400 pretrained — as a supervised baseline on DVF
   (`Dropout(0.4) → Linear(512→2)` head). → [`src/train_baseline.py`](src/train_baseline.py)
2. **Mixed-domain SupCon transfer (`run6`).** Initialise from the DVF backbone and
   fine-tune with **supervised contrastive loss** across all three domains at once
   (AEGIS target + DVF + GenVideo retention). Each batch is built by a
   **generator-aware quota sampler** so no single generator dominates the
   contrastive objective, and domains are weighted AEGIS `0.5` / DVF `0.3` /
   GenVideo `0.2`. A projection head `Linear(512→512) → ReLU → Linear(512→128)`
   is trained and its **L2-normalised 128-d output** is the embedding space.
   → [`src/train_supcon.py`](src/train_supcon.py)
3. **Projection-space prototype inference.** Build one **real** and one **fake**
   prototype as the L2-normalised mean of a small labeled support bank in
   projection space, then score each query clip by
   `score = sim_fake − sim_real` and call it AI-generated when `score ≥ 0`.
   → [`src/build_prototypes.py`](src/build_prototypes.py),
   [`src/prototype_inference.py`](src/prototype_inference.py)

<!-- Add assets/method.png and uncomment:
![Method](assets/method.png)
-->

Preprocessing (decode → resize short side to 224 → center-crop square → 24 frames
@ 24 fps) is the single source of truth in [`src/preprocess.py`](src/preprocess.py)
and is replicated exactly by the demo.

## Results

`run6`, **fixed real/fake prototype protocol**, **projection space**, decision
threshold **`0.0`** (`score = sim_fake − sim_real`). Prototypes are the
L2-normalised mean of a labeled support bank; query clips are the held-out
remainder. These measure **representation / adaptation quality**, not calibrated
deployment performance.

| Dataset | AUROC | EER | Accuracy |
|---|---|---|---|
| AEGIS (target) | 0.808 | 0.268 | 0.730 |
| DVF (retain) | 0.847 | 0.225 | 0.767 |
| GenVideo (retain) | 0.817 | 0.255 | 0.748 |

The two retention domains (DVF, GenVideo) stay strong **after** adapting to AEGIS —
the point of the mixed-domain objective. Numbers are author-reported from the
thesis under the protocol above; this repo does not re-run them (the
support/query manifests and cached clips are not bundled).

## Scope & limitations

- **Implementation, not reproduction.** This is the method and the reported
  numbers. Data is **not** redistributed (third-party licenses), and the exact
  splits and cached frame tensors are not shipped.
- **Prototype protocol caveat.** Support clips are drawn from labeled data of the
  same datasets, so the metrics reflect **representation/adaptation quality**, not
  unconditional deployment performance. The fixed `score ≥ 0` threshold is not
  calibrated per deployment.
- **Projection-space separability.** Real and fake prototypes have relatively high
  cosine similarity, which caps separation in the projection space.
- **What it detects.** **End-to-end AI-generated video only** — *not* face-swap
  deepfakes — and only across the generators seen in DVF / GenVideo / AEGIS.
  Generalisation to unseen generators or heavy post-processing is not guaranteed.

## How to run

The trained weights live on Hugging Face — you do not need to retrain anything.

```python
# pip install -r requirements.txt
from huggingface_hub import hf_hub_download
import torch, torch.nn.functional as F
import sys

repo = "jai1th/dvf_trained_transferred_aegis"
ckpt = hf_hub_download(repo, "supcon/final_best.pt")
loader_py = hf_hub_download(repo, "load_model.py")

sys.path.insert(0, __import__("os").path.dirname(loader_py))
from load_model import load_model, extract_projected_embedding

model = load_model(ckpt, map_location="cpu")        # eval mode, CPU
# Build a clip with src/preprocess.py:
from src.preprocess import video_to_clip            # raw video -> [C,T,224,224] in [0,1]
clip = video_to_clip("your_video.mp4").unsqueeze(0) # add batch dim -> [1,C,24,224,224]
emb = extract_projected_embedding(model, clip)      # L2-normalised 128-d
```

To score against prebuilt prototypes (the protocol used for the results table):

```bash
# 1) build a real/fake prototype pair from a labeled support JSONL
python src/build_prototypes.py \
  --support-jsonl <PATH>/support_bank.jsonl \
  --checkpoint    <PATH>/supcon_best_model.pt \
  --output        <PATH>/prototypes.pt \
  --embedding-source projection --num-workers 0

# 2) score a query JSONL (threshold 0.0, projection space)
python src/prototype_inference.py \
  --query-jsonl   <PATH>/query.jsonl \
  --checkpoint    <PATH>/supcon_best_model.pt \
  --prototypes-path <PATH>/prototypes.pt \
  --output-dir    ./infer_out \
  --embedding-source projection --threshold 0.0 --num-workers 0
```

The easiest way to try it on a single clip is the
[**live demo**](https://huggingface.co/spaces/jai1th/ai-generated-video-detection).

**Reproducibility knobs.** Training seed `42`; evaluation/prototype seed `123`;
clip `24` frames @ `24` fps, `224×224`, pixels in `[0, 1]`, no Kinetics norm.
The full `run6` architecture / preprocessing / training / eval-protocol values
are in [`config.json`](config.json) — the single source of truth, matching the
[model card](https://huggingface.co/jai1th/dvf_trained_transferred_aegis).

## Datasets & citation

Trained and evaluated on **DVF**, **GenVideo**, and **AEGIS**. None are
redistributed here; see [`DATASETS.md`](DATASETS.md) for sources, license notes,
and BibTeX. Downstream users must comply with each dataset's terms **and** the
terms of the underlying generators (Sora / KLing / Pika).

If you use this code or the model, please cite the thesis (see
[`CITATION.cff`](CITATION.cff)); a journal paper is in preparation and will be
added when published.

## License

- **Code:** MIT — see [`LICENSE`](LICENSE).
- **Weights:** CC BY-NC 4.0 (non-commercial), hosted on
  [Hugging Face](https://huggingface.co/jai1th/dvf_trained_transferred_aegis).
- **Data:** not included; governed by the dataset and generator licenses in
  [`DATASETS.md`](DATASETS.md).
