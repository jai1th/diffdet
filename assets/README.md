# Assets

Figures referenced by the top-level `README.md` go here.

This repository ships **no fabricated figures**. The following are placeholders —
drop in the real exported images from the thesis and the filenames below will
render automatically in the README.

| Filename | What it should show | Status |
|---|---|---|
| `method.png` | Architecture diagram: R(2+1)D backbone → SupCon projection head `Linear(512→512)→ReLU→Linear(512→128)` → L2-normalised 128-d projection space → fixed real/fake prototype scoring (`score = sim_fake − sim_real`). | **TODO — add PNG** |
| `retention.png` | Retention comparison: source-domain (DVF / GenVideo) performance before vs. after mixed-domain SupCon adaptation to AEGIS. | optional — add if exported |
| `per_generator.png` | Per-generator FPR / FNR breakdown on the test sets. | optional — add if exported |

The README's results table stands on its own without these images. When you add
`method.png`, uncomment the image line under the **Method** section of `README.md`.
