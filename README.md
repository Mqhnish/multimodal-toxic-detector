# Multimodal Toxic Content Detector 

A content moderation pipeline that analyses **text, images, and audio** independently, then fuses the scores into a single verdict. Built using pre-trained transformer models from HuggingFace.

Built while learning ML before starting a CS degree.

---

## What it does

| Modality | Model | What it detects |
|---|---|---|
| **Text** | `unitary/toxic-bert` | Toxic / hate speech in written content |
| **Image** | `Falconsai/nsfw_image_detection` | NSFW / harmful visual content |
| **Audio** | `ehcalabres/wav2vec2-lg-xlsr` | Aggressive/angry emotional tone |

All three are optional — run with just text, just an image, or all three at once.

**Fusion**: simple average of available modality scores → threshold-based verdict.

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/multimodal-toxic-detector
cd multimodal-toxic-detector
pip install -r requirements.txt
python detector.py
```

> **First run downloads ~500 MB of model weights** from HuggingFace. Cached after that.

### Example session
```
Enter text: I hate all of you
Enter image path (or Enter to skip):
Enter audio path (or Enter to skip):

[Text]  toxic       score=0.9821
------------------------------
FINAL SCORE : 0.9821
VERDICT     : [!!] TOXIC / HARMFUL CONTENT DETECTED
```

---

## Architecture

```
Input ──┬── Text  → toxic-bert    → toxicity score
        ├── Image → NSFW model    → nsfw score  
        └── Audio → emotion model → aggression score
                              ↓
                    Late fusion (mean)
                              ↓
                    Threshold → Verdict
```

**Late fusion** (combining scores at the end rather than features in the middle) is the simplest multimodal fusion strategy. More advanced approaches would weight modalities by confidence or train a learned combiner.


## What I learned

- How transformer pipelines work end-to-end (tokenisation → inference → score)
- The difference between sigmoid (multi-label) and softmax (single-label) outputs
- What late fusion is and when to use it vs early/middle fusion
- How emotion recognition from audio works (wav2vec2 features → emotion classifier)
- Why model card reading matters — wrong model = wrong results even if code runs fine
