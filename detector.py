"""
detector.py
Multimodal hate speech / toxic content detector.
Analyses text, image, and audio independently then fuses scores.

Models used:
  Text  → unitary/toxic-bert         (fine-tuned BERT on toxic comment datasets)
  Image → Falconsai/nsfw_image_detection (NSFW classifier, not AI-image detector)
  Audio → ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition

Requirements: see requirements.txt
Run: python detector.py
"""

import os
import sys
import torch
from PIL import Image
from transformers import (
    pipeline,
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    logging as hf_logging,
)

hf_logging.set_verbosity_error()

# ── Device ─────────────────────────────────────────────────────────────────
DEVICE      = 0 if torch.cuda.is_available() else -1
DEVICE_STR  = "cuda" if DEVICE == 0 else "cpu"

# ── Labels known to correlate with aggression in the audio model ────────────
AGGRESSIVE_AUDIO_LABELS = {"angry", "anger", "disgust", "fear"}


# ── Model loaders (lazy, with clear error messages) ─────────────────────────

def load_text_model():
    print("  [text]  Loading toxic-bert...")
    return pipeline(
        "text-classification",
        model="unitary/toxic-bert",
        device=DEVICE,
        truncation=True,
        max_length=512,
    )


def load_image_model():
    """
    Uses Falconsai/nsfw_image_detection — a proper NSFW/safe binary classifier.
    Labels: 'normal' / 'nsfw'
    (Original code used umm-maybe/AI-image-detector which detects AI-generated
    images, not harmful content — wrong model for this purpose.)
    """
    print("  [image] Loading NSFW image classifier...")
    extractor = AutoFeatureExtractor.from_pretrained("Falconsai/nsfw_image_detection")
    model     = AutoModelForImageClassification.from_pretrained(
                    "Falconsai/nsfw_image_detection"
                ).to(DEVICE_STR)
    return extractor, model


def load_audio_model():
    print("  [audio] Loading speech emotion classifier...")
    return pipeline(
        "audio-classification",
        model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        device=DEVICE,
    )


# ── Score extractors ─────────────────────────────────────────────────────────

def score_text(text_detector, text: str) -> tuple[float, str]:
    """
    Returns (toxicity_score, label_string).
    toxic-bert outputs label 'toxic' or 'non_toxic' (note underscore).
    Score is always the model's confidence in 'toxic'.
    """
    res   = text_detector(text)[0]
    label = res["label"].lower()
    conf  = res["score"]

    # BUG FIX: original code checked 'non-toxic' (hyphen) — bert uses underscore
    if label == "toxic":
        return conf, "toxic"
    elif label == "non_toxic":
        return 1.0 - conf, "non_toxic"
    else:
        # Unknown label — be conservative
        return 0.5, label


def score_image(extractor, model, path: str) -> tuple[float, str]:
    """
    Returns (nsfw_probability, label).
    Falconsai model: 'normal' or 'nsfw'.
    """
    image   = Image.open(path).convert("RGB")
    inputs  = extractor(images=image, return_tensors="pt").to(DEVICE_STR)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs   = torch.softmax(logits, dim=-1)[0]
    id2lbl  = model.config.id2label

    # Find nsfw probability explicitly
    for idx, lbl in id2lbl.items():
        if lbl.lower() == "nsfw":
            return probs[idx].item(), "nsfw"

    # Fallback: return max prob
    top_idx = probs.argmax().item()
    return probs[top_idx].item(), id2lbl[top_idx]


def score_audio(audio_detector, path: str) -> tuple[float, str]:
    """
    Returns (aggression_score, dominant_label).
    Searches for known aggressive emotion labels; falls back to top prediction.
    """
    results = audio_detector(path, top_k=None)  # get all labels

    # Some pipeline versions return a dict instead of a list
    if isinstance(results, dict):
        results = [results]

    # BUG FIX: original code assumed fixed label strings; we now check all
    for res in results:
        if res["label"].lower() in AGGRESSIVE_AUDIO_LABELS:
            return res["score"], res["label"]
    # No aggressive label found → content is probably calm
    top = results[0]
    return 0.1, top["label"]   # low aggression score, not zero


# ── Main detector ─────────────────────────────────────────────────────────────

def run_detector():
    print("\n" + "=" * 52)
    print("     INTEGRATED MULTIMODAL CONTENT DETECTOR")
    print("=" * 52)

    # ── Load only what's needed ──
    u_text = input("\nEnter text (or Enter to skip): ").strip()
    u_img  = input("Enter image path  (or Enter to skip): ").strip()
    u_aud  = input("Enter audio path  (or Enter to skip): ").strip()

    if not any([u_text, u_img, u_aud]):
        print("No input provided. Exiting.")
        return

    print("\nLoading models (first run downloads weights ~500 MB)...")
    scores  = []
    details = []

    # ── Text ────────────────────────────────────────────────────────────────
    if u_text:
        try:
            det   = load_text_model()
            score, label = score_text(det, u_text)
            details.append(f"[Text]  {label:<10}  score={score:.4f}")
            scores.append(score)
        except Exception as e:
            print(f"  [text]  ERROR: {e}")

    # ── Image ────────────────────────────────────────────────────────────────
    if u_img:
        if not os.path.exists(u_img):
            print(f"  [image] File not found: {u_img}")
        else:
            try:
                ext, mdl  = load_image_model()
                score, label = score_image(ext, mdl, u_img)
                details.append(f"[Image] {label:<10}  score={score:.4f}")
                scores.append(score)
            except Exception as e:
                print(f"  [image] ERROR: {e}")

    # ── Audio ────────────────────────────────────────────────────────────────
    if u_aud:
        if not os.path.exists(u_aud):
            print(f"  [audio] File not found: {u_aud}")
        else:
            try:
                det   = load_audio_model()
                score, label = score_audio(det, u_aud)
                details.append(f"[Audio] {label:<10}  score={score:.4f}")
                scores.append(score)
            except Exception as e:
                print(f"  [audio] ERROR: {e}")

    # ── Report ───────────────────────────────────────────────────────────────
    if not scores:
        print("\nNo modalities could be analysed.")
        return

    print("\n--- Analysis Report ---")
    for d in details:
        print(d)

    final = sum(scores) / len(scores)
    print("-" * 30)
    print(f"Modalities used   : {len(scores)}")
    print(f"Fusion method     : simple average")
    print(f"FINAL SCORE       : {final:.4f}")

    if final > 0.70:
        verdict = "[!!] TOXIC / HARMFUL CONTENT DETECTED"
    elif final > 0.40:
        verdict = "[?]  BORDERLINE — REVIEW RECOMMENDED"
    else:
        verdict = "[OK] CONTENT APPEARS SAFE"

    print(f"VERDICT           : {verdict}")


if __name__ == "__main__":
    run_detector()
