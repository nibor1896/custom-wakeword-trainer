# custom-wakeword-trainer

Ein **lokaler Trainer für eigene [openWakeWord](https://github.com/dscripka/openWakeWord)-Wake-Words** — eine reproduzierbare Rezeptur + Skripte, mit denen man ein **beliebiges** Wake-Word ("Hey Jarvis", "Computer", "Hey Horus", …) trainiert. Läuft **lokal auf Windows + Python 3.13**, wo das offizielle openWakeWord-Colab-Notebook seit 2023 nicht mehr durchläuft.

> **"Hey Horus"** ist das erste, vollständig durchgerechnete Beispielmodell (entstanden fürs [Horus](https://github.com/nibor1896/Horus)-Projekt).

## Was am Ende rauskommt

Eine winzige Datei `<phrase>.onnx` (~840 KB). Laufzeitkette:

```
Mikro 16 kHz → melspectrogram.onnx → embedding_model.onnx → <phrase>.onnx → Wahrscheinlichkeit → Schwelle
```

Die ersten beiden ONNX-Modelle sind die generischen, für **alle** openWakeWord-Modelle gleichen Feature-Extraktoren (Apache-2.0). Nur `<phrase>.onnx` ist das trainierte, phrasen-spezifische Teil → **Standard-openWakeWord-Format**, direkt nutzbar in Home Assistant / Rhasspy / ESPHome / eigenen Runtimes.

**Warum ein trainierter Klassifikator statt Distanzvergleich, und warum überhaupt eine Eigenlösung:** siehe [Issue #1](https://github.com/nibor1896/custom-wakeword-trainer/issues/1). Kurz: reiner Embedding-Distanzvergleich (DTW/Cosine) trennt auf echter Stimme nicht (Sprecher dominiert den Inhalt), und der offizielle Trainingsweg ist kaputt.

## Voraussetzungen

- Python 3.13 (Windows getestet), NVIDIA-GPU empfohlen (CPU geht, ist langsamer).
- ~20 GB freier Platz (16 GB Negativ-Features + Zwischendaten).
- Die beiden generischen Feature-Modelle von openWakeWord (v0.5.1):
  [`melspectrogram.onnx`](https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx),
  [`embedding_model.onnx`](https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx).

## Rezept

```bash
python -m venv venv && venv\Scripts\pip install -r requirements.txt
```

1. **Negativ-Features laden** (~16 GB, vorberechnet, ~2000 Std. Sprache/Musik/Rauschen):
   `python scripts/download_negatives.py`
2. **Synthetische Positive generieren** (viele Sprecher, via [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator)):
   ```bash
   # webrtcvad baut auf 3.13 nicht -> webrtcvad-wheels nutzen (in requirements.txt)
   set PYTHONIOENCODING=utf-8
   python -m piper_sample_generator "hey horus" --model models/en_US-libritts_r-medium.pt \
          --max-samples 20000 --batch-size 128 --output-dir data/positives
   ```
3. **Eigene Aufnahmen — der Schlüssel: ZWEI saubere Sessions** (kein Auto-Labeling → kein Label-Rauschen):
   - Session A: **nur** das Wake-Word (~50×, normale Stimme, variiert) → `data/session-pos/`
   - Session B: **nur** Negatives (normal reden, Vorlesen, harte Verwechsler, Geräusche) → `data/session-neg/`
   - Als 16-kHz-Mono-WAV, je ~2 s.
4. **Trainieren** (Pfade oben in der Datei anpassen):
   `python scripts/train.py` → schreibt `<phrase>.onnx` + gibt Recall/Fehlalarme auf zurückgehaltenen Aufnahmen aus.

## Wichtigste Erkenntnisse

Ausführlich in [Issue #2](https://github.com/nibor1896/custom-wakeword-trainer/issues/2). In einem Satz: **die Datenqualität entscheidet alles** — rein synthetische Positive geben auf echter Stimme nur ~8 % Trefferquote, echte Nutzer-Aufnahmen + Augmentation heben das auf ~90 %, und Whisper-Auto-Labeling gemischter Sessions ruiniert es wieder (die zwei sauberen Sessions oben sind der Fix).

## Lizenzen

- openWakeWord + Feature-Modelle: Apache-2.0
- LibriTTS-R (piper-Positivstimmen): CC-BY-4.0 → Namensnennung
- ACAV100M: nur vorberechnete Features genutzt, kein Audio im Modell
- Eingebrachte echte Aufnahmen sind in den Gewichten mit-eintrainiert (nicht als Audio rückgewinnbar)

## Status

Siehe [Issue #1](https://github.com/nibor1896/custom-wakeword-trainer/issues/1). "Hey Horus" ist trainiert, in Horus integriert und live bestätigt (90 % Recall, 0 Alltags-Fehlalarme).
