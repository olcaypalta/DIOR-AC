# DIOR-AC

Benchmarking object detection in satellite imagery under acquisition-chain degradations.

```
pip install -r requirements.txt
python src/egit.py --arch yolo11m --seed 0 --epochs 10 --veri <dior_dir> --out results
python src/degerlendir.py --model owlv2 --seed 0 --n 300 --veri <dior_dir> --out results
python src/analiz.py
```

Data: DIOR (Li et al., 2020), Hugging Face release `HichTala/dior`.
Trained weights and raw detections: see Releases.
