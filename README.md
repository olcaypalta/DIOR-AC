# DIOR-AC

Benchmarking object detection in satellite imagery under acquisition-chain degradations.

```
pip install -r requirements.txt
python src/egit.py --arch yolo11m --seed 0 --epochs 10 --veri <dior_dir> --out results
python src/degerlendir.py --model owlv2 --seed 0 --n 300 --veri <dior_dir> --out results
python src/analiz.py
python src/onarim.py --model ft:<best.pt> --tag yolo11m_s0 --seed 0
python src/ek_kosul.py --model ft:<best.pt> --tag yolo11m_s0 --seed 0
python src/kalite.py 500 0
python src/istatistik.py
python src/onarim_analiz.py
```

Data: DIOR (Li et al., 2020), Hugging Face release `HichTala/dior`.
Trained weights and raw detections: see Releases.
