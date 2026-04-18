# SoccerNet Dense Video Captioning! 

This repository extends the [SoccerNet-Caption](https://github.com/SoccerNet/sn-caption) baseline for the task of Dense Video Captioning in soccer broadcasts.

## Project Overview

We extend the original NetVLAD-based captioning baseline with a transformer encoder, multimodal audio-visual fusion, contrastive pretraining, and a  dual LSTM decoder with modality-specific cross-attention. Our best model achieves **CIDEr 21.18** vs the baseline **18.38** (~15% relative improvement) with fewer parameters


<img width="942" height="642" alt="image" src="https://github.com/user-attachments/assets/77c4fb0a-b420-41be-96d9-0f5cc9b49b91" />

## Our Contributions to this Contest 

- **Transformer Encoder** — learned positional embeddings with stochastic positional dropout (rate=0.3), mean pooling over the temporal dimension
- **Multimodal Fusion** — joint audio-visual transformer with modality embeddings, supporting Baidu Features
- **Contrastive Pretraining** — CLIP-style pretraining of the video encoder against frozen SBERT (`all-MiniLM-L6-v2`) text embeddings on SoccerNet caption labels
- **Dual LSTM Decoder** — two parallel LSTMs attending independently over audio and video encoder sequences, fused via linear projection
- **Memory-Efficient Data Pipeline** — `np.memmap`-based feature loading replacing full RAM loading, enabling full-dataset training on Kaggle
- **Audio Embeddings** - Extracted Vggish auido features from soccer videos
- **Explainability Module** — decoder attention heatmaps and encoder self-attention capture for qualitative analysis


## Results

| Model | Params (M) | B@1 | B@2 | B@3 | B@4 | METEOR | ROUGE-L | CIDEr |
|---|---|---|---|---|---|---|---|---|
| A0 — Baseline (NetVLAD Encoder) | 30.7 | 27.2 | 16.1 | 9.2 | 6.0 | 23.8 | 23.9 | 18.4 |
| A1 — Video Transformer + Contrastive Pretraining | 9.8 | 27.7 | 16.4 | 9.6 | 6.4 | 24.0 | 24.0 | 22.7 |
| A2 — Multimodal Transformer (Video + Audio) | 16.4 | 27.8 | 16.7 | 10.0 | 6.6 | 23.8 | 24.1 | 23.1 |
| A3 — Audio-Only Transformer | 13.0 | 20.7 | 11.2 | 5.1 | 2.7 | 21.0 | 19.1 | 5.4 |
| A4 — Video Transformer (No Contrastive) | 17.3 | 27.2 | 16.2 | 9.4 | 6.2 | 23.9 | 23.9 | 18.7 |
| **A5 — Multimodal Transformer + Dual LSTM Decoder** | **22.3** | **28.6** | **17.0** | **10.0** | **6.8** | **23.9** | **24.7** | **23.5** |



<img width="2960" height="1742" alt="image" src="https://github.com/user-attachments/assets/7209801e-8ef8-4999-8959-5168e5e69c33" />

## Training

See [Benchmarks/TemporallyAwarePooling/](Benchmarks/TemporallyAwarePooling/) for training scripts and instructions.

## Authors

- Saleem Ibrahim
- Salma Nassri
- Chidiebere Ekeke 

---

*Built on the SoccerNet-Caption baseline by Mkhallati et al. (2023)*

## Original Dataset

The task of Dense Video Captioning consists in generating engaging captions describing soccer actions and localizing each caption with a timestamp. 471 videos from soccer broadcast games with captions, features extracted at 2fps including Baidu Research features from the 2021 action spotting challenge winners.


## Acknowledgements

- Thank you very much to [SoccerNet](https://github.com/SoccerNet/sn-caption) team for the dataset, Baidu feature embeddings, and original baseline codebase (Mkhallati et al., 2023)


## Presentation and Paper

[View our project slides](https://docs.google.com/presentation/d/1HdNiKnp6isEHudK0rzIi-lhSa0tTJH1YLKGeBUpi-BQ/edit?usp=sharing)


## Citation
Please cite Soccernet team's work if you use their dataset:
```
@inproceedings{mkhallati2023soccernet,
  title={SoccerNet-Caption: Dense Video Captioning for Soccer Broadcasts Commentaries},
  author={Mkhallati, Hassan and Cioppa, Anthony and Giancola, Silvio and Ghanem, Bernard and Van Droogenbroeck, Marc},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5073--5084},
  year={2023}
}
```

