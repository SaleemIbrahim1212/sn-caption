import logging
import os
import zipfile
import json
import time
from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
from dataset import SoccerNetCaptions, PredictionCaptions, collate_fn_padd
from SoccerNet.Evaluation.utils import AverageMeter, getMetaDataTask
from utils import evaluate as evaluate_spotting
from SoccerNet.Evaluation.DenseVideoCaptioning import evaluate as evaluate_dvc
from torch.nn.utils.rnn import pack_padded_sequence
from transformer import Transformer_Audio, Transformer_Video, Transformer
from loss import ContrastiveLoss
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


import wandb

def trainer(train_loader,
            model_video,
            model_text,
            optimizer,
            scheduler,
            model_name,
            criterion,
            max_epochs=20,
            log_every_n_batches=50):
    """Run the contrastive pretraining loop to align video and text embeddings.
    Trains a video encoder and a text encoder jointly using a CLIP-style contrastive
    loss so that matching video–caption pairs are pulled together in embedding space
    while non-matching pairs are pushed apart.

    Args:
        train_loader (DataLoader): Training dataloader yielding batches of
            ``(feats, caption), lengths, mask, caption_or, cap_id``.
        model_video (nn.Module): Video encoder (e.g. ``Transformer_Video``) that
            maps video features to a shared embedding space.
        model_text (nn.Module): Text encoder (e.g. ``TextEncoder``) that maps
            captions to the same shared embedding space.
        optimizer (torch.optim.Optimizer): Optimizer over both models' parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler or None): Optional
            learning-rate scheduler stepped on training loss each epoch.
        model_name (str): Name used to create the output directory under ``models/``.
        criterion (nn.Module): Contrastive loss function (e.g. ``ContrastiveLoss``).
        max_epochs (int): Number of training epochs. Default: 20.
        log_every_n_batches (int): Frequency (in batches) for logging metrics to
            the console and Weights & Biases. Default: 50.
    """
    logging.info("start contrastive training")
    os.makedirs(os.path.join("models", model_name), exist_ok=True)
    for epoch in range(max_epochs):
        loss_training = train("contrastive", train_loader, model_video=model_video, model_text=model_text,
                      optimizer=optimizer, epoch=epoch + 1,
                      log_every_n_batches=log_every_n_batches, criterion=criterion, model_name=model_name)

        logging.info(f"Epoch {epoch+1}: train_loss={loss_training:.4f}")

        if scheduler is not None:
            scheduler.step(loss_training)
    return

def train(phase, dataloader, model_video,model_text,  criterion, optimizer, epoch, model_name, log_every_n_batches=50):
    """Execute a single training epoch for contrastive video–text alignment.
    Iterates over all batches, computing the contrastive loss between video and text
    embeddings, back-propagating, and logging diagnostics (diagonal vs. off-diagonal
    cosine similarity, top-1 retrieval accuracy). Saves a checkpoint at the end of
    each epoch.

    Args:
        phase (str): Label for logging (e.g. ``'contrastive'``).
        dataloader (DataLoader): Training dataloader.
        model_video (nn.Module): Video encoder.
        model_text (nn.Module): Text encoder.
        criterion (nn.Module): Contrastive loss function.
        optimizer (torch.optim.Optimizer): Optimizer for both models.
        epoch (int): Current epoch number (1-indexed).
        log_every_n_batches (int): Logging frequency in batches. Default: 50.

    Returns:
        float: Average training loss over the epoch.
    """
    device = next(model_video.parameters()).device
    device2 = next(model_text.parameters()).device 

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model_video.train()
    model_text.train()

    end = time.time()
    with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
        for i, batch in t:
            # measure data loading time
            data_time.update(time.time() - end)
            (feats, caption), lengths, mask, caption_or, cap_id = batch
            lengths = lengths.float().to(device)
            caption = caption.to(device)
            feats = feats.to(device)
            output_video = model_video(feats)  # this should get me the embeddings for the video, ie the token from the transformer 
            output_text = model_text(caption_or, lengths) # This should get me the embeddings for the text, ie the vector for the text embeddings 
            if isinstance(output_video, tuple):
                output_video = output_video[0]
            if isinstance(output_text, tuple):
                output_text = output_text[0]
            loss = criterion(output_text, output_video)
            # measure accuracy and record loss
            losses.update(loss.item(), feats.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if log_every_n_batches > 0 and (i + 1) % log_every_n_batches == 0:
                lr = optimizer.param_groups[0]["lr"] if optimizer is not None else -1.0
                msg = (
                    f"[{phase}] epoch={epoch} batch={i + 1}/{len(dataloader)} "
                    f"loss={loss.item():.4f} avg_loss={losses.avg:.4f} lr={lr:.2e} "
                    f"data_t={data_time.val:.3f}s iter_t={batch_time.val:.3f}s"
                )
                if torch.cuda.is_available():
                    mem = torch.cuda.memory_allocated() / (1024 ** 3)
                    max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    msg += f" gpu_mem={mem:.2f}G max_gpu_mem={max_mem:.2f}G"
                logging.info(msg)

                if wandb.run is not None:
                    wandb.log({
                        f"{phase}/batch_loss": float(loss.item()),
                        f"{phase}/batch_avg_loss": float(losses.avg),
                        f"{phase}/lr": float(lr),
                        "epoch": int(epoch),
                        f"{phase}/batch_idx": int(i + 1),
                    })

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            with torch.no_grad():
                sim = torch.nn.functional.cosine_similarity(
                    output_video.unsqueeze(1), output_text.unsqueeze(0), dim=-1
                )
                mean_off_diag = (sim.sum() - sim.diag().sum()) / (sim.numel() - sim.size(0))
                mean_diag = sim.diag().mean()
                top1 = (sim.argmax(dim=1) == torch.arange(sim.size(0), device=sim.device)).float().mean()

            desc = f'Train {epoch}: '
            desc += f'Time {batch_time.avg:.3f}s '
            desc += f'(it:{batch_time.val:.3f}s) '
            desc += f'Data:{data_time.avg:.3f}s '
            desc += f'(it:{data_time.val:.3f}s) '
            desc += f'Loss {losses.avg:.4e} '
            desc += f'Diag:{mean_diag:.3f} OffDiag:{mean_off_diag:.3f} '
            desc += f'Top1 {top1:.4e} '
            t.set_description(desc)

        torch.save({
                'epoch': epoch + 1,
                'model_video': model_video.state_dict(),
                'model_text': model_text.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join("models", model_name, "best.pth"))

    return losses.avg


class TextEncoder(nn.Module):
    """Lightweight text encoder that projects sentence embeddings into a shared space.
    Uses a frozen ``all-MiniLM-L6-v2`` sentence-transformer to produce 384-dimensional
    sentence embeddings, then maps them to ``proj_dim`` via a single trainable linear
    layer. This allows efficient text encoding while only learning the projection head.

    Args:
        vocab_size (int): Vocabulary size (unused; kept for interface compatibility).
        embed_dim (int): Embedding dimension (unused; kept for interface compatibility).
        proj_dim (int): Output projection dimension, should match the video encoder's
            output dimension for contrastive alignment.
        isFrozen (bool): If True, freeze all sentence-transformer parameters so only
            the linear projection is trained. Default: True.

    Forward args:
        caption (list[str]): Raw caption strings to encode.
        lengths (Tensor): Caption lengths (unused; kept for interface compatibility).

    Returns:
        Tensor: Projected text embeddings of shape ``(batch, proj_dim)``.
    """
    def __init__(self, vocab_size, embed_dim, proj_dim, isFrozen=True):
        super().__init__()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.linear = nn.Linear(384, proj_dim)

        if isFrozen:
            for param in self.model.parameters():
                param.requires_grad = False


    def forward(self, caption, lengths):
        with torch.no_grad():
            x = self.model.encode(caption,convert_to_tensor=True,device=self.linear.weight.device)
        x = self.linear(x)
        return x 



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--SoccerNet_path", type=str, required=True)
    parser.add_argument("--features", type=str, default="ResNET_TF2_PCA512.npy")
    parser.add_argument("--split_train", nargs="+", default=["train"])
    parser.add_argument("--version", type=int, default=2)
    parser.add_argument("--framerate", type=int, default=1)  # audio memmap is 1fps-aligned
    parser.add_argument("--window_size_caption", type=int, default=45)
    parser.add_argument("--mapping_json", type=str, required=True)
    parser.add_argument("--feature_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_num_worker", type=int, default=4)
    parser.add_argument("--caption_modality", type=str, default="audio", choices=["video", "audio"])
    parser.add_argument("--master_audio_dir", type=str, default=None)
    parser.add_argument("--audio_feature_dim", type=int, default=128)
    parser.add_argument("--model_name", type=str, default="contrastive")
    args = parser.parse_args()

    if args.caption_modality == "audio" and not args.master_audio_dir:
        raise ValueError("--master_audio_dir is required when --caption_modality=audio")

    dataset_Train = SoccerNetCaptions(
        path=args.SoccerNet_path,
        features=args.features,
        split=args.split_train,
        version=args.version,
        framerate=args.framerate,
        window_size=args.window_size_caption,
        mapping_json=args.mapping_json,
        feature_file=args.feature_file,
        caption_modality=args.caption_modality,
        master_audio_dir=args.master_audio_dir if args.caption_modality == "audio" else None,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset_Train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.max_num_worker,
        pin_memory=True,
        collate_fn=collate_fn_padd,
        persistent_workers=(args.max_num_worker > 0),
    )

    if args.caption_modality == "audio":
        model_encoder = Transformer_Audio(
            audio_feat_dim=args.audio_feature_dim,
            audio_d_model=512,
            audio_nhead=8,
            audio_num_layers=2,
            audio_length=args.window_size_caption * args.framerate,
        ).to("cuda")
    else:
        model_encoder = Transformer_Video(
            video_d_model=512,
            video_feat_dim=8576,  
            video_nhead=8,
            video_num_layers=2,
            video_length=args.window_size_caption * args.framerate,
        ).to("cuda")

    model_text = TextEncoder(vocab_size=1769, embed_dim=256, proj_dim=512).to("cuda")
    loss = ContrastiveLoss()
    optimizer = torch.optim.Adam(
        list(model_encoder.parameters()) + list(model_text.parameters()),
        lr=1e-4,
    )

    trainer(
        train_loader,
        model_video=model_encoder,   
        model_text=model_text,
        criterion=loss,
        optimizer=optimizer,
        scheduler=None,
        model_name=args.model_name,
    )
