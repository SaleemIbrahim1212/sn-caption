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
    
    '''Borrowed code from captioning.py so we can do contrastive learning on the encoder
        > The attempt here is to get the embeddings aligned with the captions so we get good representations from the transformer
        > Running this script tries to align the process, the loss function is copied from the CLIP paper which tackles the same iissue '''

    logging.info("start contrastive training")
    os.makedirs(os.path.join("models", model_name), exist_ok=True)
    for epoch in range(max_epochs):
        loss_training = train("contrastive", train_loader, model_video=model_video, model_text=model_text,
                      optimizer=optimizer, epoch=epoch + 1,
                      log_every_n_batches=log_every_n_batches, criterion=criterion)

        logging.info(f"Epoch {epoch+1}: train_loss={loss_training:.4f}")

        if scheduler is not None:
            scheduler.step(loss_training)
    return

def train(phase, dataloader, model_video,model_text,  criterion, optimizer, epoch, log_every_n_batches=50):
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
            }, os.path.join("models", "contrastive", "best.pth"))

    return losses.avg


class TextEncoder(nn.Module):
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
    parser.add_argument("--framerate", type=int, default=2)
    parser.add_argument("--window_size_caption", type=int, default=45)
    parser.add_argument("--mapping_json", type=str, required=True)
    parser.add_argument("--feature_file", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_num_worker", type=int, default=4)
    args = parser.parse_args()
    dataset_Train = SoccerNetCaptions(path=args.SoccerNet_path, features=args.features, split=args.split_train, version=args.version, framerate=args.framerate, window_size=args.window_size_caption, mapping_json=args.mapping_json, feature_file=args.feature_file)
    model_video = Transformer_Video(video_d_model=512, video_feat_dim=8576, video_nhead=8, video_num_layers=2, video_length=45).to('cuda')


    model_text = TextEncoder(vocab_size=1769, embed_dim=256, proj_dim=512).to('cuda')
    loss = ContrastiveLoss()
    optimizer = torch.optim.Adam(list(model_video.parameters()) + list(model_text.parameters()),lr=1e-4)
    dataset_Train = torch.utils.data.DataLoader(dataset_Train,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.max_num_worker, pin_memory=True, collate_fn=collate_fn_padd, persistent_workers=(args.max_num_worker > 0) )
    trainer(dataset_Train, model_video=model_video, model_text=model_text, criterion=loss, optimizer=optimizer , scheduler=None, model_name='contrastive' )
    
