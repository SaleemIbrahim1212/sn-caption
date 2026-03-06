import __future__

import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from netvlad import NetVLAD, NetRVLAD
from dataset import SOS_TOKEN, EOS_TOKEN
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import random
import math

### New sections: Positional encoding + Transformer aggregator + Late fusion
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerAggregator(nn.Module):
    """
    Transformer encoder aggregator with residual connection and LayerNorm.
    INPUT:  (batch_size, seq_len, input_size)
    OUTPUT: (batch_size, d_model)
    pool: "mean", "last", or "first_last" (first+last projected to d_model).
    """

    def __init__(
        self,
        input_size=512,
        d_model=256,
        nhead=8,
        num_encoder_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        pool="first_last",
    ):
        super(TransformerAggregator, self).__init__()
        self.d_model = d_model
        self.pool = pool
        self.input_proj = nn.Linear(input_size, d_model) if input_size != d_model else nn.Identity()
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.norm = nn.LayerNorm(d_model)
        if pool == "first_last":
            self.first_last_proj = nn.Linear(2 * d_model, d_model)
        else:
            self.first_last_proj = None

    def forward(self, x):
        # x: (B, T, input_size)
        x_proj = self.input_proj(x)  # (B, T, d_model)
        x = self.pos_encoder(x_proj)
        x = self.transformer_encoder(x)  # (B, T, d_model)
        x = self.norm(x + x_proj)  # residual + LayerNorm
        if self.pool == "mean":
            x = x.mean(dim=1)  # (B, d_model)
        elif self.pool == "last":
            x = x[:, -1, :]  # (B, d_model)
        elif self.pool == "first_last":
            x = self.first_last_proj(torch.cat([x[:, 0, :], x[:, -1, :]], dim=-1))  # (B, d_model)
        else:
            raise ValueError("TransformerAggregator pool must be 'mean', 'last', or 'first_last'")
        return x


class LateFusion(nn.Module):
    """
    Late fusion of video and (optional) audio embeddings.
    When audio_embeddings is None, only video is projected to hidden_size.
    When provided, video and audio are concatenated then projected.
    """

    def __init__(self, video_dim, hidden_size, audio_dim=0):
        super(LateFusion, self).__init__()
        self.audio_dim = audio_dim
        if audio_dim > 0:
            self.fusion = nn.Linear(video_dim + audio_dim, hidden_size)
        else:
            self.fusion = nn.Linear(video_dim, hidden_size)

    def forward(self, video_enc, audio_enc=None):
        # video_enc: (B, video_dim), audio_enc: (B, audio_dim) or None
        if audio_enc is None or self.audio_dim == 0:
            return self.fusion(video_enc)
        fused = torch.cat([video_enc, audio_enc], dim=-1)
        return self.fusion(fused)


class VideoEncoder(nn.Module):
    def __init__(self, input_size=512, vlad_k=64, window_size=15, framerate=2, pool="NetVLAD"):
        """
        INPUT: a Tensor of shape (batch_size,window_size,feature_size)
        OUTPUTS: a Tensor of shape (batch_size,hidden_size)
        """

        super(VideoEncoder, self).__init__()

        self.window_size_frame=window_size * framerate
        self.input_size = input_size
        self.framerate = framerate
        self.pool = pool
        self.vlad_k = vlad_k
        
        # are feature alread PCA'ed?
        if not self.input_size == 512:   
            self.feature_extractor = nn.Linear(self.input_size, 512)
            input_size = 512
            self.input_size = 512

        if self.pool == "MAX":
            self.pool_layer = nn.MaxPool1d(self.window_size_frame, stride=1)
            self.hidden_size = input_size

        if self.pool == "MAX++":
            self.pool_layer_before = nn.MaxPool1d(int(self.window_size_frame/2), stride=1)
            self.pool_layer_after = nn.MaxPool1d(int(self.window_size_frame/2), stride=1)
            self.hidden_size = input_size * 2


        if self.pool == "AVG":
            self.pool_layer = nn.AvgPool1d(self.window_size_frame, stride=1)
            self.hidden_size = input_size

        if self.pool == "AVG++":
            self.pool_layer_before = nn.AvgPool1d(int(self.window_size_frame/2), stride=1)
            self.pool_layer_after = nn.AvgPool1d(int(self.window_size_frame/2), stride=1)
            self.hidden_size = input_size *2


        elif self.pool == "NetVLAD":
            self.pool_layer = NetVLAD(cluster_size=self.vlad_k, feature_size=self.input_size,
                                            add_batch_norm=True)
            self.hidden_size = input_size * self.vlad_k

        elif self.pool == "NetVLAD++":
            self.pool_layer_before = NetVLAD(cluster_size=int(self.vlad_k/2), feature_size=self.input_size,
                                            add_batch_norm=True)
            self.pool_layer_after = NetVLAD(cluster_size=int(self.vlad_k/2), feature_size=self.input_size,
                                            add_batch_norm=True)
            self.hidden_size = input_size * self.vlad_k



        elif self.pool == "NetRVLAD":
            self.pool_layer = NetRVLAD(cluster_size=self.vlad_k, feature_size=self.input_size,
                                            add_batch_norm=True)
            self.hidden_size = input_size * self.vlad_k

        elif self.pool == "NetRVLAD++":
            self.pool_layer_before = NetRVLAD(cluster_size=int(self.vlad_k/2), feature_size=self.input_size,
                                            add_batch_norm=True)
            self.pool_layer_after = NetRVLAD(cluster_size=int(self.vlad_k/2), feature_size=self.input_size,
                                            add_batch_norm=True)
            self.hidden_size = input_size * self.vlad_k

        self.drop = nn.Dropout(p=0.4)

    def forward(self, inputs):
        # input_shape: (batch,frames,dim_features)


        BS, FR, IC = inputs.shape
        if not IC == 512:
            inputs = inputs.reshape(BS*FR, IC)
            inputs = self.feature_extractor(inputs)
            inputs = inputs.reshape(BS, FR, -1)

        # Temporal pooling operation
        if self.pool == "MAX" or self.pool == "AVG":
            inputs_pooled = self.pool_layer(inputs.permute((0, 2, 1))).squeeze(-1)

        elif self.pool == "MAX++" or self.pool == "AVG++":
            nb_frames_50 = int(inputs.shape[1]/2)    
            input_before = inputs[:, :nb_frames_50, :]        
            input_after = inputs[:, nb_frames_50:, :]  
            inputs_before_pooled = self.pool_layer_before(input_before.permute((0, 2, 1))).squeeze(-1)
            inputs_after_pooled = self.pool_layer_after(input_after.permute((0, 2, 1))).squeeze(-1)
            inputs_pooled = torch.cat((inputs_before_pooled, inputs_after_pooled), dim=1)


        elif self.pool == "NetVLAD" or self.pool == "NetRVLAD":
            inputs_pooled = self.pool_layer(inputs)

        elif self.pool == "NetVLAD++" or self.pool == "NetRVLAD++":
            nb_frames_50 = int(inputs.shape[1]/2)
            inputs_before_pooled = self.pool_layer_before(inputs[:, :nb_frames_50, :])
            inputs_after_pooled = self.pool_layer_after(inputs[:, nb_frames_50:, :])
            inputs_pooled = torch.cat((inputs_before_pooled, inputs_after_pooled), dim=1)

        return inputs_pooled

class DecoderRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, vocab_size, num_layers=2):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.ft_extactor_1 = nn.Linear(input_size, hidden_size)
        self.ft_extactor_2 = nn.Linear(hidden_size, hidden_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.4)
        self.activation = nn.ReLU()
        self.num_layers = num_layers
    
    def forward(self, features, captions, lengths):
        #Features extraction of video encoder
        features = self.ft_extactor_2(self.activation(self.dropout(self.ft_extactor_1(features))))
        features = torch.stack([features]*self.num_layers)
        #Embdedding
        captions = self.embed(captions)
        #To reduce the computation, we pack padd sequences (lengths must be on CPU)
        captions = pack_padded_sequence(captions, lengths.cpu(), batch_first=True, enforce_sorted=False)
        #Video encoder features are used as initial states
        hiddens, _ = self.lstm(captions, (features, features))
        outputs = self.fc(hiddens[0])
        return outputs
    
    def sample(self, features, max_seq_length, temperature=0.0):
        """Generate caption. temperature=0.0 is greedy (argmax); temperature>0 samples from softmax(logits/temperature)."""
        sampled_ids = []
        #Features extraction of video encoder
        features = self.ft_extactor_2(self.activation(self.dropout(self.ft_extactor_1(features))))
        features = torch.stack([features]*self.num_layers)
        #Video encoder features are used as initial states
        states = (features, features)
        #Start token
        inputs = torch.tensor([[SOS_TOKEN]], device=features.device)
        inputs = self.embed(inputs)
        for i in range(max_seq_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.fc(hiddens.squeeze(1))
            if temperature is not None and temperature > 0:
                probs = F.softmax(outputs / temperature, dim=1)
                predicted = torch.multinomial(probs, 1).squeeze(1)
            else:
                _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            if predicted == EOS_TOKEN:
                break
            inputs = self.embed(predicted).unsqueeze(1)
        sampled_ids = torch.cat(sampled_ids)
        return sampled_ids

class Video2Caption(nn.Module):
    def __init__(self, vocab_size, weights=None, input_size=512, vlad_k=64, window_size=15, framerate=2, pool="NetVLAD", embed_size=256, hidden_size=512, teacher_forcing_ratio=1, num_layers=2, max_seq_length=50, weights_encoder=None, freeze_encoder=False):
        super(Video2Caption, self).__init__()
        self.encoder = VideoEncoder(input_size, vlad_k, window_size, framerate, pool)
        self.decoder = DecoderRNN(self.encoder.hidden_size, embed_size, hidden_size, vocab_size, num_layers)
        self.load_weights(weights=weights)
        self.load_encoder(weights_encoder=weights_encoder, freeze_encoder=freeze_encoder)
        self.vocab_size = vocab_size
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def load_weights(self, weights=None):
        if(weights is not None):
            print("=> loading checkpoint '{}'".format(weights))
            checkpoint = torch.load(weights, map_location='cpu')
            self.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(weights, checkpoint['epoch']))
            
    def load_encoder(self, weights_encoder=None, freeze_encoder=False):
        if(weights_encoder is not None):
            print("=> loading encoder '{}'".format(weights_encoder))
            checkpoint = torch.load(weights_encoder, map_location=torch.device('cpu'))
            self.load_state_dict({k :v for k, v in checkpoint['state_dict'].items() if "encoder." in k}, strict=False)
            print("=> loaded checencoderkpoint '{}' (epoch {})"
                  .format(weights_encoder, checkpoint['epoch']))
            
            if freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False
    
    def forward(self, features, captions, lengths, audio_embeddings=None, return_encoder_output=False):
        # audio_embeddings ignored (kept for API compatibility with Video2CaptionWithTransformer)
        encoded = self.encoder(features)
        batch_size = captions.size(0)
        captions = captions[:, :-1]  # Remove last word in caption to use as input
        use_teacher_forcing = random.random() < self.teacher_forcing_ratio
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            decoder_input = captions
            decoder_output = self.decoder(encoded, decoder_input, lengths)
        else:
            decoder_input = captions[:, 0].unsqueeze(1)  # <start> token
            decoder_output = torch.zeros(batch_size, captions.size(1), self.vocab_size, device=captions.device)
            for t in range(0, captions.size(1)):
                # Pass through decoder
                decoder_output_t = self.decoder(encoded, decoder_input, torch.ones_like(lengths))
                decoder_output[:, t, :] = decoder_output_t
                # Get next input from highest predicted token
                _, topi = decoder_output_t.topk(1)
                decoder_input = topi.detach()  # detach from history as input
            decoder_output = pack_padded_sequence(decoder_output, lengths.cpu(), batch_first=True, enforce_sorted=False)[0]
        if return_encoder_output:
            return decoder_output, encoded
        return decoder_output

    def sample(self, features, max_seq_length=70, temperature=0.0):
        features = self.encoder(features.unsqueeze(0))
        return self.decoder.sample(features, max_seq_length, temperature=temperature)


### New section: Transformer for captioning
class Video2CaptionWithTransformer(nn.Module):
    """
    Captioning model that uses a transformer aggregator (instead of NetVLAD/etc.)
    and late fusion for optional audio embeddings, then LSTM decoder.
    Spotting sub-module is unchanged (still uses Video2Spot with VideoEncoder).
    """

    def __init__(
        self,
        vocab_size,
        weights=None,
        input_size=512,
        window_size=15,
        framerate=2,
        # Transformer aggregator
        d_model=256,
        nhead=8,
        num_encoder_layers=2,
        dim_feedforward=512,
        encoder_dropout=0.1,
        encoder_pool="first_last",
        # Late fusion (audio_embed_dim=0 means video-only for now)
        audio_embed_dim=0,
        # Decoder (same as original)
        embed_size=256,
        hidden_size=512,
        teacher_forcing_ratio=1,
        num_layers=2,
        max_seq_length=50,
        freeze_encoder=False,
    ):
        super(Video2CaptionWithTransformer, self).__init__()
        self.video_aggregator = TransformerAggregator(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=encoder_dropout,
            pool=encoder_pool,
        )
        self.fusion = LateFusion(
            video_dim=d_model,
            hidden_size=hidden_size,
            audio_dim=audio_embed_dim,
        )
        self.decoder = DecoderRNN(hidden_size, embed_size, hidden_size, vocab_size, num_layers)
        self.vocab_size = vocab_size
        self.teacher_forcing_ratio = teacher_forcing_ratio

        if freeze_encoder:
            for param in self.video_aggregator.parameters():
                param.requires_grad = False

        self.load_weights(weights=weights)

    def load_weights(self, weights=None):
        if weights is not None:
            print("=> loading checkpoint '{}'".format(weights))
            checkpoint = torch.load(weights, map_location="cpu")
            self.load_state_dict(checkpoint["state_dict"], strict=False)
            print("=> loaded checkpoint '{}' (epoch {})".format(weights, checkpoint["epoch"]))

    def _encode(self, video_features, audio_embeddings=None):
        video_enc = self.video_aggregator(video_features)  # (B, d_model)
        fused = self.fusion(video_enc, audio_embeddings)    # (B, hidden_size)
        return fused

    def forward(self, features, captions, lengths, audio_embeddings=None, return_encoder_output=False):
        fused = self._encode(features, audio_embeddings)
        batch_size = captions.size(0)
        captions = captions[:, :-1]
        use_teacher_forcing = random.random() < self.teacher_forcing_ratio
        if use_teacher_forcing:
            decoder_input = captions
            decoder_output = self.decoder(fused, decoder_input, lengths)
        else:
            decoder_input = captions[:, 0].unsqueeze(1)
            decoder_output = torch.zeros(
                batch_size, captions.size(1), self.vocab_size, device=captions.device
            )
            for t in range(captions.size(1)):
                decoder_output_t = self.decoder(fused, decoder_input, torch.ones_like(lengths))
                decoder_output[:, t, :] = decoder_output_t
                _, topi = decoder_output_t.topk(1)
                decoder_input = topi.detach()
            decoder_output = pack_padded_sequence(
                decoder_output, lengths.cpu(), batch_first=True, enforce_sorted=False
            )[0]
        if return_encoder_output:
            return decoder_output, fused
        return decoder_output

    def sample(self, features, max_seq_length=70, audio_embeddings=None, temperature=0.0):
        if features.dim() == 2:
            features = features.unsqueeze(0)
        fused = self._encode(features, audio_embeddings)
        return self.decoder.sample(fused, max_seq_length, temperature=temperature)

class Video2Spot(nn.Module):
    def __init__(self, weights=None, input_size=512, num_classes=17, vlad_k=64, window_size=15, framerate=2, pool="NetVLAD", weights_encoder=None, freeze_encoder=False):
        """
        INPUT: a Tensor of shape (batch_size,window_size,feature_size)
        OUTPUTS: a Tensor of shape (batch_size,num_classes+1)
        """

        super(Video2Spot, self).__init__()
        self.encoder = VideoEncoder(input_size, vlad_k, window_size, framerate, pool)
        self.head = nn.Linear(self.encoder.hidden_size, num_classes+1)
        self.drop = nn.Dropout(p=0.4)
        self.sigm = nn.Sigmoid()
        self.load_weights(weights=weights)
        self.load_encoder(weights_encoder=weights_encoder, freeze_encoder=freeze_encoder)

    def load_weights(self, weights=None):
        if(weights is not None):
            print("=> loading checkpoint '{}'".format(weights))
            checkpoint = torch.load(weights, map_location='cpu')
            self.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(weights, checkpoint['epoch']))
    
    def load_encoder(self, weights_encoder=None, freeze_encoder=False):
        if(weights_encoder is not None):
            print("=> loading encoder '{}'".format(weights_encoder))
            checkpoint = torch.load(weights_encoder, map_location=torch.device('cpu'))
            self.load_state_dict({k :v for k, v in checkpoint['state_dict'].items() if "encoder." in k}, strict=False)
            print("=> loaded checencoderkpoint '{}' (epoch {})"
                  .format(weights_encoder, checkpoint['epoch']))
            
            if freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False

    def forward(self, inputs):
        # input_shape: (batch,frames,dim_features)
        inputs_pooled = self.encoder(inputs)
        # Extra FC layer with dropout and sigmoid activation
        output = self.sigm(self.head(self.drop(inputs_pooled)))

        return output


if __name__ == "__main__":

    model = Video2Spot(pool="NetVLAD++", num_classes=1, framerate=2, window_size=15)
    model.load_encoder("Benchmarks/TemporallyAwarePooling/models/ResNET_TF2_PCA512-NetVLAD++-nms-15-window-15-teacher-1-28-02-2023_12-10-03/caption/model.pth.tar")
    print(model.encoder.pool_layer_before.clusters.requires_grad)

    BS =5
    T = 15
    framerate= 2
    D = 512
    pool = "NetRVLAD++"
    vocab_size=100
    #model = VideoEncoder(pool=pool, input_size=D, framerate=framerate, window_size=T)
    model = Video2Caption(vocab_size, pool=pool, input_size=D, framerate=framerate, window_size=T)
    criterion = nn.CrossEntropyLoss()
    print(model)
    inp = torch.rand([BS,T*framerate,D])
    DATA = [
        [0, 0],
        [1, 3, 2],
        [1, 4, 5, 2],
        [1, 6, 7, 8, 9, 2],
        [1, 4, 6, 2, 9, 6, 2],
    ]
    # DATA = [
    #     [0, 0],
    #     [0, 0],
    #     [0, 0],
    #     [0, 0],
    #     [0, 0],
    # ]
    # need torch tensors for torch's pad_sequence(); this could be a part of e.g. dataset's __getitem__ instead
    captions = list(map(lambda x: torch.tensor(x), DATA))
    lengths = torch.tensor(list(map(len, DATA))).long()
    lengths = lengths - 1
    captions = pad_sequence(captions, batch_first=True)
    target = captions[:, 1:]
    target = pack_padded_sequence(target, lengths, batch_first=True, enforce_sorted=False)[0]
    mask = pack_padded_sequence(captions != 0, lengths, batch_first=True, enforce_sorted=False)[0]
    print("INPUT SHAPE :")
    print(inp.shape, captions.shape)
    output = model(inp, captions, lengths, teacher_forcing_ratio=1)
    print(criterion(target, output))
    print("OUTPUT SHAPE :")
    print(output)
    output = model(inp, captions, lengths, teacher_forcing_ratio=0)
    print(criterion(target, output))
    print("OUTPUT SHAPE :")
    print(output)
    print("TARGET")
    print(target)
    print("MASK")
    print(mask)

    print("==============SAMPLING===============")
    print(model.sample(inp[0]))