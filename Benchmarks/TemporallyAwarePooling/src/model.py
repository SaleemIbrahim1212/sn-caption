import __future__

import logging
import os

import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from netvlad import NetVLAD, NetRVLAD
from transformer import Transformer_Audio, Transformer_Video, Transformer
from dataset import SOS_TOKEN, EOS_TOKEN
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import random


def load_contrastive_video_weights(
    pooling_layer,
    contrastive_weights_path,
    freeze_contrastive_encoder,
    unfreeze_contrastive_projection,
    multimodal_fusion=False,
):
    """Load sbertcontrastive (or compatible) weights into video_proj / embedding_video / video_transformer.

    For ``Transformer_Video``, the whole ``pooling_layer`` is the video encoder. For multimodal ``Transformer``,
    only keys present in both checkpoint and module are loaded (typically ``video_*``); audio stays random init.
    Skips tensors with shape mismatch (e.g. different ``video_feat_dim`` on ``video_proj``).
    """
    if not contrastive_weights_path:
        return
    if not os.path.exists(contrastive_weights_path):
        logging.warning("Could not find contrastive checkpoint at %s; skipping preload.", contrastive_weights_path)
        return

    logging.info("Loading contrastive video weights from %s", contrastive_weights_path)
    checkpoint = torch.load(contrastive_weights_path, map_location=torch.device("cpu"))
    state_dict = checkpoint["model_video"] if isinstance(checkpoint, dict) and "model_video" in checkpoint else checkpoint

    target_sd = pooling_layer.state_dict()
    compatible = {}
    for k, v in state_dict.items():
        if k not in target_sd:
            continue
        if target_sd[k].shape != v.shape:
            logging.warning(
                "Skipping contrastive key %s: checkpoint shape %s vs model %s",
                k,
                tuple(v.shape),
                tuple(target_sd[k].shape),
            )
            continue
        compatible[k] = v

    missing, unexpected = pooling_layer.load_state_dict(compatible, strict=False)
    if compatible:
        logging.info(
            "Contrastive load: merged %d tensors (strict=False; missing keys expected for audio in fusion).",
            len(compatible),
        )
    else:
        logging.warning("No contrastive tensors matched the video branch; check architecture and input dims.")

    if not freeze_contrastive_encoder:
        return

    if multimodal_fusion:
        for name, param in pooling_layer.named_parameters():
            if name.startswith("video_"):
                param.requires_grad = False
        if unfreeze_contrastive_projection and hasattr(pooling_layer, "video_transformer"):
            logging.info("Unfreezing last video transformer encoder block (fusion model).")
            for param in pooling_layer.video_transformer.layers[-1].parameters():
                param.requires_grad = True
    else:
        for param in pooling_layer.parameters():
            param.requires_grad = False
        if unfreeze_contrastive_projection and hasattr(pooling_layer, "video_transformer"):
            logging.info("Unfreezing last video transformer encoder block.")
            for param in pooling_layer.video_transformer.layers[-1].parameters():
                param.requires_grad = True


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

class MultimodalTransformerCaption(nn.Module):
    def __init__(self, input_size=512, window_size=15, framerate=2, pool="Transformer_Video", contrastive_weights_path=None, freeze_contrastive_encoder=True, unfreeze_contrastive_projection=False, audio_feat_dim=None):
        '''
        Same as the video encoder but we have audio and a transformer now
        Using the same recepie as above but tweaking to incorporate audio as well

        Can do the following 
        Transformer on Audio embeddings only 
        Transformer on Video embeddings only 
        Transformer on Audio and Video embeddings only 
        All the above use late fusion
        '''
        super(MultimodalTransformerCaption, self).__init__()

        self.window_size_frame=window_size * framerate
        self.input_size = input_size
        self.audio_feat_dim = audio_feat_dim if audio_feat_dim is not None else input_size
        self.framerate = framerate
        self.pool = pool
        
        if self.pool == "Transformer_Video":
            self.pooling_layer = Transformer_Video(video_feat_dim=self.input_size, video_d_model=512, video_nhead=8, video_num_layers=2, video_length=self.window_size_frame)
            load_contrastive_video_weights(
                self.pooling_layer,
                contrastive_weights_path,
                freeze_contrastive_encoder,
                unfreeze_contrastive_projection,
                multimodal_fusion=False,
            )
            self.hidden_size = 512
        elif self.pool == "Transformer_Audio":
            self.pooling_layer = Transformer_Audio(
                audio_feat_dim=self.audio_feat_dim,
                audio_d_model=512,
                audio_nhead=8,
                audio_num_layers=2,
                audio_length=self.window_size_frame,
            )
            self.hidden_size = 512

        elif self.pool == "Transformer":
            self.pooling_layer = Transformer(
                audio_feat_dim=self.audio_feat_dim,
                audio_d_model=512,
                audio_nhead=8,
                audio_num_layers=2,
                audio_length=self.window_size_frame,
                video_feat_dim=self.input_size,
                video_d_model=512,
                video_nhead=8,
                video_num_layers=2,
                video_length=self.window_size_frame,
            )
            load_contrastive_video_weights(
                self.pooling_layer,
                contrastive_weights_path,
                freeze_contrastive_encoder,
                unfreeze_contrastive_projection,
                multimodal_fusion=True,
            )
            self.hidden_size = 1024
    def forward(self, audio_feats=None, video_feats=None):
        if audio_feats is None and video_feats is None:
            raise NotImplementedError
        
        elif (video_feats is not None and  audio_feats is None ):
            '''If we only have video features '''
            cls_video_token, encoder_out = self.pooling_layer(video_feats  = video_feats)
            return cls_video_token, encoder_out
        
        elif (audio_feats is not None and  video_feats is None):
            '''If we only have audio features'''
            cls_audio_token, encoder_out = self.pooling_layer(audio_feats)
            return cls_audio_token, encoder_out

        elif (video_feats is not None  and audio_feats is not None):
            '''if we have both features — pooled init + full concat sequence for decoder attention'''
            cls_audio_token, cls_video_token, encoder_seq = self.pooling_layer(audio_feats, video_feats)
            final_token = torch.concat([cls_audio_token, cls_video_token], dim=1)
            return final_token, encoder_seq
        
        
        

        


class DecoderRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, vocab_size, num_layers=2, word_dropout=0.4):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.input = input_size 
        self.ft_extactor_1 = nn.Linear(input_size, hidden_size)
        self.ft_extactor_2 = nn.Linear(hidden_size, hidden_size)
        self.lstm = nn.LSTM(embed_size +512, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.4)
        self.activation = nn.ReLU()
        self.num_layers = num_layers
        self.word_dropout = word_dropout
    
    def forward(self, features, captions, lengths, encoder_outputs):
        #Features extraction of video encoder
        #pass_to_lstm = torch.unsqueeze(features, dim=1)
        features = self.ft_extactor_2(self.activation(self.dropout(self.ft_extactor_1(features)))) 
        features = torch.stack([features]*self.num_layers)
        #Embdedding
        captions_tokens = captions
        captions = self.embed(captions_tokens)
        if self.training and self.word_dropout > 0:
            drop_mask = torch.rand(captions_tokens.shape, device=captions.device) < self.word_dropout
            drop_mask = drop_mask & (captions_tokens != 0) & (captions_tokens != SOS_TOKEN)
            captions = captions.masked_fill(drop_mask.unsqueeze(-1), 0.0)
        states = (features, features)
        B,L, E  = captions.shape 
        logits  = [] 
        for i in range (L):
            word = captions[:, i , :]
            query  = states[0][-1].unsqueeze(1) # B, 1, 512 
            context = query @  encoder_outputs.permute(0,2,1) / (512 ** 0.5) # B, 1, 512 * B, 45, 512 
            logs = context.softmax(dim = 2) #b,1 ,45 
            final_context_vector  = logs @ encoder_outputs 
            final_context_vector = final_context_vector.squeeze(1)
            inputs  = torch.concat([word, final_context_vector] , dim=1)
            hiddens, states = self.lstm(inputs.unsqueeze(1), states) 
            logit = self.fc(hiddens.squeeze(1))
            logits.append(logit)
        outputs = torch.stack(logits, dim=1)
        outputs = pack_padded_sequence(outputs, lengths, batch_first=True, enforce_sorted=False)[0]
        return outputs


            

    
    def sample(self, features, encoder_outputs, max_seq_length=50):
        sampled_ids = []
        features = self.ft_extactor_2(self.activation(self.dropout(self.ft_extactor_1(features))))
        features = torch.stack([features] * self.num_layers)
        states = (features, features)

        word = self.embed(torch.tensor([[SOS_TOKEN]], device=features.device)).squeeze(1)

        for _ in range(max_seq_length):
            query = states[0][-1].unsqueeze(1)
            context = query @ encoder_outputs.permute(0, 2, 1) / (512 ** 0.5)
            logs = context.softmax(dim=2)
            final_context_vector = (logs @ encoder_outputs).squeeze(1)
            inputs = torch.cat([word, final_context_vector], dim=1)
            hiddens, states = self.lstm(inputs.unsqueeze(1), states)
            logit = self.fc(hiddens.squeeze(1))
            predicted = logit.argmax(1)
            sampled_ids.append(predicted)
            if predicted.item() == EOS_TOKEN:
                break
            word = self.embed(predicted)

        return torch.cat(sampled_ids)


class DualModalDecoderRNN(nn.Module):
    """Separate LSTM decoders for audio and video, each attending only to its encoder stream.

    Pooled audio/video tokens initialize the respective LSTMs. Word embeddings are shared.
    Per-step logits are produced by fusing the two LSTM hidden states with a linear layer.
    """

    def __init__(
        self,
        embed_size,
        hidden_size,
        vocab_size,
        num_layers=2,
        word_dropout=0.4,
        context_dim=512,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.word_dropout = word_dropout
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.dropout = nn.Dropout(0.4)
        self.activation = nn.ReLU()

        def _init_mlp():
            return nn.ModuleDict(
                {
                    "ft1": nn.Linear(context_dim, hidden_size),
                    "ft2": nn.Linear(hidden_size, hidden_size),
                }
            )

        self.audio_init = _init_mlp()
        self.video_init = _init_mlp()
        self.lstm_a = nn.LSTM(
            embed_size + context_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.lstm_v = nn.LSTM(
            embed_size + context_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(2 * hidden_size, vocab_size)

    def _pooled_to_lstm_state(self, feat, which):
        m = self.audio_init if which == "a" else self.video_init
        x = m["ft2"](self.activation(self.dropout(m["ft1"](feat))))
        x = torch.stack([x] * self.num_layers, dim=0)
        return (x, x)

    def initial_states(self, feat_a, feat_v):
        return self._pooled_to_lstm_state(feat_a, "a"), self._pooled_to_lstm_state(feat_v, "v")

    def _attn(self, query, encoder_out):
        context = query @ encoder_out.permute(0, 2, 1) / (self.context_dim ** 0.5)
        w = context.softmax(dim=2)
        return (w @ encoder_out).squeeze(1)

    def decode_step(self, word_embed, states_a, states_v, enc_a, enc_v):
        qa = states_a[0][-1].unsqueeze(1)
        qv = states_v[0][-1].unsqueeze(1)
        ctx_a = self._attn(qa, enc_a)
        ctx_v = self._attn(qv, enc_v)
        in_a = torch.cat([word_embed, ctx_a], dim=1)
        in_v = torch.cat([word_embed, ctx_v], dim=1)
        ha, states_a = self.lstm_a(in_a.unsqueeze(1), states_a)
        hv, states_v = self.lstm_v(in_v.unsqueeze(1), states_v)
        logits = self.fc(torch.cat([ha.squeeze(1), hv.squeeze(1)], dim=1))
        return states_a, states_v, logits

    def forward(self, feat_a, feat_v, captions, lengths, enc_a, enc_v):
        states_a, states_v = self.initial_states(feat_a, feat_v)

        captions_tokens = captions
        captions = self.embed(captions_tokens)
        if self.training and self.word_dropout > 0:
            drop_mask = torch.rand(captions_tokens.shape, device=captions.device) < self.word_dropout
            drop_mask = drop_mask & (captions_tokens != 0) & (captions_tokens != SOS_TOKEN)
            captions = captions.masked_fill(drop_mask.unsqueeze(-1), 0.0)

        _, L, _ = captions.shape
        logits = []
        for i in range(L):
            word = captions[:, i, :]
            states_a, states_v, logit = self.decode_step(word, states_a, states_v, enc_a, enc_v)
            logits.append(logit)
        outputs = torch.stack(logits, dim=1)
        return pack_padded_sequence(outputs, lengths, batch_first=True, enforce_sorted=False)[0]

    def sample(self, feat_a, feat_v, enc_a, enc_v, max_seq_length=50):
        sampled_ids = []
        states_a, states_v = self.initial_states(feat_a, feat_v)
        word = self.embed(torch.tensor([[SOS_TOKEN]], device=feat_a.device)).squeeze(1)

        for _ in range(max_seq_length):
            states_a, states_v, logit = self.decode_step(word, states_a, states_v, enc_a, enc_v)
            predicted = logit.argmax(1)
            sampled_ids.append(predicted)
            if predicted.item() == EOS_TOKEN:
                break
            word = self.embed(predicted)

        return torch.cat(sampled_ids)


class Video2Caption(nn.Module):
    def __init__(self, vocab_size, weights=None, input_size=512, vlad_k=64, window_size=15, framerate=2, pool="NetVLAD", embed_size=256, hidden_size=512, teacher_forcing_ratio=1, num_layers=2, max_seq_length=50, weights_encoder=None, freeze_encoder=False, word_dropout=0.4):
        super(Video2Caption, self).__init__()
        self.encoder = VideoEncoder(input_size, vlad_k, window_size, framerate, pool)
        self.decoder = DecoderRNN(self.encoder.hidden_size, embed_size, hidden_size, vocab_size, num_layers, word_dropout=word_dropout)
        self.load_weights(weights=weights)
        self.load_encoder(weights_encoder=weights_encoder, freeze_encoder=freeze_encoder)
        self.vocab_size = vocab_size
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def load_weights(self, weights=None):
        if(weights is not None):
            print("=> loading checkpoint '{}'".format(weights))
            checkpoint = torch.load(weights)
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

    def _attention_encoder_outputs(self, features):
        """[B, H] pooled encoder output -> [B, T, 512] for DecoderRNN attention."""
        B, H = features.shape[0], features.shape[1]
        d = self.encoder.input_size
        k = self.encoder.vlad_k
        if H == d:
            return features.unsqueeze(1)
        if H == k * d:
            return features.view(B, k, d)
        return features.unsqueeze(1)

    def forward(self, features, captions, lengths):
        features = self.encoder(features)
        enc_mem = self._attention_encoder_outputs(features)
        batch_size = captions.size(0)
        captions = captions[:, :-1]  # Remove last word in caption to use as input
        decoder_lengths = [max(int(length) - 1, 1) for length in lengths]
        use_teacher_forcing = random.random() < self.teacher_forcing_ratio
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            
            decoder_input = captions
            decoder_output = self.decoder(features, decoder_input, decoder_lengths, enc_mem)
        else:
            features_extracted = self.decoder.ft_extactor_2(self.decoder.activation(self.decoder.dropout(self.decoder.ft_extactor_1(features))))
            features_extracted = torch.stack([features_extracted] * self.decoder.num_layers)
            states = (features_extracted, features_extracted)
            decoder_input = captions[:, 0].unsqueeze(1)  # <start> token
            encoder_outputs = enc_mem
            decoder_output = torch.zeros(batch_size, captions.size(1), self.vocab_size, device=captions.device)
            for t in range(0, captions.size(1)):
                word = self.decoder.embed(decoder_input).squeeze(1)
                query = states[0][-1].unsqueeze(1)
                context = query @ encoder_outputs.permute(0, 2, 1) / (512 ** 0.5)
                logs = context.softmax(dim=2)
                final_context_vector = (logs @ encoder_outputs).squeeze(1)
                inputs = torch.cat([word, final_context_vector], dim=1)
                hiddens, states = self.decoder.lstm(inputs.unsqueeze(1), states)
                outputs = self.decoder.fc(hiddens.squeeze(1))
                # Pass through decoder
                #decoder_output_t = self.decoder(features, decoder_input, torch.ones_like(lengths))
                _, predicted = outputs.max(1)
                decoder_input = predicted.unsqueeze(1)
                decoder_output[:, t, :] = outputs  
            decoder_output = pack_padded_sequence(decoder_output, decoder_lengths, batch_first=True, enforce_sorted=False)[0]
        return decoder_output

    def sample(self, features, max_seq_length=70):
        features = self.encoder(features.unsqueeze(0))
        enc_mem = self._attention_encoder_outputs(features)
        return self.decoder.sample(features, enc_mem, max_seq_length)

class SoccerNetTransformerCaption(nn.Module):
    """A transformer-based encoder-decoder model for generating captions of soccer events.
    Combines a multimodal transformer encoder (supporting video, audio, or both) with
    an LSTM decoder that uses attention over encoder outputs to produce natural language
    descriptions of soccer actions.
    Args:
        vocab_size (int): Size of the output vocabulary.
        weights (str, optional): Path to full model weights (currently unused).
        input_size (int): Dimensionality of input features. Default: 512.
        window_size (int): Number of frames in the temporal window. Default: 15.
        framerate (int): Video framerate used for feature extraction. Default: 2.
        pool (str): Encoder pooling strategy. One of 'Transformer_Video',
            'Transformer_Audio', or a multimodal variant. Default: 'Transformer_Video'.
        embed_size (int): Word embedding dimension for the decoder. Default: 256.
        hidden_size (int): Hidden state size of the decoder LSTM. Default: 512.
        teacher_forcing_ratio (float): Initial probability of using teacher forcing
            during training. Default: 1.0.
        teacher_forcing_decay (float): Amount to reduce the teacher forcing ratio
            per forward pass. Default: 0.0.
        teacher_forcing_min (float): Minimum teacher forcing ratio after decay. Default: 0.5.
        num_layers (int): Number of stacked LSTM layers in the decoder. Default: 2.
        max_seq_length (int): Maximum caption length during generation. Default: 50.
        weights_encoder (str, optional): Path to pretrained encoder checkpoint.
        freeze_encoder (bool): If True, freeze all encoder parameters. Default: False.
        contrastive_weights_path (str, optional): Path to contrastive pretraining weights
            for the encoder.
        freeze_contrastive_encoder (bool): If True, freeze the contrastive encoder
            backbone. Default: True.
        unfreeze_contrastive_projection (bool): If True, keep the contrastive projection
            head trainable even when the encoder is frozen. Default: False.
        word_dropout (float): Dropout rate applied to word embeddings. Default: 0.4.
        use_dual_lstm_decoder (bool): If True and pool is ``Transformer`` (audio+video),
            use :class:`DualModalDecoderRNN` instead of a single :class:`DecoderRNN`. Default: False.

    Forward pass:
        Encodes input features via the selected modality, then decodes captions using
        either teacher forcing (ground-truth tokens as decoder input) or autoregressive
        greedy decoding with scaled dot-product attention over encoder outputs. Returns
        packed padded decoder logits.

    Inference:
        Use ``sample()`` to autoregressively generate a caption given raw video/audio
        features.

    When ``use_dual_lstm_decoder`` is True (with multimodal pool ``Transformer``), the
    caption decoder uses :class:`DualModalDecoderRNN`: two LSTMs with modality-specific
    attention, fused logits via a linear layer on the concatenated hidden states.
    """
    def __init__(self, vocab_size, weights=None, input_size=512, window_size=15, framerate=2, pool="Transformer_Video", embed_size=256, hidden_size=512, teacher_forcing_ratio=1.0, teacher_forcing_decay=0.0, teacher_forcing_min=0.5, num_layers=2, max_seq_length=50, weights_encoder=None, freeze_encoder=False, contrastive_weights_path=None, freeze_contrastive_encoder=True, unfreeze_contrastive_projection=False, word_dropout=0.4, audio_input_size=None, use_dual_lstm_decoder=False):
        super(SoccerNetTransformerCaption, self).__init__()
        if use_dual_lstm_decoder and pool != "Transformer":
            raise ValueError(
                "use_dual_lstm_decoder=True requires multimodal encoder (pool=='Transformer'; use --transformer_modality both)."
            )
        self.use_dual_decoder = bool(use_dual_lstm_decoder)
        self.encoder = MultimodalTransformerCaption(
            input_size=input_size,
            window_size=window_size,
            framerate=framerate,
            pool=pool,
            contrastive_weights_path=contrastive_weights_path,
            freeze_contrastive_encoder=freeze_contrastive_encoder,
            unfreeze_contrastive_projection=unfreeze_contrastive_projection,
            audio_feat_dim=audio_input_size,
        )
        if self.use_dual_decoder:
            self.decoder = DualModalDecoderRNN(
                embed_size,
                hidden_size,
                vocab_size,
                num_layers,
                word_dropout=word_dropout,
            )
        else:
            self.decoder = DecoderRNN(self.encoder.hidden_size, embed_size , hidden_size, vocab_size, num_layers, word_dropout=word_dropout)
        #self.load_weights(weights=weights)
        self.load_encoder(weights_encoder=weights_encoder, freeze_encoder=freeze_encoder)
        self.vocab_size = vocab_size
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.teacher_forcing_decay = teacher_forcing_decay
        self.teacher_forcing_min = teacher_forcing_min
            
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
    
    def forward(self, features_video, features_audio, captions, lengths):
        if (self.encoder.pool == "Transformer_Video"):
            '''Get the cls token just for the video'''
            features, encoder_out = self.encoder(video_feats = features_video)
        elif (self.encoder.pool == "Transformer_Audio"):
            '''get the cls token just for the audio'''
            features, encoder_out = self.encoder(audio_feats=features_audio)
        else:
            '''get pooled fusion + time-concatenated AV encoder states for attention'''
            features, encoder_out = self.encoder(audio_feats=features_audio, video_feats=features_video)
        batch_size = captions.size(0)
        captions = captions[:, :-1]  # Remove last word in caption to use as input
        decoder_lengths = [max(int(length) - 1, 1) for length in lengths]
        use_teacher_forcing = random.random() < self.teacher_forcing_ratio
        if self.training and self.teacher_forcing_decay > 0:
            self.teacher_forcing_ratio = max(self.teacher_forcing_min, self.teacher_forcing_ratio - self.teacher_forcing_decay)
        if use_teacher_forcing:
            decoder_input = captions
            if self.use_dual_decoder:
                half_d = features.size(-1) // 2
                feat_a, feat_v = features[:, :half_d], features[:, half_d:]
                tm = encoder_out.size(1) // 2
                enc_a, enc_v = encoder_out[:, :tm], encoder_out[:, tm:]
                decoder_output = self.decoder(feat_a, feat_v, decoder_input, decoder_lengths, enc_a, enc_v)
            else:
                decoder_output = self.decoder(features, decoder_input, decoder_lengths, encoder_out)
        else:
            if self.use_dual_decoder:
                half_d = features.size(-1) // 2
                feat_a, feat_v = features[:, :half_d], features[:, half_d:]
                tm = encoder_out.size(1) // 2
                enc_a, enc_v = encoder_out[:, :tm], encoder_out[:, tm:]
                states_a, states_v = self.decoder.initial_states(feat_a, feat_v)
                decoder_input = captions[:, 0].unsqueeze(1)
                decoder_output = torch.zeros(batch_size, captions.size(1), self.vocab_size, device=captions.device)
                for t in range(0, captions.size(1)):
                    word = self.decoder.embed(decoder_input).squeeze(1)
                    states_a, states_v, outputs = self.decoder.decode_step(
                        word, states_a, states_v, enc_a, enc_v
                    )
                    _, predicted = outputs.max(1)
                    decoder_input = predicted.unsqueeze(1)
                    decoder_output[:, t, :] = outputs
                decoder_output = pack_padded_sequence(
                    decoder_output, decoder_lengths, batch_first=True, enforce_sorted=False
                )[0]
            else:
                features_extracted = self.decoder.ft_extactor_2(self.decoder.activation(self.decoder.dropout(self.decoder.ft_extactor_1(features))))
                features_extracted = torch.stack([features_extracted] * self.decoder.num_layers)
                states = (features_extracted, features_extracted)
                decoder_input = captions[:, 0].unsqueeze(1)
                decoder_output = torch.zeros(batch_size, captions.size(1), self.vocab_size, device=captions.device)
                for t in range(0, captions.size(1)):
                    word = self.decoder.embed(decoder_input).squeeze(1)
                    query = states[0][-1].unsqueeze(1)
                    context = query @ encoder_out.permute(0, 2, 1) / (512 ** 0.5)
                    logs = context.softmax(dim=2)
                    final_context_vector = (logs @ encoder_out).squeeze(1)
                    inputs = torch.cat([word, final_context_vector], dim=1)
                    hiddens, states = self.decoder.lstm(inputs.unsqueeze(1), states)
                    outputs = self.decoder.fc(hiddens.squeeze(1))
                    _, predicted = outputs.max(1)
                    decoder_input = predicted.unsqueeze(1)
                    decoder_output[:, t, :] = outputs
                decoder_output = pack_padded_sequence(decoder_output, decoder_lengths, batch_first=True, enforce_sorted=False)[0]
        return decoder_output
    
    def sample(self, features_video, features_audio, max_seq_length=70):
        if (self.encoder.pool == "Transformer_Video"):
            features, encoder_output = self.encoder(video_feats=features_video.unsqueeze(0))
        elif (self.encoder.pool == "Transformer_Audio"):
            features, encoder_output = self.encoder(audio_feats=features_audio.unsqueeze(0))
        else:
            features, encoder_output = self.encoder(
                audio_feats=features_audio.unsqueeze(0), video_feats=features_video.unsqueeze(0)
            )

        if self.use_dual_decoder:
            half_d = features.size(-1) // 2
            feat_a, feat_v = features[:, :half_d], features[:, half_d:]
            tm = encoder_output.size(1) // 2
            enc_a, enc_v = encoder_output[:, :tm], encoder_output[:, tm:]
            return self.decoder.sample(feat_a, feat_v, enc_a, enc_v, max_seq_length)
        return self.decoder.sample(features, encoder_output, max_seq_length)
 

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
            checkpoint = torch.load(weights)
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
