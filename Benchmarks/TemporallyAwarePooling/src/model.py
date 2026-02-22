import __future__

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
    def __init__(self, input_size=512, window_size=15, framerate=2, pool="Transformer_Video"):
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
        self.framerate = framerate
        self.pool = pool

        # are feature alread PCA'ed?
        if not self.input_size == 512:   
            self.feature_extractor = nn.Linear(self.input_size, 256)
            input_size = 256
            self.input_size = 256
        
        if self.pool == "Transformer_Video":
            self.pooling_layer = Transformer_Video(video_feat_dim=self.input_size, video_d_model=self.input_size, video_nhead=8, video_num_layers=2, video_length=self.window_size_frame)
            self.hidden_size = self.input_size
        elif self.pool == "Transformer_Audio":
             #TODO: @Chidi, need to get the Transformer Audio feature size! 
             self.pooling_layer = Transformer_Audio(audio_feat_dim=self.input_size, audio_d_model=self.input_size, audio_nhead=2, audio_num_layers=2, audio_length=self.window_size_frame)
             self.hidden_size = self.input_size         

        elif self.pool == "Transformer":
             #TODO: @Chidi, need to get the Transformer Audio feature size! 
             self.pooling_layer = Transformer(video_feat_dim=self.input_size, video_d_model=self.input_size, video_nhead=8, video_num_layers=2, video_length=self.window_size_frame , audio_feat_dim=self.input_size, audio_d_model=self.input_size, audio_nhead=2, audio_num_layers=2, audio_length=self.window_size_frame)
             self.hidden_size = self.input_size  * 2 
    def forward(self, audio_feats=None, video_feats=None):
        if audio_feats is None and video_feats is None:
            raise NotImplementedError
        
        elif (video_feats is not None and  audio_feats is None ):
            '''If we only have video features '''
            BS, FR, IC = video_feats.shape
            if not IC == 512:
                video_feats = video_feats.reshape(BS*FR, IC)
                video_feats = self.feature_extractor(video_feats)
                video_feats = video_feats.reshape(BS, FR, -1)
            cls_video_token = self.pooling_layer(video_feats)
            return cls_video_token
        
        elif (audio_feats is not None and  video_feats is None):
            '''If we only have audio features'''
            BS, FR, IC = audio_feats.shape

            #TODO: @Chidi, how do we know the dimensions of the audio? are we PCAIng the audip? 
            if not IC == 512:
                audio_feats = audio_feats.reshape(BS*FR, IC)
                audio_feats = self.feature_extractor(audio_feats)
                audio_feats = audio_feats.reshape(BS, FR, -1)
            cls_audio_token = self.pooling_layer(audio_feats)
            return cls_audio_token
        
        elif (video_feats is not None  and audio_feats is not None):
            '''if we have both features'''
            BS_vid, FR_vid, IC_vid = video_feats.shape
            BS_audio, FR_audio, IC_audio = audio_feats.shape

            #TODO: @Chidi, how do we know the dimensions of the audio? are we PCAIng the audip? 
            if not IC_vid == 512:
                video_feats = video_feats.reshape(BS_vid*FR_vid, IC_vid)
                video_feats = self.feature_extractor(video_feats)
                video_feats = video_feats.reshape(BS_vid, FR_vid, -1)
            if not IC_audio == 512:
                '''TODO: We need to know the dims of audio'''
                audio_feats = audio_feats.reshape(BS_audio*FR_audio, IC_audio)
                audio_feats = self.feature_extractor(audio_feats)
                audio_feats= audio_feats.reshape(BS_vid, FR_vid, -1)
            cls_audio_token, cls_video_token = self.pooling_layer(audio_feats, video_feats )
            final_token  = torch.concat([cls_audio_token, cls_video_token], dim=1)
            return final_token
        
        
        

        


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
        #To reduce the computation, we pack padd sequences
        captions = pack_padded_sequence(captions, lengths, batch_first=True, enforce_sorted=False)
        #Video encoder features are used as initial states
        hiddens, _ = self.lstm(captions, (features, features))
        outputs = self.fc(hiddens[0])
        return outputs
    
    def sample(self, features, max_seq_length):
        sampled_ids = []
        #Features extraction of video encoder
        features = self.ft_extactor_2(self.activation(self.dropout(self.ft_extactor_1(features))))
        features = torch.stack([features]*self.num_layers)
        #Video encoder features are used as initial states
        states = (features, features)
        #Start token
        inputs = torch.tensor([[SOS_TOKEN]], device=features.device)
        #Start token
        inputs = self.embed(inputs)
        #Sample at most max_seq_length token
        for i in range(max_seq_length):
            hiddens, states = self.lstm(inputs, states) 
            outputs = self.fc(hiddens.squeeze(1))
            #Sample the most likely word
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            if predicted == EOS_TOKEN:
                #end of sampling
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
    
    def forward(self, features, captions, lengths):
        features = self.encoder(features)
        batch_size = captions.size(0)
        captions = captions[:, :-1]  # Remove last word in caption to use as input
        use_teacher_forcing = random.random() < self.teacher_forcing_ratio
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            decoder_input = captions
            decoder_output = self.decoder(features, decoder_input, lengths)
        else:
            decoder_input = captions[:, 0].unsqueeze(1)  # <start> token
            decoder_output = torch.zeros(batch_size, captions.size(1), self.vocab_size, device=captions.device)
            for t in range(0, captions.size(1)):
                # Pass through decoder
                decoder_output_t = self.decoder(features, decoder_input, torch.ones_like(lengths))
                decoder_output[:, t, :] = decoder_output_t
                # Get next input from highest predicted token
                _, topi = decoder_output_t.topk(1)
                decoder_input = topi.detach()  # detach from history as input
            decoder_output = pack_padded_sequence(decoder_output, lengths, batch_first=True, enforce_sorted=False)[0]
        return decoder_output
    
    def sample(self, features, max_seq_length=70):
        features = self.encoder(features.unsqueeze(0))
        return self.decoder.sample(features, max_seq_length)

class SoccerNetTransformerCaption(nn.Module):
    def __init__(self, vocab_size, weights=None, input_size=512, window_size=15, framerate=2, pool="Transformer_Video", embed_size=128, hidden_size=256, teacher_forcing_ratio=1, num_layers=2, max_seq_length=50, weights_encoder=None, freeze_encoder=False):
        super(SoccerNetTransformerCaption, self).__init__()
        self.encoder = MultimodalTransformerCaption(input_size, window_size, framerate, pool)
        self.decoder = DecoderRNN(self.encoder.hidden_size, embed_size, hidden_size, vocab_size, num_layers)
        #self.load_weights(weights=weights)
        self.load_encoder(weights_encoder=weights_encoder, freeze_encoder=freeze_encoder)
        self.vocab_size = vocab_size
        self.teacher_forcing_ratio = teacher_forcing_ratio
            
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
            features = self.encoder(features_video)
        elif (self.encoder.pool == "Transformer_Audio"):
            '''get the cls token just for the audio'''
            features = self.encoder( features_audio)
        else:
            '''get the cls token for the combined versions'''
            features = self.encoder(features_audio , features_video )
        batch_size = captions.size(0)
        captions = captions[:, :-1]  # Remove last word in caption to use as input
        use_teacher_forcing = random.random() < self.teacher_forcing_ratio
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            decoder_input = captions
            decoder_output = self.decoder(features, decoder_input, lengths)
        else:
            decoder_input = captions[:, 0].unsqueeze(1)  # <start> token
            decoder_output = torch.zeros(batch_size, captions.size(1), self.vocab_size, device=captions.device)
            for t in range(0, captions.size(1)):
                # Pass through decoder
                decoder_output_t = self.decoder(features, decoder_input, torch.ones_like(lengths))
                decoder_output[:, t, :] = decoder_output_t
                # Get next input from highest predicted token
                _, topi = decoder_output_t.topk(1)
                decoder_input = topi.detach()  # detach from history as input
            decoder_output = pack_padded_sequence(decoder_output, lengths, batch_first=True, enforce_sorted=False)[0]
        return decoder_output
    
    def sample(self, features_video, features_audio, max_seq_length=70):
        if (self.encoder.pool == "Transformer_Video"):
            features = self.encoder(features_video.unsqueeze(0))
        elif (self.encoder.pool == "Transformer_Audio"):
            features = self.encoder(features_audio.unsqueeze(0))
        else:
            features = self.encoder(video_feats= features_video.unsqueeze(0), audio_feats = features_audio.unsqueeze(0))

        return self.decoder.sample(features, max_seq_length)
 

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