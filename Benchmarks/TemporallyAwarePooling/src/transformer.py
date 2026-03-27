import torch.nn as nn
import torch 


'''
A suite of transformer pooling layers to test things out 
'''

class Transformer(nn.Module):
    '''
    Simple MultiModal Transformer 

    Taked in audio and video embeddings, provides the (global representation) of each embedding in it's forward pass
    Encoding: Learned Encoding y the model 

    TODO: handle device -> gpu/cpu option 
    NOTE: This handles both the audio and video embeddings together,it uses cls token, but we should probably set it up to use mean pooling like Transformer_Video, based on preferenc

    '''
    def __init__(
        self,

        audio_feat_dim,
        audio_d_model,
        audio_nhead,
        audio_num_layers,

        video_feat_dim,
        video_d_model,
        video_nhead,
        video_num_layers,

        audio_length, 
        video_length):
        super().__init__()


        self.audio_proj = nn.Linear(audio_feat_dim, audio_d_model)
        self.audio_length = audio_length
        self.video_length = video_length
        self.pos_dropout_rate = 0.3

        self.cls_audio = nn.Parameter(torch.randn(1, 1, audio_d_model))
        audio_layer = nn.TransformerEncoderLayer(audio_d_model, audio_nhead, dropout=0.3, batch_first=True)
        self.audio_transformer = nn.TransformerEncoder(audio_layer, num_layers=audio_num_layers)
        self.embedding_audio = nn.Embedding(audio_length+1, audio_d_model)
        self.video_proj = nn.Linear(video_feat_dim, video_d_model)
        self.cls_video = nn.Parameter(torch.randn(1, 1, video_d_model))
        self.embedding_video = nn.Embedding(video_length+1, video_d_model)
        video_layer = nn.TransformerEncoderLayer(video_d_model, video_nhead, dropout=0.3, batch_first=True)
        self.video_transformer = nn.TransformerEncoder(video_layer, num_layers=video_num_layers)

    def forward(self, audio_feats , video_feats):

        '''
        batch_audio, _ , _ = audio_feats.shape
        batch_video, _, _ = video_feats.shape 
        x = self.audio_proj(audio_feats) + self.embedding_audio(torch.arange(1, audio_feats.shape[1] + 1, device=audio_feats.device))
        cls_aud = self.cls_audio.expand(batch_audio, -1, -1) 
        cls_aud = cls_aud+ self.embedding_audio(torch.zeros(1, dtype=torch.long, device=video_feats.device))
        x = torch.concat([cls_aud, x], dim=1 )
        x  = self.audio_transformer(x) 
        audio_token  = x[:,0, :]


        x = self.video_proj(video_feats) + self.embedding_video(torch.arange(1, video_feats.shape[1] + 1, device=video_feats.device))
        cls_video = self.cls_video.expand(batch_video, -1, -1) 
        cls_video = cls_video + self.embedding_video(torch.zeros(1, dtype=torch.long, device=video_feats.device))
        x = torch.concat([cls_video, x], dim=1 )
        x  = self.video_transformer(x)
        video_token  = x[:, 0, : ]

        ''' 
        raise NotImplementedError


class Transformer_Video(nn.Module):
    """Video-only transformer encoder that produces a global representation from video features.

    Projects input video features into a learned embedding space, adds learned positional
    encodings (with stochastic dropout during training), and passes the sequence through
    a stack of transformer encoder layers. The global representation is obtained by
    mean-pooling over the temporal dimension.

    Args:
        video_feat_dim (int): Dimensionality of the raw input video features.
        video_d_model (int): Hidden dimension of the transformer (projection target).
        video_nhead (int): Number of attention heads in each transformer layer.
        video_num_layers (int): Number of stacked transformer encoder layers.
        video_length (int): Maximum sequence length (number of frames), used to
            define the learned positional embedding table.

    Forward args:
        video_feats (Tensor): Input video features of shape ``(batch, seq_len, video_feat_dim)``.

    Returns:
        tuple[Tensor, Tensor]:
            - **video_token** — Mean-pooled global representation of shape ``(batch, video_d_model)``.
            - **x** — Full encoder output sequence of shape ``(batch, seq_len, video_d_model)``,
              useful for downstream cross-attention (e.g. in a decoder).

    PLEASE NOTE:
        During training, the learned positional encodings are stochastically dropped
        (with probability ``pos_dropout_rate = 0.3``) on a per-batch-element basis to
        encourage the model not to over-rely on absolute position information. Please remove this if you find it not helpful! 
    """
    def __init__(
        self,
        video_feat_dim,
        video_d_model,
        video_nhead,
        video_num_layers,

        video_length):
        super().__init__()


        self.video_length = video_length

        self.video_proj = nn.Linear(video_feat_dim, video_d_model)
        #self.cls_video = nn.Parameter(torch.randn(1, 1, video_d_model)) This can be removed 
        self.embedding_video = nn.Embedding(video_length, video_d_model)
        self.pos_dropout_rate = 0.3
        video_layer = nn.TransformerEncoderLayer(video_d_model, video_nhead, dropout=0.3, batch_first=True)
        self.video_transformer = nn.TransformerEncoder(video_layer, num_layers=video_num_layers)

    def forward(self, video_feats):

        batch_video, _, _ = video_feats.shape 
        pos = self.embedding_video(torch.arange(0, video_feats.shape[1], device=video_feats.device))
        if self.training:
            mask = (torch.rand(batch_video, 1, 1, device=video_feats.device) > self.pos_dropout_rate).float()
            pos = pos * mask

        x = self.video_proj(video_feats) + pos
        x  = self.video_transformer(x)
        video_token  = x.mean(dim=1)
        return video_token, x

class Transformer_Audio(nn.Module):
    '''
    Simple MultiModal Transformer , just for audio 

    Taked in audio and video embeddings, provides the cls token (global representation) of each embedding in it's forward pass
    Encoding: Learned Encoding y the model 

    TODO: handle device -> gpu/cpu option 
    NOTE: This is for an audio only pipeline, we can handle it the same way as the video embedding but figured to put it here for the ablation studies if we come to that 

    '''
    def __init__(
        self,

        audio_feat_dim,
        audio_d_model,
        audio_nhead,
        audio_num_layers,

        audio_length):
        super().__init__()


        self.audio_proj = nn.Linear(audio_feat_dim, audio_d_model)
        self.audio_length = audio_length
        self.cls_audio = nn.Parameter(torch.randn(1, 1, audio_d_model))
        audio_layer = nn.TransformerEncoderLayer(audio_d_model, audio_nhead, dropout=0.3, batch_first=True)
        self.audio_transformer = nn.TransformerEncoder(audio_layer, num_layers=audio_num_layers)
        self.embedding_audio = nn.Embedding(audio_length +1, audio_d_model)
        self.pos_dropout = nn.Dropout(p=0.1)
    def forward(self, audio_feats):
        
        ''' 
        batch_audio, _ , _ = audio_feats.shape
        x = self.audio_proj(audio_feats) + self.embedding_audio(torch.arange(1, audio_feats.shape[1] + 1, device=audio_feats.device))
        cls_aud = self.cls_audio.expand(batch_audio, -1, -1) 
        cls_aud = cls_aud + self.embedding_audio(torch.zeros(1, dtype=torch.long, device=audio_feats.device))
        x = torch.concat([cls_aud, x], dim=1 )
        x = self.pos_dropout(x)
        x  = self.audio_transformer(x) 
        audio_token  = x[:,0, :]
        '''
        raise NotImplementedError




if __name__ == "__main__":
    Transformer_vid = Transformer_Video(video_feat_dim =512,video_d_model=512,video_nhead=8,video_num_layers=2,video_length=120 )
    feat_in = torch.rand((3,120,512)) # Batches, video length, Features
    print("in", feat_in.shape)
    feat_out = Transformer_vid(feat_in)
    print("out", feat_out.shape)
    print('result' , feat_out)
