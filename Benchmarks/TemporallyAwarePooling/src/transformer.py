import torch.nn as nn
import torch 


'''
A suite of transformer pooling layers to test things out 
'''

class Transformer(nn.Module):
    '''
    Simple MultiModal Transformer 

    Taked in audio and video embeddings, provides the cls token (global representation) of each embedding in it's forward pass
    Encoding: Learned Encoding y the model 

    TODO: handle device -> gpu/cpu option 

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
        video_length    ):
        super().__init__()


        self.audio_proj = nn.Linear(audio_feat_dim, audio_d_model)
        self.audio_length = audio_length
        self.video_length = video_length

        self.cls_audio = nn.Parameter(torch.randn(1, 1, audio_d_model))
        audio_layer = nn.TransformerEncoderLayer(audio_d_model, audio_nhead, batch_first=True)
        self.audio_transformer = nn.TransformerEncoder(audio_layer, num_layers=audio_num_layers)
        self.embedding_audio = nn.Embedding(audio_length+1, audio_d_model)
        self.video_proj = nn.Linear(video_feat_dim, video_d_model)
        self.cls_video = nn.Parameter(torch.randn(1, 1, video_d_model))
        self.embedding_video = nn.Embedding(video_length+1, video_d_model)
        video_layer = nn.TransformerEncoderLayer(video_d_model, video_nhead, batch_first=True)
        self.video_transformer = nn.TransformerEncoder(video_layer, num_layers=video_num_layers)

    def forward(self, audio_feats , video_feats):

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

        return audio_token, video_token


class Transformer_Video(nn.Module):
    '''
    Simple MultiModal Transformer, just for videos 

    Taked in audio and video embeddings, provides the cls token (global representation) of each embedding in it's forward pass
    Encoding: Learned Encoding y the model 
 

    '''
    def __init__(
        self,
        video_feat_dim,
        video_d_model,
        video_nhead,
        video_num_layers,

        video_length    ):
        super().__init__()


        self.video_length = video_length

        self.video_proj = nn.Linear(video_feat_dim, video_d_model)
        #self.cls_video = nn.Parameter(torch.randn(1, 1, video_d_model))
        self.embedding_video = nn.Embedding(video_length, video_d_model)
        video_layer = nn.TransformerEncoderLayer(video_d_model, video_nhead, batch_first=True)
        self.video_transformer = nn.TransformerEncoder(video_layer, num_layers=video_num_layers)

    def forward(self, video_feats):


        batch_video, _, _ = video_feats.shape 
        x = self.video_proj(video_feats) + self.embedding_video(torch.arange(0, video_feats.shape[1] , device=video_feats.device))
        x  = self.video_transformer(x)
        video_token  = x.mean(dim=1)
        return video_token, x

class Transformer_Audio(nn.Module):
    '''
    Simple MultiModal Transformer , just for audio 

    Taked in audio and video embeddings, provides the cls token (global representation) of each embedding in it's forward pass
    Encoding: Learned Encoding y the model 

    TODO: handle device -> gpu/cpu option 

    '''
    def __init__(
        self,

        audio_feat_dim,
        audio_d_model,
        audio_nhead,
        audio_num_layers,

        audio_length, ):
        super().__init__()


        self.audio_proj = nn.Linear(audio_feat_dim, audio_d_model)
        self.audio_length = audio_length
        self.cls_audio = nn.Parameter(torch.randn(1, 1, audio_d_model))
        audio_layer = nn.TransformerEncoderLayer(audio_d_model, audio_nhead, batch_first=True)
        self.audio_transformer = nn.TransformerEncoder(audio_layer, num_layers=audio_num_layers)
        self.embedding_audio = nn.Embedding(audio_length +1, audio_d_model)
    def forward(self, audio_feats):

        batch_audio, _ , _ = audio_feats.shape
        x = self.audio_proj(audio_feats) + self.embedding_audio(torch.arange(1, audio_feats.shape[1] + 1, device=audio_feats.device))
        cls_aud = self.cls_audio.expand(batch_audio, -1, -1) 
        cls_aud = cls_aud + self.embedding_audio(torch.zeros(1, dtype=torch.long, device=audio_feats.device))
        x = torch.concat([cls_aud, x], dim=1 )
        x  = self.audio_transformer(x) 
        audio_token  = x[:,0, :]

        return audio_token




if __name__ == "__main__":
    Transformer_vid = Transformer_Video(video_feat_dim =512,video_d_model=512,video_nhead=8,video_num_layers=2,video_length=120 )
    feat_in = torch.rand((3,120,512)) # Batches, video length, Features
    print("in", feat_in.shape)
    feat_out = Transformer_vid(feat_in)
    print("out", feat_out.shape)
    print('result' , feat_out)
