from torch.utils.data import Dataset

import numpy as np
import random
import os
import time


from tqdm import tqdm

import torch

import logging
import json
from functools import lru_cache
from collections import Counter
from torchtext.vocab import vocab

from SoccerNet.Downloader import getListGames
from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.Evaluation.utils import getMetaDataTask
from torch.utils.data import default_collate
import numpy as np

PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2

def _infer_memmap_shape(feature_file, mapping, default_feature_dim=8576, dtype=np.float32):
    itemsize = np.dtype(dtype).itemsize
    file_size = os.path.getsize(feature_file)

    max_end = 0
    for entry in mapping.values():
        try:
            half1_end = int(entry["half1_start"]) + int(entry["half1_len"])
            half2_end = int(entry["half2_start"]) + int(entry["half2_len"])
            max_end = max(max_end, half1_end, half2_end)
        except (KeyError, TypeError, ValueError):
            continue

    if max_end > 0 and file_size % (max_end * itemsize) == 0:
        feature_dim = file_size // (max_end * itemsize)
        if feature_dim > 0:
            return int(max_end), int(feature_dim)

    row_bytes = default_feature_dim * itemsize
    if file_size % row_bytes != 0:
        raise ValueError(
            f"Cannot infer memmap shape for {feature_file}. "
            f"file_size={file_size} is not divisible by default row_bytes={row_bytes}."
        )
    return int(file_size // row_bytes), int(default_feature_dim)

def _build_game_to_mapping_key(mapping):
    if not mapping:
        return {}
    keys = list(mapping.keys())
    if not all(str(k).isdigit() for k in keys):
        return {}
    full_games = getListGames(["train", "valid", "test"], task="caption")
    return {game: str(i) for i, game in enumerate(full_games) if str(i) in mapping}

def _resolve_mapping_entry(mapping, game_name, game_id, game_to_mapping_key):
    if game_name in mapping:
        return mapping[game_name]
    mapped_key = game_to_mapping_key.get(game_name)
    if mapped_key is not None and mapped_key in mapping:
        return mapping[mapped_key]
    local_key = str(game_id)
    if local_key in mapping:
        return mapping[local_key]
    raise KeyError(
        f"No mapping entry found for game '{game_name}' (local id={game_id}). "
        "Check mapping_json consistency with the selected split."
    )

def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    captions = [t[-1] for t in batch]
    idx = [t[-3:-1] for t in batch]
    ## padd
    tokens = [([SOS_TOKEN] + t[-4] + [EOS_TOKEN]) if t[-4] else [PAD_TOKEN, PAD_TOKEN] for t in batch]
    tokens = [torch.Tensor(t).long() for t in tokens ]
    ## get sequence lengths
    lengths = torch.tensor([ len(t) for t in tokens ])
    tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
    ## compute mask
    mask = (tokens != PAD_TOKEN)
    return default_collate([t[:-4] for t in batch ]) + [tokens], lengths, mask, captions, idx


def feats2clip(feats, stride, clip_length, padding = "replicate_last", off=0):
    if padding =="zeropad":
        print("beforepadding", feats.shape)
        pad = feats.shape[0] - int(feats.shape[0]/stride)*stride
        print("pad need to be", clip_length-pad)
        m = torch.nn.ZeroPad2d((0, 0, clip_length-pad, 0))
        feats = m(feats)
        print("afterpadding", feats.shape)
        # nn.ZeroPad2d(2)

    idx = torch.arange(start=0, end=feats.shape[0]-1, step=stride)
    idxs = []
    for i in torch.arange(-off, clip_length-off):
        idxs.append(idx+i)
    idx = torch.stack(idxs, dim=1)

    if padding=="replicate_last":
        idx = idx.clamp(0, feats.shape[0]-1)
    # print(idx)
    return feats[idx,...]

class SoccerNetClips(Dataset):
    """
    This class is used to download and pre-compute clips from the SoccerNet dataset for spotting training phase.
    """
    def __init__(self, path, features="ResNET_PCA512.npy", split=["train"], version=2, 
                framerate=2, window_size=15):
        self.path = path
        self.listGames = getListGames(split, task="caption")
        self.features = features
        self.window_size_frame = window_size*framerate
        self.version = version
        labels, num_classes, dict_event, _ = getMetaDataTask("caption", "SoccerNet", version)
        self.labels = labels
        self.num_classes = num_classes
        self.dict_event = dict_event

        logging.info("Checking/Download features and labels locally")
        downloader = SoccerNetDownloader(path)
        for s in split:
            if s == "challenge":
                downloader.downloadGames(files=[f"1_{self.features}", f"2_{self.features}"], split=[s], task="caption", verbose=False,randomized=True)
            else:
                downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=[s], task="caption", verbose=False,randomized=True)

        logging.info("Pre-compute clips")

        self.game_feats = list()
        self.game_labels = list()

        for game in tqdm(self.listGames):
            # Load features
            feat_half1 = np.load(os.path.join(self.path, game, "1_" + self.features))
            feat_half1 = feat_half1.reshape(-1, feat_half1.shape[-1])
            feat_half2 = np.load(os.path.join(self.path, game, "2_" + self.features))
            feat_half2 = feat_half2.reshape(-1, feat_half2.shape[-1])

            feat_half1 = feats2clip(torch.from_numpy(feat_half1), stride=self.window_size_frame, clip_length=self.window_size_frame)
            feat_half2 = feats2clip(torch.from_numpy(feat_half2), stride=self.window_size_frame, clip_length=self.window_size_frame)

            # Load labels
            labels = json.load(open(os.path.join(self.path, game, self.labels)))

            label_half1 = np.zeros((feat_half1.shape[0], self.num_classes+1))
            label_half1[:,0]=1 # those are BG classes
            label_half2 = np.zeros((feat_half2.shape[0], self.num_classes+1))
            label_half2[:,0]=1 # those are BG classes


            for annotation in labels["annotations"]:

                time = annotation["gameTime"]
                event = annotation["label"]

                half = int(time[0])

                minutes, seconds = time.split(' ')[-1].split(':')
                minutes, seconds = int(minutes), int(seconds)
                frame = framerate * ( seconds + 60 * minutes ) 

                
                if event not in self.dict_event or half > 2:
                    continue
                label = self.dict_event[event]

                # if label outside temporal of view
                if half == 1 and frame//self.window_size_frame>=label_half1.shape[0]:
                    continue
                if half == 2 and frame//self.window_size_frame>=label_half2.shape[0]:
                    continue

                if half == 1:
                    label_half1[frame//self.window_size_frame][0] = 0 # not BG anymore
                    label_half1[frame//self.window_size_frame][label+1] = 1 # that's my class

                if half == 2:
                    label_half2[frame//self.window_size_frame][0] = 0 # not BG anymore
                    label_half2[frame//self.window_size_frame][label+1] = 1 # that's my class
            
            self.game_feats.append(feat_half1)
            self.game_feats.append(feat_half2)
            self.game_labels.append(label_half1)
            self.game_labels.append(label_half2)

        self.game_feats = np.concatenate(self.game_feats)
        self.game_labels = np.concatenate(self.game_labels)



    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            clip_feat (np.array): clip of features.
            clip_labels (np.array): clip of labels for the segmentation.
            clip_targets (np.array): clip of targets for the spotting.
        """
        return self.game_feats[index,:,:], self.game_labels[index,:]

    def __len__(self):
        return len(self.game_feats)

class SoccerNetClipsTesting(Dataset):
    """
    This class is used to download and pre-compute clips from the SoccerNet dataset for spotting inference phase.
    """
    def __init__(self, path, features="ResNET_PCA512.npy", split=["test"], version=2, 
                framerate=2, window_size=15):
        self.path = path
        self.listGames = getListGames(split, task="caption")
        self.features = features
        self.window_size_frame = window_size*framerate
        self.framerate = framerate
        self.version = version
        self.split=split
        labels, num_classes, dict_event, _ = getMetaDataTask("caption", "SoccerNet", version)
        self.labels = labels
        self.num_classes = num_classes
        self.dict_event = dict_event

        logging.info("Checking/Download features and labels locally")
        downloader = SoccerNetDownloader(path)
        for s in split:
            if s == "challenge":
                downloader.downloadGames(files=[f"1_{self.features}", f"2_{self.features}"], split=[s], task="caption", verbose=False,randomized=True)
            else:
                downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=[s], task="caption", verbose=False,randomized=True)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            feat_half1 (np.array): features for the 1st half.
            feat_half2 (np.array): features for the 2nd half.
            label_half1 (np.array): labels (one-hot) for the 1st half.
            label_half2 (np.array): labels (one-hot) for the 2nd half.
        """
        # Load features
        feat_half1 = np.load(os.path.join(self.path, self.listGames[index], "1_" + self.features))
        feat_half2 = np.load(os.path.join(self.path, self.listGames[index], "2_" + self.features))


        label_half1 = np.zeros((feat_half1.shape[0], self.num_classes))
        label_half2 = np.zeros((feat_half2.shape[0], self.num_classes))


        # check if annoation exists
        if os.path.exists(os.path.join(self.path, self.listGames[index], self.labels)):
            labels = json.load(open(os.path.join(self.path, self.listGames[index], self.labels)))

            for annotation in labels["annotations"]:

                time = annotation["gameTime"]
                event = annotation["label"]

                half = int(time[0])

                minutes, seconds = time.split(' ')[-1].split(':')
                minutes, seconds = int(minutes), int(seconds)
                frame = self.framerate * ( seconds + 60 * minutes ) 

                
                if event not in self.dict_event or half > 2:
                    continue
                label = self.dict_event[event]

                value = 1
                if "visibility" in annotation.keys():
                    if annotation["visibility"] == "not shown":
                        value = -1

                if half == 1:
                    frame = min(frame, feat_half1.shape[0]-1)
                    label_half1[frame][label] = value

                if half == 2:
                    frame = min(frame, feat_half2.shape[0]-1)
                    label_half2[frame][label] = value

        
            

        feat_half1 = feats2clip(torch.from_numpy(feat_half1), 
                        stride=1, off=int(self.window_size_frame/2), 
                        clip_length=self.window_size_frame)

        feat_half2 = feats2clip(torch.from_numpy(feat_half2), 
                        stride=1, off=int(self.window_size_frame/2), 
                        clip_length=self.window_size_frame)

        
        return self.listGames[index], feat_half1, feat_half2, label_half1, label_half2

    def __len__(self):
        return len(self.listGames)

class SoccerNetCaptions(Dataset):
    """
    This class is used to download and pre-compute clips and captions from the SoccerNet dataset for captining training phase.
    """
    def __init__(self, path, features="ResNET_TF2_PCA512.npy", split=["train"], version=2, framerate=2, window_size=15, mapping_json = "mapping.json", feature_file = "features.dat" ):
        self.path = path
        split = [s for s in split if s!= "challenge"]
        self.listGames = getListGames(split, task="caption")
        self.features = features
        self.window_size_frame = window_size*framerate
        self.version = version
        self.labels, self.num_classes, self.dict_event, _ = getMetaDataTask("caption", "SoccerNet", version)

        logging.info("Checking/Download features and labels locally")
        downloader = SoccerNetDownloader(path)
        downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], task="caption",split=split, verbose=False,randomized=True)

        self.data = list()
        self.l_pad = self.window_size_frame//2 + self.window_size_frame%2
        self.r_pad = self.window_size_frame//2
        with open(mapping_json, "r") as f:
            self.mapping = json.load(f)
        self.game_to_mapping_key = _build_game_to_mapping_key(self.mapping)
        memmap_rows, memmap_dim = _infer_memmap_shape(feature_file, self.mapping)
        self.memmap = np.memmap(feature_file, mode='r', shape=(memmap_rows, memmap_dim), dtype='float32')

        for game_id, game in enumerate(tqdm(self.listGames, desc="Building caption index")):
            # Load labels only (features loaded lazily in __getitem__)
            labels = json.load(open(os.path.join(self.path, game, self.labels)))

            for caption_id, annotation in enumerate(labels["annotations"]):
                

                time = annotation["gameTime"]
                event = annotation["label"]

                event_clean = str(event).strip().lower()
                #Removing these labels as the paper suggested these are completely out of distribution labels 
                if event_clean in {"funfact", "attendance"}:
                    continue
                half = int(time[0])
                if event not in self.dict_event or half > 2:
                    continue

                minutes, seconds = time.split(' ')[-1].split(':')
                minutes, seconds = int(minutes), int(seconds)
                frame = framerate * ( seconds + 60 * minutes)

                self.data.append(((game_id, half-1, frame) , (caption_id, annotation['anonymized'])))

        #launch a VideoProcessor that will create a clip around a caption
        self.video_processor = SoccerNetVideoProcessor(self.window_size_frame)
        #launch a TextProcessor that will tokenize a caption
        self.text_processor = SoccerNetTextProcessor(self.getCorpus(split=["train"]))
        self.vocab_size = len(self.text_processor.vocab)

    def __len__(self):
        return len(self.data)

    def _load_game_features(self, game_id):
        return self._cached_load(game_id) # Adding this to help with loading data 

    def _cached_load(self, game_id):
        """Load and pad features for a single game (lazy loading)."""
        game_name = self.listGames[game_id]
        entry = _resolve_mapping_entry(self.mapping, game_name, game_id, self.game_to_mapping_key)
        half1_start = int(entry['half1_start'])
        half1_len = int(entry['half1_len'])
        half2_start = int(entry['half2_start'])
        half2_len = int(entry['half2_len'])
        feat_half1 = self.memmap[half1_start : half1_start + half1_len]
        feat_half2 = self.memmap[half2_start: half2_start + half2_len]

        return (feat_half1, feat_half2)


    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        Returns:
            vfeats (np.array): clip of features.
            caption_tokens (np.array): tokens of captions.
            clip_id (np.array): clip id.
            caption_id (np.array): caption id.
            caption (List[strings]): list of original captions.
        """
        clip_id, (caption_id, caption) = self.data[idx]
        game_id = clip_id[0]
        game_feats = self._load_game_features(game_id)
        # VideoProcessor expects feats[video_id][half]; pass a list with only this game at index game_id
        feats_for_processor = [None] * game_id + [game_feats]
        vfeats = self.video_processor(clip_id, feats_for_processor)
        caption_tokens = self.text_processor(caption)

        return vfeats, caption_tokens, clip_id[0], caption_id, caption

    def getCorpus(self, split=["train"]):
        """
        Args:
            split (string): split of dataset
        Returns:
            corpus (List[string]): vocabulary build from split.
        """
        corpus = [annotation['anonymized'] for game in getListGames(split, task="caption") for annotation in json.load(open(os.path.join(self.path, game, self.labels)))["annotations"]]
        return corpus
    
    def detokenize(self, tokens, remove_EOS=True):
        """
        Args:
            tokens (List[int]): tokens of caption
        Returns:
            caption (string): string obtained after replacing each token by its corresponding word
        """
        string = self.text_processor.detokenize(tokens)
        return string.rstrip(f" {self.text_processor.vocab.lookup_token(EOS_TOKEN)}") if remove_EOS else string

class SoccerNetVideoProcessor(object):
    """video_fn is a tuple of (video_id, half, frame)."""

    def __init__(self, clip_length):
        self.clip_length = clip_length

    def __call__(self, video_fn, feats):
        video_id, half, frame = video_fn
        video_feature = feats[video_id][half]
        #make sure that the clip lenght is right
        start = min(frame, video_feature.shape[0] - self.clip_length)
        video_feature = video_feature[start : start + self.clip_length]

        return video_feature

class SoccerNetTextProcessor(object):
    """
    A generic Text processor
    tokenize a string of text on-the-fly.
    """

    def __init__(self, corpus, min_freq=5):
        import spacy
        spacy_token = spacy.load("en_core_web_sm").tokenizer
        # Add special case rule
        spacy_token.add_special_case("[PLAYER]", [{"ORTH": "[PLAYER]"}])
        spacy_token.add_special_case("[COACH]", [{"ORTH": "[COACH]"}])
        spacy_token.add_special_case("[TEAM]", [{"ORTH": "[TEAM]"}])
        spacy_token.add_special_case("([TEAM])", [{"ORTH": "([TEAM])"}])
        spacy_token.add_special_case("[REFEREE]", [{"ORTH": "[REFEREE]"}])
        # self.tokenizer = lambda s: [c.text for c in spacy_token(s)]
        self.spacy_token = spacy_token
        self.min_freq = min_freq
        self.build_vocab(corpus)
    
    def tokenizer(self, s):
        """Tokenize a string using spaCy. This method is picklable for multiprocessing."""
        return [c.text for c in self.spacy_token(s)]
    
    def build_vocab(self, corpus):
        counter = Counter([token for c in corpus for token in self.tokenizer(c)])
        voc = vocab(counter, min_freq=self.min_freq, specials=["[PAD]", "[SOS]", "[EOS]", "[UNK]", "[MASK]", "[CLS]"])
        voc.set_default_index(voc['[UNK]'])
        self.vocab = voc
    
    def __call__(self, text):
        return self.vocab(self.tokenizer(text))
    
    def detokenize(self, tokens):
        return " ".join(self.vocab.lookup_tokens(tokens))

class PredictionCaptions(Dataset):
    def __init__(self, SoccerNetPath, PredictionPath, features="ResNET_TF2_PCA512.npy", split=["train"], version=2, framerate=2, window_size=15 ,mapping_json = "mapping.json", feature_file = "features.dat"):
        self.path = SoccerNetPath
        self.PredictionPath = PredictionPath
        self.listGames = getListGames(split, task="caption")
        self.features = features
        self.window_size_frame = window_size*framerate
        self.version = version
        self.labels, _, self.dict_event, _ = getMetaDataTask("caption", "SoccerNet", version)
        self.split = split
        with open(mapping_json, "r") as f:
            self.mapping = json.load(f)
        self.game_to_mapping_key = _build_game_to_mapping_key(self.mapping)
        memmap_rows, memmap_dim = _infer_memmap_shape(feature_file, self.mapping)
        self.memmap = np.memmap(feature_file, mode='r', shape=(memmap_rows, memmap_dim), dtype='float32')

        
     
        logging.info("Checking/Download features and labels locally")
        downloader = SoccerNetDownloader(self.path)
        downloader.downloadGames(files=[f"1_{self.features}", f"2_{self.features}"], task="caption", split=split, verbose=False,randomized=True)

        self.data = list()
        self.l_pad = self.window_size_frame//2 + self.window_size_frame%2
        self.r_pad = self.window_size_frame//2

        for game_id, game in enumerate(tqdm(self.listGames, desc="Building prediction index")):
            preds = json.load(open(os.path.join(self.PredictionPath, game, "results_spotting.json")))

            for caption_id, annotation in enumerate(preds["predictions"]):

                if annotation["label"] not in self.dict_event:
                    continue

                time = annotation["gameTime"]
                half = int(time[0])
                if half > 2:
                    continue

                minutes, seconds = time.split(' ')[-1].split(':')
                minutes, seconds = int(minutes), int(seconds)
                frame = framerate * ( int(seconds) + 60 * int(minutes))

                self.data.append(((game_id, half-1, frame), caption_id))

        #launch a VideoProcessor that will create a clip around a caption
        self.video_processor = SoccerNetVideoProcessor(self.window_size_frame)
        #launch a TextProcessor that will tokenize a caption
        self.text_processor = SoccerNetTextProcessor(self.getCorpus(split=["train"]))
        self.vocab_size = len(self.text_processor.vocab)

    def _load_game_features(self, game_id):
        return self._cached_load(game_id) # Adding this to help with loading data 

    def _cached_load(self, game_id):
        """Load and pad features for a single game (lazy loading)."""
        game_name = self.listGames[game_id]
        entry = _resolve_mapping_entry(self.mapping, game_name, game_id, self.game_to_mapping_key)
        half1_start = int(entry['half1_start'])
        half1_len = int(entry['half1_len'])
        half2_start = int(entry['half2_start'])
        half2_len = int(entry['half2_len'])
        feat_half1 = self.memmap[half1_start : half1_start + half1_len]
        feat_half2 = self.memmap[half2_start: half2_start + half2_len]


        #game = self.listGames[game_id]
        #feat_half1 = np.load(os.path.join(self.path, game, "1_" + self.features))
        #
        #feat_half2 = 
        #feat_half2 = np.pad(feat_half2.reshape(-1, feat_half2.shape[-1]), ((self.l_pad, self.r_pad), (0, 0)), "edge")
        return (feat_half1, feat_half2)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        Returns:
            vfeats (np.array): clip of features.
            clip_id (np.array): clip id.
            caption_id (np.array): caption id.
        """
        clip_id, caption_id = self.data[idx]
        game_id = clip_id[0]
        game_feats = self._load_game_features(game_id)
        feats_for_processor = [None] * game_id + [game_feats]
        vfeats = self.video_processor(clip_id, feats_for_processor)
        return vfeats, clip_id[0], caption_id


    def detokenize(self, tokens, remove_EOS=True):
        """
        Args:
            tokens (List[int]): tokens of caption
        Returns:
            caption (string): string obtained after replacing each token by its corresponding word
        """
        string = self.text_processor.detokenize(tokens)
        return string.rstrip(f" {self.text_processor.vocab.lookup_token(EOS_TOKEN)}") if remove_EOS else string
    
    def getCorpus(self, split=["train"]):
        """
        Args:
            split (string): split of dataset
        Returns:
            corpus (List[string]): vocabulary build from split.
        """
        corpus = [annotation['anonymized'] for game in getListGames(split, task="caption") for annotation in json.load(open(os.path.join(self.path, game, self.labels)))["annotations"]]
        return corpus


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    torch.manual_seed(0)
    np.random.seed(0)
    root = "/scratch/users/hmkhallati/SoccerNet/"
    dataset_Test  = SoccerNetCaptions(path=root, features="ResNET_TF2_PCA512.npy", split="test", version=2, framerate=2, window_size=15)
    test_loader = torch.utils.data.DataLoader(dataset_Test,
        batch_size=1, shuffle=False, pin_memory=True)
    batch = next(iter(test_loader))
    (feats, caption), lengths, mask, caption_or, idx = batch
    print(feats, caption)
    print(test_loader.detokenize([55, 22, 33, 2]))
    print(idx)