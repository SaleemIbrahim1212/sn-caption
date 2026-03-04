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
    def __init__(self, path, features="ResNET_TF2_PCA512.npy", split=["train"], version=2, framerate=2, window_size=15):
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

        for game_id, game in enumerate(tqdm(self.listGames, desc="Building caption index")):
            # Load labels only (features loaded lazily in __getitem__)
            labels = json.load(open(os.path.join(self.path, game, self.labels)))

            for caption_id, annotation in enumerate(labels["annotations"]):

                time = annotation["gameTime"]
                event = annotation["label"]
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
        return self._cached_load(game_id)

    @lru_cache(maxsize=16)
    def _cached_load(self, game_id):
        """Load and pad features for a single game (lazy loading)."""
        game = self.listGames[game_id]
        feat_half1 = np.load(os.path.join(self.path, game, "1_" + self.features))
        feat_half1 = np.pad(feat_half1.reshape(-1, feat_half1.shape[-1]), ((self.l_pad, self.r_pad), (0, 0)), "edge")
        feat_half2 = np.load(os.path.join(self.path, game, "2_" + self.features))
        feat_half2 = np.pad(feat_half2.reshape(-1, feat_half2.shape[-1]), ((self.l_pad, self.r_pad), (0, 0)), "edge")
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


class SoccerNetCaptionsMaster(Dataset):
    """
    Caption dataset that reads precomputed embeddings from a single memory-mapped file.
    Uses np.memmap / mmap_mode to avoid loading the full dataset into RAM and to keep
    the GPU fed without CPU I/O bottleneck. Expects embeddings built for window_size_caption=45, 1fps.
    """
    def __init__(self, path, master_dir, features="ResNET_TF2_PCA512.npy", split=["train"], version=2, framerate=1, window_size=45):
        self.path = path
        self.master_dir = os.path.abspath(master_dir)
        split = [s for s in split if s != "challenge"]
        self.listGames = getListGames(split, task="caption")
        self.features = features
        self.window_size_frame = window_size * framerate
        self.framerate = framerate
        self.version = version
        self.labels, self.num_classes, self.dict_event, _ = getMetaDataTask("caption", "SoccerNet", version)

        logging.info("Using master embeddings from %s (no feature download)", self.master_dir)
        downloader = SoccerNetDownloader(path)
        downloader.downloadGames(files=[self.labels], task="caption", split=split, verbose=False, randomized=True)

        mapping_path = os.path.join(self.master_dir, "mapping.json")
        if not os.path.isfile(mapping_path):
            raise FileNotFoundError("Master mapping not found: %s" % mapping_path)
        with open(mapping_path) as f:
            self._mapping = json.load(f)

        # Build caption index from labels only
        self.data = list()
        for game_id, game in enumerate(tqdm(self.listGames, desc="Building caption index")):
            labels_path = os.path.join(self.path, game, self.labels)
            if not os.path.isfile(labels_path):
                logging.warning("Labels not found for %s, skipping", game)
                continue
            labels = json.load(open(labels_path))
            for caption_id, annotation in enumerate(labels["annotations"]):
                time_str = annotation["gameTime"]
                event = annotation["label"]
                half = int(time_str[0])
                if event not in self.dict_event or half > 2:
                    continue
                minutes, seconds = time_str.split(" ")[-1].split(":")
                minutes, seconds = int(minutes), int(seconds)
                frame = framerate * (seconds + 60 * minutes)
                self.data.append(((game_id, half - 1, frame), (caption_id, annotation["anonymized"])))

        self.text_processor = SoccerNetTextProcessor(self.getCorpus(split=["train"]))
        self.vocab_size = len(self.text_processor.vocab)

        # Open memory-mapped embeddings
        self._master_mmap, self._feature_dim = self._open_master_embeddings()

    def _open_master_embeddings(self):
        """Open the master embeddings file as read-only memory-mapped array."""
        npy_path = os.path.join(self.master_dir, "embeddings.npy")
        bin_path = os.path.join(self.master_dir, "embeddings.bin")
        meta_path = os.path.join(self.master_dir, "metadata.json")

        if os.path.isfile(npy_path):
            arr = np.load(npy_path, mmap_mode="r")
            if arr.ndim != 2:
                raise ValueError("Master embeddings .npy must be 2D (total_frames, feature_dim), got shape %s" % (arr.shape,))
            total_frames, feature_dim = arr.shape
            logging.info("Master embeddings: %s shape=%s (memmapped)", npy_path, arr.shape)
            return arr, feature_dim

        if os.path.isfile(bin_path):
            if not os.path.isfile(meta_path):
                raise FileNotFoundError("embeddings.bin found but metadata.json missing in %s" % self.master_dir)
            with open(meta_path) as f:
                meta = json.load(f)
            feature_dim = int(meta["feature_dim"])
            dtype = np.dtype(meta.get("dtype", "float32"))
            total_from_mapping = 0
            for k, v in self._mapping.items():
                end = v["half2_start"] + v["half2_len"]
                total_from_mapping = max(total_from_mapping, end)
            arr = np.memmap(bin_path, dtype=dtype, mode="r", shape=(total_from_mapping, feature_dim))
            logging.info("Master embeddings: %s shape=(%s, %s) (memmap)", bin_path, total_from_mapping, feature_dim)
            return arr, feature_dim

        raise FileNotFoundError(
            "No master embeddings file in %s. Expected embeddings.npy or embeddings.bin (+ metadata.json)" % self.master_dir
        )

    def getCorpus(self, split=["train"]):
        corpus = [
            annotation["anonymized"]
            for game in getListGames(split, task="caption")
            for annotation in json.load(open(os.path.join(self.path, game, self.labels)))["annotations"]
        ]
        return corpus

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        clip_id, (caption_id, caption) = self.data[idx]
        game_id, half, frame = clip_id
        key = str(game_id)
        if key not in self._mapping:
            raise KeyError("Game id %s not in master mapping (listGames order may not match)", game_id)
        m = self._mapping[key]
        if half == 0:
            half_start = m["half1_start"]
            half_len = m["half1_len"]
        else:
            half_start = m["half2_start"]
            half_len = m["half2_len"]
        start = min(frame, half_len - self.window_size_frame)
        start = max(0, start)
        end = half_start + start + self.window_size_frame
        vfeats = np.array(self._master_mmap[half_start + start : end], dtype=np.float32, copy=True)
        caption_tokens = self.text_processor(caption)
        return vfeats, caption_tokens, clip_id[0], caption_id, caption

    def detokenize(self, tokens, remove_EOS=True):
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
    def __init__(self, SoccerNetPath, PredictionPath, features="ResNET_TF2_PCA512.npy", split=["train"], version=2, framerate=2, window_size=15):
        self.path = SoccerNetPath
        self.PredictionPath = PredictionPath
        self.listGames = getListGames(split, task="caption")
        self.features = features
        self.window_size_frame = window_size*framerate
        self.version = version
        self.labels, _, self.dict_event, _ = getMetaDataTask("caption", "SoccerNet", version)
        self.split = split

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
        return self._cached_load(game_id)

    @lru_cache(maxsize=16)
    def _cached_load(self, game_id):
        """Load and pad features for a single game (lazy loading)."""
        game = self.listGames[game_id]
        feat_half1 = np.load(os.path.join(self.path, game, "1_" + self.features))
        feat_half1 = np.pad(feat_half1.reshape(-1, feat_half1.shape[-1]), ((self.l_pad, self.r_pad), (0, 0)), "edge")
        feat_half2 = np.load(os.path.join(self.path, game, "2_" + self.features))
        feat_half2 = np.pad(feat_half2.reshape(-1, feat_half2.shape[-1]), ((self.l_pad, self.r_pad), (0, 0)), "edge")
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
