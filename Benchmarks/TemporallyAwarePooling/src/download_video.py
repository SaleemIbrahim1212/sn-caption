import os
from  SoccerNet.Downloader import SoccerNetDownloader as SNdl
from config import SOCCERNET_PASSWORD, BASE_PATH, PATH_224P, PATH_720P
from config.setup import validate_config, print_config
from SoccerNet.Evaluation.utils import getMetaDataTask


def download_videos(
    resolutions=None,
    splits=None,
    features=None,
    download_embeddings=True,
    download_videos_flag=True,
    download_captions=False,
    version=2,
):
    """
    Download SoccerNet videos (and optionally embeddings) to configured paths.
    
    Args:
        resolutions: List of resolutions to download. Options: ['224p', '720p']
                    Default: both resolutions
        splits: List of dataset splits to download. 
               Default: ['train', 'valid', 'test', 'challenge']
        features: List of embedding feature filenames to download.
                  Example: ['baidu_soccer_embeddings.npy']
                  Default: ['baidu_soccer_embeddings.npy'] when download_embeddings is True
        download_embeddings: Whether to download embedding features (SoccerNet caption task files).
        download_videos_flag: Whether to download video files.
        download_captions: Whether to download caption label files.
        version: SoccerNet version for caption labels.
    """
    # Validate configuration
    validate_config()
    print_config()
    print()
    
    if resolutions is None:
        resolutions = ['224p', '720p']
    if splits is None:
        splits = ['train', 'valid', 'test', 'challenge']
    if download_embeddings and features is None:
        features = ['baidu_soccer_embeddings.npy']
    
    # Download 224p videos
    if download_videos_flag and '224p' in resolutions:
        print("=" * 50)
        print("Downloading 224p videos...")
        print(f"Destination: {PATH_224P}")
        print("=" * 50)
        
        os.makedirs(PATH_224P, exist_ok=True)
        downloader_224 = SNdl(LocalDirectory=PATH_224P)
        downloader_224.password = SOCCERNET_PASSWORD
        downloader_224.downloadGames(
            files=["1_224p.mkv", "2_224p.mkv"], 
            split=splits
        )
        print("224p download complete!\n")
    
    # Download 720p videos
    if download_videos_flag and '720p' in resolutions:
        print("=" * 50)
        print("Downloading 720p videos...")
        print(f"Destination: {PATH_720P}")
        print("=" * 50)
        
        os.makedirs(PATH_720P, exist_ok=True)
        downloader_720 = SNdl(LocalDirectory=PATH_720P)
        downloader_720.password = SOCCERNET_PASSWORD
        downloader_720.downloadGames(
            files=["1_720p.mkv", "2_720p.mkv", "video.ini"], 
            split=splits
        )
        print("720p download complete!\n")

    # Download embeddings/features
    if download_embeddings:
        print("=" * 50)
        print("Downloading embeddings/features...")
        print(f"Destination: {BASE_PATH}")
        print(f"Features: {', '.join(features)}")
        if download_captions:
            print("Including caption labels")
        print("=" * 50)

        os.makedirs(BASE_PATH, exist_ok=True)
        downloader_feat = SNdl(LocalDirectory=BASE_PATH)
        downloader_feat.password = SOCCERNET_PASSWORD

        feature_files = []
        for feat in features:
            feature_files.extend([f"1_{feat}", f"2_{feat}"])
        if download_captions:
            labels, _, _, _ = getMetaDataTask("caption", "SoccerNet", version)
            feature_files = [labels] + feature_files

        downloader_feat.downloadGames(
            files=feature_files,
            split=splits,
            task="caption"
        )
        print("Embeddings/features download complete!\n")
    
    print("All downloads complete!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Download SoccerNet videos')
    parser.add_argument(
        '--resolution', 
        nargs='+', 
        choices=['224p', '720p'], 
        default=['224p', '720p'],
        help='Video resolutions to download (default: both)'
    )
    parser.add_argument(
        '--split', 
        nargs='+', 
        choices=['train', 'valid', 'test', 'challenge'], 
        default=['train', 'valid', 'test', 'challenge'],
        help='Dataset splits to download (default: all)'
    )
    parser.add_argument(
        '--features',
        nargs='+',
        default=['baidu_soccer_embeddings.npy'],
        help='Embedding feature filenames to download (default: baidu_soccer_embeddings.npy)'
    )
    parser.add_argument(
        '--no-embeddings',
        action='store_true',
        help='Skip downloading embeddings/features'
    )
    parser.add_argument(
        '--no-videos',
        action='store_true',
        help='Skip downloading videos'
    )
    parser.add_argument(
        '--captions',
        action='store_true',
        help='Download caption label files along with embeddings'
    )
    parser.add_argument(
        '--version',
        type=int,
        default=2,
        help='SoccerNet version for caption labels (default: 2)'
    )
    
    args = parser.parse_args()
    download_videos(
        resolutions=args.resolution,
        splits=args.split,
        features=args.features,
        download_embeddings=not args.no_embeddings,
        download_videos_flag=not args.no_videos,
        download_captions=args.captions,
        version=args.version
    )
