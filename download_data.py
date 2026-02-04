from SoccerNet.Downloader import SoccerNetDownloader as SNdl

PASSWORD = "s0cc3rn3t"

mySNdl = SNdl(LocalDirectory="data/SoccerNet")
# mySNdl.downloadDataTask(task="caption-2023", split=["train","valid", "test","challenge"]) # SN challenge 2023
mySNdl.downloadDataTask(task="caption-2024", split=["train","valid", "test","challenge"]) # SN challenge 2024

mySNdl.password = PASSWORD
# mySNdl.downloadGames(files=["1_224p.mkv", "2_224p.mkv"], split=["train","valid","test","challenge"])
# mySNdl.downloadGames(files=["1_720p.mkv", "2_720p.mkv", "video.ini"], split=["train","valid","test","challenge"])


'''
Can train/run model by running:

python Benchmarks/TemporallyAwarePooling/src/main.py \
  --SoccerNet_path data/SoccerNet/caption-2024 \
  --features baidu_soccer_embeddings.npy
'''