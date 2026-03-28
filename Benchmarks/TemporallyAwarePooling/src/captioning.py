import os
import logging
from datetime import datetime
import time
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from dataset import SoccerNetCaptions, PredictionCaptions, collate_fn_padd
from model import Video2Caption, SoccerNetTransformerCaption
from train import trainer, test_captioning, validate_captioning

from utils import valid_probability

import wandb

def resolve_device(args):
    if getattr(args, "device", None) is not None:
        return torch.device(args.device)
    if getattr(args, "GPU", -1) >= 0 and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_caption_pool(args):
    caption_type = str(getattr(args, "caption_type", "baseline")).strip().lower()
    if caption_type != "transformer":
        return args.pool

    modality = str(getattr(args, "transformer_modality", "video")).strip().lower()
    modality_to_pool = {
        "video": "Transformer_Video",
        "audio": "Transformer_Audio",
        "both": "Transformer",
    }
    if modality not in modality_to_pool:
        raise ValueError(
            f"Incorrect modality --transformer_modality='{modality}'. "
        )
    return modality_to_pool[modality]


def caption_dataset_kw(args):
    cap_mod = (
        str(args.transformer_modality).strip().lower()
        if str(args.caption_type).strip().lower() == "transformer"
        else "video"
    )
    mad = getattr(args, "master_audio_dir", None)
    if isinstance(mad, str) and mad.strip() == "":
        mad = None
    if cap_mod in ("audio", "both") and not mad:
        raise ValueError(
            "Set --master_audio_dir to the folder with audio_mapping.json and audio_features.dat "
            "when using --transformer_modality audio or both."
        )
    return dict(
        path=args.SoccerNet_path,
        features=args.features,
        version=args.version,
        framerate=args.framerate,
        window_size=args.window_size_caption,
        mapping_json=args.mapping_json,
        feature_file=args.feature_file,
        caption_modality=cap_mod,
        master_audio_dir=mad if cap_mod in ("audio", "both") else None,
    )


def resolve_caption_feature_dims(args, dataset_Test):
    s0 = dataset_Test[0]
    if str(args.caption_type).strip().lower() == "transformer":
        mod = str(args.transformer_modality).strip().lower()
        if mod == "both":
            if args.feature_dim is None:
                args.feature_dim = s0[0].shape[-1]
            args.audio_feature_dim = s0[1].shape[-1]
        elif mod == "audio":
            if args.feature_dim is None:
                args.feature_dim = s0[0].shape[-1]
            args.audio_feature_dim = args.feature_dim
        else:
            if args.feature_dim is None:
                args.feature_dim = s0[0].shape[-1]
            args.audio_feature_dim = None
    else:
        if args.feature_dim is None:
            args.feature_dim = s0[0].shape[-1]
        args.audio_feature_dim = None
    print("feature_dim:", args.feature_dim, "audio_feature_dim:", getattr(args, "audio_feature_dim", None))


def main(args):
    device = resolve_device(args)
    caption_pool = resolve_caption_pool(args)

    logging.info("Parameters:")
    for arg in vars(args):
        logging.info(arg.rjust(15) + " : " + str(getattr(args, arg)))

    d_kw = caption_dataset_kw(args)
    # create dataset
    if not args.test_only:
        dataset_Train = SoccerNetCaptions(split=args.split_train, **d_kw)
        dataset_Valid = SoccerNetCaptions(split=args.split_valid, **d_kw)
        dataset_Valid_metric  = SoccerNetCaptions(split=args.split_valid, **d_kw)
    dataset_Test  = SoccerNetCaptions(split=args.split_test, **d_kw)

    resolve_caption_feature_dims(args, dataset_Test)
    # create model

    if str(args.caption_type).strip().lower() == "transformer":
        model = SoccerNetTransformerCaption(vocab_size=dataset_Test.vocab_size, weights=args.load_weights, input_size=args.feature_dim,
                  window_size=args.window_size_caption, 
                  framerate=args.framerate,
                  pool=caption_pool,
                  num_layers=args.num_layers,
                  teacher_forcing_ratio=args.teacher_forcing_ratio,
                  word_dropout=args.word_dropout,
                  freeze_encoder=args.freeze_encoder,
                  weights_encoder=args.weights_encoder,
                  contrastive_weights_path=args.contrastive_weights_path,
                  freeze_contrastive_encoder=args.freeze_contrastive_encoder,
                  unfreeze_contrastive_projection=args.unfreeze_contrastive_projection,
                  audio_input_size=getattr(args, "audio_feature_dim", None)).to(device)
    else:
        model = Video2Caption(vocab_size=dataset_Test.vocab_size, weights=args.load_weights, input_size=args.feature_dim,
                    window_size=args.window_size_caption, 
                    vlad_k = args.vlad_k,
                    framerate=args.framerate,
                    pool=args.pool,
                    num_layers=args.num_layers,
                    teacher_forcing_ratio=args.teacher_forcing_ratio, word_dropout=args.word_dropout, freeze_encoder=args.freeze_encoder, weights_encoder=args.weights_encoder).to(device)
        
    logging.info(model)
    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    parameters_per_layer  = [p.numel() for p in model.parameters() if p.requires_grad]
    logging.info("Total number of parameters: " + str(total_params))

    if not args.test_only:
        #rng = np.random.default_rng(args.seed)  
        #rng.shuffle(dataset_Train.data) # Doing this because shuffling has been turned off
        train_loader = torch.utils.data.DataLoader(dataset_Train,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.max_num_worker, pin_memory=True, collate_fn=collate_fn_padd, persistent_workers=(args.max_num_worker > 0) )

        val_loader = torch.utils.data.DataLoader(dataset_Valid,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.max_num_worker, pin_memory=True, collate_fn=collate_fn_padd, persistent_workers = (args.max_num_worker > 0) )

        val_metric_loader = torch.utils.data.DataLoader(dataset_Valid_metric,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.max_num_worker, pin_memory=True, collate_fn=collate_fn_padd,  persistent_workers = (args.max_num_worker > 0))


    # training parameters
    if not args.test_only:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        if str(args.caption_type).strip().lower() == "transformer":
            named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
            layer0_names = (
                "encoder.pooling_layer.video_transformer.layers.0.",
                "encoder.pooling_layer.audio_transformer.layers.0.",
            )
            layer1_names = (
                "encoder.pooling_layer.video_transformer.layers.1.",
                "encoder.pooling_layer.audio_transformer.layers.1.",
            )
            encoder_layer0_params = [param for name, param in named_params if any(token in name for token in layer0_names)]
            encoder_layer1_params = [param for name, param in named_params if any(token in name for token in layer1_names)]
            layer_param_ids = {id(p) for p in encoder_layer0_params + encoder_layer1_params}
            other_encoder_params = [
                param for name, param in named_params
                if name.startswith("encoder.") and id(param) not in layer_param_ids
            ]
            other_params = [param for name, param in named_params if not name.startswith("encoder.")]
            param_groups = []
            if other_params:
                param_groups.append({"params": other_params, "lr": args.LR})
            if other_encoder_params:
                param_groups.append({"params": other_encoder_params, "lr": 1e-5})
            if encoder_layer0_params:
                param_groups.append({"params": encoder_layer0_params, "lr": 1e-5})
            if encoder_layer1_params:
                param_groups.append({"params": encoder_layer1_params, "lr": 2e-5})
            optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
            logging.info(
                "Transformer optimizer groups: non-encoder=%d (lr=%.2e), encoder other=%d (lr=1e-5), layer0=%d (lr=1e-5), layer1=%d (lr=2e-5)",
                len(other_params), args.LR, len(other_encoder_params), len(encoder_layer0_params), len(encoder_layer1_params)
            )
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.LR,
                                        betas=(0.9, 0.999), eps=1e-08,
                                        weight_decay=0, amsgrad=False)


        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=args.patience)

        # start training
        trainer("caption", train_loader, val_loader, val_metric_loader, 
                model, optimizer, scheduler, criterion,
                model_name=args.model_name,
                max_epochs=args.max_epochs, evaluation_frequency=args.evaluation_frequency,
                log_every_n_batches=args.log_every_n_batches)

    # For the best model only
    checkpoint = torch.load(os.path.join("models", args.model_name, "caption","model.pth.tar"), map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)

    # validate caption generation on groundtruth spots on multiple splits [test/challenge]
    for split in args.split_test:

        dataset_Test  = SoccerNetCaptions(split=[split], **d_kw)

        test_loader = torch.utils.data.DataLoader(dataset_Test,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.max_num_worker, pin_memory=True, collate_fn=collate_fn_padd)

        try:
            results = validate_captioning(test_loader, model, args.model_name)
        except ImportError as e:
            logging.warning(f"Skipping caption validation: {e}")
            continue
        if results is None:
            continue

        logging.info("Best Performance at end of training in generating captions")
        logging.info(f'| Bleu_1: {results["Bleu_1"]}')
        logging.info(f'| Bleu_2: {results["Bleu_2"]}')
        logging.info(f'| Bleu_3: {results["Bleu_3"]}')
        logging.info(f'| Bleu_4: {results["Bleu_4"]}')
        logging.info(f'| METEOR: {results["METEOR"]}')
        logging.info(f'| ROUGE_L: {results["ROUGE_L"]}')
        logging.info(f'| CIDEr: {results["CIDEr"]}')

        wandb.log({f"{k}_{split}_gt" : v for k, v in results.items()})


    return 

def dvc(args):
    device = resolve_device(args)
    caption_pool = resolve_caption_pool(args)

    logging.info("Parameters:")
    for arg in vars(args):
        logging.info(arg.rjust(15) + " : " + str(getattr(args, arg)))

    d_kw = caption_dataset_kw(args)
    dataset_Test  = SoccerNetCaptions(split=args.split_test, **d_kw)

    resolve_caption_feature_dims(args, dataset_Test)
    # create model

    if str(args.caption_type).strip().lower() == "transformer":
        model = SoccerNetTransformerCaption(vocab_size=dataset_Test.vocab_size, weights=args.load_weights, input_size=args.feature_dim,
                  window_size=args.window_size_caption, 
                  framerate=args.framerate,
                  pool=caption_pool,
                  num_layers=args.num_layers,
                  teacher_forcing_ratio=args.teacher_forcing_ratio,
                  word_dropout=args.word_dropout,
                  freeze_encoder=args.freeze_encoder,
                  weights_encoder=args.weights_encoder,
                  contrastive_weights_path=args.contrastive_weights_path,
                  freeze_contrastive_encoder=args.freeze_contrastive_encoder,
                  unfreeze_contrastive_projection=args.unfreeze_contrastive_projection,
                  audio_input_size=getattr(args, "audio_feature_dim", None)).to(device)
    else: 
        model = Video2Caption(vocab_size=dataset_Test.vocab_size, weights=args.load_weights, input_size=args.feature_dim,
                    window_size=args.window_size_caption, 
                    vlad_k = args.vlad_k,
                    framerate=args.framerate,
                    pool=args.pool,
                    num_layers=args.num_layers,
                    teacher_forcing_ratio=args.teacher_forcing_ratio, word_dropout=args.word_dropout).to(device)
    logging.info(model)
    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    parameters_per_layer  = [p.numel() for p in model.parameters() if p.requires_grad]
    logging.info("Total number of parameters: " + str(total_params))

    # For the best model only
    checkpoint = torch.load(os.path.join("models", args.model_name, "caption","model.pth.tar"), map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)

    # generate dense caption on multiple splits [test/challenge]
    for split in args.split_test:
        PredictionPath = os.path.join("models", args.model_name, f"outputs/{split}")
        dataset_Test  = PredictionCaptions(
            SoccerNetPath=args.SoccerNet_path,
            PredictionPath=PredictionPath,
            features=d_kw["features"],
            split=[split],
            version=d_kw["version"],
            framerate=d_kw["framerate"],
            window_size=d_kw["window_size_caption"],
            mapping_json=d_kw["mapping_json"],
            feature_file=d_kw["feature_file"],
            caption_modality=d_kw["caption_modality"],
            master_audio_dir=d_kw["master_audio_dir"],
        )

        test_loader = torch.utils.data.DataLoader(dataset_Test,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.max_num_worker, pin_memory=True)

        results = test_captioning(test_loader, model, args.model_name)
        if results is None:
            continue

        logging.info("Best Performance at end of training in dense video captioning")
        logging.info(f'| Bleu_1_tight: {results["Bleu_1_tight"]}')
        logging.info(f'| Bleu_2_tight: {results["Bleu_2_tight"]}')
        logging.info(f'| Bleu_3_tight: {results["Bleu_3_tight"]}')
        logging.info(f'| Bleu_4_tight: {results["Bleu_4_tight"]}')
        logging.info(f'| METEOR_tight: {results["METEOR_tight"]}')
        logging.info(f'| ROUGE_L_tight: {results["ROUGE_L_tight"]}')
        logging.info(f'| CIDEr_tight: {results["CIDEr_tight"]}')
        logging.info(f'| Recall_tight: {results["Recall_tight"]}')
        logging.info(f'| Precision_tight: {results["Precision_tight"]}')

        logging.info(f'| Bleu_1_loose: {results["Bleu_1_loose"]}')
        logging.info(f'| Bleu_2_loose: {results["Bleu_2_loose"]}')
        logging.info(f'| Bleu_3_loose: {results["Bleu_3_loose"]}')
        logging.info(f'| Bleu_4_loose: {results["Bleu_4_loose"]}')
        logging.info(f'| METEOR_loose: {results["METEOR_loose"]}')
        logging.info(f'| ROUGE_L_loose: {results["ROUGE_L_loose"]}')
        logging.info(f'| CIDEr_loose: {results["CIDEr_loose"]}')
        logging.info(f'| Recall_loose: {results["Recall_loose"]}')
        logging.info(f'| Precision_loose: {results["Precision_loose"]}')

        wandb.log({f"{k}_{split}_pt" : v for k, v in results.items()})


if __name__ == '__main__':


    parser = ArgumentParser(description='SoccerNet-Caption: Captioning training', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--SoccerNet_path',   required=False, type=str,   default="/kaggle/input/datasets/salzeem/soccernet/data",     help='Path for SoccerNet' )
    parser.add_argument('--features',   required=False, type=str,   default="baidu_soccer_embeddings.npy",     help='Video features' )
    parser.add_argument('--mapping_json', required=False, type=str, default="/kaggle/input/datasets/salzeem/soccernet-densefile-at-45-1fps/mapping.json", help='Path to memmap row mapping json')
    parser.add_argument('--feature_file', required=False, type=str, default="/kaggle/input/datasets/salzeem/soccernet-densefile-at-45-1fps/features.dat", help='Path to memmap feature file')
    parser.add_argument('--master_audio_dir', required=False, type=str, default=None, help='Directory with audio_mapping.json and audio_features.dat (e.g. ../../data/master_audio)')
    parser.add_argument('--max_epochs',   required=False, type=int,   default=100,     help='Maximum number of epochs' )
    parser.add_argument('--load_weights',   required=False, type=str,   default=None,     help='weights to load' )
    parser.add_argument('--model_name',   required=False, type=str,   default="NetVLAD-Transformer-memapfixed",     help='named of the model to save' )
    parser.add_argument('--test_only',   required=False, action='store_true',  help='Perform testing only' )

    parser.add_argument('--split_train', nargs='+', default=["train"], help='list of split for training')
    parser.add_argument('--split_valid', nargs='+', default=["valid"], help='list of split for validation')
    parser.add_argument('--split_test', nargs='+', default=["test"], help='list of split for testing')

    parser.add_argument('--version', required=False, type=int,   default=2,     help='Version of the dataset' )
    parser.add_argument('--feature_dim', required=False, type=int,   default=None,     help='Number of input features' )
    parser.add_argument('--evaluation_frequency', required=False, type=int,   default=1,     help='Number of chunks per epoch' )
    parser.add_argument('--log_every_n_batches', required=False, type=int, default=20, help='Log caption batch stats every N batches (<=0 disables)' )
    parser.add_argument('--framerate', required=False, type=int,   default=1,     help='Framerate of the input features' )
    parser.add_argument('--window_size_caption', required=False, type=int,   default=45,     help='Size of the chunk (in seconds)' )
    parser.add_argument('--pool',       required=False, type=str,   default="NetVLAD++", help='How to pool for non-transformer captioning' )
    parser.add_argument('--transformer_modality', required=False, type=str, choices=["video", "audio", "both"], default="video", help='Transformer modality to run when --caption_type=Transformer' )
    parser.add_argument('--vlad_k',       required=False, type=int,   default=64, help='Size of the vocabulary for NetVLAD' )
    parser.add_argument('--min_freq',       required=False, type=int,   default=5, help='Minimum word frequency to the vocabulary for caption generation' )
    
    parser.add_argument('--teacher_forcing_ratio',  required=False, type=valid_probability,   default=1.0, help='Teacher forcing ratio to use' )
    parser.add_argument('--word_dropout', required=False, type=valid_probability, default=0.01, help='Word dropout probability in decoder teacher forcing path')
    parser.add_argument('--num_layers',  required=False, type=int,   default=2, help='Teacher forcing ratio to use' )
    parser.add_argument('--freeze_encoder',  required=False, type=bool, default=False)
    parser.add_argument('--pretrain',   required=False, action='store_true',  help='Perform testing only' )
    parser.add_argument('--weights_encoder',  required=False, type=str, default=None)
    parser.add_argument('--contrastive_weights_path', required=False, type=str, default="/kaggle/input/models/salzeem/sbertcontrastive/pytorch/default/1/best.pth", help='Path to contrastive encoder checkpoint to preload Transformer_Video')
    parser.add_argument('--freeze_contrastive_encoder', dest='freeze_contrastive_encoder', action='store_true', help='Freeze Transformer_Video encoder after loading --contrastive_weights_path')
    parser.add_argument('--no_freeze_contrastive_encoder', dest='freeze_contrastive_encoder', action='store_false', help='Do not freeze Transformer_Video encoder after loading --contrastive_weights_path')
    parser.add_argument('--unfreeze_contrastive_projection', action='store_true', help='When --freeze_contrastive_encoder is set, keep encoder.pooling_layer.video_proj trainable')
    parser.set_defaults(freeze_contrastive_encoder=True)
    parser.set_defaults(unfreeze_contrastive_projection=True)
    parser.add_argument('--first_stage',  required=False, type=str,  choices=["spotting", "caption"], default="spotting")

    parser.add_argument('--batch_size', required=False, type=int,   default=256,     help='Batch size' )
    parser.add_argument('--LR',       required=False, type=float,   default=1e-03, help='Learning Rate' )
    parser.add_argument('--LRe',       required=False, type=float,   default=1e-06, help='Learning Rate end' )
    parser.add_argument('--patience', required=False, type=int,   default=10,     help='Patience before reducing LR (ReduceLROnPlateau)' )

    parser.add_argument('--GPU',        required=False, type=int,   default=0,     help='ID of the GPU to use' )
    parser.add_argument('--device',     required=False, type=str,   default="cuda",   help='torch device (e.g., cpu, cuda, cuda:0)' )
    parser.add_argument('--max_num_worker',   required=False, type=int,   default=2, help='number of worker to load data')
    parser.add_argument('--seed',   required=False, type=int,   default=0, help='seed for reproducibility')
    parser.add_argument('--caption_type',   required=False, type=str, choices=['Transformer', 'Baseline', 'transformer', 'baseline'], default='Transformer', help='Caption model type')

    parser.add_argument('--loglevel',   required=False, type=str,   default='INFO', help='logging level')

    args = parser.parse_args()

    # for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)

    os.makedirs(os.path.join("models", args.model_name), exist_ok=True)
    log_path = os.path.join("models", args.model_name,
                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log'))

    run = wandb.init(
    project="NetVLAD-caption",
    name=args.model_name
    )

    wandb.config.update(args)

    logging.basicConfig(
        level=numeric_level,
        format=
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ])

    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
    if args.device is None:
        args.device = "cuda" if args.GPU >= 0 and torch.cuda.is_available() else "cpu"


    start=time.time()
    logging.info('Starting main function')
    main(args)
    logging.info(f'Total Execution Time is {time.time()-start} seconds')
