import argparse

import dill
import numpy as np
import torch
import os, random
from modules.CIDGMed import CIDGMed
from modules.causal_construction_easyuse import CausaltyGraph4Visit
# from modules.causal_construction import CausaltyGraph4Visit
from training import Test, Train
from util import buildPrjSmiles
from icecream import ic

def str2bool(v):
    """Method to map string to bool for argument parser"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')



def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser()
    # mode
    parser.add_argument("--debug", default=True, type=str2bool,
                        help="debug mode, the number of samples, "
                             "the number of generations run are very small, "
                             "designed to run on cpu, the development of the use of")
    parser.add_argument("--Test", default=False, type=str2bool, help="test mode")
    parser.add_argument('--savepath', default="data/saved/", type=str,
                        help='save folder for models and results')
    # environment
    parser.add_argument('--dataset', default='mimic3', help='mimic3/mimic4')
    parser.add_argument('--resume_path', default="../saved/mimic3/trained_model_0.5546", type=str,
                        help='path of well trained model, only for evaluating the model, needs to be replaced manually')
    parser.add_argument('--device', type=int, default=0, help='gpu id to run on, negative for cpu')

    # parameters
    parser.add_argument('--dim', default=64, type=int, help='model dimension')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--dp', default=0.7, type=float, help='dropout ratio')
    parser.add_argument("--regular", type=float, default=0.005, help="regularization parameter")
    parser.add_argument('--target_ddi', type=float, default=0.06, help='expected ddi for training')
    parser.add_argument('--coef', default=2.5, type=float, help='coefficient for DDI Loss Weight Annealing')
    parser.add_argument('--epochs', default=25, type=int, help='the epochs for training')
    parser.add_argument('--seed', default=25, type=int, help='seed')
    parser.add_argument('--lambda_graph', default=0.1, type=float, help='weight for graph bias in differential attention')

    parser.add_argument("--usedemographics", default=False, type=str2bool, help="use gender and age year")
    parser.add_argument("--usenotes", default=False, type=str2bool, help="use doctors notes")
    parser.add_argument("--uselabevents", default=False, type=str2bool, help="use lab events")
    
    args = parser.parse_args()
    if args.Test and args.resume_path is None:
        raise FileNotFoundError('Can\'t Load Model Weight From Empty Dir')

    return args


if __name__ == '__main__':
    
    args = parse_args()
    
    print(args)
    set_seed(args.seed) #1234
    if not torch.cuda.is_available() or args.device < 0:
        device = torch.device("cpu")
        if not args.Test:
            print("GPU unavailable, switch to debug mode")
            args.debug = True
    else:
        device = torch.device(f'cuda:{args.device}')

    if not os.path.exists(args.savepath): # args.savepath = data/saved/args.dataset
        os.makedirs(args.savepath)
        
    data_path = f'../data/{args.dataset}/output/records_final_augmented.pkl'
    voc_path = f'../data/{args.dataset}/output/voc_final.pkl'
    ddi_adj_path = f'../data/{args.dataset}/output/ddi_A_final.pkl'
    ddi_mask_path = f'../data/{args.dataset}/output/ddi_mask_H.pkl'
    molecule_path = f'../data/{args.dataset}/input/idx2drug.pkl'
    # relevance_diag_med_path = f'../data/{args.dataset}/graphs/Diag_Med_relevance.pkl'
    # relevance_proc_med_path = f'../data/{args.dataset}/graphs/Proc_Med_relevance.pkl'
    relevance_diag_med_path = f'../data/{args.dataset}/graphs/Diag_Med_causal_effect.pkl'
    relevance_proc_med_path = f'../data/{args.dataset}/graphs/Proc_Med_causal_effect.pkl'

    with open(ddi_adj_path, 'rb') as Fin:
        ddi_adj = torch.from_numpy(dill.load(Fin)).to(device)
    with open(ddi_mask_path, 'rb') as Fin:
        ddi_mask_H = torch.from_numpy(dill.load(Fin)).to(device)
    with open(data_path, 'rb') as Fin:
        data = dill.load(Fin)
        adm_id = 0
        for patient in data:
            for adm in patient:
                adm.append(adm_id)
                adm_id += 1
        if args.debug:
            data = data[:5]
    
    with open(voc_path, 'rb') as Fin:
        voc = dill.load(Fin)
    with open(molecule_path, 'rb') as Fin:
        molecule = dill.load(Fin)
    with open(relevance_proc_med_path, 'rb') as Fin:
        relevance_proc_med = dill.load(Fin)
    with open(relevance_diag_med_path, 'rb') as Fin:
        relevance_diag_med = dill.load(Fin)

    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    voc_size = [
        len(diag_voc.idx2word),
        len(pro_voc.idx2word),
        len(med_voc.idx2word)
    ]

    # split_point = int(len(data) * 2 / 3)
    # data_train = data[:split_point]
    # eval_len = int(len(data[split_point:]) / 2)
    # data_test = data[split_point:split_point + eval_len]
    # data_eval = data[split_point + eval_len:]
    
    train_split = int(len(data) * 0.8)
    val_split = int(len(data) * 0.1)
    data_train = data[:train_split]
    data_eval = data[train_split:train_split + val_split]
    data_test = data[train_split + val_split:]

    ic(len(data), len(data_train), len(data_eval), len(data_test))
    # ic(data[0][1])
    ## len(data[0]) -> number of admissions (visit) for one patient
    ## len(data[0][0]) -> diagnoses for ith visit
    ## len(data[0][1]) -> procedures for ith visit
    ## len(data[0][2]) -> medications for ith visit
    ## len(data[0][3]) -> gender and age for ith visit (remains same for all admissions) -> ['F', 2087]
    ## len(data[0][4]) -> doctor's notes for ith visit [list of sentences]
    ## len(data[0][5]) -> lab events for ith visit [list of lists of lab id and values], [[id1, val1], [id2, val2]...]
    
    binary_projection, average_projection, smiles_list = buildPrjSmiles(molecule, med_voc.idx2word)

    relevance_diag_mole = np.dot(relevance_diag_med.to_numpy(), binary_projection)
    relevance_proc_mole = np.dot(relevance_proc_med.to_numpy(), binary_projection)
    relevance_med_mole = average_projection
    mole_relevance = [relevance_diag_mole, relevance_proc_mole, relevance_med_mole, binary_projection]
    voc_size.append(relevance_med_mole.shape[1])

    causal_graph = CausaltyGraph4Visit(data, data_train, voc_size[0], voc_size[1], voc_size[2], args.dataset)

    model = CIDGMed(
        causal_graph=causal_graph,
        mole_relevance=mole_relevance,
        tensor_ddi_adj=ddi_adj,
        dropout=args.dp,
        emb_dim=args.dim,
        voc_size=voc_size,
        device=device,
        lambda_graph=args.lambda_graph
    ).to(device)
    ic("args in main", args)
    print("1.Training Phase")
    ic(args.Test)
    if args.Test:
        print("Test mode, skip training phase")
        with open(args.resume_path, 'rb') as Fin:
            model.load_state_dict(torch.load(Fin, map_location=device))
    else:
        model = Train(model, device, data_train, data_eval, voc_size, args)

    print("2.Testing Phase")
    Test(model, device, data_test, voc_size, args)
