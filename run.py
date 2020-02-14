import torch
import pickle
import argparse
import numpy as np
from data import *
from model import Encoder,Decoder
from sklearn.metrics import f1_score

USE_CUDA = torch.cuda.is_available()

def main():
    parser = argparse.ArgumentParser(description="Inference script for performing joint tasks on ATIS datasets.")
    parser.add_argument("--train_path", type=str,
                        help="path of train dataset.")
    parser.add_argument("--test_path", type=str,
                        help="path of test dataset.")
    parser.add_argument("--model_dir", type=str, default="./models/",
                        help='path for saved trained models.')

    parser.add_argument('--max_length', type=int , default=60,
                        help='max sequence length')
    parser.add_argument('--embedding_size', type=int , default=128,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=100,
                        help='dimension of lstm hidden states')

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    _, word2index, tag2index, intent2index = preprocessing(args.train_path,args.max_length)
    index2tag = {v:k for k,v in tag2index.items()}
    index2intent = {v:k for k,v in intent2index.items()}

    # Load model
    print("Loading model...")
    encoder = Encoder(len(word2index),args.embedding_size,args.hidden_size)
    decoder = Decoder(len(tag2index),len(intent2index),len(tag2index)//3,args.hidden_size*2)
    encoder.load_state_dict(torch.load(os.path.join(args.model_dir,'jointnlu-encoder.pkl'), map_location=torch.device("cuda" if USE_CUDA else "cpu")))
    decoder.load_state_dict(torch.load(os.path.join(args.model_dir,'jointnlu-decoder.pkl'), map_location=torch.device("cuda" if USE_CUDA else "cpu")))
    encoder.eval()
    decoder.eval()

    if USE_CUDA:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    
    test = open(args.test_path,"r").readlines()
    test = [t[:-1] for t in test]
    test = [[t.split("\t")[0].split(" "),t.split("\t")[1].split(" ")[:-1],t.split("\t")[1].split(" ")[-1]] for t in test]
    test = [[t[0][1:-1],t[1][1:],t[2].split("#")[0]] for t in test]

    slot_f1 = []
    intent_err = []

    for index in range(len(test)):
        test_raw = test[index][0]
        test_in = prepare_sequence(test_raw,word2index)
        test_mask = Variable(torch.BoolTensor(tuple(map(lambda s: s ==0, test_in.data)))).cuda() if USE_CUDA else Variable(torch.BoolTensor(tuple(map(lambda s: s ==0, test_in.data)))).view(1,-1)
        start_decode = Variable(torch.LongTensor([[word2index['<SOS>']]*1])).cuda().transpose(1,0) if USE_CUDA else Variable(torch.LongTensor([[word2index['<SOS>']]*1])).transpose(1,0)

        output, hidden_c = encoder(test_in.unsqueeze(0),test_mask.unsqueeze(0))
        tag_score, intent_score = decoder(start_decode,hidden_c,output,test_mask)

        v,i = torch.max(tag_score,1)
        slot_pred = list(map(lambda ii:index2tag[ii],i.data.tolist()))
        slot_gt = test[index][1]
        slot_f1.append(f1_score(slot_gt, slot_pred, average="micro"))

        
        v,i = torch.max(intent_score,1)
        intent_pred = index2intent[i.data.tolist()[0]]
        intent_gt = test[index][2]
        if intent_pred != intent_gt:
            intent_err.append([test[index][0], intent_gt, intent_pred])

        print("Input Sentence\t: ", *test[index][0])

        print("Truth\t\t: ", *slot_gt)
        print("Prediction\t: ", *slot_pred)

        print("Truth\t\t: ", intent_gt)
        print("Prediction\t: ", intent_pred)

        print()

    # print("Got slot err ", len(slot_err[0]))
    # print(*slot_err, sep="\n")
    print("Got intent err ", len(intent_err))
    print(*intent_err, sep="\n")
    print("Total ", len(test))
    print("Slot f1_micro avg %f" % np.average(slot_f1))
    print("Intent acc %f" % (1 - len(intent_err)/len(test)))

if __name__ == "__main__":
    main()