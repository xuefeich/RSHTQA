import os
import pickle
import argparse
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer

from transformers.models.bert import BertTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--passage_length_limit", type=int, default=457)
parser.add_argument("--question_length_limit", type=int, default=46)
parser.add_argument("--encoder", type=str, default="roberta")
parser.add_argument("--mode", type=str, default='train')
parser.add_argument("--roberta_model", type=str, default='')
parser.add_argument("--data_format", type=str, default="tatqa_and_hqa_field_{}.json")
parser.add_argument("--num_arithmetic_operators",type=int,default=6)
parser.add_argument("--tapas_path", type=str, default='')

args = parser.parse_args()

if args.encoder == 'roberta':
    from data.tatqa_dataset import TagTaTQATestReader, TagTaTQAReader
    # tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    tokenizer = RobertaTokenizer.from_pretrained(args.roberta_model)
    sep = '<s>'
    tokenizer.add_special_tokens({'additional_special_tokens':['<OPT>']})
elif args.encoder == 'bert':
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    sep = '[SEP]'
elif args.encoder == 'tapas':
    from transformers import TapasTokenizer
    from tag_op.data.tapas_dataset import TagTaTQAReader, TagTaTQATestReader
    tokenizer = TapasTokenizer.from_pretrained(args.tapas_path + "/tapas.large")
    tokenizer.add_special_tokens({'additional_special_tokens':['[OPT]']})
    sep = '[SEP]'


if args.mode == 'dev':
    data_reader = TagTaTQATestReader(tokenizer, args.passage_length_limit, args.question_length_limit, sep=sep)
    data_mode = ["dev"]
elif args.mode == 'test':
    data_reader = TagTaTQATestReader(tokenizer, args.passage_length_limit, args.question_length_limit, sep=sep)
    data_mode = ["test"]
else:
    data_reader = TagTaTQAReader(tokenizer, args.passage_length_limit, args.question_length_limit, sep=sep)
    data_mode = ["train"]

data_format = args.data_format
print(f'==== NOTE ====: encoder:{args.encoder}, mode:{args.mode}')

for dm in data_mode:
    dpath = os.path.join(args.input_path, data_format.format(dm))
    data = data_reader._read(dpath)
    print("skipping number: ", data_reader.skip_count)
    data_reader.skip_count = 0
    print("Save data to {}.".format(os.path.join(args.output_dir, f"tagop_{args.encoder}_cached_{dm}.pkl")))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, f"tagop_{args.encoder}_cached_{dm}.pkl"), "wb") as f:
        pickle.dump(data, f)
