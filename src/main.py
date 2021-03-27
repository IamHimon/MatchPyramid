import argparse
from src.utils import init_logger, load_w2v
from src.model import MatchPyramidMatcher
from src.dataset import MSRPDataset

parser = argparse.ArgumentParser(
    description='Train and Evaluate MatchPyramid on MSRP dataset')

# main parameters
parser.add_argument("--data_path", type=str, default="/Users/himon/Jobs/class/paper8/part3/MatchPyramid_torch/data/MSRC/",
                    help="")
parser.add_argument("--dump_path", type=str, default="./dump/",
                    help="")
parser.add_argument("--embedding_path", type=str, default="/Users/himon/resource/glove.6B.50d.txt",
                    help="")
parser.add_argument("--max_seq_len", type=int, default=50,
                    help="")
parser.add_argument("--batch_size", type=int, default=16,
                    help="")
parser.add_argument("--lr", type=float, default=0.0001,
                    help="")
parser.add_argument("--n_epochs", type=int, default=4,
                    help="")

# model parameters
parser.add_argument("--dim_embedding", type=int, default=50,
                    help="")
parser.add_argument("--dim_output", type=int, default=2,
                    help="")

parser.add_argument("--conv1_size", type=str, default="5_5_8",
                    help="第一层filter")
parser.add_argument("--pool1_size", type=str, default="10_10",
                    help="地一层pooling层结果")
parser.add_argument("--conv2_size", type=str, default="3_3_16",
                    help="第二层filter")
parser.add_argument("--pool2_size", type=str, default="5_5",
                    help="地二层pooling层结果")
parser.add_argument("--mp_hidden", type=int, default="128",
                    help="")
parser.add_argument("--dim_out", type=int, default="2",
                    help="")

# parse arguments
params = parser.parse_args()

# check parameters

logger = init_logger(params)

params.word2idx, params.glove_weight = load_w2v(params.embedding_path, params.dim_embedding)

train_data = MSRPDataset(params.data_path, data_type="msr_paraphrase_train")
test_data = MSRPDataset(params.data_path, data_type="msr_paraphrase_test")

params.train_data = train_data
params.test_data = test_data

mp_model = MatchPyramidMatcher(params)
mp_model.run()
