import argparse
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nGPU", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prepare",
        type=utils.str2bool,
        default=True,
        help='Prepare training or testing data. Only need to run once.'
    )
    parser.add_argument("--mode",
                        type=str,
                        default="train",
                        choices=['train', 'test', 'train_test', 
                                 'create_embeddings',
                                 'gen_discuss_data',
                                 'split_data',
                                 'test_baseline',
                                 'ad_hoc'])
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="../data/Forum_train",
    )
    parser.add_argument(
        "--test_data_dir",
        type=str,
        default="../data/Forum_dev",
    )
    parser.add_argument("--model_dir", type=str, default='../model')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--npratio", type=int, default=4)
    parser.add_argument("--enable_gpu", type=utils.str2bool, default=True)
    parser.add_argument("--filter_num", type=int, default=3)
    parser.add_argument("--log_steps", type=int, default=100)

    parser.add_argument("--model", type=str, default=None, choices=['NAML', 'NRMS'])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0001)

    parser.add_argument("--num_words_title", type=int, default=20)
    parser.add_argument("--num_words_abstract", type=int, default=50)
    parser.add_argument("--user_log_length", type=int, default=50)
    parser.add_argument("--word_embedding_dim", type=int, default=300)
    parser.add_argument("--glove_embedding_path", type=str, default='../data/glove.840B.300d.txt')
    parser.add_argument("--bpemb_embedding_path", type=str, default='../data/bpemb.320K.300d.txt.gz')
    parser.add_argument("--freeze_embedding", type=utils.str2bool, default=False)
    parser.add_argument("--news_dim", type=int, default=400)
    parser.add_argument("--news_query_vector_dim", type=int, default=200)
    parser.add_argument("--user_query_vector_dim", type=int, default=200)
    parser.add_argument("--num_attention_heads", type=int, default=20)
    parser.add_argument("--user_log_mask", type=utils.str2bool, default=False)
    parser.add_argument("--drop_rate", type=float, default=0.2)
    parser.add_argument("--save_steps", type=int, default=10000)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--load_ckpt_name", type=str, default=None)
    parser.add_argument("--use_category", type=utils.str2bool, default=False)
    parser.add_argument("--use_subcategory", type=utils.str2bool, default=False)
    parser.add_argument("--category_emb_dim", type=int, default=100)
    parser.add_argument("--show_news_doc_sim", type=utils.str2bool, default=False)
    
    parser.add_argument("--data_dir", type=str, default='../data/discuss')
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--start_date", type=str, default=None)
    parser.add_argument("--test_date", type=str, default=None)
    parser.add_argument("--skip_count_sample", type=utils.str2bool, default=False)



    args = parser.parse_args()
    return args
