cd ../src

devices1="0,1,2"
devices2="3,4,5"
devices3="6,7"

# LongRAG baseline
# CUDA_VISIBLE_DEVICES=$devices python -W "ignore" main_longrag_0114.py --temperature 0 --model llama3.1-8b-instruct --r_path processed/200_2_2_e5 --version 0114_100_12_p1_01 --dataset 2wikimultihopqa --top_k1 100 --top_k2 12 --rb --rl --ext --fil --ext_fil --ext_rb --rb_ext_fil --prompt_version 1
# CUDA_VISIBLE_DEVICES=$devices python -W "ignore" main_longrag_0114.py --temperature 0 --model llama3.1-8b-instruct --r_path processed/200_2_2_e5 --version 0114_100_12_p1_01 --dataset hotpotqa --top_k1 100 --top_k2 12 --rb --rl --ext --fil --ext_fil --ext_rb --rb_ext_fil --prompt_version 1
# CUDA_VISIBLE_DEVICES=$devices python -W "ignore" main_longrag_0114.py --temperature 0 --model llama3.1-8b-instruct --r_path processed/200_2_2_e5 --version 0114_100_12_p1_01 --dataset musique --top_k1 100 --top_k2 12 --rb --rl --ext --fil --ext_fil --ext_rb --rb_ext_fil --prompt_version 1

# # # 2wikimultihopqa with chunk_ext=2
CUDA_VISIBLE_DEVICES=$devices1 python -W "ignore" main_macrag.py --temperature 0 --model llama3.1-8b-instruct --r_path processed/sum_450_300_raw_1500_500_e5 --version 0208_100_7_01_wo_merge --dataset 2wikimultihopqa --top_k1 100 --top_k2 7  --rb --rl --ext --fil --ext_fil --ext_rb --rb_ext_fil --chunk_ext 0 --merge_version 0 --upscaling 4 --with_reranking 1 --prompt_version 1
# CUDA_VISIBLE_DEVICES=$devices2 python -W "ignore" main_macrag.py --temperature 0 --model llama3.1-8b-instruct --r_path processed/sum_450_300_raw_1500_500_e5 --version 0208_100_7_01_w_merge --dataset 2wikimultihopqa --top_k1 100 --top_k2 7  --rb --rl --ext --fil --ext_fil --ext_rb --rb_ext_fil --chunk_ext 0 --merge_version 1 --upscaling 4 --with_reranking 1 --prompt_version 1
# CUDA_VISIBLE_DEVICES=$devices3 python -W "ignore" main_macrag.py --temperature 0 --model llama3.1-8b-instruct --r_path processed/sum_450_300_raw_1500_500_e5 --version 0208_100_7_01_w_merge --dataset 2wikimultihopqa --top_k1 100 --top_k2 7  --rb --rl --ext --fil --ext_fil --ext_rb --rb_ext_fil --chunk_ext 1 --merge_version 1 --upscaling 4 --with_reranking 1 --prompt_version 1

# # # hotpotqa with chunk_ext=1
CUDA_VISIBLE_DEVICES=$devices1 python -W "ignore" main_macrag.py --temperature 0 --model llama3.1-8b-instruct --r_path processed/sum_450_300_raw_1500_500_e5 --version 0208_100_7_01_wo_merge --dataset hotpotqa --top_k1 100 --top_k2 7  --rb --rl --ext --fil --ext_fil --ext_rb --rb_ext_fil --chunk_ext 0 --merge_version 0 --upscaling 4 --with_reranking 1 --prompt_version 1
# CUDA_VISIBLE_DEVICES=$devices2 python -W "ignore" main_macrag.py --temperature 0 --model llama3.1-8b-instruct --r_path processed/sum_450_300_raw_1500_500_e5 --version 0208_100_7_01_w_merge --dataset hotpotqa --top_k1 100 --top_k2 7  --rb --rl --ext --fil --ext_fil --ext_rb --rb_ext_fil --chunk_ext 0 --merge_version 1 --upscaling 4 --with_reranking 1 --prompt_version 1
# CUDA_VISIBLE_DEVICES=$devices3 python -W "ignore" main_macrag.py --temperature 0 --model llama3.1-8b-instruct --r_path processed/sum_450_300_raw_1500_500_e5 --version 0208_100_7_01_w_merge --dataset hotpotqa --top_k1 100 --top_k2 7  --rb --rl --ext --fil --ext_fil --ext_rb --rb_ext_fil --chunk_ext 1 --merge_version 1 --upscaling 4 --with_reranking 1 --prompt_version 1

# # # musique with chunk_ext=1
CUDA_VISIBLE_DEVICES=$devices1 python -W "ignore" main_macrag.py --temperature 0 --model llama3.1-8b-instruct --r_path processed/sum_450_300_raw_1500_500_e5 --version 0208_100_7_01_wo_merge --dataset musique --top_k1 100 --top_k2 7  --rb --rl --ext --fil --ext_fil --ext_rb --rb_ext_fil --chunk_ext 0 --merge_version 0 --upscaling 4 --with_reranking 1 --prompt_version 1
# CUDA_VISIBLE_DEVICES=$devices2 python -W "ignore" main_macrag.py --temperature 0 --model llama3.1-8b-instruct --r_path processed/sum_450_300_raw_1500_500_e5 --version 0208_100_7_01_w_merge --dataset musique --top_k1 100 --top_k2 7  --rb --rl --ext --fil --ext_fil --ext_rb --rb_ext_fil --chunk_ext 0 --merge_version 1 --upscaling 4 --with_reranking 1 --prompt_version 1
# CUDA_VISIBLE_DEVICES=$devices3 python -W "ignore" main_macrag.py --temperature 0 --model llama3.1-8b-instruct --r_path processed/sum_450_300_raw_1500_500_e5 --version 0208_100_7_01_w_merge --dataset musique --top_k1 100 --top_k2 7  --rb --rl --ext --fil --ext_fil --ext_rb --rb_ext_fil --chunk_ext 1 --merge_version 1 --upscaling 4 --with_reranking 1 --prompt_version 1



