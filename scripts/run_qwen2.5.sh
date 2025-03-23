cd ../src

devices="6,7"

# LongRAG baseline
CUDA_VISIBLE_DEVICES=$devices python -W "ignore" main_longrag.py --temperature 0 --model qwen2.5-7b-instruct --r_path processed/200_2_2_e5 --version 0114_100_7_p1_01 --dataset 2wikimultihopqa --top_k1 100 --top_k2 7 --ext_rb --rb_ext_fil --prompt_version 1
CUDA_VISIBLE_DEVICES=$devices python -W "ignore" main_longrag.py --temperature 0 --model qwen2.5-7b-instruct --r_path processed/200_2_2_e5 --version 0114_100_7_p1_01 --dataset hotpotqa --top_k1 100 --top_k2 7 --ext_rb --rb_ext_fil --prompt_version 1
CUDA_VISIBLE_DEVICES=$devices python -W "ignore" main_longrag.py --temperature 0 --model qwen2.5-7b-instruct --r_path processed/200_2_2_e5 --version 0114_100_7_p1_01 --dataset musique --top_k1 100 --top_k2 7 --ext_rb --rb_ext_fil --prompt_version 1

# 2wikimultihopqa with chunk_ext=2
CUDA_VISIBLE_DEVICES=$devices python -W "ignore" main_macrag.py --temperature 0 --model qwen2.5-7b-instruct --r_path processed/sum_450_300_raw_1500_500_e5 --version 0114_100_7_01 --dataset 2wikimultihopqa --top_k1 100 --top_k2 7  --rb --rl --ext --fil --ext_fil --ext_rb --rb_ext_fil --chunk_ext 2 --with_reranking 1 --prompt_version 1
CUDA_VISIBLE_DEVICES=$devices python -W "ignore" main_macrag.py --temperature 0 --model qwen2.5-7b-instruct --r_path processed/sum_600_400_raw_1500_500_e5 --version 0114_100_7_01 --dataset 2wikimultihopqa --top_k1 100 --top_k2 7  --rb --rl --ext --fil --ext_fil --ext_rb --rb_ext_fil --chunk_ext 2 --with_reranking 1 --prompt_version 1

# hotpotqa with chunk_ext=1
CUDA_VISIBLE_DEVICES=$devices python -W "ignore" main_macrag.py --temperature 0 --model qwen2.5-7b-instruct --r_path processed/sum_450_300_raw_1500_500_e5 --version 0114_100_7_01 --dataset hotpotqa --top_k1 100 --top_k2 7  --ext_rb --rb_ext_fil --chunk_ext 1 --with_reranking 1 --prompt_version 1
CUDA_VISIBLE_DEVICES=$devices python -W "ignore" main_macrag.py --temperature 0 --model qwen2.5-7b-instruct --r_path processed/sum_600_400_raw_1500_500_e5 --version 0114_100_7_01 --dataset hotpotqa --top_k1 100 --top_k2 7  --ext_rb --rb_ext_fil --chunk_ext 1 --with_reranking 1 --prompt_version 1

# musique with chunk_ext=1
CUDA_VISIBLE_DEVICES=$devices python -W "ignore" main_macrag.py --temperature 0 --model qwen2.5-7b-instruct --r_path processed/sum_450_300_raw_1500_500_e5 --version 0114_100_7_01 --dataset musique --top_k1 100 --top_k2 7  --ext_rb --rb_ext_fil --chunk_ext 1 --with_reranking 1 --prompt_version 1
CUDA_VISIBLE_DEVICES=$devices python -W "ignore" main_macrag.py --temperature 0 --model qwen2.5-7b-instruct --r_path processed/sum_600_400_raw_1500_500_e5 --version 0114_100_7_01 --dataset musique --top_k1 100 --top_k2 7  --rb --rl --ext --fil --ext_fil --ext_rb --rb_ext_fil --chunk_ext 1 --with_reranking 1 --prompt_version 1



