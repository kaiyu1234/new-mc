# score path for main experiments or robustness experiments
# example: ./results/mmw_book_report/Llama_3.2_3B_Instruct/main_exp/score.txt
# score_path='./results/mmw_book_report/Mistral_7B_Instruct_v0.3/mcmark/score_back_translation.txt'
# score_path='./results/mmw_book_report/Mistral_7B_Instruct_v0.3/mcmark/score_random_token_replacement_eps_0_05.txt'
# score_path='./results/mmw_book_report/Mistral_7B_Instruct_v0.3/mcmark/score_gpt_rephrase.txt'
# score_path='./results/mmw_fake_news/Mistral_7B_Instruct_v0.3/mcmark/score_random_token_replacement_eps_0_1.txt'
score_path='./results/c4_subset/Mistral_7B_Instruct_v0.3/none/score.txt'
# autodl-tmp/results/finance_qa/Mistral_7B_Instruct_v0.3/mcmark/score.txt
# autodl-tmp/results/longform_qa/Mistral_7B_Instruct_v0.3/mcmark/score.txt
# autodl-tmp/results/mmw_story/Mistral_7B_Instruct_v0.3/mcmark/score.txt
# autodl-tmp/results/c4_subset/Mistral_7B_Instruct_v0.3/mcmark/score.txt
# autodl-tmp/results/dolly_cw/Mistral_7B_Instruct_v0.3/mcmark/score.txt
fpr=0.00001

# for baselines
# python ./evaluations/get_baselines_acc.py \
#     --fpr_thres $fpr \
#     --score_path $score_path
baseline_score_path='./results/c4_subset/Mistral_7B_Instruct_v0.3/none/score.txt'

# autodl-tmp/results/c4_subset/Mistral_7B_Instruct_v0.3/none-mcmark/score.txt
# for MC-mark:
python ./evaluations/get_mcmark_acc.py \
    --fpr_thres $fpr \
    --baseline_score_path $baseline_score_path \
    --score_path $score_path
