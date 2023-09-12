#PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/tag_op python tag_op/prepare_dataset.py --input_path ./dataset_extra_field/ --output_dir tag_op/data/ --encoder roberta --mode train --roberta_model ./model/roberta.large
#PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/tag_op python tag_op/prepare_dataset.py --input_path ./dataset_test_hqa --output_dir tag_op/data/test/ --encoder roberta --mode test --roberta_model ./model/roberta.large --data_format tathqa_dataset_{}.json
CUDA_VISIBLE_DEVICES=1,2,3,4 PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/tag_op nohup python tag_op/trainer.py --data_dir tag_op/data/both --save_dir tag_op/model_L2I --batch_size 32 --eval_batch_size 32 --max_epoch 50 --warmup 0.06 --optimizer adam --learning_rate 5e-4  --weight_decay 5e-5 --seed 123 --gradient_accumulation_steps 4 --bert_learning_rate 1.5e-5 --bert_weight_decay 0.01 --log_per_updates 100 --eps 1e-6  --encoder roberta --test_data_dir tag_op/data/ --roberta_model ./model/roberta.large --cross_attn_layer 0 --do_finetune 0 --num_ops 6 --data_dir ./tag_op/data/ &

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:$(pwd) python tag_op/predictor.py --data_dir tag_op/data/test --test_data_dir tag_op/data/test --save_dir tag_op/model_L2I --eval_batch_size 32 --model_path tag_op/model_L2I --encoder roberta --roberta_model ./model/roberta.large --cross_attn_layer 0 --num_ops 6
