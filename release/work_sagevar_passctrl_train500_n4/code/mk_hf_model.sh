hf_model_path="YOUR PATH TO THE TOKENIZER"

for number in 50 100 150
# for example, you want to test the model at 50, 100, 150 global steps.
do
target_dir="/YOUR PATH TO THE TARGET DIRECTORY/global_step_$number"
python scripts/model_merger.py \
--local_dir "/YOUR PATH TO THE CHECKPOINT DIRECTORY/global_step_$number/actor" \
--target_dir $target_dir \
--backend fsdp \
--hf_model_path $hf_model_path \

cp $hf_model_path/tokenizer_config.json $target_dir/tokenizer_config.json
cp $hf_model_path/tokenizer.json $target_dir/tokenizer.json
cp $hf_model_path/vocab.json $target_dir/vocab.json
done