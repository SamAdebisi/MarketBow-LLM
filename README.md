
1. Go to https://github.com/settings/tokens and create new token by clicking `Generate New Token` button. Give read access to public repositories.
2. Copy the access token and set the env variable via `export GH_ACCESS_TOKEN=<copied access token>`.
3. `cd dataset_creation` and Run `python clone_hf_repos.py`
4. The data in `hf_public_repos` folder in current repo should look like below:
```
accelerate          candle   datasets       diffusers               notebooks  pytorch-image-models       tokenizers    trl
alignment-handbook  chat-ui  deep-rl-class  diffusion-models-class  peft       text-generation-inference  transformers
```
5. Download nltk punkt
```python
import nltk
nltk.download('punkt')
```
6. Run Data Pipeline on a machine with 16 CPUs:
```
python pipeline.py
```
7. Collate and push to hub:
```
python prepare_hf_dataset.py
```
8. Create tokenizer:
```
cd ../tokenizer_creation
python create_tokenizer.py
```
9. Training the model from scratch using DeepSpeed on 8 A100 GPUs
10. Look at the loss plots
11. No evaluations were carried out because a 7B more training from scratch require atleast a ~1 Trillion tokens and have only trained on ~110 Million tokens which is 10^4 order less.

Note:
1. if you are getting dataset related issue `Loading a streaming dataset cached in a LocalFileSystem is not supported yet.`, run `pip install -U datasets`

### SFT
1. Go to SFT subfloder and run the below command
```
cd Module5/sft
bash run_sft.sh
```

### DPO
1. Go to DPO subfloder and run the below command after the SFt finetuning is completed.
```
cd Module5/dpo
bash run_dpo.sh
```

### PPO/RLHF
1. Go to PPO subfloder and open the notebook and run all the cells.