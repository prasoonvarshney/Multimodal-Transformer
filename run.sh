python main.py --wandb
python main.py --wandb --self_attention_only
python main.py --wandb --use_bottleneck

python main.py --wandb --normalize 
python main.py --wandb --normalize --self_attention_only
python main.py --wandb --normalize --use_bottleneck 

python main.py --wandb --aligned
python main.py --wandb --aligned --self_attention_only
python main.py --wandb --aligned --use_bottleneck

python main.py --wandb --aligned --normalize 
python main.py --wandb --aligned --normalize --self_attention_only
python main.py --wandb --aligned --normalize --use_bottleneck

python main.py --wandb --use_bottleneck --fusion_layer 3
python main.py --wandb --use_bottleneck --n_bottlenecks 6
python main.py --wandb --use_bottleneck --n_bottlenecks 10
python main.py --wandb --use_bottleneck --n_bottlenecks 16

python main.py --wandb --use_bottleneck --lonly
python main.py --wandb --use_bottleneck --aonly
python main.py --wandb --use_bottleneck --vonly

# python main.py --wandb
# python main.py --wandb --lonly
# python main.py --wandb --aonly
# python main.py --wandb --vonly
# python main.py --aligned 
# python main.py --aligned --lonly
# python main.py --aligned --aonly
# python main.py --aligned --vonly
# python main.py --aligned --wandb --use_bottleneck
# python main.py --aligned --wandb --use_bottleneck --lonly
# python main.py --aligned --wandb --use_bottleneck --aonly
# python main.py --aligned --wandb --use_bottleneck --vonly