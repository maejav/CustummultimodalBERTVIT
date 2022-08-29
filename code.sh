# python3.7 -m pip install numpy
cd vqamed2019/
# python3.7 train.py --run_name PREmedOrgan --epochs 100 --vocab_size 17 --category Organ --model_dir ../VQAmedSave/recorderPREmedOrganFINAL_acc.pt
#### len class organ 17
### len class in plane 16
### len class abnormality
####len class alll 
 #### in pre med general 30522

# python3.7 train.py --run_name PREmedPlane --epochs 400 --vocab_size 30522  --category Plane
# python3.7 train.py --run_name PREmedAbnormality --epochs 200 --vocab_size 1671  --category Abnormality --model_dir ../VQAmedSave/recorderpremedAbnormality230_acc.pt
# python3.7 train.py --run_name PREmedAllcat --epochs 400 --vocab_size 30522  --category all
# python3.7 train.py --run_name PREmedAllcat --epochs 400  --category all
python3.7 train.py --run_name PREmedAbnormality --epochs 200 --vocab_size 1671  --category Abnormality --model_dir ../VQAmedSave/recorderPREmedAbnormalityFINAL200_acc.pt



cd vqarad/
# source ../MMBERT/env_rad/bin/activate
# bert-base-multilingual-uncased
# bert-base-multilingual-cased
# bert-base-uncased
# bert-base-cased   #### NEED PRETAINING
# python3.7 train_vqarad.py --run_name PreRadpropose --epoch 400   ###515
