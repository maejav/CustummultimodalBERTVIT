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




# cd vqamed2019/
# source ../MMBERT/env_rad/bin/activate
# bert-base-multilingual-uncased
# bert-base-multilingual-cased
# bert-base-uncased
# bert-base-cased   #### NEED PRETAINING
# python3.7 train_vqarad.py --run_name PreRadpropose --epoch 400   ###515
# python3.7 train.py --run_name PREmedAbnormalityRE2 --epochs 400  --vocab_size 1671  --category Abnormality --model_dir ../VQAmedSave/recorderPREmedAbnormalityRE1FINAL_acc.pt
# python3.7 train.py --run_name PREmedOrganRE1 --epochs 400 --vocab_size 17 --category Organ --model_dir ../../MMBERT/VQAmedSave/recorderPREmedOrganFINAL_acc.pt
# python3.7 train.py --run_name PREmedAllcatRE1 --epochs 1  --vocab_size 1749 --category all --model_dir ../../MMBERT/VQAmedSave/recorderPREmedAllcat22FINAL_acc.pt



# python3.7 -m pip install numpy
# cd vqamed2019/
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




# cd vqamed2019/
# source ../MMBERT/env_rad/bin/activate
# bert-base-multilingual-uncased
# bert-base-multilingual-cased
# bert-base-uncased
# bert-base-cased   #### NEED PRETAINING
# python3.7 train_vqarad.py --run_name PreRadpropose --epoch 400   ###515
# python3.7 train.py --run_name PREmedAbnormalityRE1 --epochs 400  --vocab_size 1671  --category Abnormality --model_dir ../../MMBERT/VQAmedSave/recorderPREmedAbnormalityRE1FINAL_acc.pt
# python3.7 train.py --run_name Organfinal --epochs 200  --category Organ 
# python3.7 train.py --run_name OrganfinalORIG --epochs 3   --vocab_size 119547   --model_dir ../VQAmedSave/recorder_preMEDcasedkomakOIG30.pt  --category Organ --bert_model bert-base-uncased
  
# ####Final effort

# python3.7 train.py --run_name OrganfinalORIGTwo12 --epochs  50  --vocab_size 17   --model_dir ../VQAmedSave/recorderOrganfinalORIGTwo1FINAL50_acc.pt  --category Organ --bert_model bert-base-uncased
# python3.7 train.py --run_name PlanefinalORIGTwo12 --epochs 50   --vocab_size 16   --model_dir   ../VQAmedSave/recorderPlanefinalORIGTwo1FINAL50_acc.pt --category Plane --bert_model bert-base-uncased
# python3.7 train.py --run_name AllcatfinalORIGTwo --epochs 200   --vocab_size 30522   --model_dir  ../roco_mlm/PREMEDcasedFinal510BESTVAL.pt --category all --bert_model bert-base-uncased
# python3.7 train.py --run_name AbnormalityfinalORIGTwo --epochs 10   --vocab_size 30522   --model_dir  ../roco_mlm/PREMEDcasedFinal510BESTVAL.pt --category Abnormality --bert_model bert-base-uncased
# python3.7 train.py --run_name Organfinal600test --epochs  20  --vocab_size  17    --model_dir ../VQAmedSave/recorderOrganfinal600FINAL10_acc.pt --category Organ --bert_model bert-base-uncased   --hidden_dropout_prob 0.7

# ##############   HELOOOOOOOOOO  Programmer
# python3.7 train.py --run_name OrganTesttt  --epochs  15  --vocab_size 30522   --model_dir ../roco_mlm/PREMEDcasedFinal41BESTVAL.pt  --category Organ --bert_model bert-base-uncased   --hidden_dropout_prob 0.3
python3.7 train.py --run_name PlaneTestttFinal --epochs  15  --vocab_size 16    --model_dir ../VQAmedSave/PlaneTestttBESTVALPlane.pt  --category Plane --bert_model bert-base-uncased   --hidden_dropout_prob 0.3
# python3.7 train.py --run_name AllCATfinal800 --epochs 5  --vocab_size 1741    --model_dir ../VQAmedSave/AllCATfinal800FINAL70.pt --category  all --bert_model bert-base-uncased  --factor 0.1  --patience 10    --hidden_dropout_prob 0.5  --lr 0.00001
# python3.7 train.py --run_name Abnormalityfinal800 --epochs 30   --vocab_size   1670  --model_dir ../VQAmedSave/Abnormalityfinal800FINAL20.pt  --category Abnormality --bert_model bert-base-uncased  --factor 0.1  --patience 5   --hidden_dropout_prob 0.2






# python3.7 train.py --run_name AllcatfinalORIGThree --epochs  100  --vocab_size 30522    --model_dir ../roco_mlm/PREMEDcasedFinal510BESTVAL.pt --category  all  --bert_model  bert-base-uncased   







# python3.7 train.py --run_name modaltest --epochs  5  --vocab_size 30522   --model_dir  ../roco_mlm/PREMEDcasedFinal510BESTVAL.pt --category Modality --bert_model bert-base-uncased   
# python3.7 train.py --run_name modaltest --epochs  5  --vocab_size 47   --model_dir  ../VQAmedSave/recordermodaltestFINAL_acc.pt --category Modality --bert_model bert-base-uncased   

# PREMEDcasedFinal55BESTVAL.pt
# python3.7 train.py --run_name PlanefinalORIGTwo12 --epochs 50   --vocab_size 16   --model_dir   ../VQAmedSave/recorderPlanefinalORIGTwo1FINAL50_acc.pt --category Plane --bert_model bert-base-uncased
# python3.7 train.py --run_name AllcatfinalORIGTwo --epochs 200   --vocab_size 30522   --model_dir  ../roco_mlm/PREMEDcasedFinal510BESTVAL.pt --category all --bert_model bert-base-uncased
# python3.7 train.py --run_name AbnormalityfinalORIGTwo --epochs 200   --vocab_size 30522   --model_
# PREMEDcasedFinal53BESTVAL.pt
# python3.7 train.py --run_name OrganfinalORIG --epochs 100   --category Organ
# python3.7 train.py --run_name PlanefinalORIG --epochs 100   --category Plane 
# python3.7 train.py --run_name AbnormalityfinalORIG --epochs 100   --category Abnormality 
# python3.7 train.py --run_name AllcatfinalORIG --epochs 100    --category all

# python3.7 train.py --run_name Planefinal --epochs 200  --category Plane 
# python3.7 train.py --run_name Abnormalityfinal --epochs 200  --category Abnormality 
# python3.7 train.py --run_name ALLCATfinal --epochs 200  --category all 


# python3.7 train.py --run_name ORGANfinal --epochs 400 --vocab_size 17 --category Organ --model_dir ../VQAmedSave/recorderPREmedOrganRE1FINAL_acc.pt ##haminjasave shodeh
# python3.7 train.py --run_name ABNORMALITYfinal --epochs 400  --vocab_size 1671  --category Abnormality --model_dir ../VQAmedSave/recorderPREmedAbnormalityRE1FINAL_acc.pt
# python3.7 train.py --run_name PREmedAllcatRE1 --epochs 400 --vocab_size 1749 --category all --model_dir ../../MMBERT/VQAmedSave/recorderPREmedAllcat22FINAL_acc.pt

