echo Training Conditional Generators.....

# TRAINING CASE 1
echo CONDITIONAL GENERATOR CASE 1: UTILIZES PRETRAINED CLASSIFIER
python main_conditional.py TrainScripts/train_conditional_CASE1_SFD0_GyroData.txt
#read -p "Press enter to continue: "
clear
python main_conditional.py TrainScripts/train_conditional_CASE1_SFD1_GyroData.txt
#read -p "Press enter to continue: "
clear
python main_conditional.py TrainScripts/train_conditional_CASE1_SFD0_SportsData.txt
#read -p "Press enter to continue: "
clear
python main_conditional.py TrainScripts/train_conditional_CASE1_SFD1_SportsData.txt
#read -p "Press enter to continue: "
clear

# TRAINING CASE 2
echo CONDITIONAL GENERATOR CASE 2: NO CLASSIFIER
python main_conditional_CASE2_NO_CLASSIFIER.py TrainScripts/train_conditional_CASE2_SFD0_GyroData.txt
#read -p "Press enter to continue: "
clear
python main_conditional_CASE2_NO_CLASSIFIER.py TrainScripts/train_conditional_CASE2_SFD1_GyroData.txt
#read -p "Press enter to continue: "
clear
python main_conditional_CASE2_NO_CLASSIFIER.py TrainScripts/train_conditional_CASE2_SFD0_SportsData.txt
#read -p "Press enter to continue: "
clear
python main_conditional_CASE2_NO_CLASSIFIER.py TrainScripts/train_conditional_CASE2_SFD1_SportsData.txt
#read -p "Press enter to continue: "
clear

# TRAINING CASE 3
echo CONDITIONAL GENERATOR CASE 3: CLASSIFIER TRAINS WITH GENERATOR
python main_conditional_CASE3_NOT_PRETRAINED.py TrainScripts/train_conditional_CASE3_SFD0_GyroData.txt
#read -p "Press enter to continue: "
clear
python main_conditional_CASE3_NOT_PRETRAINED.py TrainScripts/train_conditional_CASE3_SFD1_GyroData.txt
#read -p "Press enter to continue: "
clear
python main_conditional_CASE3_NOT_PRETRAINED.py TrainScripts/train_conditional_CASE3_SFD0_SportsData.txt
#read -p "Press enter to continue: "
clear
python main_conditional_CASE3_NOT_PRETRAINED.py TrainScripts/train_conditional_CASE3_SFD1_SportsData.txt