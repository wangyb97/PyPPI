import numpy
import os
from biologicalFeatures import *
from structuralFeatures import *
from LMFeatures import *
from fileOperation import *
from getFeatures import *


def generateStructuralFeatures(dssp=False, RSA=False, windowSize=1):
    if (RSA==True and dssp==True):
        print("The features generated using dssp and RSA are both protein structural features, but the two methods are different while computing, please choose one.")
        exit(1)
    fileName=os.environ.get('INPUT_FN')
    if RSA==True:
        Features=getRSA(RSA, windowSize)
        return Features
    if dssp==True:
        # seq_name = getID('seq_2.fasta')  # ['2hrkB','1ktzA']
        seq_name = getID(fileName)
        for i in seq_name:
            protein = i[0:4]
            chain = i[-1]
            Features=getDSSP(protein, chain)

    folder_path = 'dataset_dssp_process/dssp/'  # 指定文件夹路径

    data_list = []  # 用于存储加载的数据
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):  # 仅处理以'.npy'结尾的文件
            file_path = os.path.join(folder_path, filename)  # 构建完整的文件路径
            dssp_feature = np.load(file_path)  # 加载Numpy文件
            data_list.append(dssp_feature)  # 将加载的数据添加到列表中

    # 按照第二列拼接所有数据
    concatenated_data = np.concatenate(data_list, axis=0)
    print(concatenated_data.shape) #(氨基酸個數，14)
    return concatenated_data

def generateBIOFeatures(ECO=False, HSP=False, scoringMatrix=1,RAA=False, Pro2Vec_1D=False, PSSM=False, Anchor=False, windowSize=1):
    count=0  # count is used for observing whether the feature is used to merge or used as the first feature
    if ECO == True:
        if os.path.exists('feature_computation/txtFeatures/ECO.txt'):
            print('The ECO features have already been generated, please do not generate it again!')
            count += 1
        else:
            count+=1
            print("ECO computing!")
            cmd = PRO_DIR + "/feature_computation/ECO/run_ECO.sh"
            os.system(cmd)
        Features=enter(ECO=ECO, WINDOW_SIZE=windowSize)  # feature_3d_npArray
    if HSP == True:
        if os.path.exists('feature_computation/txtFeatures/HSP.txt'):
            print('The HSP features have already been generated, please do not generate it again!')
            count += 1
        else:
            count += 1
            if scoringMatrix not in [1, 2, 3]:
                raise Exception('The value of scoringMatrix should be in [1, 2, 3].')
            print("HSP computing!")
            cmd = PRO_DIR + '/feature_computation/HSP/run_HSP.sh ' + str(scoringMatrix)
            os.system(cmd)
        if (count == 1):
            Features = enter(HSP=HSP, WINDOW_SIZE=windowSize)  # feature_3d_npArray
        else:
            HSPFeature = enter(HSP=HSP, WINDOW_SIZE=windowSize)  # feature_3d_npArray
            Features = np.concatenate((Features, HSPFeature), axis=2)
    if RAA == True:
        count += 1
        print("RAA computing!")
        cmd = PRO_DIR + "/feature_computation/RAA/run_RAA.sh"
        os.system(cmd)
        if (count == 1):
            Features = enter(RAA=RAA, WINDOW_SIZE=windowSize)
        else:
            RAAFeature = enter(RAA=RAA, WINDOW_SIZE=windowSize)  # feature_3d_npArray
            Features = np.concatenate((Features, RAAFeature), axis=2)
    if Pro2Vec_1D == True:
        count += 1
        print("Pro2Vec_1D computing!")
        cmd = PRO_DIR + "/feature_computation/Pro2Vec_1D/run_Pro2Vec_1D.sh"
        os.system(cmd)
        if (count == 1):
            Features = enter(Pro2Vec_1D=Pro2Vec_1D, WINDOW_SIZE=windowSize)
        else:
            Pro2Vec_1DFeature = enter(Pro2Vec_1D=Pro2Vec_1D, WINDOW_SIZE=windowSize)  # feature_3d_npArray
            Features = np.concatenate((Features, Pro2Vec_1DFeature), axis=2)
    if PSSM == True:
        if os.path.exists('feature_computation/txtFeatures/PSSM20.txt'):
            print('The PSSM features have already been generated, please do not generate it again!')
            count += 1
        else:
            count += 1
            print("computing PSSM!")
            os.system("cd utils && mkdir -p TMP_DIR/splitFile/ && ./split.sh INPUT_FN TMP_DIR/splitFile/ && cd ..")
            os.system("python feature_computation/PSSM/pssmQueryDB.py")
            os.system("python feature_computation/PSSM/featuresCompute.py")
            # cmd="cd ../.."
            # os.system(cmd)
        if (count == 1):
            Features = enter(PSSM=PSSM, WINDOW_SIZE=windowSize)
        else:
            PSSMFeature = enter(PSSM=PSSM, WINDOW_SIZE=windowSize)  # feature_3d_npArray
            Features = np.concatenate((Features, PSSMFeature), axis=2)
    if Anchor == True:
        if os.path.exists('feature_computation/txtFeatures/Anchor.txt'):
            print('The Anchor features have already been generated, please do not generate it again!')
            count += 1
        else:
            count += 1
            print("Anchor computing!")
            os.system("cd feature_computation/Anchor && python getAnchor.py && cd ../..")
        if (count == 1):
            Features = enter(Anchor=Anchor, WINDOW_SIZE=windowSize)
        else:
            AnchorFeature = enter(Anchor=Anchor, WINDOW_SIZE=windowSize)
            Features = np.concatenate((Features, AnchorFeature), axis=2)

    if(count==0):
        print("Please select at least one biological feature!")
        exit(1)
    return Features


def generatePhysChemFeatures(HYD=False, PHY_Prop=False, PKA=False, PHY_Char=False, windowSize=1):
    count=0
    if HYD == True:
        count+=1
        print("HYD computing!")
        cmd = PRO_DIR + "/feature_computation/HYD/run_HYD.sh"
        os.system(cmd)
        Features=enter(HYD=HYD, WINDOW_SIZE=windowSize)
        # print(Features)
    if PHY_Prop == True:
        count += 1
        print("PHY_Prop computing!")
        cmd = PRO_DIR + "/feature_computation/PHY_Prop/run_PHY_Prop.sh"
        os.system(cmd)
        if (count == 1):
            Features = enter(PHY_Prop=PHY_Prop, WINDOW_SIZE=windowSize)
        else:
            PHY_PropFeature = enter(PHY_Prop=PHY_Prop, WINDOW_SIZE=windowSize)
            Features = np.concatenate((Features, PHY_PropFeature), axis=2)
        # print(PHY_PropFeature)
    if PKA == True:
        count += 1
        print("PKA computing!")
        cmd = PRO_DIR + "/feature_computation/PKA/run_PKA.sh"
        os.system(cmd)
        if (count == 1):
            Features = enter(PKA=PKA, WINDOW_SIZE=windowSize)
        else:
            PKAFeature = enter(PKA=PKA, WINDOW_SIZE=windowSize)
            Features = np.concatenate((Features, PKAFeature), axis=2)
        # print(PKAFeature)
    if PHY_Char == True:
        count += 1
        print("PHY_Char computing!")
        cmd = PRO_DIR + "/feature_computation/PHY_Char/run_PHY_Char.sh"
        os.system(cmd)
        if (count == 1):
            Features = enter(PHY_Char=PHY_Char, WINDOW_SIZE=windowSize)
        else:
            PHY_CharFeature = enter(PHY_Char=PHY_Char, WINDOW_SIZE=windowSize)
            Features = np.concatenate((Features, PHY_CharFeature), axis=2)
        # print(PHY_CharFeature)
    if(count==0):
        print("Please select at least one physicochemical feature!")
        exit(1)
    # print(Features)
    return Features


def generateLanguageModelFeatures(model='ProtT5'):
    if(model not in ['ProtT5', 'ESM_1b', 'ProGen2']):
        raise Exception("The value of model should be among ['ProtT5', 'ESM_1b', 'ProGen2'], please check your model value!")
    if(model=='ProtT5'):
        Features = generateProtT5Features()
    if(model=='ESM_1b'):
        Features = generateESM1bFeatures()        
    if(model=='ProGen2'):
        Features = generateProgenFeatures()
    return Features