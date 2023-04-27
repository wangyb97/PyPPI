from structuralFeatures import *
from fileOperation import *

# biofeature=generateBIOFeatures(ECO=True, RAA=True,HSP=False)
# PhysChemFeatures=generatePhysChemFeatures(HYD=True, PHY_Prop=True, PKA=True, PHY_Char=True)
# print('PhysChemFeatures:',PhysChemFeatures.shape)


seq_name=getID('seq_2.fasta') # ['2hrkB','1ktzA']
for i in seq_name:
    protein=i[0:4]
    chain=i[-1]
    # print(protein,chain)
    # print(isinstance(protein,str)) #ÅÐ¶ÏÊÇ·ñÎª×Ö·û´®
    getDSSP(protein,chain)