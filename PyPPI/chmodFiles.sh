#!/bin/bash
# set -x

cd utils
chmod +x split.sh
cd ../feature_computation/ECO
chmod +x run_ECO.sh
chmod +x cleanPSSM.sh
chmod +x genMSA.sh
chmod +x extractBlosum.sh
cd ../HSP
chmod +x run_HSP.sh
cd ../HYD
chmod +x run_HYD.sh
cd ../PHY_Char
chmod +x run_PHY_Char.sh
cd ../PHY_Prop
chmod +x run_PHY_Prop.sh
cd ../PKA
chmod +x run_PKA.sh
cd ../Pro2Vec_1D
chmod +x run_Pro2Vec_1D.sh
cd ../RAA
chmod +x run_RAA.sh
cd ../RSA
chmod +x run_RSA.sh

echo all done, have fun!
