#!/bin/bash
# usage: `./generate_hw_ini DDR` or `./generate_hw_ini HBM <number of channels>`

hw_config=../src/spmspv.ini

cat <<EOF > $hw_config
[connectivity]
sp=spmspv.vector:HBM[30]
sp=spmspv.result:HBM[31]
EOF

if [ $1 = "DDR" ];then
cat <<EOF >> spmspv.ini
sp=spmspv.mat_0:DDR[0]
EOF
elif [ "$1" = "HBM" ];then
  for (( i = 0; i < $2; i++ )); do
    echo "sp=spmspv.mat_$i:HBM[$i]" >> $hw_config
  done
fi
