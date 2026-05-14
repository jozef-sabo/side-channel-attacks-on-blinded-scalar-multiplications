#!/bin/bash

END=50


for CURVE in secp256k1; do
	for R_num in 64; do
		for ERR_num in 15 ; do
			for idx in $(seq 1 $END); do
				echo "running numgen for -e $ERR_num -m $R_num $CURVE idx $idx";
				python3 numbers_generator.py --original -e "$ERR_num" -m "$R_num" "$CURVE" 10000 "${CURVE}Curve_e${ERR_num}_r${R_num}_10000-${idx}_original.pkl";
			done
		done
	done
done
