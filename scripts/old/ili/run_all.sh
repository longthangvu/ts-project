#!/bin/bash
./scripts/ili/FEDformer.sh > ./logs/ili/FEDformer.log
./scripts/ili/ForecastPFN.sh > ./logs/ili/ForecastPFN.log
./scripts/ili/iTransformer.sh > ./logs/ili/iTransformer.log
./scripts/ili/Nonstationary_Transformer.sh > ./logs/ili/Nonstationary_Transformer.log
./scripts/ili/PatchTST.sh > ./logs/ili/PatchTST.log
./scripts/ili/PAttn.sh > ./logs/ili/PAttn.log