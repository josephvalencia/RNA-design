#!/bin/bash
source nfs6/BB/Hendrix_Lab/valejose/bioseq2seq/venv/bin/activate 
cat ${1} | parallel --gnu --lb -j 44 --tmpdir .  eval {} 
