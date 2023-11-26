#!/bin/bash
source /nfs6/BB/Hendrix_Lab/valejose/miniconda3/bin/activate struct
cat ${1} | parallel --gnu --lb -j 4 --tmpdir .  eval {} --device {= '$_ = $job->slot() - 1' =}
