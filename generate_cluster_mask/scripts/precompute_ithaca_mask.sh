#!/bin/bash
set -e
ntotal_parts=50
for ((ipart=0; ipart < $ntotal_parts; ipart++)) do
    sbatch scripts/run.sh python generate_mask_ithaca365.py total_part=$ntotal_parts part=$ipart
done