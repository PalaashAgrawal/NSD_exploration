#!/bin/bash
#SBATCH --job-name=RDM_Calculation
#SBATCH --array=0-3
#SBATCH --time=92:00:00
#SBATCH --ntasks=1
#SBATCH --mem=400000

subject=$1
i=0
for region in 'v1' 'v2' 'v3' 'v4';
do 
	regions[$i]="$region"
	i=$[$i+1]
done

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running subject $subject, region ${regions[$SLURM_ARRAY_TASK_ID]}"

python buildRDM.py $subject "${regions[$SLURM_ARRAY_TASK_ID]}"
