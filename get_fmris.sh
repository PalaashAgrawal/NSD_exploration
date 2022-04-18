subject=$1

mkdir betas_shared1000/subj0$subject

aws s3 cp s3://natural-scenes-dataset/nsddata_betas/ppdata/subj0$subject/func1pt8mm/betas_fithrf_GLMdenoise_RR/ betas_shared1000/subj0$subject --recursive --exclude "*" --include "*.hdf5"

aws s3 cp s3://natural-scenes-dataset/nsddata/ppdata/subj0$subject/func1pt8mm/roi/lh.prf-visualrois.nii.gz betas_shared1000/subj0$subject

aws s3 cp s3://natural-scenes-dataset/nsddata/ppdata/subj0$subject/func1pt8mm/roi/rh.prf-visualrois.nii.gz betas_shared1000/subj0$subject
