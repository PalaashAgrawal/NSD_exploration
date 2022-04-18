import numpy as np
import pandas as pd
from pathlib import Path
import h5py
from scipy.stats import pearsonr
import pickle
from tqdm import tqdm
import nibabel as nib
import sys
import traceback
#FORMAT FOR USE: python buildRDM.py subject region pathway(optional)
#subject: 1-8
#region: v1,v2,v3,v4
#pathway: v,d. If nothing is specified, both v and d will be considered. 
print('starting buildRDM script')

path = Path('/om2/user/palaash/nsd')
info_file_path = Path(path)/'betas_shared1000/shared_1000_info.csv'
session_file_path = Path(path)/'betas_shared1000' #where to find the fmri files (subject wise)


#VARIABLE S TO CHANGE
subject = int(sys.argv[1])
fmri_path = session_file_path/f'subj{subject:02d}'
region = sys.argv[2]
pathway = sys.argv[3] if len(sys.argv)>3 else None
if pathway is not None: assert pathway in ('v', 'd'), f'pathway should be v or d (for ventral and dorsal respectively)'

def get_stim(subject, info_file_path):
	info_file = pd.read_csv(info_file_path)
	fmri_order_list=[]
	for i,vals in (info_file.iterrows()): 
		
		#FOR THE CASE WHEN ALL REPS ARE AVAILABLE. BUT SADLY, NOT ALL PATIENTS SAW ALL THE IMAGES, 
		#moreover LAST 3 SESSIONS NOT PROVIDED. ie, fmris will be of different lengths, and RDMs cannot be calculated this way
		
		#orders = [vals[f'subject{subject}_rep{rep}'] for rep in range(3)]
		#orders = list(filter(lambda x: x<=27750, orders))
		#if orders: fmri_order_list.append(orders) #get list of rep0,rep1 and rep2 values for each row
		
		order = vals[f'subject{subject}_rep0']
		session = order//750 + 1
		if get_file_path(fmri_path,session).exists(): fmri_order_list.append(order)
	stim = sorted(fmri_order_list) #, key = lambda x: x[0]) #use key when you're appending a list of 3 items in the fmri_order_list
	#stim = {i: order for i,order in enumerate(stim)}
	return stim 


def get_file_path(fmri_path,session_id):
	return fmri_path/f'betas_session{session_id:02d}.hdf5'

def get_chunkified_stimulus_order(stimulus_order:dict):
	f'Its easier to access all the files from a hdf5 file at once rather than opening the hdf5 file repeatedly for each stimulus. Helper function for that'
	chunks = {}
	for order in stimulus_order:
		session = order//750 + 1 #session files are 1-ordered
		if session not in chunks: chunks[session] = [(order-1)%750]
		else: chunks[session].append((order-1)%750)
		#order is provided as 1-ordered. But to access them efficiently, we convert them to 0 ordered. 
	
	for session in sorted(chunks.keys()): yield session,chunks[session]

def apply_tfms(x, tfms: list):
	for tfm in tfms: x = tfm(x)
	return x

def noop(x): return x


def get_fmri_list(stimulus_order:dict, subj:int, tfms = None): #we arent using subj in our function, but we'll use it in openmind, where each fmri betas are stored under the corresponding subj folder
	f'''
	get list of fmri's from the hdf5 files, on which tfms are applied (eg ROIs),
	given the stimulus_order dictionary which corresponds to the subject "subj" is being considered
	'''
	tfms = list(tfms) if tfms else [noop]
	fmri_list = []
	for session, indices in get_chunkified_stimulus_order(stimulus_order):
		fmri_file = fmri_path/f'betas_session{session:02d}.hdf5'
		if fmri_file.exists():
			with h5py.File(fmri_file,"r") as f: 
				beta = f['betas'][()]
				fmris = apply_tfms(beta[indices], tfms)
				fmri_list.extend(list(fmris))
	return fmri_list


_prf_visualroi_index = {
	'v1v':1.,
	'v1d':2.,
	'v2v':3.,
	'v2d':4.,
	'v3v':5.,
	'v3d':6.,
	'v4':7. #hv4
}



def get_roi_data(roi_file_paths):
	roi_file_paths = list(roi_file_paths)
	return [nib.load(roi_path).get_fdata().transpose(2,1,0) for roi_path in roi_file_paths]

def get_roi_regions(roi_file_index, region, pathway):
	if pathway is None: pathway = ''
	keys = [key for key in roi_file_index if key.startswith(region+pathway)]
	print('keys extracted: ',keys)
	return [roi_file_index[key] for key in keys]


class apply_roi():
	def __init__(self, roi_file_paths,roi_file_index =_prf_visualroi_index, region = 'v1', pathway = None):
		f'''
		roi_file_paths( str or Path or list of str/Paths): path of roi file (eg. <path>/'lh.prf-visualrois.nii.gz'). Many a times, ROI files corresponding to any region are present as multiple files. 
		eg: lh.prf-visualrois and rh.prf-visualrois. You can provide a list of paths, provided the roi_file_index mapping is common for all of them. 
        
		roi_file_index: dictionary mapping region name to index in the roi file. 
		region (str): v1,v2,v3 or v4
		pathway: None, 'v' (ventral) or 'd' (dorsal). If pathway =None, both ventral and dorsal are chosen
		'''
		
		#Assertions
		assert roi_file_paths is not None
		if pathway: assert pathway in ('v','d'), f"invalid pathway. Should be None, 'v' or 'd'"
		if region=='v4': assert pathway is None, f"ventral or dorsal pathways dont exist for the hv4 region. Use pathway=None"
		#____________________________________________________________________________________________________
		self.roi =  sum((roi==reg).astype(float) for roi in get_roi_data(roi_file_paths) for reg in get_roi_regions(roi_file_index,region, pathway))

	def __call__(self,fmri):
		return np.multiply(fmri,self.roi)



def construct_RDM(activations):
	num_images = len(activations)
	RDM = np.zeros((num_images, num_images))

	for x in range(num_images):
		for y in range(num_images):
			if x<=y: #because they're symmetric
			# get the pearson correlation
				correl = 1 - (pearsonr(activations[x].flatten(), activations[y].flatten()))[0]
				RDM[x][y] = correl
				RDM[y][x] = correl
	return RDM.astype(float)


try:
	stim = get_stim(subject, info_file_path)
	print('got stim')
	roi_paths = [fmri_path/'lh.prf-visualrois.nii.gz',fmri_path/'rh.prf-visualrois.nii.gz']
	tfms = apply_roi(roi_paths, region = region, pathway = pathway)
	print('got tfms')
	activations = get_fmri_list(stim, subject, tfms = None)
	print('got activations')
	rdm = construct_RDM(activations)
	print('got rdm')

	with open(path/f'RDM/RDM_subj{subject}_{region}.pkl','wb') as f: pickle.dump(rdm, f)
except Exception as e:
	print('SOMETHING WENT WRONG. Tried to score {0} and failed. Here is the Exception'.format(sys.argv[1]))
	print(e);
	#print('printing error stack')
	traceback.print_exc()
