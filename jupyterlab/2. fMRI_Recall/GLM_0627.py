#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from nltools.file_reader import onsets_to_dm
from nltools.stats import regress, zscore
from nltools.data import Brain_Data, Design_Matrix
from nltools.stats import find_spikes 
from nilearn.plotting import view_img, glass_brain, plot_stat_map
from nilearn import image
from bids import BIDSLayout, BIDSValidator


# In[2]:


root_dir = '/Users/janny/Desktop/2019_Gamble_fMRI'
layout = BIDSLayout(root_dir, 
                    derivatives = True)


# In[17]:


#참가자 변수 설정 
subject = layout.get_subject()
subject.sort()
subject.remove('02') #원래 데이터에서도 없음

#task 제한: "gamble"
task = layout.get_task()
task.remove('resting')
task.remove('recog')


# In[18]:


print(subject)
print(task)
len(subject)


# In[20]:


#first_first : 참가자별 contrast 값을 먼저 구함 

tr = layout.get_tr() #전체 데이터에서 가져오는 것 
n_tr = nib.load(layout.get(subject = subject[0],
                              scope = 'raw',  

                              task = task,
                              suffix = 'bold',
                              extension = 'nii.gz')[0].path).shape[-1] #모든 run 마다 동일하기 때문에 for-loop 밖에 둠. 


root_dir = '/Users/janny/Desktop/2019_Gamble_fMRI'
root_dir2 = '/Users/janny/Desktop/2019_Gamble_fMRI/derivatives/fmriprep/'
dm_concat = pd.DataFrame();
mc_covs = pd.DataFrame();
spikes = pd.DataFrame();
full = pd.DataFrame();
regressor_names = ['a_comp_cor_0'+str(i) for i in range(6)[1:]] #preprocessing에서 중요한 외생변수들을 정리해놓은 변수들임 

c_reject = [ 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
c_accept = [ 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for sub in subject:
    event_filenames = glob.glob(os.path.join(root_dir, f"sub-{sub}" , 'func', '*task-gamble*events.tsv'))
    event_filenames = [file for file in event_filenames if not file.endswith("recog_events.tsv")]
    event_filenames.sort()
    
    cov_filenames = glob.glob(os.path.join(root_dir2, f"sub-{sub}" ,'func','*gamble*regressor*.tsv'))
    cov_filenames.sort()

    brain_filenames = glob.glob(os.path.join(root_dir2,f"sub-{sub}",'func','*gamble*AROMA*.gz'))
    brain_filenames.sort()

    num_runs = [0,1,2,3]
    dm_concat = pd.DataFrame();
    mc_covs = pd.DataFrame();
    
    for i in num_runs:
    # 1) Load in onsets for this run
        onsets = pd.read_csv(event_filenames[i], sep = '\t')
        
        onsets['trial_type'] = onsets['gamble_choice']+onsets['image_condition']
        #onsets의 index 순서는 각 run에서 어떤 변수가 먼저 나왔는지에 따라 달라질 수 있기 때문이다. 그래서reindex가 중요하다. 
        onset = onsets[['onset', 'duration', 'trial_type']]
        onset.columns=['Onset','Duration','Stim'] #각 열의 이름을 함수가 이해할 수 있게 바꿔줘야 함. 
        dm = onsets_to_dm(onset, sampling_freq = 1/tr, run_length=n_tr) 

    # 2) Convolve them with the hrf, and add run_id 
        dm_conv = dm.convolve()
        dm_conv = dm_conv.reindex(sorted(dm_conv.columns), axis=1) # run 별 조건 순서를 동일하게 재정렬해준 것임(re-ordering) 
        dm_conv_run = dm_conv.add_poly() #run에 대한 값 입력 - b0 값을 추가해준 것임. 
        colnames = dm_conv_run.columns.tolist() #run 변수 값에 대한 열 이름 변경 <----???? 다시 한번 확인해보기 이해 안된당 
        colnames[-1] = f'run_{i + 1}'
        dm_conv_run.columns = colnames
        dm_concat = pd.concat([dm_concat, dm_conv_run]) #동일한 형태의 데이터 합치기       

    dm_concat = dm_concat.fillna(0) #결측치 -> 0 
    dm_concat = dm_concat.drop(dm_concat.columns[-1], axis = 1)
    dm_concat = dm_concat.reset_index(drop=True) 

    # 4) covariates
    #mc_covs = pd.DataFrame();
    for i in num_runs:
        covariates = pd.read_csv(cov_filenames[i], sep = '\t')
        mc = covariates[regressor_names] # 좀더 알아보고 싶은게 있을 땐 추가하는 거 같은데 은주님께서 어떤걸본건지 모르겠음. 
        mc.fillna(value=0, inplace=True)
        mc_cov = Design_Matrix(mc, sampling_freq=1/tr)  
        mc_covs = pd.concat([mc_covs,mc_cov])
    mc_covs.reset_index(drop=True, inplace = True)        
 
    #5) Brain_data_for_spike
    for i in num_runs:       
        datas = image.concat_imgs(brain_filenames[i])  
    data = Brain_Data(datas)
    spike = data.find_spikes(global_spike_cutoff=2.5, diff_spike_cutoff=2.5)
    spike = Design_Matrix(spikes.iloc[:,1:], sampling_freq=1/tr)
    spikes = pd.concat([spikes, spike])
    spikes.reset_index(drop=True, inplace=True)
    spikes.dropna(axis = 1, inplace = True)
    
# 6) Join the onsets and covariates together, and heatmap plotting
    full = pd.concat([dm_concat, mc_covs, spikes], axis=1) #dm마다 크기가 다름...  #전체 dm 
    full['Intercept'] = 1
# 6.5) save heatmap 
    fig, ax = plt.subplots()
    plt.style.use(['classic'])
    plt.imshow(full, aspect ='auto', cmap = 'gray',interpolation = 'none')
    ax.set_title(f'sub-{sub}'.format(subject=subject))
    ax.set_ylabel('trials')
    ax.set_xticks(np.arange(len([x for x in full.columns])))
    ax.set_xticklabels([x for x in full.columns], rotation = 90)
    plt.show()
    fig.savefig(os.path.join(root_dir,'janny_practice','fist_level1',f'sub-{sub}_design_matrix.png'.format(subject = sub, task = task)))

# 7) estimate GLM = (fitting model to data) - no smoothing, AROMA처리를 한 데이터이여서  
    data = Brain_Data(brain_filenames)
    data.X = full # brain data instance에 dm을 assign 
    result = data.regress() # data.regress = estimator 
        
            # 6) save all of the betas with join()를 호출하고 있는 실행 파일의 어떤 한 디렉토리에 새로운 파일 생성 
    result['beta'].write(os.path.join(root_dir,'janny_practice','fist_level1', 
                                             f"sub-{sub}_betas_denoised.nii.gz"))
            # 7) save separate file for eqach contrast 
    accept = result['beta'] * c_accept
    reject = result['beta'] * c_reject
    accept.write(os.path.join(root_dir,'janny_practice','contrast',  
                                             f"sub-{sub}_accept_contrast_denoised.nii.gz"))
    reject.write(os.path.join(root_dir,'janny_practice','contrast',  
                                             f"sub-{sub}_reject_contrast_denoised.nii.gz"))


# In[23]:


#one-sample t-#uncorrexted threshold for each condition : arbitrary threshold, p < .001
root_dir = '/Users/janny/Desktop/2019_Gamble_fMRI'
conditions = {0:'accep', 1:'reject'}
for i in [0,1]:
    con_file_lists = glob.glob(os.path.join(root_dir,'janny_practice','contrast',f'sub*_{conditions[i]}*nii.gz'))
    print(con_file_lists)
    con_file_lists.sort()
    con_dats = Brain_Data(con_file_lists)
    con_dats.mean().plot()
    con1_stats = con_dats.ttest(threshold_dict={'unc':.001})
    con1_stats['thr_t'].plot()
    con1_stats['thr_t'].write(os.path.join(root_dir, 'janny_practice','group_level',    #"one-sample t-test 결과 저장 
                                             f"gamble_{conditions[i]}_uncorrected_ttest_denoised.nii.gz"))


# In[24]:


#참가자들 간의 평균 구하고 FDR 하기 
#FDR threshold for each condition 
for i in [0,1]:
    con_file_lists = glob.glob(os.path.join(root_dir,'janny_practice','contrast',f'sub*_{conditions[i]}*nii.gz'))
    con_file_lists.sort()
    con_dat = Brain_Data(con_file_lists)
    con_dats.mean().plot()
    con1_stats = con_dat.ttest(threshold_dict={'fdr':.05})
    con1_stats['thr_t'].plot()
    con1_stats['thr_t'].write(os.path.join(root_dir, 'janny_practice','group_level',    #"one-sample t-test 결과 저장 
                                             f"gamble_{conditions[i]}_FDR_ttest_denoised.nii.gz"))
    con_dats.append(con_dat)
    #print(con_dats)


# In[ ]:


#fist_second : 각 참가자별 조건별 베타값 및 t 값 저장 


# ## contrast  - 6/29에 이어서 할 부분 
# 
# 1. 각 조건당 전체 참가자의 데이터를 불러온다 ->  집단 수준에서 contrast : 이때 평균값을 사용하면 안되는 이유는 평균값에는 분산 값이 포함되어 있지 않기 때문이다. 
# 2. 각 참가자당 개별 contrast를 구한다 -> 그리고 이 값을 평균을 내고 임계값 설정을 한다. >> 교수님은 이 방법 추천,, ?
# 
# contrast가 정확히 어떤걸 어떻게 측정하는 건지 잘모르겠다.. 다시 공부하기 
# 
# Contrasts are a very important concept in fMRI data analysis as they provide the statistical inference underlying the subtraction method of making inferences.
# 
# dm_heatmap & dm 은 group_level 참가자별 contrast 는 contrast 파일에 있음 
#  

# In[ ]:





# In[20]:


#first_first : 참가자별 contrast 값을 먼저 구함 

tr = layout.get_tr() #전체 데이터에서 가져오는 것 
n_tr = nib.load(layout.get(subject = subject[0],
                              scope = 'raw',  

                              task = task,
                              suffix = 'bold',
                              extension = 'nii.gz')[0].path).shape[-1] #모든 run 마다 동일하기 때문에 for-loop 밖에 둠. 


root_dir = '/Users/janny/Desktop/2019_Gamble_fMRI'
root_dir2 = '/Users/janny/Desktop/2019_Gamble_fMRI/derivatives/fmriprep/'
dm_concat = pd.DataFrame();
mc_covs = pd.DataFrame();
spikes = pd.DataFrame();
full = pd.DataFrame();
regressor_names = ['a_comp_cor_0'+str(i) for i in range(6)[1:]]
c_reject = [ 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
c_accept = [ 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for sub in subject:
    event_filenames = glob.glob(os.path.join(root_dir, f"sub-{sub}" , 'func', '*task-gamble*events.tsv'))
    event_filenames = [file for file in event_filenames if not file.endswith("recog_events.tsv")]
    event_filenames.sort()
    
    cov_filenames = glob.glob(os.path.join(root_dir2, f"sub-{sub}" ,'func','*gamble*regressor*.tsv'))
    cov_filenames.sort()

    brain_filenames = glob.glob(os.path.join(root_dir2,f"sub-{sub}",'func','*gamble*AROMA*.gz'))
    brain_filenames.sort()

    num_runs = [0,1,2,3]
    dm_concat = pd.DataFrame();
    mc_covs = pd.DataFrame();
    
    for i in num_runs:
    # 1) Load in onsets for this run
        onsets = pd.read_csv(event_filenames[i], sep = '\t')
        
        onsets['trial_type'] = onsets['gamble_choice']+onsets['image_condition']
        onset = onsets[['onset', 'duration', 'trial_type']]
        onset.columns=['Onset','Duration','Stim'] #각 열의 이름을 함수가 이해할 수 있게 바꿔줘야 함. 
        dm = onsets_to_dm(onset, sampling_freq = 1/tr, run_length=n_tr) 

    # 2) Convolve them with the hrf, and add run_id 
        dm_conv = dm.convolve()
        dm_conv = dm_conv.reindex(sorted(dm_conv.columns), axis=1) # run 별 조건 순서를 동일하게 재정렬해준 것임(re-ordering) 
        dm_conv_run = dm_conv.add_poly() #run에 대한 값 입력 - b0 값을 추가해준 것임. 
        colnames = dm_conv_run.columns.tolist() #run 변수 값에 대한 열 이름 변경 <----???? 다시 한번 확인해보기 이해 안된당 
        colnames[-1] = f'run_{i + 1}'
        dm_conv_run.columns = colnames
        dm_concat = pd.concat([dm_concat, dm_conv_run]) #동일한 형태의 데이터 합치기       

    dm_concat = dm_concat.fillna(0) #결측치 -> 0 
    dm_concat = dm_concat.drop(dm_concat.columns[-1], axis = 1)
    dm_concat = dm_concat.reset_index(drop=True) 

    # 4) covariates
    #mc_covs = pd.DataFrame();
    for i in num_runs:
        covariates = pd.read_csv(cov_filenames[i], sep = '\t')
        mc = covariates[regressor_names] # 좀더 알아보고 싶은게 있을 땐 추가하는 거 같은데 은주님께서 어떤걸본건지 모르겠음. 
        mc.fillna(value=0, inplace=True)
        mc_cov = Design_Matrix(mc, sampling_freq=1/tr)  
        mc_covs = pd.concat([mc_covs,mc_cov])
    mc_covs.reset_index(drop=True, inplace = True)        
 
    #5) Brain_data_for_spike
    for i in num_runs:       
        datas = image.concat_imgs(brain_filenames[i])  
    data = Brain_Data(datas)
    spike = data.find_spikes(global_spike_cutoff=2.5, diff_spike_cutoff=2.5)
    spike = Design_Matrix(spikes.iloc[:,1:], sampling_freq=1/tr)
    spikes = pd.concat([spikes, spike])
    spikes.reset_index(drop=True, inplace=True)
    spikes.dropna(axis = 1, inplace = True)
    
# 6) Join the onsets and covariates together, and heatmap plotting
    full = pd.concat([dm_concat, mc_covs, spikes], axis=1) #dm마다 크기가 다름...  #전체 dm 
    full['Intercept'] = 1
# 6.5) save heatmap 
    fig, ax = plt.subplots()
    plt.style.use(['classic'])
    plt.imshow(full, aspect ='auto', cmap = 'gray',interpolation = 'none')
    ax.set_title(f'sub-{sub}'.format(subject=subject))
    ax.set_ylabel('trials')
    ax.set_xticks(np.arange(len([x for x in full.columns])))
    ax.set_xticklabels([x for x in full.columns], rotation = 90)
    plt.show()
    fig.savefig(os.path.join(root_dir,'janny_practice','fist_level1',f'sub-{sub}_design_matrix.png'.format(subject = sub, task = task)))

# 7) estimate GLM = (fitting model to data) - no smoothing, AROMA처리를 한 데이터이여서  
    data = Brain_Data(brain_filenames)
    data.X = full # brain data instance에 dm을 assign 
    result = data.regress() # data.regress = estimator 
        
            # 6) save all of the betas with join()를 호출하고 있는 실행 파일의 어떤 한 디렉토리에 새로운 파일 생성 
    result['beta'].write(os.path.join(root_dir,'janny_practice','fist_level1', 
                                             f"sub-{sub}_betas_denoised.nii.gz"))
            # 7) save separate file for eqach contrast 
    

