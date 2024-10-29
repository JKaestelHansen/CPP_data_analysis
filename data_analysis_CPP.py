# %%
import tifffile as tiff
import os
import matplotlib.pyplot as plt
from cellpose import utils
import tifffile as tiff
from cellpose import models
from cellpose.io import logger_setup
from stardist.models import StarDist2D, StarDist3D, Config3D, Config2D
from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
from cellpose import plot
import pickle
import numpy as np
from scipy.ndimage import gaussian_filter
import copy
from tqdm import tqdm


def flatten_list(l):
    return [item for sublist in l for item in sublist]


dirname = 'data_CPP/'
path = 'MCF-7 1uM CCPP15-30min/MCF-7 1uM CCPP15 30min-07052023-4/'
filename = '_Process_13741_/stack1/frame_t_0.pkl'

rootnames = []
filenames = []
for root, dirs, files in os.walk(dirname):
    for name in files:
        if name.endswith(".pkl") and 'frame_t_0' in name:
            filenames.append(os.path.join(root,name))
            rootnames.append(root.split(dirname)[-1])

# sort filenames and rootnames by rootnames
rootnames, filenames = zip(*sorted(zip(rootnames, filenames)))
rootnames = list(rootnames)
filenames = list(filenames)
print(len(filenames))
for f, r in zip(filenames, rootnames):
    assert f.split(dirname)[-1].split('/frame_t')[0] == r
# %%

labels = {
14405: 'hek293_pre',
14406: 'hek293_post_ccp',
14407: 'hek293_post_mem',

14408: 'hek293_pre',
14409: 'hek293_post_ccp',
14410: 'hek293_post_mem',

14411: 'hek293_pre',
14412: 'hek293_post_ccp',
14413: 'hek293_post_mem',

14456: 'hek293_pre',
14457: 'hek293_post_ccp',
14458: 'hek293_post_mem',

14025: 'HT-29_pre',
14026: 'HT-29_post_ccp',
14027: 'HT-29_post_mem',

14028: 'HT-29_pre',
14029: 'HT-29_post_ccp',
14030: 'HT-29_post_mem',

14031: 'HT-29_pre',
14032: 'HT-29_post_ccp',
14033: 'HT-29_post_mem',

14034: 'HT-29_pre',
14035: 'HT-29_post_ccp',
14036: 'HT-29_post_mem',

14037: 'Hela_pre',
14038: 'Hela_post_ccp',
14039: 'Hela_post_mem',

14040: 'Hela_pre',
14041: 'Hela_post_ccp',
14042: 'Hela_post_mem',

14043: 'Hela_pre',
14044: 'Hela_post_ccp',
14045: 'Hela_post_mem',

14046: 'Hela_pre',
14047: 'Hela_post_ccp',
14048: 'Hela_post_mem',

13741: 'mcf7_pre',
13742: 'mcf7_post_ccp',
13743: 'mcf7_post_mem',

13362: 'mcf7_pre',
13363: 'mcf7_post_ccp',
13364: 'mcf7_post_mem',

13365: 'mcf7_pre',
13368: 'mcf7_post_ccp',
13369: 'mcf7_post_mem',

13370: 'mcf7_pre',
13373: 'mcf7_post_ccp',
13374: 'mcf7_post_mem',

# # error in image saving and processing
# 13744: 'mcf7_pre_tat',
# 13745: 'mcf7_post_TAT', # last couple of frames are blank
# 13746: 'mcf7_post_tatmem', # last couple of frames look like post_TAT images have been corrupted and mixed with other files somehow

13747: 'mcf7_pre_tat', 
13748: 'mcf7_post_TAT',
13749: 'mcf7_post_tatmem',

13750: 'mcf7_pre_tat',
13751: 'mcf7_post_TAT',
13752: 'mcf7_post_tatmem',

13753: 'mcf7_pre_tat',
13754: 'mcf7_pre_tat',
13755: 'mcf7_post_TAT',
13756: 'mcf7_post_tatmem',

14364: 'mcf7_100nm_pre',
14365: 'mcf7_100nm_post_ccp',
14366: 'mcf7_100nm_post_mem',

14367: 'mcf7_100nm_pre',
14368: 'mcf7_100nm_post_ccp',
14369: 'mcf7_100nm_post_mem',

14370: 'mcf7_100nm_pre',
14371: 'mcf7_100nm_post_ccp',
14372: 'mcf7_100nm_post_mem',

14373: 'mcf7_300nm_pre',
14374: 'mcf7_300nm_post_ccp',
14375: 'mcf7_300nm_post_mem',

14376: 'mcf7_300nm_pre',
14377: 'mcf7_300nm_post_ccp',
14378: 'mcf7_300nm_post_mem',

14379: 'mcf7_300nm_pre',
14380: 'mcf7_300nm_post_ccp',
14381: 'mcf7_300nm_post_mem',

14382: 'mcf7_500nm_pre',
14383: 'mcf7_500nm_post_ccp',
14384: 'mcf7_500nm_post_mem',

14385: 'mcf7_500nm_pre',
14386: 'mcf7_500nm_post_ccp',
14387: 'mcf7_500nm_post_mem',

14388: 'mcf7_500nm_pre',
14389: 'mcf7_500nm_post_ccp',
14390: 'mcf7_500nm_post_mem',

14391: 'mcf7_3um_pre',
14392: 'mcf7_3um_post_ccp',
14393: 'mcf7_3um_post_mem',

14394: 'mcf7_3um_pre',
14397: 'mcf7_3um_post_ccp',
14398: 'mcf7_3um_post_mem',

14399: 'mcf7_3um_pre',
14400: 'mcf7_3um_post_ccp',
14401: 'mcf7_3um_post_mem',

14402: 'mcf7_5um_pre',
14403: 'mcf7_5um_post_ccp',
14404: 'mcf7_5um_post_mem',

14459: 'mcf7_5um_pre',
14460: 'mcf7_5um_post_ccp',
14461: 'mcf7_5um_post_mem',

14462: 'mcf7_5um_pre',
14463: 'mcf7_5um_post_ccp',
14464: 'mcf7_5um_post_mem',

13884: 'SH-SY5Y_pre',
13885: 'SH-SY5Y_post_ccp',
13886: 'SH-SY5Y_post_mem',

14001: 'SH-SY5Y_pre',
14002: 'SH-SY5Y_post_ccp',
14003: 'SH-SY5Y_post_mem',

14004: 'SH-SY5Y_pre',
14005: 'SH-SY5Y_post_ccp',
14006: 'SH-SY5Y_post_mem',

14007: 'SH-SY5Y_pre',
14008: 'SH-SY5Y_post_ccp',
14009: 'SH-SY5Y_post_mem',

13725: 'U20S_pre',
13726: 'U20S_post_ccp',
13727: 'U20S_post_mem',

13728: 'U20S_pre',
13729: 'U20S_post_ccp',
13730: 'U20S_post_mem',

13731: 'U20S_pre',
13734: 'U20S_post_ccp',
13735: 'U20S_post_mem',

13738: 'U20S_pre',
13739: 'U20S_post_ccp',
13740: 'U20S_post_mem',

15032: 'RPE-1_pre',
15033: 'RPE-1_post_ccp',
15034: 'RPE-1_post_mem',

15029: 'RPE-1_pre',
15030: 'RPE-1_post_ccp',
15031: 'RPE-1_post_mem',

15035: 'RPE-1_pre',
15036: 'RPE-1_post_ccp',
15037: 'RPE-1_post_mem',

15041: 'RPE-1_pre',
15042: 'RPE-1_post_ccp',
15043: 'RPE-1_post_mem',

}

# %%

i = 0

imgs = pickle.load(open(filenames[i], 'rb'))
imgs1 = pickle.load(open(filenames[i+1], 'rb'))
imgs2 = pickle.load(open(filenames[i+2], 'rb'))
print(filenames[i], imgs.shape)
print(filenames[i+1], imgs1.shape)
print(filenames[i+2], imgs2.shape)


imgs = pickle.load(open(filenames[i], 'rb'))
fig, ax = plt.subplots(1,2, figsize=(6,6))
ax[0].imshow(imgs[0,:,:]/np.mean(imgs[0,:,:]), 
             cmap='Greys_r', vmin=0.5, vmax=1.2)
ax[1].imshow(imgs[-1,:,:]/np.mean(imgs[-1,:,:]), 
             cmap='Greys_r', vmin=0.5, vmax=1.2)
ax[0].set_title('Frame 0 '+filenames[i].split('/')[-3])
ax[1].set_title('Frame 60 '+filenames[i].split('/')[-3])

fig, ax = plt.subplots(1,5, figsize=(15,6))
ax[0].imshow(imgs1[0,:,:]/np.mean(imgs1[0,:,:]), 
             cmap='Greys_r', vmin=0.5, vmax=1.2)
ax[1].imshow(imgs1[10,:,:]/np.mean(imgs1[0,:,:]), 
             cmap='Greys_r', vmin=0.5, vmax=1.2)
ax[2].imshow(imgs1[30,:,:]/np.mean(imgs1[0,:,:]),
                cmap='Greys_r', vmin=0.5, vmax=1.2)
ax[3].imshow(imgs1[40,:,:]/np.mean(imgs1[0,:,:]),
                cmap='Greys_r', vmin=0.5, vmax=1.2)
ax[4].imshow(imgs1[59,:,:]/np.mean(imgs1[0,:,:]),
                cmap='Greys_r', vmin=0.5, vmax=1.2)
ax[0].set_title('Frame 0 '+filenames[i+1].split('/')[-3])
ax[1].set_title('Frame 10 '+filenames[i+1].split('/')[-3])
ax[2].set_title('Frame 30 '+filenames[i+1].split('/')[-3])
ax[3].set_title('Frame 40 '+filenames[i+1].split('/')[-3])
ax[4].set_title('Frame 60 '+filenames[i+1].split('/')[-3])

imgs2 = pickle.load(open(filenames[i+2], 'rb'))
fig, ax = plt.subplots(1,2, figsize=(6,6))
ax[0].imshow(imgs2[0,:,:]/np.mean(imgs2[0,:,:]), 
             cmap='Greys_r', vmin=0.5, vmax=1.2)
ax[1].imshow(imgs2[59,:,:]/np.mean(imgs2[-1,:,:]), 
             cmap='Greys_r', vmin=0.5, vmax=1.2)
ax[0].set_title('Frame 0 '+filenames[i+2].split('/')[-3])
ax[1].set_title('Frame 60 '+filenames[i+2].split('/')[-3])
plt.tight_layout()

(np.sum(imgs2[-1,:,:])-np.sum(imgs2[0,:,:]))/np.sum(imgs2[0,:,:])

# %%
my_cmap = copy.copy(plt.cm.get_cmap('hsv')) 
my_cmap.set_bad(alpha=0) 
my_cmap.set_under((0,0,0,0))

i = -1
img_mem_f_ = imgs[i,:,:]/gaussian_filter(imgs[i,:,:], sigma=30)

import torch
use_GPU = True if torch.cuda.is_available() else False
model_type = 'cyto'
model = models.Cellpose(gpu=use_GPU, model_type=model_type)
channels = [[0,0]]
masks, flows, styles, diams = model.eval(img_mem_f_,
                                         diameter=75, 
                                         min_size=100,
                                         channels=channels, 
                                         z_axis=0,
                                         normalize=True,
                                         flow_threshold=0.75,
                                         )

fig, ax = plt.subplots(1,2, figsize=(12,6))
ax[0].imshow(img_mem_f_, 
             cmap='Greys_r', vmin=0.5, vmax=1.2)
ax[1].imshow(img_mem_f_, 
             cmap='Greys_r', vmin=0.5, vmax=1.2)
ax[1].imshow(masks, cmap=my_cmap, vmin=0.1, alpha=0.5)
pickle.dump(masks, open('testmasks.pkl', 'wb'))

# %%

cell_types = ['mcf7', 'Hela', 'HT-29', 'hek293', 'SH-SY5Y', 'U20S', 'RPE-1']

intensities_list_per_cell = []
ccp_f_all = []
for cell_type in tqdm(cell_types):
    ccp_names = []
    mem_names = []
    for k in labels.keys():
        if cell_type in labels[k]:
            if '_ccp' in labels[k] or 'post_TAT' in labels[k]:
                ccp_names.append(k)
            elif '_mem' in labels[k] or 'post_tatmem' in labels[k]:
                mem_names.append(k)

    intensities_list = []
    for c,m in zip(ccp_names, mem_names):
        ccp_f = [f for f in filenames 
            if 'Process_'+str(c) in f][0]

        img_ccp_f = pickle.load(open(ccp_f, 'rb'))
        sigma = 40
        norm_smooth_img = gaussian_filter(img_ccp_f[0,:,:], sigma=sigma)/np.max(gaussian_filter(img_ccp_f[0,:,:], sigma=sigma))

        img_ccp_f = np.array([img_ccp_f[i,:,:]/norm_smooth_img for i in range(img_ccp_f.shape[0])])

        intensities = [(
            np.sum(img_ccp_f[i,:,:])-np.median(img_ccp_f[0,:,:])*len(np.hstack(img_ccp_f[0,:,:])))\
                /(np.median(img_ccp_f[0,:,:])*len(np.hstack(img_ccp_f[0,:,:])))
                       for i in range(img_ccp_f.shape[0])]
        
        intensities_list.append(intensities)
        ccp_f_all.append(ccp_f)

    intensities_list_per_cell.append(intensities_list)
# %%


ccp_f_all 

intensities_list_per_cell_flatten = flatten_list(intensities_list_per_cell)

exp_names = np.array([f.split('/')[1] for f in ccp_f_all])
print(exp_names)
for e in np.unique(exp_names):


    filtering = [e in f for f in exp_names]
    x = np.arange(60)
    
    y = [intensities_list_per_cell_flatten[i] for i in range(len(intensities_list_per_cell_flatten)) if filtering[i]]
    ccp_f_names = [ccp_f_all[i] for i in range(len(ccp_f_all)) if filtering[i]]

    y_avg = np.mean(y, axis=0)
    y_std = np.std(y, axis=0, ddof=1)

    plt.figure(figsize=(5,4))
    for i in y:
        plt.plot(x, i, color='grey', alpha=0.75)
    plt.plot(x, y_avg, color='black')
    plt.fill_between(x, y_avg-y_std, y_avg+y_std, alpha=0.25)
    plt.title(e)
    plt.ylabel('Intensity increase relative to bg (a.f.u.)')
    plt.xlabel('Frame')
    plt.tight_layout()
    plt.savefig('data_CPP/figures/'+e.replace(' ', '_')+'_ccp_intensity_curves.pdf', bbox_inches='tight')
    plt.show()

intensities_list_per_cell_flatten = flatten_list(intensities_list_per_cell)
print(np.unique(exp_names))

plt.figure(figsize=(6,5))
for e in np.sort(np.unique(exp_names)):
    filtering = [e in f for f in exp_names]
    if 'MCF-7' not in e:
        continue
    if 'TAT' in e:
        continue

    if '1uM' in e:
        label = 'MCF-7 1uM'
    elif '100nM' in e:
        label = 'MCF-7 0.1uM'
    elif '300nM' in e:
        label = 'MCF-7 0.3uM'
    elif '500nM' in e:
        label = 'MCF-7 0.5uM'
    elif '3uM' in e:
        label = 'MCF-7 3uM'
    elif '5uM' in e:
        label = 'MCF-7 5uM'
    else:
        label = 'error in legends'

    x = np.arange(60)
    
    y = [intensities_list_per_cell_flatten[i] for i in range(len(intensities_list_per_cell_flatten)) if filtering[i]]
    ccp_f_names = [ccp_f_all[i] for i in range(len(ccp_f_all)) if filtering[i]]

    y_avg = np.mean(y, axis=0)
    y_std = np.std(y, axis=0, ddof=1)

    plt.plot(x, y_avg, label=label)
    plt.fill_between(x, y_avg-y_std, y_avg+y_std, alpha=0.25)

# sort legend of plot alphabetically
handles, labels_ = plt.gca().get_legend_handles_labels()
order = [1,2,3,0,4,5]
order = [1,2,4,0,3,5]
plt.legend([handles[idx] for idx in order],[labels_[idx] for idx in order])

plt.ylabel('Intensity increase relative to bg (a.f.u.)')
plt.xlabel('Frame')
plt.tight_layout()
plt.savefig('data_CPP/figures/mcf7_ccp_intensity_curves.pdf', bbox_inches='tight')

plt.figure(figsize=(6,5))
for e in np.sort(np.unique(exp_names)):
    filtering = [e in f for f in exp_names]
    if 'MCF-7' not in e:
        continue
    if 'TAT' in e:
        continue

    if '1uM' in e:
        label = 'MCF-7 1uM'
    elif '100nM' in e:
        label = 'MCF-7 0.1uM'
    elif '300nM' in e:
        label = 'MCF-7 0.3uM'
    elif '500nM' in e:
        label = 'MCF-7 0.5uM'
    elif '3uM' in e:
        continue
        label = 'MCF-7 3uM'
    elif '5uM' in e:
        continue
        label = 'MCF-7 5uM'
    else:
        label = 'error in legends'

    x = np.arange(60)
    
    y = [intensities_list_per_cell_flatten[i] for i in range(len(intensities_list_per_cell_flatten)) if filtering[i]]
    ccp_f_names = [ccp_f_all[i] for i in range(len(ccp_f_all)) if filtering[i]]

    y_avg = np.mean(y, axis=0)
    y_std = np.std(y, axis=0, ddof=1)

    plt.plot(x, y_avg, label=label)
    plt.fill_between(x, y_avg-y_std, y_avg+y_std, alpha=0.25)

# sort legend of plot alphabetically
handles, labels_ = plt.gca().get_legend_handles_labels()
order = [1,2,3,0]
plt.legend([handles[idx] for idx in order],[labels_[idx] for idx in order])
plt.ylabel('Intensity increase relative to bg (a.f.u.)')
plt.xlabel('Frame')
plt.tight_layout()
plt.savefig('data_CPP/figures/mcf7_ccp_intensity_curves_zoom.pdf', bbox_inches='tight')

# %%
my_cmap = copy.copy(plt.cm.get_cmap('hsv')) 
my_cmap.set_bad(alpha=0) 
my_cmap.set_under((0,0,0,0))


cell_types = ['mcf7', 'Hela', 'HT-29', 'hek293', 'SH-SY5Y', 'U20S']

def get_intensity_curves(i,c,m, filenames):
    #for i, (c,m) in enumerate(zip(ccp_names[check_i:check_i+1], mem_names[check_i:check_i+1])):
    ccp_f = [f for f in filenames 
            if 'Process_'+str(c) in f][0]
    mem_f = [f for f in filenames
            if 'Process_'+str(m) in f][0]

    img_ccp_f = pickle.load(open(ccp_f, 'rb'))
    img_mem_f = pickle.load(open(mem_f, 'rb'))

    import torch
    use_GPU = True if torch.cuda.is_available() else False
    model_type = 'cyto'
    model = models.Cellpose(gpu=use_GPU, model_type=model_type)
    channels = [[0,0]]
    norm_smooth_img = gaussian_filter(img_mem_f[0,:,:], sigma=40)/np.max(gaussian_filter(img_mem_f[0,:,:], sigma=40))
    img_mem_f_ = [img_mem_f[i,:,:]/(norm_smooth_img) for i in range(img_mem_f.shape[0])]
    img_mem_f_ = np.array(img_mem_f_)
    masks, flows, styles, diams = model.eval(img_mem_f,
                                            diameter=75, 
                                            min_size=1000,
                                            channels=channels, 
                                            z_axis=0,
                                            normalize=True,
                                            flow_threshold=0.75,
                                            )

    assert img_ccp_f.shape == img_mem_f.shape == masks.shape

    new_masks = masks.copy()
    ccp_in_mask_intensity_all = []
    bg_in_mask_intensity_all = []
    sum_mask_intensity_all = []

    # gaussian smoothing
    bg_image = gaussian_filter(img_ccp_f[0], sigma=20)

    for idx, m in enumerate(masks):
        used_mi = []

        for mi in np.unique(m):
            if mi == 0:
                continue
            new_mi = np.random.randint(1,np.max(np.unique(m)))
            while new_mi in used_mi:
                new_mi = np.random.randint(1,np.max(np.unique(m)))
            new_masks[idx][m==mi] = new_mi

        # norm_img_ccp_f_i = img_ccp_f[idx]/np.max(img_ccp_f)

        norm_smooth_ccp_img = gaussian_filter(img_ccp_f[0,:,:], sigma=40)/np.max(gaussian_filter(img_ccp_f[0,:,:], sigma=40))

        norm_img_ccp_f_i = img_ccp_f[idx]/(norm_smooth_ccp_img)
        sum_mask_intensity_all.append(np.sum(norm_img_ccp_f_i[m>0]))
        bg_in_mask_intensity = np.median(norm_img_ccp_f_i[m==0])
        bg_in_mask_intensity_all.append(bg_in_mask_intensity*np.sum(m>0))

        ccp_in_mask_intensity = (np.sum(norm_img_ccp_f_i[m>0])-bg_in_mask_intensity*np.sum(m>0))/bg_in_mask_intensity

        if idx == 0:
            initial_values = ccp_in_mask_intensity/np.sum(m>0)

        ccp_in_mask_intensity = ccp_in_mask_intensity/np.sum(m>0)-initial_values
        ccp_in_mask_intensity_all.append(ccp_in_mask_intensity)
        # print(ccp_in_mask_intensity)
        # print(bg_in_mask_intensity*np.sum(m>0), np.sum(norm_img_ccp_f_i[m>0]))
        # print(sdf)

    fig, ax = plt.subplots(1,4, figsize=(12,8))
    ax[0].imshow(normalize(img_mem_f[0,:,:]/gaussian_filter(img_mem_f[0,:,:], sigma=30), 0.0, 99.),
                cmap='Greys_r', vmin=0.1, vmax=0.9)
    ax[0].set_title('Membrane channel')

    ax[1].imshow(normalize(img_mem_f[0,:,:]/gaussian_filter(img_mem_f[0,:,:], sigma=30), 0.0, 99.),
                cmap='Greys_r', vmin=0.1, vmax=0.9)
    ax[1].imshow(new_masks[0,:,:], cmap=my_cmap, 
                vmin=0.5, alpha=0.5)
    ax[1].set_title('Membrane channel + masks')

    ax[2].imshow(normalize(img_ccp_f[0,:,:]/gaussian_filter(img_ccp_f[0,:,:], sigma=30), 0.0, 99.),
                cmap='Greys_r', vmin=0.1, vmax=0.9)
    ax[2].imshow(new_masks[0,:,:], cmap=my_cmap,
                vmin=0.5, alpha=0.5)
    ax[2].set_title('CCP channel frame 0 + masks')

    ax[3].imshow(normalize(img_ccp_f[-1,:,:]/gaussian_filter(img_ccp_f[-1,:,:], sigma=30), 0.0, 99.),
                cmap='Greys_r', vmin=0.1, vmax=0.9)
    ax[3].imshow(new_masks[-1,:,:], cmap=my_cmap,
                vmin=0.5, alpha=0.5)
    ax[3].set_title('CCP channel frame 60 + masks')
    
    plt.savefig(
        dirname+'figures/'+ccp_f.split('/')[1]+'_prediction_example_process_v1_'+str(c)+'.pdf', 
        bbox_inches='tight')
    # plt.show()
    # plt.close()


    fig, ax = plt.subplots(1,4, figsize=(12,8))
    ax[0].imshow(normalize(img_ccp_f[0,:,:]/gaussian_filter(img_ccp_f[0,:,:], sigma=30), 0.0, 99.),
                cmap='Greys_r', vmin=0.1, vmax=0.9)
    ax[0].imshow(new_masks[0,:,:], cmap=my_cmap, 
                vmin=0.5, alpha=0.5)
    
    ax[1].imshow(normalize(img_ccp_f[1,:,:]/gaussian_filter(img_ccp_f[1,:,:], sigma=30), 0.0, 99.),
                cmap='Greys_r', vmin=0.1, vmax=0.9)
    ax[1].imshow(new_masks[1,:,:], cmap=my_cmap, 
                vmin=0.5, alpha=0.5)
  
    ax[2].imshow(normalize(img_ccp_f[30,:,:]/gaussian_filter(img_ccp_f[30,:,:], sigma=30), 0.0, 99.),
                cmap='Greys_r', vmin=0.1, vmax=0.9)
    ax[2].imshow(new_masks[30,:,:], cmap=my_cmap,
                vmin=0.5, alpha=0.5)

    ax[3].imshow(normalize(img_ccp_f[-1,:,:]/gaussian_filter(img_ccp_f[-1,:,:], sigma=30), 0.0, 99.),
                cmap='Greys_r', vmin=0.1, vmax=0.9)
    ax[3].imshow(new_masks[-1,:,:], cmap=my_cmap,
                vmin=0.5, alpha=0.5)
    plt.savefig(
        dirname+'figures/'+ccp_f.split('/')[1]+'_prediction_example_process_v2_'+str(c)+'.pdf',
          bbox_inches='tight')
    # plt.show()
    # plt.close()

    print('0', np.min(img_ccp_f[0,:,:]/norm_smooth_ccp_img)/bg_in_mask_intensity)
    print('-1', np.min(img_ccp_f[-1,:,:]/norm_smooth_ccp_img)/bg_in_mask_intensity)
    print('0', np.max(img_ccp_f[0,:,:]/norm_smooth_ccp_img)/bg_in_mask_intensity)
    print('-1', np.max(img_ccp_f[-1,:,:]/norm_smooth_ccp_img)/bg_in_mask_intensity)
    print('0', np.mean(img_ccp_f[0,:,:]/norm_smooth_ccp_img)/bg_in_mask_intensity)
    print('-1', np.mean(img_ccp_f[-1,:,:]/norm_smooth_ccp_img)/bg_in_mask_intensity)
    print('0', np.median(img_ccp_f[0,:,:]/norm_smooth_ccp_img)/bg_in_mask_intensity)
    print('-1', np.median(img_ccp_f[-1,:,:]/norm_smooth_ccp_img)/bg_in_mask_intensity)

    fig, ax = plt.subplots(1,4, figsize=(12,8))
    ax[0].imshow((img_ccp_f[0,:,:]/norm_smooth_ccp_img)/bg_in_mask_intensity,
                cmap='Greys_r', vmin=0.8, vmax=1.5)
    
    ax[1].imshow((img_ccp_f[1,:,:]/norm_smooth_ccp_img)/bg_in_mask_intensity,
                cmap='Greys_r', vmin=0.8, vmax=1.5)

    ax[2].imshow((img_ccp_f[30,:,:]/norm_smooth_ccp_img)/bg_in_mask_intensity,
                cmap='Greys_r', vmin=0.8, vmax=1.5)

    ax[3].imshow((img_ccp_f[-1,:,:]/norm_smooth_ccp_img)/bg_in_mask_intensity,
                cmap='Greys_r', vmin=0.8, vmax=1.5)
    plt.savefig(
        dirname+'figures/'+ccp_f.split('/')[1]+'_prediction_example_process_v3_'+str(c)+'.pdf',
          bbox_inches='tight')
    #plt.show()
    # plt.close()

    plt.figure()      
    plt.plot(ccp_in_mask_intensity_all)
    plt.savefig(
        dirname+'figures/'+ccp_f.split('/')[1]+'_prediction_example_process_v5_'+str(c)+'.pdf',
          bbox_inches='tight')
    plt.figure()
    plt.plot(sum_mask_intensity_all, label='np.sum(norm_img_ccp_f_i[m>0])')
    plt.plot(bg_in_mask_intensity_all, label='bg_in_mask_intensity*np.sum(m>0)')
    plt.legend()
    plt.show()
    plt.savefig(
        dirname+'figures/'+ccp_f.split('/')[1]+'_prediction_example_process_v6_'+str(c)+'.pdf',
          bbox_inches='tight')
    # plt.close()

    print(ccp_f, ccp_in_mask_intensity_all[0], ccp_in_mask_intensity_all[-1])
    return ccp_in_mask_intensity_all, ccp_f, mem_f# sum_mask_intensity_all, bg_in_mask_intensity_all, 


from joblib import Parallel, delayed


for cell_type in tqdm(cell_types):
    if os.path.exists(dirname+cell_type+'_ccp_intensity_curves.pkl'):
        print('already done')
        continue
    ccp_names = []
    mem_names = []
    for k in labels.keys():
        if cell_type in labels[k]:
            if '_ccp' in labels[k]:
                ccp_names.append(k)
            elif '_mem' in labels[k]:
                mem_names.append(k)

    # chi = 0
    # offsets = 8
    # results = Parallel(n_jobs=30)(
    #                 delayed(get_intensity_curves)(i,c,m) 
    #                 for i, (c,m) in enumerate(zip(ccp_names[chi:chi+offsets], mem_names[chi:chi+offsets])))
    
    results = Parallel(n_jobs=20)(
                    delayed(get_intensity_curves)(i,c,m, filenames) 
                    for i, (c,m) in enumerate(zip(ccp_names, mem_names)))


    ccp_intensity_curves = np.array([r[0] for r in results])
    ccp_f_all = np.array([r[1] for r in results])
    mem_f_all = np.array([r[2] for r in results])

    pickle.dump(ccp_f_all, open(dirname+cell_type+'_ccp_f_all.pkl', 'wb'))
    pickle.dump(mem_f_all, open(dirname+cell_type+'_mem_f_all.pkl', 'wb'))
    pickle.dump(
    ccp_intensity_curves, open(dirname+cell_type+'_ccp_intensity_curves.pkl', 'wb'))

# %%
ccp_intensity_curves

# %%
import pickle
import numpy as np
import matplotlib.pyplot as plt

dirname = 'data_CCP/'
cell_type = 'mcf7'

ccp_f_all = pickle.load(open(dirname+cell_type+'_ccp_f_all.pkl', 'rb'))
mem_f_all = pickle.load(open(dirname+cell_type+'_mem_f_all.pkl', 'rb'))
ccp_in_mask_intensity_all = pickle.load(open(dirname+cell_type+'_ccp_intensity_curves.pkl', 'rb'))


ccp_in_mask_intensity_all = np.array(ccp_in_mask_intensity_all)

print(ccp_in_mask_intensity_all.shape, len(ccp_f_all), len(mem_f_all))
print(ccp_in_mask_intensity_all[0].shape)
print(ccp_f_all)

plt.figure()
exp_names = np.array([f.split('/')[1] for f in ccp_f_all])
for e in np.unique(exp_names):
    x = np.arange(60)
    y = np.mean(ccp_in_mask_intensity_all[exp_names==e], axis=0)
    yerr = np.std(ccp_in_mask_intensity_all[exp_names==e], axis=0, ddof=1)/np.sqrt(np.sum(exp_names==e))
    plt.fill_between(x, y-yerr, y+yerr, alpha=0.25)
    plt.plot(x, y, label=e)
plt.legend()

plt.figure()
exp_names = np.array([f.split('/')[1] for f in ccp_f_all])
for e in np.unique(exp_names):
    x = np.arange(60)
    y = np.mean(ccp_in_mask_intensity_all[exp_names==e], axis=0)
    yerr = np.std(ccp_in_mask_intensity_all[exp_names==e], axis=0, ddof=1)
    plt.fill_between(x, y-yerr, y+yerr, alpha=0.25)
    plt.plot(x, y, label=e)
plt.legend()

plt.figure()
exp_names = np.array([f.split('/')[1] for f in ccp_f_all])
for e in np.unique(exp_names):
    x = np.arange(60)
    y = np.mean(ccp_in_mask_intensity_all[exp_names==e], axis=0)
    yerr = np.std(ccp_in_mask_intensity_all[exp_names==e], axis=0, ddof=1)/np.sqrt(np.sum(exp_names==e))
    plt.fill_between(x, y-yerr, y+yerr, alpha=0.25)
    plt.plot(x, y, label=e)
plt.legend()
plt.savefig('data_CPP/figures/mcf7_ccp_intensity_curves.pdf', bbox_inches='tight')


plt.figure()
exp_names = np.array([f.split('/')[1] for f in ccp_f_all])
for e in np.unique(exp_names):
    x = np.arange(60)
    y = np.mean(ccp_in_mask_intensity_all[exp_names==e], axis=0)
    yerr = np.std(ccp_in_mask_intensity_all[exp_names==e], axis=0, ddof=1)
    plt.fill_between(x, y-yerr, y+yerr, alpha=0.25)
    plt.plot(x, y, label=e)
plt.legend()
plt.ylim(0,.15)
plt.savefig('data_CPP/figures/mcf7_ccp_intensity_curves_zoom.pdf', bbox_inches='tight')

# %%

plt.figure()
exp_names = np.array([f.split('/')[1] for f in ccp_f_all])
for ei, e in enumerate(np.unique(exp_names)):
    x = np.arange(60)
    ys = ccp_in_mask_intensity_all[exp_names==e]
    for i, y in enumerate(ys):
        if i == 0:
            plt.plot(x, y, label=e, color='C'+str(ei))
        else:
            plt.plot(x, y, color='C'+str(ei))
plt.legend()
plt.savefig('data_CPP/figures/mcf7_ccp_intensity_curves_individual.pdf', bbox_inches='tight')


plt.figure()
exp_names = np.array([f.split('/')[1] for f in ccp_f_all])
for ei, e in enumerate(np.unique(exp_names)):
    x = np.arange(60)
    ys = ccp_in_mask_intensity_all[exp_names==e]
    for i, y in enumerate(ys):
        if i == 0:
            plt.plot(x, y, label=e, color='C'+str(ei))
        else:
            plt.plot(x, y, color='C'+str(ei))
plt.legend()
plt.ylim(0,.15)
plt.savefig('data_CPP/figures/mcf7_ccp_intensity_curves_individual_zoom.pdf', bbox_inches='tight')


# %%

cell_types = ['mcf7', 'Hela', 'HT-29', 'hek293', 'SH-SY5Y', 'U20S', 'RPE-1']
position_in_plot = [[0,0], [0,1], [0,2], [0,3], [1,0], [1,1], [1,2]]

fig, ax = plt.subplots(2,4, figsize=(12,8))
for cell_type in cell_types:
    p0, p1 = position_in_plot[cell_types.index(cell_type)]
    if cell_type == 'RPE-1':
        cell_type = 'U20S'
    ccp_f_all = pickle.load(open(dirname+cell_type+'_ccp_f_all.pkl', 'rb'))
    mem_f_all = pickle.load(open(dirname+cell_type+'_mem_f_all.pkl', 'rb'))
    ccp_in_mask_intensity_all = pickle.load(open(dirname+cell_type+'_ccp_intensity_curves.pkl', 'rb'))

    ccp_in_mask_intensity_all = np.array(ccp_in_mask_intensity_all)

    exp_names = np.array([f.split('/')[1] for f in ccp_f_all])
    print(np.unique(exp_names))
    for e in np.unique(exp_names):
        if 'MCF-7' in e and '1uM' not in e:
            continue
        x = np.arange(60)
        y = np.mean(ccp_in_mask_intensity_all[exp_names==e], axis=0)
        yerr = np.std(ccp_in_mask_intensity_all[exp_names==e], axis=0, ddof=1)/np.sqrt(np.sum(exp_names==e))
        ax[p0,p1].fill_between(x, y-yerr, y+yerr, alpha=0.25)
        ax[p0,p1].plot(x, y, label=e)
        ax[p0,p1].set_title(cell_type)
# remove the empty plot in the last position
fig.delaxes(ax[1,3])


plt.savefig('data_CPP/figures/ccp_intensity_curves_celltypes.pdf', bbox_inches='tight')

# %%
