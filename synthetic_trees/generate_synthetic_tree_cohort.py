import matplotlib.pyplot as plt
import os
from openalea.plantgl.algo.view import *
import importlib.util
import torch
from openalea.lpy import *
from openalea.plantgl.all import *
import multiprocessing
from joblib import Parallel, delayed
import tqdm
import numpy as np
import argparse

# # testing
from turtle import *
import turtle as turtle
import matplotlib.pyplot as plt
import os
from openalea.plantgl.all import *
from imp import reload
from openalea.plantgl.algo.view import *
from openalea.plantgl.algo.view import *

def view_2(scene: Scene, imgsize : tuple = (800,800), perspective : bool = True, zoom : float = 2, azimuth : float = 0 , elevation : float = 0, savepath=None, ax=None) -> None:
    """
    Display an orthographic view of a scene.
    :param scene: The scene to render
    :param imgsize: The size of the image
    :param azimuth: The azimuth (in degrees) of view to render
    :param elevation: The elevation (in degrees) of view to render
    """
    if perspective:
        img = perspectiveimage(scene, imgsize=imgsize, zoom=zoom, azimuth=azimuth, elevation=elevation)
    else:
        img = orthoimage(scene, imgsize=imgsize, azimuth=azimuth, elevation=elevation)
    if not img is None:
        # import matplotlib.pyplot as plt
        if ax is not None:
            ax.imshow(img, cmap='binary')
        else:
            fig, ax = plt.subplots(figsize=(9, 9))
            ax.imshow(img)
            if savepath is not None:
                plt.savefig(os.path.abspath(savepath))
            
            # plt.show()







def main():
    package= 'openalea.lpy'
    is_present = importlib.util.find_spec(package) #find_spec will look for the package
    if is_present is None:
        print(package +" is not installed")
    else:
        print ("Successfull")
        

        
        
def generate_tree(i, args):
    file_save_root = args.experiment_root
    file_save_name = file_save_root+f"tree_{i}.csv"
    gt_load = args.gt_path
    gt_info = torch.load(gt_load)

    gt_path = file_save_root + f"gt_info.pkl"
    
    # Always save the ground truth info as 'gt_info.pkl' in the folder called file_save_root
    torch.save(gt_info, gt_path)
    variables = {
        'file_path':file_save_name,
          'config_path':args.config_path,
            'load_path':gt_path, 
            'stop_id':args.stop_id,
              'lobe_path': args.lobe_path,
                'lobe_of_choice':args.lobe_of_choice
                }
    lsystem = Lsystem(args.lpy_file, variables)
    lstring = lsystem.derive()
    scene = lsystem.sceneInterpretation(lstring)
    az= 90
    el = 0

    view_2(scene, perspective=True, azimuth=az,elevation=el, savepath= file_save_root + f'tree_{i}.png')


main()

new_root = os.getcwd() 
print('Current directory', new_root)


new_path = new_root+f"/new_tree_cohort/"
print("Starting path", new_path)


# exist or not
if not os.path.exists(new_path):
    # if the demo_folder directory is not present
    # then create it
    os.makedirs(new_path)



parser = argparse.ArgumentParser(description="Generation of Synthetic Trees")

parser.add_argument("--N", default=5, type=int)
parser.add_argument("--gt_path", default=f'{os.getcwd()}/ATM_example/gt_info.pkl', type=str)
parser.add_argument("--config_path", default=f'{os.getcwd()}/ATM_example/tree_template_df.csv', type=str)
parser.add_argument("--stop_id", default=177, type=int)
parser.add_argument("--lobe_of_choice", default=3, type=int)
parser.add_argument("--lobe_path", default=f'{os.getcwd()}/ATM_example/lobe_template.npy', type=str)
parser.add_argument("--experiment_root", default=new_path, type=str)
parser.add_argument("--lpy_file",
                        default=f"synthetic_tree_generation.lpy",
                    type=str)
args = parser.parse_args('')


Parallel(n_jobs=2, verbose=10)(delayed(generate_tree)(i,  args) for i in range(args.N))




