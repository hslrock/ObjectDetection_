import glob
import shutil
import os
from torch.utils.tensorboard import SummaryWriter

base_path="experiments"
def delete_files(base_path):
    print("Following files are going to be removed:")
    for filename in glob.iglob(base_path + '**/**', recursive=True):
        print(f"\t{filename}" )
    print("Continue? (y,n):")
    x=input()

    if x=='y':
        file_name=os.listdir(base_path)
        for sub_path in file_name:
            path=os.path.join(base_path,sub_path)
            shutil.rmtree(path)
        print("Cleaned")
    else:
        print("Cancelling Operation")


def _mk_directory(base_name="experiment",exp_name="None"):
    trial_num=1

    directory=os.path.join(base_name,exp_name,str(trial_num))
    while  os.path.exists(directory):
        trial_num+=1
        directory=os.path.join(base_name,exp_name,str(trial_num))
    os.makedirs(directory) #Create Exp Folder
    return directory

def _mk_SummaryWriter(root,exp_name):
    path=_mk_directory(root,exp_name)
    summary_writer=SummaryWriter(log_dir=path)   
    print(f"Event are saved under {summary_writer.get_logdir()}")
    return summary_writer

class Writer():
    def __init__(self,exp_name="None",base_name="experiment"):
        self.writer=_mk_SummaryWriter(base_name,exp_name)
        
    def add_events(self,name,data,step):
        lookups = [
        self.writer.add_scalar,
        self.writer.add_figure,
        self.writer.add_image
        ]
        for lookup in lookups:
            try:
                lookup(name,data,step)
                info=True
                # exit the loop on success
                break    
            except:
                info=False
                # repeat the loop on failure
                continue
        if not info:
            raise Exception('Something Went Wrong, Currently tried add_scalar,add_figure,add_image. For debugging, we advise you to debug by using Writer.writer.add_{}')
        self.writer.flush()
        
