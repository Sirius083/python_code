from datetime import datetime
# from each run create a directory
def get_logdir():
    now = datetime.now().strftime("%m%d%H%M") # local time as log dir name
    logdir = "run-{}/".format(now)
    return logdir


# ckpt file path
if True:
   checkpoint = tf.train.latest_checkpoint(ckpt_path) # restore from the lastest checkpoint file
else: # build new ckpt_path
   checkpoint = ckpt_path + '/' +  checkpoint_name   # restore from given checkpoint name
    
    


