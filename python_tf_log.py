from datetime import datetime
# from each run create a directory
now = datetime.now().strftime("%m%d%H%M")
logdir = "run-{}/".format(now)
