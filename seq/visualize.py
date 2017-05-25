import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import pandas as pd 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utils.actions.read
import utils.paths.dirs

def visualize_features(in_path,out_path,dataset_format='cp_dataset'):
    read_actions=utils.actions.read.ReadActions(dataset_format,img_seq=False)
    actions=read_actions(in_path)
    action_plots=[ (action_i.name,get_features(action_i)) 
                   for action_i in actions]
    utils.paths.dirs.make_dir(out_path)
    for action_plots_i in action_plots:
        action_out_path= out_path+'/'+action_plots_i[0]
        save_action_plots(action_out_path,action_plots_i[1])	

def save_action_plots(name_i,plots):
    print(name_i)
    utils.paths.dirs.make_dir(name_i)
    for i,plot_i in enumerate(plots):
        out_plot_i=name_i+'/img'+str(i)+'.png'
        ax=plot_i.plot()
        ax.get_figure()
        plt.savefig(out_plot_i)
        plt.clf()
        plt.close()

def get_features(action_i):
    all_features=action_i.to_series() 
    plots=[ get_feature_plot(feature_i)
              for feature_i in all_features]
    return plots

def get_feature_plot(feature):
    if(type(feature)!=list):
        feature=list(feature)
    size=len(feature)
    ts = pd.Series(feature, index=range(size))
    return ts#ts.plot()

if __name__ == "__main__":
    in_path='../../ultimate/simple3/seq'
    out_path='../../ultimate/viz'
    visualize_features(in_path,out_path)