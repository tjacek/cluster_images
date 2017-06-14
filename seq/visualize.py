import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import pandas as pd 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utils.actions.read
import utils.paths
import utils.paths.dirs	
from collections import defaultdict

def visualize_category(in_path,out_path,dataset_format='cp_dataset'):
    read_actions=utils.actions.read.ReadActions(str(dataset_format),img_seq=False)
    actions=read_actions(in_path)
    actions=by_category(actions)
    categories={ name:category_feat(actions) 
                 for name,actions in actions.items()} 
    make_dir_plots(out_path,categories)

def visualize_features(in_path,out_path,dataset_format='cp_dataset'):
    read_actions=utils.actions.read.ReadActions(str(dataset_format),img_seq=False)
    actions=read_actions(in_path)
    action_plots=[ (action_i.name,get_features(action_i)) 
                   for action_i in actions]
    #save_by_action(out_path,action_plots)
    save_by_category(out_path,action_plots)

feat_names={0:'area_feat',1:'std_x',2:'std_y',3:'std_y',4:'skew_x',5:'skew_y',6:'skew_z',7:'corl_x',8:'corl_y',9:'corl_z',10:'corl_x',11:'corl_y',12:'corl_z'}

def by_category(actions):
    actions_by_category = defaultdict(lambda: [])
    for action_i in actions:
        actions_by_category[action_i.cat].append(action_i)
    return actions_by_category

def category_feat(cat_actions):
    def feat_helper(j):
        return { action_i.name:action_i.to_series()[j] 
                  for action_i in cat_actions}
    category_i={ name:feat_helper(index)
               for index,name in feat_names.items()}
    return category_i 

def make_dir_plots(out_path,categories):
    dir_paths=make_dir_structure(out_path,categories.keys())
    for name_i,cat_i in categories.items():
        cat_path=dir_paths[name_i]
        make_feat_plots(cat_path,cat_i)
    print(dir_paths)

@utils.paths.path_args
def make_dir_structure(out_path,dir_names):
    utils.paths.dirs.make_dir(out_path)
    dir_paths={}
    for name_i in dir_names:
        cat_path=out_path + name_i
        utils.paths.dirs.make_dir(cat_path)
        dir_paths[name_i]=cat_path
    return dir_paths

def make_feat_plots(cat_path,cat_i):
    for name_i,actions_i in cat_i.items():
        feat_path=cat_path+name_i
        make_action_series(feat_path,actions_i)

def make_action_series(feat_path,actions_i):
    #data_frame=pd.DataFrame.from_items(actions_i.items())
    print(feat_path)
    fig, ax = plt.subplots()
    colors=['blue','green','red','cyan','magenta','yellow']
    for i,data_i in enumerate(actions_i.values()):
        x=range(len(data_i))
        c_i=colors[i % len(colors)]
        ax.plot(x, data_i, color=c_i)#,label=name_i)
    plt.savefig(str(feat_path))
    plt.clf()
    plt.close()

    #save_plot(feat_path,fig)
    #ax.legend(loc='upper left')
    #print(actions_i.keys())

@utils.paths.path_args
def save_by_category(out_path,action_plots):
    utils.paths.dirs.make_dir(out_path)
    feat_size=len(feat_names)
    def get_feat_plots(j):
    	return [ (action_plots_i[0],action_plots_i[1][j]) 
    	         for action_plots_i in action_plots]
    for i in range(feat_size):
        feature_i=feat_names[i]
        feat_plots=get_feat_plots(i)
        feat_path_i=out_path+feature_i
        utils.paths.dirs.make_dir(feat_path_i)
        for feat_plot_j in feat_plots:
            name=feat_plot_j[0]
            name=name.split('.')[0]+'.png'
            plot_j=feat_plot_j[1]
            out_path_ij=feat_path_i+name
            save_plot(out_path_ij,plot_j)
        

@utils.paths.path_args
def save_by_feature(out_path,action_plots):
    utils.paths.dirs.make_dir(out_path)
    for action_plots_i in action_plots:
        action_out_path= out_path+action_plots_i[0]
        save_dir_plots(action_out_path,action_plots_i[1])	

@utils.paths.path_args
def save_by_action(out_path,action_plots):
    utils.paths.dirs.make_dir(out_path)
    for action_plots_i in action_plots:
        action_out_path= out_path+action_plots_i[0]
        save_dir_plots(action_out_path,action_plots_i[1])	

@utils.paths.path_args
def save_dir_plots(path_i,plots):
    print(path_i)
    utils.paths.dirs.make_dir(path_i)
    for i,plot_i in enumerate(plots):
        name=feat_names[i]+'.png'
        out_plot_i=path_i+name
        save_plot(out_plot_i,plot_i)

def save_plot(out_path_i,plot_i):
    ax=plot_i.plot()
    ax.get_figure()
    plt.savefig(str(out_path_i))
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
    out_path='../../exper/cat_plots'
    visualize_category(in_path,out_path)