import pickle
from grid_search import * 
if __name__ == "__main__":
    data_path = '/home/yam/workspace/repos/big_exp.pkl'
    with open(data_path, 'rb')as f:
        exp_summerys = pickle.load(f)

    fig, ax = plt.subplots(4, 2, figsize=(18, 34))

    for k, data_dir in enumerate(data_axis):
        mat_name = data_dir[0] + '_train_loss'
        ax[k, 0].imshow(exp_summerys[k,:,:,0], aspect='auto')
        ax[k, 0].set_title(mat_name, fontsize=25)
        ax[k, 0].set_xlabel('nn depth', fontsize=25)
        ax[k, 0].set_ylabel('nn width', fontsize=25)
        ax[k, 0].set_xticks(ticks =list(range(len(hidden_layers_axis))),labels= hidden_layers_axis, fontsize=25)
        ax[k, 0].set_yticks(ticks=list(range(len(hidden_features_axis))), labels= hidden_features_axis, fontsize=25)

        mat_name = data_dir[0] + '_hight_res_loss'
        ax[k, 1].imshow(exp_summerys[k,:,:,1],  aspect='auto')
        ax[k, 1].set_title(mat_name, fontsize=25)
        ax[k, 1].set_xlabel('nn depth', fontsize=25)
        ax[k, 1].set_ylabel('nn width', fontsize=25)
        ax[k, 1].set_xticks(ticks =list(range(len(hidden_layers_axis))),labels= hidden_layers_axis, fontsize=25)
        ax[k, 1].set_yticks(ticks=list(range(len(hidden_features_axis))), labels= hidden_features_axis, fontsize=25)
    fig.suptitle('title', fontsize=40)
    plt.savefig(output_path / ('convergence_data2.png'))