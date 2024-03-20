import torch 
import matplotlib.pyplot as plt
from trainer import run_exp, TrainConfig
from pathlib import Path

data_axis = [['48_test_2', '256_test_2'], ['48_test_5', '256_test_5'], ['48_test_10', '256_test_10'], ['48_test_bigger', '256_test_bigger']]
hidden_features_axis = [64, 128, 256, 512]
hidden_layers_axis = [2, 4, 6, 8, 10]
data_path = Path('/home/yam/workspace/data/cognetive/data/')
output_path = data_path / 'results'
images_pairs_names = [['buy', 'return_purchase'], ['price_tag_euro', 'price_tag_usd'], ['return_purchase','shopping_cart']]


if __name__ == "__main__":
    data_path = Path('/home/yam/workspace/data/cognetive/data/')
    train_data_path = data_path / '48_test'
    high_res_data_path = data_path/ '256_test'
    output_path = data_path / 'results'

    exp_summerys = []
    for k, data_dir in enumerate(data_axis):
        
        exp_summerys_i = []
        for i, hidden_features in enumerate(hidden_features_axis):
            exp_summerys_ij = []
            for j, hidden_layers in enumerate(hidden_layers_axis):
                omega_0 = 30
                outermost = 'linear'
                total_steps = 10
                steps_til_summary=10
                lr = 1e-4
                train_data_path = data_path / data_dir[0]
                high_res_data_path = data_path / data_dir[1]
                assert train_data_path.exists(), str(train_data_path) + ' does not exits'
                assert high_res_data_path.exists(), str(high_res_data_path) + ' does not exits'
                print('run test summery  -' '_' + data_dir[0] + ' depth ' + str(hidden_layers) + ' width ' + str(hidden_features))
                output_path_i = output_path / (data_dir[0] + '_depth_' + str(hidden_layers) + '_width_' + str(hidden_features))
                net_architecture = {'hidden_features': hidden_features, 'hidden_layers': hidden_layers, 'omega_0': omega_0, 'outermost': outermost}
                train_config = TrainConfig(total_steps = total_steps, steps_til_summary=steps_til_summary, lr = lr, net_params=net_architecture)
                exp_summery_ijk = run_exp(train_data_path, high_res_data_path, output_path_i,images_pairs_names, train_config)
                train_loss = exp_summery_ijk['loss']['train_loss'][-1]
                test_loss = exp_summery_ijk['loss']['test_loss'][-1]

                exp_summerys_ij.append([train_loss, test_loss]) 
            exp_summerys_i.append(exp_summerys_ij)
        exp_summerys.append(exp_summerys_i)

    exp_summerys = torch.tensor(exp_summerys)

    fig, ax = plt.subplots(exp_summerys.shape[0], 2, figsize=(6, 12))
    if ax.ndim > 1:
        for k, data_dir in enumerate(data_axis):
            mat_name = data_dir[0] + '_train_loss'
            ax[k, 0].imshow(exp_summerys[k,:,:,0])
            ax[k, 0].set_title(mat_name)
            ax[k, 0].set(xlabel='nn depth', ylabel='nn width'   )
            # ax[k, 0].ylabel('nn width')
            ax[k, 0].set_xticks(ticks =list(range(len(hidden_layers_axis))),labels= hidden_layers_axis)
            ax[k, 0].set_yticks(ticks=list(range(len(hidden_features_axis))), labels= hidden_features_axis)

            mat_name = data_dir[0] + '_hight_res_loss'
            ax[k, 1].imshow(exp_summerys[k,:,:,1])
            ax[k, 1].set_title(mat_name)
    elif ax.ndim == 1:
        data_dir = data_axis[0]
        mat_name = data_dir[0] + '_train_loss'
        ax[0].imshow(exp_summerys[k,:,:,0])
        ax[0].set_title(mat_name)
        ax[0].set(xlabel='nn depth', ylabel='nn width'   )
        # ax[k, 0].ylabel('nn width')
        ax[0].set_xticks(ticks =list(range(len(hidden_layers_axis))),labels= hidden_layers_axis)
        ax[0].set_yticks(ticks=list(range(len(hidden_features_axis))), labels= hidden_features_axis)

        mat_name = data_dir[0] + '_hight_res_loss'
        ax[1].imshow(exp_summerys[k,:,:,1])
        ax[1].set_title(mat_name)
    plt.savefig(output_path / 'search_grid.png')

    print('done!')

0