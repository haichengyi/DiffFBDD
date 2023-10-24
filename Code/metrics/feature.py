import csv
import os
from rdkit import Chem
from rdkit.Chem import PandasTools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import csv

def num_bonds(path):
    bonds = {'single':0,'double':0,'trigle':0}
    f_num = 0
    num_bond = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            if str(f).split('.')[1] == 'sdf':
                try:
                    supplier = Chem.SDMolSupplier(path + f)
                    # 遍历SDF文件中的分子
                except:
                    continue
                for molecule in supplier:
                    if molecule is not None:
                        f_num = f_num + 1
                        # 计算各元素的原子个数
                        bonds_type = molecule.GetBonds()
                        for i in bonds_type:
                            bond = i.GetBondType()
                            if bond == Chem.BondType.SINGLE:
                                bonds['single'] += 1
                            if bond == Chem.BondType.DOUBLE:
                                bonds['double'] += 1
                            if bond == Chem.BondType.TRIPLE:
                                bonds['trigle'] += 1
                # except:
                #     continue
    for key in bonds:
        bonds[key] = float(bonds[key]) / f_num
    sum = 0
    for key in bonds:
        sum += bonds[key]
    for key in bonds:
        bonds[key] = float(bonds[key]) / sum
    return bonds
def num_node(path):
    element_counts = {'C': 0, 'N': 0, 'O': 0, 'Other': 0}
    f_num = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            if str(f).split('.')[1] == 'sdf':
                try:
                    supplier = Chem.SDMolSupplier(path+f)
                except:
                    continue
                    # 遍历SDF文件中的分子
                for molecule in supplier:
                    if molecule is not None:
                        f_num = f_num + 1
                            # 计算各元素的原子个数
                        atoms = [atom.GetSymbol() for atom in molecule.GetAtoms()]
                        for atom_symbol in atoms:
                            # print(atom_symbol)
                            if atom_symbol in element_counts:
                                element_counts[atom_symbol] += 1
                            else:
                                element_counts['Other'] += 1
    # print(element_counts)
    for key in element_counts:
        element_counts[key] = float(element_counts[key])/f_num
    sum = 0
    for key in element_counts:
        sum += element_counts[key]
    for key in element_counts:
        element_counts[key] = float(element_counts[key]) / sum
    return element_counts

def plot_ele_counts(element_counts,ax):
    x = np.array(['CrossDocked','OURS'])
    y1 = np.array([element_counts[0][0]*100,element_counts[1][0]*100])
    y2 = np.array([element_counts[0][1]*100,element_counts[1][1]*100])
    y3 = np.array([element_counts[0][2]*100, element_counts[1][2]*100])
    y4 = np.array([element_counts[0][3]*100, element_counts[1][3]*100])
    print('atom',x,y1,y2,y3,y4)
    # 绘制柱状堆叠图，设置柱子颜色和标签
    N = len(y1)
    width = 0.45
    ind = np.arange(N)

    bar_plot1 = ax.bar(ind, y1, width, color='b', alpha=0.7, label='C')
    bar_plot2 = ax.bar(ind, y2, width, bottom=y1, color='g', alpha=0.7, label='N')
    bar_plot3 = ax.bar(ind, y3, width, bottom=np.array(y1)+np.array(y2), color='r', alpha=0.7, label='O')
    bar_plot4 = ax.bar(ind, y4, width, bottom=np.array(y1)+np.array(y2)+np.array(y3), color='m', alpha=0.7, label='Others')

    # 添加标题、标签和图例
    ax.set_title('ATOMS', pad=50,fontsize=20)

    ax.set_yticklabels([f'{val:.0f}%' for val in ax.get_yticks()],fontsize = 20)

    new_x = [i for i in x]
    ax.set_xticks([0,1])
    ax.set_xticklabels(new_x,fontsize=20)
    ax.legend(ncol=4, loc=[-0.05,1.01], fontsize=20,frameon = False)
    # 显示图表
    # plt.show()

def plot_bonds_counts(element_counts,ax):
    x = np.array(['CrossDocked','OURS'])
    y1 = np.array([element_counts[0][0]*100,element_counts[1][0]*100])
    y2 = np.array([element_counts[0][1]*100,element_counts[1][1]*100])
    y3 = np.array([element_counts[0][2]*100, element_counts[1][2]*100])
    print('bond', x, y1, y2, y3)
    # 绘制柱状堆叠图，设置柱子颜色和标签
    N = len(y1)
    width = 0.45
    ind = np.arange(N)
    bar_plot1 = ax.bar(ind, y1, width, color='b', alpha=0.7, label='SINGLE')
    bar_plot2 = ax.bar(ind, y2, width, bottom=y1, color='r', alpha=0.7, label='DOUBLE')
    bar_plot3 = ax.bar(ind, y3, width, bottom=np.array(y1)+np.array(y2), color='g', alpha=0.7, label='TRIPLE')

    # 添加标题、标签和图例
    ax.set_title('BONDS', pad=50,fontsize=20)
    ax.set_yticklabels([f'{val:.0f}%' for val in ax.get_yticks()],fontsize = 20)
    # ax.legend()
    new_x = [i for i in x]
    ax.set_xticks([0,1])
    ax.set_xticklabels(new_x,fontsize=20)

    # 显示图表
    # plt.show()
    ax.legend(ncol=4, loc=[-0.05,1.01], fontsize=20,frameon = False)

def write_fea_csv(my_list,s,n):
    # with open('/home/linux/'+s+str(n)+'.csv', mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     for row in my_list:
    #         writer.writeline(row)
    my_list.to_csv('/home/linux/'+s+str(n)+'.csv',index=True)
def get_feature(path1,path2,ax,s):
    # 读取第一个CSV文件
    df1 = pd.read_csv(path1)
    # 读取第二个CSV文件
    df2 = pd.read_csv(path2)

    # 提取两个文件中的SA项
    sa_values1 = [round(sa,1) for sa in df1[s]]
    sa_values2 = [round(sa,1) for sa in df2[s]]

    # 使用Pandas的value_counts函数统计每个文件中的SA值的个数
    sa_counts1 = pd.Series(sa_values1).value_counts().sort_index()
    sa_counts2 = pd.Series(sa_values2).value_counts().sort_index()
    # print('1',sa_values1,sa_counts1)
    # print('2',sa_values2,sa_counts2)
    write_fea_csv(sa_counts1,s,1)
    write_fea_csv(sa_counts2,s,2)
    x1 = sa_counts1.index
    y1 = sa_counts1.values
    x_smooth = np.linspace(min(x1),max(x1),100)
    # print(x_smooth)
    spline = make_interp_spline(x1,y1)
    y_smoth = spline(x_smooth)

    # 在ax1上绘制第一个文件的SA值折线图
    ax.plot(x1,y1, marker='o', linestyle='--', label='CrossDocked',color = 'blue')
    ax.set_xlabel(f'{s} Distribution',fontsize=20)
    ax.set_ylabel('Count')
    # ax.set_title(f'{s} Distribution')
    ax.legend()
    x2 = sa_counts2.index
    y2 = sa_counts2.values
    x_smooth_2 = np.linspace(min(x2), max(x2), 100)
    spline = make_interp_spline(x2, y2)
    y_smoth_2 = spline(x_smooth_2)
    # 在ax2上绘制第二个文件的SA值折线图
    ax.plot(x2,y2, marker='o', linestyle='-', label='OURS', color='red')

    ax.set_xlabel(f'{s} Distribution',fontsize=20)
    ax.set_ylabel('Count',fontsize=20)
    # ax.set_title(f'{s} Distribution')
    ax.set_yticklabels([f'{val:.0f}' for val in ax.get_yticks()], fontsize=20)
    ax.set_xticklabels([f'{val:.1f}' for val in ax.get_xticks()], fontsize=15)
    ax.grid(False)
    ax.legend()

    ax.legend(ncol=4, loc=[0.10, 1.01], fontsize=20, frameon=False)

def get_feture(element_counts_orl,element_counts_gen):
    h = []
    s = [element_counts_orl[i] for i in element_counts_orl]
    t = [element_counts_gen[i] for i in element_counts_gen]
    h.append(s)
    h.append(t)
    return h

if __name__ == "__main__":
    path_orl = '/home/linux/Documents/DiffSBDD-Data/data_pre/processed_crossdock_noH_full_temp/test/'
    path_gen = '/home/linux/Downloads/121_epoch/result_last_121/processed/'
#     path_orl = '/home/linux/Documents/DiffSBDD-Data/data_pre/processed_noH_full/test/'
#     path_gen = '/home/linux/Downloads/process_result/result_121_processed_2/processed/'
    # 读文件，统计原子数（主要对比不同数据集上每个原子的个数，QED,SA）的分布
    element_counts_orl = num_node(path_orl)
    element_counts_gen = num_node(path_gen)

    bonds_type_orl = num_bonds(path_orl)
    bonds_type_gen = num_bonds(path_gen)
    # print(element_counts_gen,element_counts_orl)

    path_sa_orl = '/home/linux/DiffFBDD/results_all.csv'
    path_sa_gen = '/home/linux/DiffFBDD/result/121_epoch/results/results_all.csv'
    # path_sa_orl = '/home/linux/DiffFBDD/BindingMoad/results/results.csv'
    # path_sa_gen = '/home/linux/DiffFBDD/BindingMoad/results/results_all.csv'

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    # plt.suptitle('CrossDocked',fontsize=20)
    # first_figure
    plot_ele_counts(get_feture(element_counts_orl,element_counts_gen),ax[0,0])
    # second_figure
    plot_bonds_counts(get_feture(bonds_type_orl,bonds_type_gen),ax[0,1])
    # third_figure
    get_feature(path_sa_orl,path_sa_gen,ax[1,0],'SA')
    #forth_figure
    get_feature(path_sa_orl, path_sa_gen, ax[1, 1], 'Vina_score')
    # ax[1,1].arrow(-10, 20, -10, 20, length_includes_head=False, head_width=0.05, fc='b', ec='k')
    # ax[1,1].annotate('', xy=(-10.5,40), xytext=(-7.5, 40), arrowprops=dict(arrowstyle='->', lw=7.))
    # ax[1, 1].annotate('Better', xy=(-10, 20), xytext=(-10.2, 43), fontsize=25)
    # ax[1, 0].annotate('', xy=(0.93, 550), xytext=(0.7, 550), arrowprops=dict(arrowstyle='->', lw=7.))
    # ax[1,0].annotate('Better', xy=(0.2, 300), xytext=(0.7, 590),fontsize=25)
    ax[1, 1].annotate('', xy=(-12.7, 40), xytext=(-9, 40), arrowprops=dict(arrowstyle='->', lw=7.))
    ax[1, 1].annotate('Better', xy=(-10, 20), xytext=(-12.2, 43), fontsize=25)
    ax[1, 0].annotate('', xy=(0.7, 250), xytext=(0.52, 250), arrowprops=dict(arrowstyle='->', lw=7.))
    ax[1, 0].annotate('Better', xy=(0.2, 300), xytext=(0.52, 270), fontsize=25)
    fig.text(0.49,0.01,'CrossDocked',ha = 'center',fontsize=25)

    # 调整子图布局
    plt.tight_layout()
    # 显示合并后的图
    plt.savefig('/home/linux/Crossdocked.png', dpi=800, bbox_inches='tight')
    plt.show()

