import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import constants.constants as plot
from matplotlib.gridspec import GridSpec

def univariate_plot(df, col, _hue='Default'):    
    fix, ax = plt.subplots(1, 2, figsize=(20, 4), gridspec_kw={'width_ratios': [1,4]})    
    sns.violinplot(x='Default', y=col, data=df, palette=['#ccd5ae', '#d4a373'], ax=ax[0])
    ax[0].set_title(f'Distribution of {col}', fontname=plot.FONT, fontsize=plot.TITLE_FONT_SIZE)
    ax[0].set_xlabel(f'{col}', fontname=plot.FONT, fontsize=plot.LABEL_FONT_SIZE)
    ax[0].set_ylabel('Frequency', fontname=plot.FONT, fontsize=plot.LABEL_FONT_SIZE)

    sns.kdeplot(df, x = col, hue=_hue, palette=['#ccd5ae', '#d4a373'], fill=True,ax=ax[1]);
    ax[1].set_title(f'Distribution of {col}', fontname=plot.FONT, fontsize=plot.TITLE_FONT_SIZE)
    ax[1].set_xlabel(f'{col}', fontname=plot.FONT, fontsize=plot.LABEL_FONT_SIZE)
    ax[1].set_ylabel('Frequency', fontname=plot.FONT, fontsize=plot.LABEL_FONT_SIZE)
    plt.savefig(f"images/univariate_plot_{col}.png")
    plt.show()

def bivariate_plot(df, _x, _y, _hue='Default', x_margin=0, y_margin=0, kde_levels=10):
    x_min, x_max = df[_x].min()-x_margin, df[_x].max()+x_margin
    y_min, y_max = df[_y].min()-y_margin, df[_y].max()+y_margin
    
    g = sns.jointplot(data=df, x=_x, y=_y, hue=_hue, 
                      marginal_ticks=True, 
                      space=0.1, marker="+",s=100,ratio=4,
                      xlim=(x_min, x_max),
                      ylim=(y_min, y_max),
                      alpha=1,
                      palette=['#f1dca7', '#7f5539'],
                      height=7
                 )
    g.plot_joint(sns.kdeplot, palette=['#9b9b7a', '#7f5539']  , zorder=0, levels=kde_levels, alpha=0.7, bw_adjust=0.5)
    sns.move_legend(g.ax_joint, "lower right", title='Species')
    g.set_axis_labels(_x,_y, fontname=plot.FONT, fontsize=plot.LABEL_FONT_SIZE);
    plt.xticks(rotation=30, ha='center', va='top')
    plt.savefig(f"images/bivariate_plot_{_x}_{_y}_{_hue}.png")
    plt.show()

def bivariate_gridplot(df, _x, _y, _hue='Default'):
    df = df.replace({'Default': {0:"No", 1:"Yes"}})
    fix, ax = plt.subplots(1, 3, figsize=(20, 4), gridspec_kw={'width_ratios': [1,1,2]})    
    sns.barplot(data=df,x=_hue,y=_y,hue=_x, 
                width=0.7, edgecolor="white", linewidth=1.5,
                palette=['#9b9b7a', '#7f5539', '#997b66', '#d4a373'], 
                ax = ax[0])
    ax[0].legend(loc="upper left", ncols=4, bbox_to_anchor=(-0.2, -.12))
    ax[0].set_xlabel("Loan Defaulters")
    
    sns.countplot(data=df,x=_hue,hue=_x, 
                  width=0.7, edgecolor="white", linewidth=1.5,
                  palette=['#9b9b7a', '#7f5539', '#997b66', '#d4a373'], ax = ax[1], legend=False)
    
    sns.violinplot(data=df, x=_x, y=_y, hue=_hue, 
                   palette=['#d4a373', '#7f5539', '#997b66', '#9b9b7a'], ax=ax[2])
    ax[2].legend(loc="upper left", ncols=4, bbox_to_anchor=(-0.1, -.12))
    
    fix.suptitle("Relation of Loan Defaulters with LoanAmount and Education", fontsize=16)
    plt.savefig(f"images/bivariate_gridplot_{_x}_{_y}_{_hue}.png")
    plt.show()


def bivariate_plots(df, _x, _y, _hue = 'Default', _bg='IncomeGroup', title=''):
    fig = plt.figure(constrained_layout=True, figsize=(20, 10))    

    gs = GridSpec(2, 4, figure=fig)
    
    ax1 = fig.add_subplot(gs[0:, :-2])
    ax2 = fig.add_subplot(gs[0, -2:])
    ax3 = fig.add_subplot(gs[1, -2:])
        
    df = df.replace({'Default': {0:"No", 1:"Yes"}})
    
    sns.scatterplot(data=df.query('Default=="Yes"'), x=_x, y=_y, size='CreditScore', hue='LoanPurpose', 
                    legend=True, sizes=(20, 200),
                    palette=['#d4a373', '#7f5539', '#997b66', '#9b9b7a'], 
                    style='Default', ax=ax1)
    ax1.legend(ncols=6, bbox_to_anchor=(0.8, -0.08))
    ax1.set_title(f"{_y} - {_x} distribution", fontsize=16)
    ax1.set_xlabel(_x)
    ax1.set_ylabel(_y)

    sns.kdeplot(data=df.query('Default=="Yes"'), x=_x, y=_y, hue=_hue, 
                palette=['#7f5539','#ccd5ae', '#d4a373', '#fefae0', '#f1dca7'],            
                common_norm=False, zorder=0, levels=5, alpha=0.5, bw_adjust=0.5, fill=True, ax=ax2)
    ax2.set_title(f"{_y} - {_x} density chart", fontsize=16)
    ax2.set_xlabel(_x)
    ax2.set_ylabel(_y)
    
    if _bg not in df.columns.tolist():
        df['IncomeGroup'] = pd.qcut(df['Income'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
        
    sns.boxenplot(data=df, x=_bg, y=_y, ax=ax3, hue=_hue,palette=['#d4a373', '#7f5539', '#997b66', '#9b9b7a'])
    ax3.legend(ncols=6, bbox_to_anchor=(0.6, -0.15))
    ax3.set_title(f"Per {_y} - {_bg} distribution per {_hue}")
    ax3.set_xlabel(_bg)
    ax3.set_ylabel(_y)
    
    # depict illustration
    if title == '':
        title=f"{_x} and {_y} distribution for loan defaulters";
        
    fig.suptitle(title, fontsize=20)
    plt.savefig(f"images/bivariate_plots_{_x}_{_y}_{_hue}_{_bg}.png")
    plt.show()

def bivariate_gridplot_alpha(df, _x, _y, _hue='Default'):
    fig = plt.figure(constrained_layout=True, figsize=(20, 10))    

    gs = GridSpec(2, 3, figure=fig)
    gs.set_width_ratios([1,1,2])
     
    # create sub plots as grid
    ax2 = fig.add_subplot(gs[0, :-1])
    ax3 = fig.add_subplot(gs[0:, -1:])
    ax4 = fig.add_subplot(gs[-1, 0])
    ax5 = fig.add_subplot(gs[-1, -2])

    df = df.replace({'Default': {0:"No", 1:"Yes"}})
    
    sns.violinplot(data=df, x=_y, y=_x, hue="Default", 
                   palette=['#d4a373', '#7f5539', '#997b66', '#9b9b7a'], 
                   ax=ax2)
    ax2.legend(ncols=4, bbox_to_anchor=(0.2, -0.15))
    ax2.set_title(f"Per {_y} - {_x} distribution")
        
    sns.barplot(data=df,x=_hue,y=_x,hue=_y, 
                palette=['#9b9b7a', '#7f5539', '#997b66', '#d4a373', '#ccd5ae'],
                width=0.7, 
                edgecolor="white", 
                linewidth=1.5,
                ax = ax4)
    ax4.legend(ncols=2, bbox_to_anchor=(0.8, -0.15))
    ax4.set_title(f"{_x} {_y} distribution")
    ax4.set_xlabel("Loan Defaulters")
    
    sns.countplot(data=df,x=_hue,hue=_y, 
                  palette=['#9b9b7a', '#7f5539', '#997b66', '#d4a373', '#ccd5ae'], 
                  width=0.7, 
                  edgecolor="white", 
                  linewidth=1.5,
                  legend=True, 
                  ax = ax5)
    ax5.set_title(f"{_y} count")
    ax5.set_xlabel("Loan Defaulters")
    ax5.legend(ncols=2, bbox_to_anchor=(0.8, -0.15))  
    
    sns.histplot(data=df.query('Default == "Yes"'),x=_x,hue=_y, 
                 palette=['#9b9b7a', '#7f5539', '#997b66', '#d4a373','#ccd5ae'], 
                 shrink=0.8,
                 multiple="dodge", 
                 legend=True,
                 ax = ax3)
    ax3.set_title("Only Loan Defaulters")
    ax3.set_xlabel(_x)
    
    fig.suptitle(f"Relation of Loan Defaulters with {_x} and {_y}", fontsize=20)
    
    plt.savefig(f"images/bivariate_gridplot_alpha_{_x}_{_y}_{_hue}.png")
    plt.show()

    
def bivariate_gridplot_category(df, _x, _y, _hue='Default'):
    fig = plt.figure(constrained_layout=True, figsize=(20, 12))

    gs = GridSpec(3, 3, figure=fig)
    gs.set_width_ratios([1,1,2])
     
    # create sub plots as grid
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :-1])
    ax3 = fig.add_subplot(gs[1:, -1])
    ax4 = fig.add_subplot(gs[-1, 0])
    ax5 = fig.add_subplot(gs[-1, -2])
    
    df = df.replace({'Default': {0:"No", 1:"Yes"}})
    
    sns.kdeplot(df, x = _x, hue=_hue, 
                palette=['#9b9b7a', '#d4a373'], 
                fill=True, 
                multiple='stack', 
                ax=ax1);
    
    sns.violinplot(data=df, x=_y, y=_x, hue="Default", 
                   palette=['#d4a373', '#7f5539', '#997b66', '#9b9b7a'], 
                   ax=ax2)
    ax2.legend(ncols=4, bbox_to_anchor=(0.2, -0.15))
    ax2.set_title(f"Per {_y} - {_x} distribution")
    
    
    sns.barplot(data=df,x=_hue,y=_x,hue=_y, 
                palette=['#9b9b7a', '#7f5539', '#997b66', '#d4a373'],
                width=0.7, 
                edgecolor="white", 
                linewidth=1.5,
                ax = ax4)
    ax4.legend(ncols=2, bbox_to_anchor=(0.8, -0.15))
    ax4.set_title(f"{_x} {_y} distribution")
    ax4.set_xlabel("Loan Defaulters")
    
    sns.countplot(data=df,x=_hue,hue=_y, 
                  palette=['#9b9b7a', '#7f5539', '#997b66', '#d4a373'], 
                  width=0.7, 
                  edgecolor="white", 
                  linewidth=1.5,
                  legend=True, 
                  ax = ax5)
    ax5.set_title(f"{_y} count")
    ax5.set_xlabel("Loan Defaulters")
    ax5.legend(ncols=2, bbox_to_anchor=(0.8, -0.15))  
    
    sns.histplot(data=df.query('Default == "Yes"'),x=_x,hue=_y, 
                 palette=['#9b9b7a', '#7f5539', '#997b66', '#d4a373'], 
                 shrink=0.8,
                 multiple="dodge", 
                 legend=True,
                 ax = ax3)
    ax3.set_title("Only Loan Defaulters")
    ax3.set_xlabel(_x)
    
    fig.suptitle(f"Relation of Loan Defaulters with {_x} and {_y}", fontsize=20)
    
    plt.savefig(f"images/bivariate_gridplot_category_{_x}_{_y}_{_hue}.png")
    plt.show()

def plot_two_score(df, y1, y1_range, y1_label,y2, y2_range, y2_label, title):
    width = 0.3
    font = {'family': 'sans-serif', 'color':  '#7f5539', 'weight': 'regular', 'size': 14}
    
    x = np.arange(len(df.index))  
    y1 = df[y1]  
    y2 = df[y2]   
    
    fig, ax1 = plt.subplots(layout="constrained", figsize=(16, 7))
    
    bars1 = ax1.bar(x - width/2, y1, width=width, color='#9b9b7a', alpha=0.7, label=y1_label)
    
    # Set the y-axis limits and label for the first feature
    ax1.set_ylim(y1_range[0], y1_range[1])
    ax1.set_ylabel(y1_label, color='#9b9b7a', fontdict=font)
    ax1.tick_params(axis='y', labelcolor='#9b9b7a')
    
    # Create a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()
    
    
    # Plot the second feature on ax2 (right y-axis)
    bars2 = ax2.bar(x + width/2, y2, width=width, color='#7f5539', alpha=0.7, label=y2_label)
    
    # Set the y-axis limits and label for the second feature
    ax2.set_ylim(0, 0.04)
    ax2.set_ylabel(y2_label, color='#7f5539', fontdict=font)
    ax2.tick_params(axis='y', labelcolor='#7f5539')
    
    # Configure the x-axis labels (optional)
    font['size']=12
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{i}' for i in df.index], rotation=45, ha='right', fontdict=font)
    
    # Create a combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title(title)
    plt.tight_layout()
    plt.show()