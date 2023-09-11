from tkinter import font
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches

sns.set(font_scale=1.2)
sns.set_style('whitegrid')
sns.set_context('talk')

###############################################################################
###############################################################################

def plot_regression(x, y_obs, x_mod, y_mean, y_hpdi, y_predci, ci, **kwargs):
    '''
    Bit of a messy function - could do with a cleaning of the inputs.
    '''
    # Sort values for plotting by x axis
    sns.set(font_scale=1.2)
    sns.set_style('whitegrid')

    # Plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
    ax.plot(x_mod, y_mean)
    ax.plot(x, y_obs, "o")
    ax.fill_between(
        x_mod, y_predci[0,:], y_predci[1,:], 
        alpha=0.2, color='C1', interpolate=True,
        label="Prediction\nuncertainty"
    )
    ax.fill_between(
        x_mod, y_hpdi[0,:], y_hpdi[1,:],
        alpha=0.4, color='C0', interpolate=True,
        label="Model\nuncertainty"
    )

    ax.set_xlabel(kwargs.get('xlabel', 'Log Water Level'), labelpad=10)
    ax.set_ylabel(kwargs.get('ylabel', 'Log Discharge'), labelpad=10)
    ax.set_title(kwargs.get('title', 'Regression line with {}% CI'.format(int(ci * 100))), pad=15)

    ax.legend(loc='center left', bbox_to_anchor=(1.05,0.5))

###############################################################################
###############################################################################

def draw_fit(x_obs,y_obs,x_pred,y_pred = None,y_sample=None,**kwargs):
    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.add_subplot(111)

    if not y_sample is None:
        ax1.plot(x_pred, y_sample[0,:].T,'-',color='xkcd:light grey',alpha=0.4,label='Samples')
        ax1.plot(x_pred, y_sample[1:,:].T,'-',color='xkcd:light grey',alpha=0.4)

    # if not y is None:
    ax1.plot(x_obs, y_obs, 'o',color='xkcd:dark grey',label='Observed')

    if not y_pred is None:
        ax1.plot(x_pred, y_pred, 'C0', label='Predicted')

    # # plot mean
    # ax1.plot(xSample, predyMean, 'k', label='Mean')

    ax1.set_xlabel(kwargs.get('xlabel','Log Water Level'),labelpad=10)
    ax1.set_ylabel(kwargs.get('ylabel','Log Disharge'),labelpad=10)

    if kwargs.get('log_scale',False):
        ax1.set_xscale('log')
        ax1.set_yscale('log')

    ax1.set_title(kwargs.get('title','Frequentist Fit'),pad=10)
    # place legend outside the plot
    plt.legend(loc='center', bbox_to_anchor=(1.2, 0.5))
    plt.show()
    
###############################################################################
###############################################################################

def plot_rc(data,a=None,b=None, mu=None, log_scale=True,retfig=False):
    """
    Plot the raw values as a rating curve
    """
    fig = plt.figure(figsize=(7.5, 5))
    ax1 = fig.add_subplot(111)

    sns.scatterplot(
        x='water_level', y='discharge', hue='year', 
        palette='viridis',
        data=data, ax=ax1)

    if not a is None and not b is None:
        dummy_x = np.linspace(data['water_level'].min(), data['water_level'].max(), 100)
        dummy_y = a + b * dummy_x
        sns.lineplot(x=dummy_x, y=dummy_y, color='C3', linestyle='--', ax=ax1)
    
    if not mu is None:
        mod_data = data.copy()
        mod_in = pd.DataFrame(data=np.exp(mu.squeeze().T), columns=['mod_dis_{}'.format(_) for _ in np.arange(mu.shape[0])])
        mod_data = pd.concat([mod_data[['date','water_level']], mod_in], axis=1)
        mod_data = mod_data.melt(id_vars=['date','water_level'], var_name='mod', value_name='modelled_discharge')
        sns.scatterplot(
            x='water_level', y='modelled_discharge',
            data=mod_data, s=5, ax=ax1,
            label = 'Modelled Discharge'
        )

    if log_scale:
        ax1.set_yscale('log')
        ax1.set_xscale('log')
    ax1.set_xlabel(r'Water Level ($m$)')
    ax1.set_ylabel(r'Discharge ($m^3/s$)')

    # ax1.set_ylim([data['discharge'].min(), data['discharge'].max()])

    plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.)

    if retfig:
        return fig
    else:      
        plt.show()


###############################################################################
###############################################################################


