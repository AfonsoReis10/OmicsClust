import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import os
from .SOM import _gene_search

def genegrid_dict(classification):
    """
    Builds dictionary genegrid, with each node as a key and its corresponding genes and their data as values.
    Parameters:
        classification (gema.Classification Object): Classification object.
    Returns:
        genegrid (dictionary): each key coordinate of the grid and corresponding to
        it are the names of the genes and their data.
    """
    genegrid = {}
    for i in range(len(classification.classification_map)):
        x_coord = classification.classification_map['x'][i]
        y_coord = classification.classification_map['y'][i]
        label = classification.classification_map['labels'][i]
        data = classification.classification_map['data'][i]

        if (x_coord, y_coord) in genegrid:
            genegrid[(x_coord, y_coord)].append((label,data))
        else:
            genegrid[(x_coord, y_coord)] = [(label, data)]
    return genegrid

#%% 5. Correlação, Variância e Entropia

# 5.1 Correlação

def correlation(main_map, classification, metagene_map):

    """
    Computes and plots the average correlation between each gene associated with 
    a grid node and its respective metagene.

    Parameters:
        main_map: An object that contains the grid map structure, with a specified 
        map size (`map_size`).
        classification: Classification data, which indicates which genes are 
        associated with each grid node.
        metagene_map: A map of metagenes that represents the averaged gene 
        expression values per grid node.
    
    Returns:
        The function displays a heatmap of the correlation values, using a color map 
        (coolwarm) to differentiate high and low correlations, with a color bar 
        indicating the range.
    """
    

    genegrid=genegrid_dict(classification)
    correlation_map = []
    mcounter=0
    for i in range(main_map.map_size):  
        for j in range(main_map.map_size):
            check_for_genes=genegrid.get((i,j))
            if check_for_genes:
                gene_expression=[]
                correlations=[]
                for k in range(len(genegrid[(i,j)])):
                    gene_expression.append(genegrid[(i,j)][k][1]) #temos todas as expressões dos genes alocados a um neurónio
                    correlation=np.corrcoef(metagene_map[mcounter],gene_expression[k])[0][1]
                    correlations.append(correlation)
                mean_correlation = np.mean(correlations)
                correlation_map.append(mean_correlation)
            else:
                correlation_map.append(0.5) #mudar o nó que não tem genes para branco(com outro colormap)
            mcounter+=1
    correlation_map=np.reshape(np.array(correlation_map),(main_map.map_size,main_map.map_size))

    vmin=np.min(correlation_map)
    vmax=np.max(correlation_map)
    plt.figure(figsize=(8, 6))
    plt.imshow((correlation_map), cmap='coolwarm',interpolation='none', vmin=vmin, vmax=vmax, origin='lower')
    plt.colorbar(ticks=[vmin, vmax])
    plt.title('Mean Gene-Metagene Correlation')
    plt.show()

# 5.2 Gene-Metagene Variance (não está em uso)

def avg_variance(main_map, genegrid):
    """
    Computes and plots the average variance of gene expression levels for all genes 
    associated with each grid node.

    Parameters:
        main_map: An object that contains the grid map structure, with a specified 
                  map size (`map_size`).
        genegrid: A dictionary mapping each grid node (i, j) to a list of genes and 
                  their associated expression values.    
    Returns:
        Displays a heatmap of the logged variance values, using a color map 
        (coolwarm) to show regions with high and low variance, and a color bar 
        indicating the range.
    """
    avg_variance_map = []
    mcounter=0
    for i in range(main_map.map_size): 
        for j in range(main_map.map_size):
            check_for_genes=genegrid.get((i,j))
            if check_for_genes:
                gene_expression=[]
                correlations=[]
                for k in range(len(genegrid[(i,j)])):
                    gene_expression.append(genegrid[(i,j)][k][1]) #temos todas as expressões dos genes alocados a um neurónio
                variance=np.var(gene_expression)
                avg_variance_map.append(variance)
            else:
                avg_variance_map.append(1)
            mcounter+=1

    avg_variance_map=np.log(avg_variance_map)
    avg_variance_map=np.reshape(np.array(avg_variance_map),(main_map.map_size,main_map.map_size))

    vmin=np.min(avg_variance_map)
    vmax=np.max(avg_variance_map)
    plt.figure(figsize=(8, 6))
    plt.imshow((avg_variance_map), cmap='coolwarm',interpolation='none', vmin=vmin, vmax=vmax, origin='lower')
    plt.colorbar(ticks=[vmin, vmax])
    plt.title('Gene-Metagene Variance')
    plt.show()

# 5.3 Metagene Variance

def variance(main_map, metagene_map):
    """
    Computes and plots the variance of the metagene associated with each grid node.

    Parameters:
        main_map: An object that contains the grid map structure, with a specified 
                  map size (`map_size`).
        genegrid: A dictionary mapping each grid node (i, j) to a list of genes and 
                  their associated expression values.    
    Returns:
        Displays a heatmap of the logged variance values, using a color map 
        (coolwarm) to show regions with high and low variance, and a color bar 
        indicating the range.
    """
    variance_map = []
    for m in range(len(metagene_map)):
        sum_var = []
        for k in range(len(metagene_map[m])):
            delta = (metagene_map[m][k] - np.mean(metagene_map[m])) ** 2
            sum_var.append(delta / len(metagene_map[m]))
        variance_map.append(np.sum(sum_var))
    
    variance_map = np.log(variance_map)
    reshaped_variance_map = np.reshape(np.array(variance_map), (main_map.map_size, main_map.map_size))
    
    vmin = np.min(reshaped_variance_map)
    vmax = np.max(reshaped_variance_map)
    
    vmin_e = np.exp(vmin)
    vmax_e = np.exp(vmax)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(reshaped_variance_map, cmap='coolwarm', interpolation='none', vmin=vmin, vmax=vmax, origin='lower')
    
    cbar = plt.colorbar()
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f'{vmin_e:.2f}', f'{vmax_e:.2f}'])
    
    plt.title('Metagene Variance')
    plt.show()

# 5.4 Metagene Entropy

def entropy(main_map, metagene_map):
    """
    Plots entropy grid of the metagenes, differentiating them between three expression states:
    underexpressed, overexpressed, and inconclusive.

    Parameters:
        main_map: Object containing map details, particularly the grid size (map_size).
        metagene_map: 2D array containing metagene expression values for each node.

    Returns:
        Displays a heatmap visualizing the entropy across the grid nodes.
    """
    flat_metagenes=metagene_map.flatten()
    percentile25=np.percentile(flat_metagenes,25)
    percentile75=np.percentile(flat_metagenes,75)
    entropy_map=[]
    for i in range(len(metagene_map)):
        rho1=0 #underexpressed (under 25)
        rho2=0 #inconclusive
        rho3=0 #overexpressed (over75)
        for j in range(len(metagene_map[i])):
            if metagene_map[i][j]<=percentile25:
                rho1+=1
            elif metagene_map[i][j]>=percentile75:
                rho3+=1
            else:
                rho2+=1
        
        total=rho1 + rho2 + rho3
        p1 = rho1 / total
        p2 = rho2 / total
        p3 = rho3 / total

        state1 = p1 * np.log2(p1) if p1 > 0 else 0
        state2 = p2 * np.log2(p2) if p2 > 0 else 0
        state3 = p3 * np.log2(p3) if p3 > 0 else 0

        entropy_sum = -(state1 + state2 + state3)
        entropy_map.append(entropy_sum)

    entropy_map=np.reshape(np.array(entropy_map),(main_map.map_size,main_map.map_size))
    vmin=np.min(entropy_map)
    vmax=np.max(entropy_map)
    plt.figure(figsize=(8, 6))
    plt.imshow((entropy_map), cmap='coolwarm',interpolation='none', vmin=vmin, vmax=vmax, origin='lower')
    cbar = plt.colorbar()
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f'{vmin:.2f}', f'{vmax:.2f}'])
    plt.title('Metagene Entropy')
    plt.show()

#%% 6. Averaged Maps w/ Gene Labels
def averaged_maps(main_map_avg, dataset, nreplicates, genelist, classification, ensemblid, n_rows, n_col):
    """
    Plots the averaged Self-Organizing Maps (SOMs) for the dataset. Highlights specific gene coordinates on each plot.
    
    Parameters:
        main_map_avg: List of averaged SOM maps to be plotted.
        dataset: DataFrame containing sample names to use as titles.
        nreplicates: Number of replicates per sample.
        genelist: List of gene names to locate within the classification map.
        classification: Classification object with SOM node classifications.
        ensemblid: Identifier map for genes.
        n_rows: Number of rows for subplots in the figure.
        n_col: Number of columns for subplots in the figure.
        
    Returns:
        Displays a grid of SOM maps with highlighted gene locations.
    """

    stage=_gene_search(genenames=genelist, classification_map=classification.classification_map, ensembl_id=ensemblid)

    n_col=n_col
    n_rows=n_rows
    title=0
    fig = plt.figure(figsize=(20, 7*n_rows))
    gs = fig.add_gridspec(n_rows, n_col)
    xscatter=[]
    yscatter=[]
    for i in range(len(stage)): 
        xscatter.append(stage[i][0])
        yscatter.append(stage[i][1]) 

    for i, map_index in enumerate(range(len(main_map_avg))):
        row = i // n_col
        col = i % n_col
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(main_map_avg[map_index], cmap='jet', interpolation='none', origin='lower')
        ax.scatter(yscatter, xscatter, c='#000000', marker='o')
        ax.set_title(dataset.columns[title])
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        fig.colorbar(im, ax=ax, shrink=0.2, ticks=[np.min(main_map_avg[map_index]),np.min(main_map_avg[map_index])/2, 0, np.max(main_map_avg[map_index])/2, np.max(main_map_avg[map_index])])
        title+=nreplicates
    fig.tight_layout()
    plt.show()
#%% 7. All Maps (Atualizar descrição)

def allmaps(main_map, dataset, ncolumns, output_folder_name):
    """
    Plots every sample and its corresponding SOM map, arranging plots in a grid.
    
    Parameters:
        main_map: Object with SOM weights (typically a 3D array).
        dataset: DataFrame containing sample names used as plot titles.
        ncolumns: Number of columns in the figure layout.
        output_folder_name: Name of the folder to save individual sample maps.
    
    Returns:
        Displays a grid of SOM maps for all samples and saves each sample plot individually.
    """

    output_folder = output_folder_name
    os.makedirs(output_folder, exist_ok=True)

    if len(dataset.loc[0])%ncolumns==0:
        n_rows= int(len(dataset.loc[0])/ncolumns)
    else:
        n_rows = int(len(dataset.loc[0])//ncolumns +1)
    
    n_col = ncolumns
    fig, axs = plt.subplots(n_rows, n_col, figsize=(20, 7*n_rows))
    sum = 0

    for i in range(n_rows):
        for j in range(n_col):  
            if sum < main_map.weights.shape[2]:

                titulo=dataset.columns[sum]
                for char in titulo:
                    if char == '"':
                        titulo=titulo.replace('"', '')

                ax = axs[i, j]
                im = ax.imshow(main_map.weights[:,:,sum], cmap='jet', interpolation='none', origin='lower')
                ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
                ax.set_title(titulo)
                cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
                
                
                single_fig, single_ax = plt.subplots()
                single_im = single_ax.imshow(main_map.weights[:,:,sum], cmap='jet', interpolation='none', origin='lower')
                single_ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
                single_ax.set_title(dataset.columns[sum])
                single_cbar = single_fig.colorbar(single_im, ax=single_ax, fraction=0.05, pad=0.04)
                
                
                file_name = os.path.join(output_folder, f"figure_{titulo}.png")
                single_fig.savefig(file_name, bbox_inches='tight')
                plt.close(single_fig)
                
                sum += 1
            else:
                axs[i, j].axis('off')

    fig.tight_layout()
    plt.show()
    plt.close(fig)

# %% 8. Maps in Absolute Scale

def scaled_maps(main_map, dataset, nrows, ncolumns):
    """
    Plots all SOM maps with a unified color scale across all maps.

    Parameters:
        main_map: Object with SOM weights (3D array representing each sample's map).
        dataset: DataFrame containing sample names used as plot titles.
        nrows: Number of rows in the figure grid layout.
        ncolumns: Number of columns in the figure grid layout.
    
    Returns:
        Displays all SOM maps with a single color scale.
    """

    n_rows = nrows
    n_col = ncolumns
    fig, axs = plt.subplots(n_rows, n_col)
    fig.set_figwidth(20)
    fig.set_figheight(15)
    images = []
    sum = 0
    title = 0
    
    try:
        for j in range(n_rows):
            for i in range(n_col):
                try:
                    titulo = dataset.columns[title]
                    titulo = titulo.replace('"', '')
                except IndexError:
                    titulo = "Placeholder"
                
                try:
                    images.append(axs[j, i].imshow(main_map.weights[:, :, sum], cmap='jet', interpolation='none', origin='lower'))
                except IndexError:  
                    images.append(axs[j, i].imshow(np.zeros((40, 40)), cmap='jet', interpolation='none', origin='lower'))

                axs[j, i].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
                axs[j, i].set_title(titulo)
                axs[j, i].grid(False)
                
                sum += 1
                title += 1

    except IndexError as e:
        pass  

    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)
    fig.colorbar(images[0], ax=axs, fraction=.1)
    plt.show()