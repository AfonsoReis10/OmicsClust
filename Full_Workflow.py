#%% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import GEMA as gema
from pyensembl import EnsemblRelease
from sklearn.cluster import KMeans
import pickle
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from goatools.base import download_go_basic_obo
from goatools.base import download_ncbi_associations
from goatools.obo_parser import GODag
from goatools.anno.genetogo_reader import Gene2GoReader
from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS
import matplotlib as mpl
import seaborn as sns
import textwrap
import gseapy as gp
import json

#%% 1. Dataset Load, Function and Variable Setup

"""
Loads your data (file) into a pandas DataFrame (dataset).
Set gene_id to 'True' if your data uses EnsemblIDs and 'False' if using gene symbols.
Load Ensembl IDs for use in gene_search function (if gene_id==True).
Set nreplicates (int) to the number of experimental replicates.

TERMINAL: pyensembl install --release 77 --species human
"""

file='C:/Users/reisa/Documents/IST/5ºano/TESe/OmicsClust-main/OmicsClust-main/Datasets/logCPM_Frank_cent(conf).csv'
dataset= pd.read_csv(file, delimiter=",", header=0, index_col=0)
gene_id=False
Ensembl=EnsemblRelease(release=77, species='human')
nreplicates=3

def tratamento(dataset):

    """
    Creates a second dataframe (dados) ONLY for later use in SOM classification to keep data labelled.
    Parameters:
        dataset(DataFrame): Obtained from loading raw data into a pandas DataFrame.
    Returns:
        dados (numpy.array):array of shape (number of genes, number of samples) with gene identification for later use ONLY in SOM classification.
    """
    row_names = dataset.index.tolist()
    dataset.reset_index(drop=True, inplace=True)
    dados = dataset.to_numpy()
    dados = dados.astype(object)
    dados = np.insert(dados, 0, row_names, axis=1)
    return dados

dados=tratamento(dataset)

def gene_search(genenames, classification_map, gene_id):

    """
    Searches for the coordinates of genes in SOM grid.
    Parameters:
        genenames (numpy.ndarray): gene symbols to search for.
        classification_map (DataFrame): output of classification function after SOM mapping.
        gene_id (boolean): set True if genes are identified through EnsemblID or False if through gene symbols.
    Returns:
        genesinSOM (numpy.ndarray): array of tuples with each gene's corresponding coordinates (y,x), in the order that they are given.
    """
    genesinSOM=[]
    if gene_id==False:
        for i in range(len(genenames)):
            found=False
            for j in range(len(classification_map)):
                if genenames[i]==classification_map.iloc[j,0]:
                    position=classification_map.iloc[j,2],classification_map.iloc[j,3]
                    print("Gene '{}' found in index at position {}: {}".format(genenames[i], position, classification_map.iloc[j,0]))
                    print("Coordinates in SOM are'{}'".format(position))
                    genesinSOM.append(position)
                    found = True
                    break
            
            if not found:
                print("Gene '{}' not found.".format(genenames[i]))

        return genesinSOM
    
    if gene_id==True:
        genetranslation=[]
        for i in range(len(genenames)):
            #falta uma condição dentro deste ciclo porque ele buga caso um gene não exista no ensembl tipo o tbxt que na verdade é o tbx5
            found = False
            gene = Ensembl.genes_by_name(genenames[i])
            gene_id = gene[0].gene_id
            print(gene_id)
            genetranslation.append(gene_id)
            for j in range(len(classification_map)):
                if genetranslation[i]==classification_map.iloc[j,0]:
                    position=classification_map.iloc[j,2],classification_map.iloc[j,3]
                    print("Gene ID:'{}', name:'{}' found in index at position {}: {}".format(genetranslation[i], gene[0].gene_name, position, classification_map.iloc[j,0]))
                    print("Coordinates in SOM are'{}'".format(position))
                    genesinSOM.append(position)
                    found = True
                    break
            
            if not found:
                print("Gene '{}' not found.".format(genetranslation[i]))
    
        return genesinSOM


def avgmaps(main_map, nreplicates):

    """
    Obtains map from the average of replicates (does not support missing data).
    Parameters:
        main_map (gema.Map Object): SOM object generated after training.
        main_map.weights is an array of shape (map size, map size, sample number) which represents the SOM maps created for each sample.
        nreplicates (int): number of experimental replicates.
    Returns:
        main_map_avg (numpy.ndarray): array of shape (map size, map size, number of samples/number of replicates).
    """
    main_map_avg = []
    num_samples = len(main_map.weights[0][0])
    for i in range(0, num_samples, nreplicates):
        if i + nreplicates <= num_samples:
            avg_map = np.mean([main_map.weights[:, :, j] for j in range(i, i + nreplicates)], axis=0)
            main_map_avg.append(avg_map)
    return main_map_avg

#%% 2. SOM e Classificação

def SOM(dataset, dados, map_size, period, learning_rate):
    """
    Train a SOM and perform classification.
    Parameters:
        dataset (DataFrame): Input data for training.
        dados (numpy.array): Input data for classification.
        map_size (int): Length of the edges of squared SOM. Map size is actually (map_size*map_size).
        period (int): Training period. Must be greater than 0. Use dataset.values.shape[0]*n, if you want n iterations over the entire dataset (with presentation='sequential').
        learning_rate (float): Initial learning rate. Must be greater than 0.
    Returns:
        main_map (gema.Map Object): SOM object generated after training.
        classification (gema.Classification Object): Classification object.
    """
    main_map = gema.Map(dataset.values, size=map_size, period=period, initial_lr=learning_rate,
                        distance='euclidean', use_decay=True, normalization='none',
                        presentation='sequential', weights='PCA')
    classification = gema.Classification(main_map, dados, tagged=True)
    print('Quantization Error:', classification.quantization_error)
    print('Topographic Error:', classification.topological_error)
    return main_map, classification

main_map, classification = SOM(dataset, dados, map_size=40, period=dataset.values.shape[0]*185, learning_rate=0.05)
main_map_avg=avgmaps(main_map,nreplicates)
#%% 3. SOMSaver
"""
Saves or loads the trained and mapped SOMs as pickle files.

Select 'True' to save or load and 'False' otherwise.
Confirm your save by typing 'yes'.
Also executes main_map_avg, in case the user is re-starting the kernel.
"""

Save=False
Load=False
if Save == True:
    
    confirmation = input("Are you sure you want to save? (yes/no): ")

    if confirmation.lower() == 'yes':
        with open('map_Frank_185.pkl', 'wb') as f:
            pickle.dump(main_map, f)

        with open ('class_Frank_185.pkl', 'wb') as c:
            pickle.dump(classification, c)
    else:
        print("Saving canceled.")
if Load==True:
    with open('main_map_mbranco_182.pkl', 'rb') as f:
        main_map = pickle.load(f)

    with open('classification_mbranco_182.pkl', 'rb') as c:
        classification = pickle.load(c)

main_map_avg=avgmaps(main_map,3)

#%% 4. Metagene_map

def metagenes(main_map):
    """
    Builds a list with all metagenes, which correspond to the final values returned by the SOM for every node.
    Parameters:
        main_map (gema.Map Object): SOM object generated after training.
    Returns:
        metagene_map (numpy.array): array of shape (main_map x main_map, number of samples)
    """
    metagene_map = []
    for i in range(main_map.weights.shape[0]):
        for j in range(main_map.weights.shape[1]):
            metagene=[]
            for k in range(main_map.weights.shape[2]):
                metagene.append(main_map.weights[i,j,k])
            metagene_map.append(metagene)
    metagene_map=np.array(metagene_map)
    return metagene_map

metagene_map=metagenes(main_map)

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

genegrid = genegrid_dict(classification)

def geneid_dict(classification):
    """
    Builds dictionary geneid_grid, with each node as a key and its corresponding gene labels as values.
    Parameters:
        classification (gema.Classification Object): Classification object.
    Returns:
        geneid_grid (dictionary): each key coordinate of the grid and corresponding to
        it are the names of the genes as they are given in the raw data.
    """
    geneid_grid = {}
    for i in range(len(classification.classification_map)):
        x_coord = classification.classification_map['x'][i]
        y_coord = classification.classification_map['y'][i]
        label = classification.classification_map['labels'][i]

        if (x_coord, y_coord) in geneid_grid:
            geneid_grid[(x_coord, y_coord)].append(label)
        else:
            geneid_grid[(x_coord, y_coord)] = [(label)]
    return geneid_grid

geneid_grid=geneid_dict(classification)

def genename_dict(classification):
    """
    Builds dictionary genename_grid, with each node as a key and its corresponding genes and their data as values.
    Parameters:
        classification (gema.Classification Object): Classification object.
    Returns:
        geneid_grid (dictionary): each key coordinate of the grid and corresponding to
        it are the genesymbols, in case the raw data uses EnsemblIDs.
    """   
    genename_grid = {}

    for i in range(len(classification.classification_map)):
        x_coord = classification.classification_map['x'][i]
        y_coord = classification.classification_map['y'][i]
        label = classification.classification_map['labels'][i]

        try:
            genetranslation = Ensembl.gene_by_id(label)
            gene_name = genetranslation.gene_name
            if (x_coord, y_coord) in genename_grid:
                genename_grid[(x_coord, y_coord)].append(gene_name)
            else:
                genename_grid[(x_coord, y_coord)] = [(gene_name)]
        except ValueError as e:
            print("Gene not found:", label)
    return genename_grid

genename_grid = genename_dict(classification)

#%% 5. Correlação, Variância e Entropia

# 5.1 Correlação

def correlation():
    """
    Plots average correlation between all genes allocated to a node of the grid
    and its metagene
    """
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

correlation_map=correlation()

# 5.2 Gene-Metagene Variance

def avg_variance():
    """
    Plots variance grid between all genes allocated to a node of the grid
    and its metagene
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

avg_variance_map=avg_variance()

# 5.3 Metagene Variance

def variance():
    """
    Plots variance grid of the metagenes
    """
    variance_map=[]
    for m in range(len(metagene_map)):
        sum=[]
        for k in range(len(metagene_map[m])):
            delta=(metagene_map[m][k]-np.mean(metagene_map[m]))**2
            sum.append(delta/(len(metagene_map[m])))
        variance_map.append(np.sum(sum))
    
    variance_map=np.log(variance_map)
    variance_map=np.reshape(np.array(variance_map),(main_map.map_size,main_map.map_size))
    vmin=np.min(variance_map)
    vmax=np.max(variance_map)
    plt.figure(figsize=(8, 6))
    plt.imshow((variance_map), cmap='coolwarm',interpolation='none', vmin=vmin, vmax=vmax, origin='lower')
    plt.colorbar(ticks=[vmin, vmax])
    plt.title('Metagene Variance')
    plt.show()

variance_map=variance()

# 5.4 Metagene Entropy

def entropy():
    """
    Plots entropy grid of the metagenes, differentiating them between 3 expressions states,
    underexpressed, overexpressed and inconclusive.
    """
    flat_metagenes=metagene_map.flatten()
    percentile25=np.percentile(flat_metagenes,25)
    percentile75=np.percentile(flat_metagenes,75)
    entropy_map=[]
    for i in range(len(metagene_map)):
        rho1=1 #underexpressed (under 25)
        rho2=1 #inconclusive
        rho3=1 #overexpressed (over75)
        for j in range(len(metagene_map[i])):
            if metagene_map[i][j]<=percentile25:
                rho1+=1
            elif metagene_map[i][j]>=percentile75:
                rho3+=1
            else:
                rho2+=1
        state1=rho1*np.log2(rho1)
        state2=rho2*np.log2(rho2)
        state3=rho3*np.log2(rho3)
        sum=state1+state2+state3
        entropy_map.append(-sum)

    entropy_map=np.reshape(np.array(entropy_map),(main_map.map_size,main_map.map_size))
    vmin=np.min(entropy_map)
    vmax=np.max(entropy_map)
    plt.figure(figsize=(8, 6))
    plt.imshow((entropy_map), cmap='coolwarm',interpolation='none', vmin=vmin, vmax=vmax, origin='lower')
    plt.colorbar(ticks=[vmin, vmax])
    plt.title('Metagene Entropy')
    plt.show()

entropy_map=entropy()

#%% 6. Averaged Maps w/ Gene Labels

"""
Plots the averaged SOMs.
Function gene_search is used to find coordinates of certain genes to plot on top of the SOM.
Set n_col (int) to the number of columns of your figure and n_rows (int) to the number of rows.
Title (int) should stay at 0. Used to iterate through the sample names to give each figure its respective name.
"""

pluripotency=gene_search(genenames=['POU5F1','NANOG','SOX2','LIN28A' ,'ZFP42' ,'THY1'], classification_map=classification.classification_map, gene_id=gene_id)
# mesoderm=gene_search(genenames=['TBXT', 'ANPEP', 'MIXL1', 'ROR2'], classification_map=classification.classification_map, gene_id=gene_id)
# cardiac_mesoderm=gene_search(genenames=['MESP1', 'KDR', 'KIT', 'CXCR4', 'PDGFRA'], classification_map=classification.classification_map, gene_id=gene_id)
# cardiac_progenitor=gene_search(genenames=['ISL1', 'NKX2-5','GATA4', 'TBX5', 'TBX20', 'MEF2C'], classification_map=classification.classification_map, gene_id=gene_id)
# immature_cardio=gene_search(genenames=['MYH6','TNNT2', 'TNNI3', 'MYL2', 'EMILIN2', 'SIRPA'], classification_map=classification.classification_map, gene_id=gene_id)

n_col=11
n_rows=2
title=0
fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(n_rows, n_col)
xscatter=[]
yscatter=[]
for i in range(len(pluripotency)): 
    xscatter.append(pluripotency[i][0])
    yscatter.append(pluripotency[i][1]) 

for i, map_index in enumerate(range(len(main_map_avg))):
    row = i // n_col
    col = i % n_col
    ax = fig.add_subplot(gs[row, col])
    im = ax.imshow(main_map_avg[map_index], cmap='jet', interpolation='none', origin='lower')
    ax.scatter(yscatter, xscatter, c='#000000', marker='o')
    ax.set_title(dataset.columns[title][:-2])
    ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    fig.colorbar(im, ax=ax, shrink=0.2, ticks=[np.min(main_map_avg[map_index]),np.min(main_map_avg[map_index])/2, 0, np.max(main_map_avg[map_index])/2, np.max(main_map_avg[map_index])])
    title+=nreplicates
fig.tight_layout()
plt.show()
#%% 7. All Maps
"""
Plots every sample and its corresponding SOM. Set 'reps' as the number of lines in your figure and 'samps' as the
number of columns. Does not support missing data.
Set n_col (int) to the number of columns of your figure and n_rows (int) to the number of rows.
"""

n_rows = 7
n_col = 3
fig, axs = plt.subplots(n_rows, n_col, figsize=(20, 7*n_rows))
images = []
sum = 0
for i in range(n_rows):
    for j in range(n_col):  
        if sum < main_map.weights.shape[2]:  
            images.append(axs[i, j].imshow((main_map.weights[:,:,sum]), cmap='jet', interpolation='none', origin='lower'))
            axs[i, j].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
            axs[i, j].set_title(dataset.columns[sum][:-2]) # [:-n]deletes the n last characters in the title
            cbar = fig.colorbar(images[-1], ax=axs[i, j], fraction=0.05, pad=0.04)
            sum = sum + 1
        else:
            axs[i, j].axis('off') 

fig.tight_layout()
plt.show()

# %% 8. Maps in Absolute Scale
"""
Plots all maps with a single colorscale.
Set n_col (int) to the number of columns of your figure and n_rows (int) to the number of rows.
"""

n_rows = 3
n_col = 7
fig, axs = plt.subplots(n_rows, n_col)
fig.set_figwidth(20)
fig.set_figheight(15)
images = []
sum=0
title=0
try:
    for i in range(n_col):
        for j in range(n_rows):
            images.append(axs[j, i].imshow(main_map.weights[:,:,sum], cmap='jet', interpolation='none', origin='lower'))
            axs[j, i].tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
            axs[j,i].set_title(dataset.columns[title])
            sum=sum+1
            title+=1
except IndexError as e:
    images.append(axs[j, i].imshow(np.zeros((40, 40)), cmap='jet', interpolation='none', origin='lower'))
    axs[j, i].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    axs[j, i].set_title("Placeholder")
vmin = min(image.get_array().min() for image in images)
vmax = max(image.get_array().max() for image in images)
norm = colors.Normalize(vmin=vmin, vmax=vmax)
for im in images:
    im.set_norm(norm)
fig.colorbar(images[0], ax=axs, fraction=.1)

plt.show()

#%% 9. K-Testing (!) Work In Progress (ver datasettest)

#%% 10. KMeans
"""

Creates the clustered_genes dictionary, in which each entry is a cluster and 
corresponding to it is every gene within that cluster.
"""

def KMeans_clustering(n_clusters):
    """
    Performs the KMeans algorithm on the metagenes and creates a plot for easy cluster visualization with a 
    custom colormap. Does not support more than 20 clusters.
    Parameters:
        n_clusters (int): number of clusters
    Returns:
        cluster_labels_grid (list): list of shape (map_size, map_size) where each entry is the cluster to which the node in the sam eposition belongs to.
    """
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42, verbose=1,init="k-means++").fit(metagene_map)
    cluster_labels = kmeans.labels_
    cluster_labels_grid = cluster_labels.reshape((main_map.map_size, main_map.map_size))
    custom_colormap=['#9e0142','#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#ffffbf', '#d9ef8b', '#abdda4', '#66c2a5', '#3288bd', '#5e4fa2', '#ffffff', '#878787', '#1a1a1a', '#c51b7d', '#b2abd2', '#4d9221', '#35978f', '#313695', '#8c510a']
    cmap = colors.ListedColormap(custom_colormap)
    plt.matshow((cluster_labels_grid), cmap=cmap, origin='lower')
    plt.title('KMeans Clustering of Metagenes')
    plt.colorbar(label='Cluster Label').set_ticks(np.arange(0,n_clusters,1))
    plt.show()
    return cluster_labels_grid

cluster_labels_grid=KMeans_clustering(n_clusters=20)

def clustered_names_dict(cluster_labels_grid, map_size):
    """
    Builds dictionary genegrid, where each key is a cluster and corresponding to it is every gene within it, by symbol.
    Parameters:
        cluster_labels_grid (list): list of shape (map_size, map_size) where each entry is the cluster to which the node in the sam eposition belongs to.
        map_size (int): Length of the edges of squared SOM. Map size is actually (map_size*map_size).
    Returns:
        clustered_genes_names (dictionary): dictionary with every cluster and its corresponding gene symbols.
    """
    clustered_genes_names={}
    for y in range(map_size):
        for x in range(map_size):
            try:
                if cluster_labels_grid[y][x] in clustered_genes_names:
                    clustered_genes_names[cluster_labels_grid[y][x]].extend(genename_grid[(x, y)])
                else:
                    clustered_genes_names[cluster_labels_grid[y][x]] = list(genename_grid[(x, y)])
            except KeyError:
                continue
    return clustered_genes_names

clustered_genes_names=clustered_names_dict(cluster_labels_grid, main_map.map_size)

def clustered_ids_dict(cluster_labels_grid, map_size):
    """
    Builds dictionary genegrid, where each key is a cluster and corresponding to it is every gene within it, by EnsemblID.
    Parameters:
        cluster_labels_grid (list): list of shape (map_size, map_size) where each entry is the cluster to which the node in the sam eposition belongs to.
        map_size (int): Length of the edges of squared SOM. Map size is actually (map_size*map_size).
    Returns:
        clustered_genes_ids (dictionary)_ dictionary with every cluster and its corresponding gene IDs.
    """
    clustered_genes_ids={}
    for y in range(map_size):
        for x in range(map_size):
            try:
                if cluster_labels_grid[y][x] in clustered_genes_ids:
                    clustered_genes_ids[cluster_labels_grid[y][x]].extend(geneid_grid[(x, y)])
                else:
                    clustered_genes_ids[cluster_labels_grid[y][x]] = list(geneid_grid[(x, y)])
            except KeyError:
                continue
    return clustered_genes_ids

clustered_genes_ids=clustered_ids_dict(cluster_labels_grid, main_map.map_size)

#%% Gene Ontology Setup
"""
Follow the instructions in the video (How to do gene ontology analysis in python - Sanbomics) to build the background gene set from NCBI before use.
Import GENEID2NT from the file created.
"""
from genes_ncbi_human import GENEID2NT as GeneID2nt_human
def goatools_setup():
    """
    Setup of gene ontology program. 
    Must be always called to the variables 'mapper', 'goeaobj', 'GO_items', 'inv_map', in this order.
    Returns:
        mapper (dict): dictionary where each key is a gene symbol and its value is the corresponding label the file created before.
        inv_map (dict): the inverse of the 'mapper' dictionary.
        goeaobj (goatools.goea.go_enrichment_ns.GOEnrichmentStudyNS Obejct): Initializes Gene Ontology Object.
        GO_items (list): list of all GO terms that are duplicated.
    """
    obo_fname = download_go_basic_obo()
    fin_gene2go = download_ncbi_associations()
    obodag = GODag("go-basic.obo")
    
    mapper = {}
    for key in GeneID2nt_human:
        mapper[GeneID2nt_human[key].Symbol] = GeneID2nt_human[key].GeneID
    objanno = Gene2GoReader(fin_gene2go, taxids=[9606])
    ns2assoc = objanno.get_ns2assc()
    goeaobj = GOEnrichmentStudyNS(
            GeneID2nt_human.keys(), 
            ns2assoc, 
            obodag, 
            propagate_counts = False,
            alpha = 0.05, 
            methods = ['fdr_bh']) 
    inv_map = {v: k for k, v in mapper.items()}
    GO_items = []
    temp = goeaobj.ns2objgoea['BP'].assoc
    for item in temp:
        GO_items += temp[item]

    temp = goeaobj.ns2objgoea['CC'].assoc
    for item in temp:
        GO_items += temp[item]

    temp = goeaobj.ns2objgoea['MF'].assoc
    for item in temp:
        GO_items += temp[item]
    
    return mapper, goeaobj, GO_items, inv_map

mapper, goeaobj, GO_items, inv_map = goatools_setup()
#%%
def go_it(test_genes, fdr_thresh=0.05):
    """
    Performs Gene Ontology analysis on a given set of genes.
    Parameters:
        test_genes (dictionary): genes to analyzed.
        fdr_thresh (int): threshold for False Discovery Rate.
    Returns:
        GO(DataFrame): post Gene Ontology DataFrame.
    """
    print(f'input genes: {len(test_genes)}')
    
    mapped_genes = []
    for gene in test_genes:
        try:
            mapped_genes.append(mapper[gene])
        except:
            pass
    print(f'mapped genes: {len(mapped_genes)}')
    
    goea_results_all = goeaobj.run_study(mapped_genes)
    goea_results_sig = [r for r in goea_results_all if r.p_fdr_bh < fdr_thresh]
    GO = pd.DataFrame(list(map(lambda x: [x.GO, x.goterm.name, x.goterm.namespace, x.p_uncorrected, x.p_fdr_bh,\
                   x.ratio_in_study[0], x.ratio_in_study[1], x.ratio_in_pop[0], x.ratio_in_pop[1], GO_items.count(x.GO), list(map(lambda y: inv_map[y], x.study_items)),\
                   ], goea_results_sig)), columns = ['GO', 'Term', 'class', 'p', 'p_corr', 'n_genes',
                                                    'n_study', 'pop_count', 'pop_n', 'n_go', 'study_genes'])

    GO = GO[GO.n_genes > 1]
    return GO

#%% Gene Ontology for clusters
"""
Performs gene ontology on the genes of the selected cluster. Select the cluster number according to KMeans.
"""

def cluster_go(clustered_genes, cluster_number, type=None):
    """
    Build a DataFrame of 10 gene ontologies with highest fold enrichment scores.
    Parameters:
        clustered_genes_names (dictionary): dictionary with every cluster and its corresponding gene symbols. Use the according dictionary
        (clustered_genes_names if gene_IDs==True and clustered_genes_ids if gene_IDs==False).
        cluster_number (int): label that identifies the cluster to be analyzed
        type (str): Optional parameter. Specifies ontology type. Available options include:
                    - 'biological_process'
                    - 'cellular_component'
                    - 'molecular_function'
    Returns:
        df (DataFrame): pandas DataFrame of gene ontologies, their class, p-values, corrected p-values, number of genes in study, expected number in population and the gene symbols.
    """
    if type:
        df = go_it(clustered_genes[cluster_number])
        df = df[df['class'] == type]
        df['Fold Enrichment'] = (df.n_genes/df.n_study)/(df.pop_count/df.pop_n)
        df=df.sort_values(by=['Fold Enrichment'], ascending=False)
        df=df[0:10]
        return df
    else:
        df = go_it(clustered_genes[cluster_number])
        df['Fold Enrichment'] = (df.n_genes/df.n_study)/(df.pop_count/df.pop_n)
        df=df.sort_values(by=['Fold Enrichment'], ascending=False)
        df=df[0:10]
        return df


bp=cluster_go(clustered_genes=clustered_genes_names, cluster_number=3, type='biological_process')

def cluster_to_go(clustered_genes, cluster_number, filename):
    """
    Writes gene symbols to a text file for later use in other gene ontology repositories.
    Parameters:
        clustered_genes (dictionary): dictionary with every cluster and its corresponding gene symbols. Use the according dictionary
        (clustered_genes_names if gene_IDs==True and clustered_genes_ids if gene_IDs==False).
        cluster_number (int): label that identifies the cluster to be analyzed
        filename (str): name of the file where gene symbols will be saved. Must include file type (ex.:.txt)
    Returns:
        File with gene symbols. Also prints the same gene symbols.
    """
    if cluster_number in clustered_genes:
        with open(filename, 'w') as file:
            for value in clustered_genes[cluster_number]:
                file.write(value + '\n')
                print('\n'.join(map(str, clustered_genes[cluster_number])))
    else:
        print(f"Cluster number {cluster_number} not found in the dictionary.")

cluster_to_go(clustered_genes=clustered_genes_names, cluster_number=3, filename='testing.txt')

def gontology(df):
    """
    Plots a bar plot of the top 10 gene ontologies found where each bar represents fold enrichment
    and its color is the false discovery rate. 
    Parameters:
        df (DataFrame): the resulting DataFrame from the cluster_go function.
    Returns:
        A bar plot.
    """

    unique_levels = df['Fold Enrichment'].unique()

    fig = plt.figure(figsize=(8, 10))
    gs = mpl.gridspec.GridSpec(1, 2, width_ratios=[20, 1])
    ax_bar = plt.subplot(gs[0])

    cmap = mpl.cm.coolwarm
    palette = [mpl.colors.rgb2hex(cmap(val)) for val in np.linspace(0, 1, len(unique_levels))]

    ax_bar = sns.barplot(data=bp, x='Fold Enrichment', y='Term', hue='Fold Enrichment', dodge=False, palette=palette, ax=ax_bar, legend=False)
    y_ticks = np.arange(len(bp))
    y_labels = [textwrap.fill(e, 22) for e in bp['Term']]
    ax_bar.set_yticks(y_ticks)
    ax_bar.set_yticklabels(y_labels)
    ax_cb = plt.subplot(gs[1])
    norm = mpl.colors.Normalize(vmin=bp.p_corr.min(), vmax=bp.p_corr.max())
    cbar = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=norm, orientation='vertical')
    cbar.set_label('False Discovery Rate')
    plt.tight_layout()
    plt.show()

gontology(bp)

#%% GSEA
gene_id=True
#Ranking
def gsea_ranking(cluster_number, clustered_genes, classification_map, gene_id):
    """
    Ranks genes based on maximum expression values.
    Parameters:
        cluster_number (int): label of the cluster to be analyzed
        clustered_genes (dictionary): dictionary with every cluster and its corresponding gene symbols. Use the according dictionary
        (clustered_genes_names if gene_IDs==True and clustered_genes_ids if gene_IDs==False).
        classification_map (gema.Classification Object): Classification object.
        gene_id (boolean): set True if genes are identified through EnsemblID or False if through gene symbols.
    Returns:
        df_sorted (DataFrame): pandas DataFrame with 2 columns, 'gene_symbol 'and 'max_expr'.
        Organized by descending maximum expression values.
    """
    print('Number of genes:%d' % len(clustered_genes[cluster_number]))
    dataframe = []
    if gene_id==True:
        for i, label in enumerate(classification_map['labels']):
            if label in clustered_genes[cluster_number]:
                gene_symbol = Ensembl.gene_by_id(label).gene_name
                max_exp = np.max((classification_map['data'][i]))
                dataframe.append((gene_symbol, max_exp))
        df = pd.DataFrame(dataframe, columns=['gene_symbol', 'max_exp'])
        df_sorted = df.sort_values(by='max_exp', ascending=False)

    else:
        for i, label in enumerate(classification_map['labels']):
            if label in clustered_genes[cluster_number]:
                max_exp = np.max((classification_map['data'][i]))
                dataframe.append((label, max_exp))
        df = pd.DataFrame(dataframe, columns=['gene_symbol', 'max_exp'])
        df_sorted = df.sort_values(by='max_exp', ascending=False)

    return df_sorted

#Indicar aqui o cluster
ranking=gsea_ranking(cluster_number=3,clustered_genes=clustered_genes_ids,classification_map=classification.classification_map, gene_id=gene_id)

#Gene Set Enrichment

def write_geneset(jsonfile):
    """
    Transforms .json file with gene set information into dictionary, for building custom genes sets.
    Parameters:
        jsonfile (.json file): File with gene sets.
    Returns:
        geneset_dict (dictionary): each key is a term and its avlues are the associated genes.
    """
    geneset_dict = {}
    with open(jsonfile, 'r') as f:
        geneset = json.load(f)
    for key, value in geneset.items():
        name = key
        gene_symbols = value.get('geneSymbols', [])
        geneset_dict[name] = gene_symbols
    return geneset_dict

custom_geneset=write_geneset('c2.cp.v2023.2.Hs.json')

def enrichment(ranking, geneset):
    """
    Executes gene set enrichment analysis.
    Parameters:
        ranking (DataFrame): pandas DataFrame with ONLY 2 columns. First, the gene symbols and second, their maximum expression.
        geneset (dict or string): use your custom gene set from 'write_geneset' (dictionary) or choose one form the Enrichr library (str).
    Returns:
        out_df (DataFrame): pandas DataFrame with Term, False Discovery Rate(fdr), Enrichment Score (es), and Normalized Enrichment Score (nes).
        Plots Enrichment Score and Ranked Metric acording to Gene Rank.
        Prints the out_df dataframe.
    """
    gp.get_library_name()
    pre_res = gp.prerank(rnk = ranking, gene_sets = geneset, seed = 6, permutation_num = 100)
    out = []

    for term in list(pre_res.results):
        out.append([term,
                pre_res.results[term]['fdr'],
                pre_res.results[term]['es'],
                pre_res.results[term]['nes']])

    out_df = pd.DataFrame(out, columns = ['Term','fdr', 'es', 'nes']).sort_values(by=['fdr','es'], ascending=[True,True]).reset_index(drop = True)
    print(out_df)
    term_to_graph = out_df.iloc[0].Term

    gp.plot.gseaplot(**pre_res.results[term_to_graph], rank_metric=pre_res.ranking, term=term_to_graph)
    return out_df

enrich_df=enrichment(ranking=ranking, geneset='GTEx_Tissues_V8_2023')
# %%
