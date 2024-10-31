import pandas as pd
import numpy as np
import GEMA as gema
import pickle
from pyensembl import EnsemblRelease

#%% 1. Dataset Load, Function and Variable Setup

"""
Loads your data (file) into a pandas DataFrame (dataset).
Set ensembl_id to 'True' if your data uses EnsemblIDs and 'False' if using gene symbols.
Load Ensembl IDs for use in gene_search function (if ensembl_id==True).
Set nreplicates (int) to the number of experimental replicates.

TERMINAL: pyensembl install --release 77 --species human
"""

file='Datasets/logCPM_Frank_cent(conf).csv'
dataset= pd.read_csv(file, delimiter=",", header=0, index_col=0)
ensembl_id=False
Ensembl=EnsemblRelease(release=111, species='human')
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


def _gene_search(genenames, classification_map, ensembl_id):

    """
    Searches for the coordinates of genes in SOM grid.
    Parameters:
        genenames (numpy.ndarray): gene symbols to search for.
        classification_map (DataFrame): output of classification function after SOM mapping.
        ensembl_id (boolean): set True if genes are identified through EnsemblID or False if through gene symbols.
    Returns:
        genesinSOM (numpy.ndarray): array of tuples with each gene's corresponding coordinates (y,x), in the order that they are given.
    """
    genesinSOM=[]
    if ensembl_id==False:
        for i in range(len(genenames)):
            found=False
            for j in range(len(classification_map)):
                if genenames[i]==classification_map.iloc[j,0]:
                    position=classification_map.iloc[j,2],classification_map.iloc[j,3]
                    print("Gene '{}' found in index at position {}: {}".format(genenames[i], position, classification_map.iloc[j,0]))
                    print("Coordinates in SOM are'{}'".format((position[1],position[0])))
                    genesinSOM.append(position)
                    
                    found = True
                    break
            
            if not found:
                print("Gene '{}' not found.".format(genenames[i]))

            

        return genesinSOM
    
    if ensembl_id==True:
        genetranslation=[]
        for i in range(len(genenames)):
            #falta uma condição (try) dentro deste ciclo porque ele buga caso um gene não exista no ensembl tipo o tbxt que na verdade é o tbx5
            found = False
            gene = Ensembl.genes_by_name(genenames[i])
            gene_ensembl_id = gene[0].gene_id
            genetranslation.append(gene_ensembl_id)
            for j in range(len(classification_map)):
                if genetranslation[i]==classification_map.iloc[j,0]:
                    position=classification_map.iloc[j,2],classification_map.iloc[j,3]
                    print("Gene ID:'{}', name:'{}' found in index at position {}: {}".format(genetranslation[i], gene[0].gene_name, position, classification_map.iloc[j,0]))
                    print("Coordinates in SOM are'{}'".format((position[1],position[0])))
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

#%% 3. SOMSaver
def SOMSaver(mode, main_map_name, classification_name,main_map_object=None,classification_object=None):
    """
    Saves or loads the trained and mapped SOMs as pickle files.
    Parameters:
        mode (str): either 'save' or 'load'.
        main_map_object (GEMA Object): Only if mode is 'save'. main_map object given by the SOM which the user intends to save.
        classification_object (GEMA Object): Only if mode is 'save'. classification object of the SOM which the user intends to save.
        main_map_name (str): name of the main_map object to be saved/loaded.
        classification_name (str): name of the classification object to be saved/loaded.
    Output (Only if mode is 'load'):
        main_map (GEMA Object): Previously saved map and its attributes.
        classification (GEMA Object): Previously saved classifications and attibutes.
        
    Select 'True' to save or load and 'False' otherwise.
    Confirm your save by typing 'yes'.
    """

    if mode == 'save':
        
        confirmation = input("Are you sure you want to save? (yes/no): ")

        if confirmation.lower() == 'yes':
            with open(f'{main_map_name}', 'wb') as f:
                pickle.dump(main_map_object, f)

            with open (f'{classification_name}', 'wb') as c:
                pickle.dump(classification_object, c)
        else:
            print("Saving canceled.")

    if mode == 'load':
        with open(f'{main_map_name}', 'rb') as f:
            main_map = pickle.load(f)

        with open(f'{classification_name}', 'rb') as c:
            classification = pickle.load(c)

        return main_map, classification

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

def geneid_dict(classification, ensembl_id):
    """
    Builds dictionary genesymbol_grid, with each node as a key and its corresponding genes and their data as values.
    Parameters:
        classification (gema.Classification Object): Classification object.
    Returns:
        geneid_grid (dictionary): each key coordinate of the grid and corresponding to
        it are the genesymbols, in case the raw data uses EnsemblIDs.
    """   
    geneid_grid = {}

    for i in range(len(classification.classification_map)):
        x_coord = classification.classification_map['x'][i]
        y_coord = classification.classification_map['y'][i]
        label = classification.classification_map['labels'][i]

        if ensembl_id==False:
            try:
                gene_id = Ensembl.gene_ids_of_gene_name(label)
                if (x_coord, y_coord) in geneid_grid:
                    if len(gene_id)==1:
                        geneid_grid[(x_coord, y_coord)].extend(gene_id)
                    else:
                        geneid_grid[(x_coord, y_coord)].append(gene_id)
                else:
                    geneid_grid[(x_coord, y_coord)] = [(gene_id)]
            except ValueError as e:
                print("Gene not found when converting symbol to EnsemblID:", label)
        
        else:
            if (x_coord, y_coord) in geneid_grid:
                geneid_grid[(x_coord, y_coord)].append(label)
            else:
                geneid_grid[(x_coord, y_coord)] = [(label)]
    return geneid_grid


def genename_dict(classification, ensembl_id):
    """
    Builds dictionary genesymbol_grid, with each node as a key and its corresponding genes and their data as values.
    Parameters:
        classification (gema.Classification Object): Classification object.
    Returns:
        geneid_grid (dictionary): each key coordinate of the grid and corresponding to
        it are the genesymbols, in case the raw data uses EnsemblIDs.
    """   
    genesymbol_grid = {}

    for i in range(len(classification.classification_map)):
        x_coord = classification.classification_map['x'][i]
        y_coord = classification.classification_map['y'][i]
        label = classification.classification_map['labels'][i]

        if ensembl_id==True:
            try:
                genetranslation = Ensembl.gene_by_id(label)
                gene_name = genetranslation.gene_name
                if (x_coord, y_coord) in genesymbol_grid:
                    genesymbol_grid[(x_coord, y_coord)].append(gene_name)
                else:
                    genesymbol_grid[(x_coord, y_coord)] = [(gene_name)]
            except ValueError as e:
                print("Gene not found when converting EnsemblID to symbol:", label)
        
        else:
            if (x_coord, y_coord) in genesymbol_grid:
                genesymbol_grid[(x_coord, y_coord)].append(label)
            else:
                genesymbol_grid[(x_coord, y_coord)] = [(label)]
    return genesymbol_grid
