import pandas as pd
import seaborn as sb
import gseapy as gp
import numpy as np
import json
import textwrap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gspec
import matplotlib.colors as colors
import matplotlib.colorbar as cbar
import matplotlib.cm as cm
from goatools.base import download_go_basic_obo
from goatools.base import download_ncbi_associations
from goatools.obo_parser import GODag
from goatools.anno.genetogo_reader import Gene2GoReader
from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS
from genes_ncbi_human import GENEID2NT as GeneID2nt_human

#%% 13. Gene Ontology Setup
"""
Follow the instructions in the video (How to do gene ontology analysis in python - Sanbomics) to build the background gene set from NCBI before use.
Import GENEID2NT from the file created.
"""
def _goatools_setup(GeneID2nt_human):
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

#%% 13.5
def _go_it(test_genes, mapper, goeaobj, GO_items, inv_map, fdr_thresh=0.05):
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

#%% 14. Gene Ontology for clusters
"""
Performs gene ontology on the genes of the selected cluster. Select the cluster number according to KMeans.
"""

def cluster_go(clustered_genes, cluster_number=None, type=None):
    """
    Build a DataFrame of 10 gene ontologies with highest fold enrichment scores.
    Parameters:
        clustered_genes_names (dictionary): dictionary with every cluster and its corresponding gene symbols. Use the according dictionary
        (clustered_genes_names if ensembl_ids==True and clustered_genes_ids if ensembl_ids==False).
        cluster_number (int): Optional parameter. label that identifies the cluster to be analyzed
        type (str): Optional parameter. Specifies ontology type. Available options include:
                    - 'biological_process'
                    - 'cellular_component'
                    - 'molecular_function'
    Returns:
        df (DataFrame): pandas DataFrame of gene ontologies, their class, p-values, corrected p-values, number of genes in study, expected number in population and the gene symbols.
    """
    mapper, goeaobj, GO_items, inv_map= _goatools_setup(GeneID2nt_human)
    if type:
        if cluster_number:
            df = _go_it(clustered_genes[cluster_number], mapper, goeaobj, GO_items, inv_map,)
        else:
            df= _go_it(clustered_genes, mapper, goeaobj, GO_items, inv_map,)
        df = df[df['class'] == type]
        df['Fold Enrichment'] = (df.n_genes/df.n_study)/(df.pop_count/df.pop_n)
        df=df.sort_values(by=['Fold Enrichment'], ascending=False)
        df=df[0:10]
        return df
    else:
        if cluster_number:
            df = _go_it(clustered_genes[cluster_number], mapper, goeaobj, GO_items, inv_map,)
        else:
            df= _go_it(clustered_genes, mapper, goeaobj, GO_items, inv_map,)
        df['Fold Enrichment'] = (df.n_genes/df.n_study)/(df.pop_count/df.pop_n)
        df=df.sort_values(by=['Fold Enrichment'], ascending=False)
        df=df[0:10]
        return df

def cluster_to_go(clustered_genes, cluster_number, filename):
    """
    Writes gene symbols to a text file for later use in other gene ontology repositories.
    Parameters:
        clustered_genes (dictionary): dictionary with every cluster and its corresponding gene symbols. Use the according dictionary
        (clustered_genes_names if ensembl_ids==True and clustered_genes_ids if ensembl_ids==False).
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
    gs = gspec.GridSpec(1, 2, width_ratios=[20, 1])
    ax_bar = plt.subplot(gs[0])

    cmap = cm.coolwarm
    palette = [colors.rgb2hex(cmap(val)) for val in np.linspace(0, 1, len(unique_levels))]

    ax_bar = sb.barplot(data=df, x='Fold Enrichment', y='Term', hue='Fold Enrichment', dodge=False, palette=palette, ax=ax_bar, legend=False)
    y_ticks = np.arange(len(df))
    y_labels = [textwrap.fill(e, 22) for e in df['Term']]
    ax_bar.set_yticks(y_ticks)
    ax_bar.set_yticklabels(y_labels)
    ax_cb = plt.subplot(gs[1])
    norm = colors.Normalize(vmin=df.p_corr.min(), vmax=df.p_corr.max())
    colbar = cbar.ColorbarBase(ax_cb, cmap=cmap, norm=norm, orientation='vertical')
    colbar.set_label('False Discovery Rate')
    plt.tight_layout()
    plt.show()

#%% 15. GSEA
"A função de ranking pode ser simplificada."
#Ranking
def gsea_ranking(clustered_genes, classification_map, Ensembl, ensembl_id, cluster_number=None):
    """
    Ranks genes based on maximum expression values.
    Parameters:
        cluster_number (int): label of the cluster to be analyzed
        clustered_genes (dictionary): dictionary with every cluster and its corresponding gene symbols. Use the according dictionary
        (clustered_genes_names if ensembl_ids==True and clustered_genes_ids if ensembl_ids==False).
        classification_map (gema.Classification Object): Classification object.
        ensembl_id (boolean): set True if genes are identified through EnsemblID or False if through gene symbols.
    Returns:
        df_sorted (DataFrame): pandas DataFrame with 2 columns, 'gene_symbol 'and 'max_expr'.
        Organized by descending maximum expression values.
    """
    dataframe = []
    if cluster_number:
        print('Number of genes:%d' % len(clustered_genes[cluster_number]))
        if ensembl_id==True:
            for i, label in enumerate(classification_map['labels']):
                if label in clustered_genes[cluster_number]:
                    try:
                        gene_symbol = Ensembl.gene_by_id(label).gene_name
                        max_exp = np.max((classification_map['data'][i]))
                        dataframe.append((gene_symbol, max_exp))
                    except ValueError:
                        continue
            df = pd.DataFrame(dataframe, columns=['gene_symbol', 'max_exp'])
            df_sorted = df.sort_values(by='max_exp', ascending=False)

        else:
            for i, label in enumerate(classification_map['labels']):
                if label in clustered_genes[cluster_number]:
                    max_exp = np.max((classification_map['data'][i]))
                    dataframe.append((label, max_exp))
            df = pd.DataFrame(dataframe, columns=['gene_symbol', 'max_exp'])
            df_sorted = df.sort_values(by='max_exp', ascending=False)

    else:
        print('Number of genes:%d' % len(clustered_genes))
        if ensembl_id==True:
            for i, label in enumerate(classification_map['labels']):
                if label in clustered_genes:
                    try:
                        gene_symbol = Ensembl.gene_by_id(label).gene_name
                        max_exp = np.max((classification_map['data'][i]))
                        dataframe.append((gene_symbol, max_exp))
                    except ValueError:
                        continue
            df = pd.DataFrame(dataframe, columns=['gene_symbol', 'max_exp'])
            df_sorted = df.sort_values(by='max_exp', ascending=False)

        else:
            for i, label in enumerate(classification_map['labels']):
                if label in clustered_genes:
                    max_exp = np.max((classification_map['data'][i]))
                    dataframe.append((label, max_exp))
            df = pd.DataFrame(dataframe, columns=['gene_symbol', 'max_exp'])
            df_sorted = df.sort_values(by='max_exp', ascending=False)

    df_sorted.replace("", np.nan, inplace=True)
    df_sorted.dropna(subset=['gene_symbol'], inplace=True)
    df_sorted.dropna(inplace=True)
    return df_sorted

#Gene Set Enrichment

def geneset_into_dict(jsonfile):
    """
    Transforms .json file with gene set information into dictionary, for building custom genes sets.
    Parameters:
        jsonfile (.json file): File with gene sets.
    Returns:
        geneset_dict (dictionary): each key is a term and its values are the associated genes.
    """
    geneset_dict = {}
    with open(jsonfile, 'r') as f:
        geneset = json.load(f)
    for key, value in geneset.items():
        name = key
        gene_symbols = value.get('geneSymbols', [])
        geneset_dict[name] = gene_symbols
    return geneset_dict

def enrichment(ranking, geneset, min_size, max_size, term_to_plot=0):
    """
    Executes gene set enrichment analysis.
    Parameters:
        ranking (DataFrame): pandas DataFrame with ONLY 2 columns. First, the gene symbols and second, their maximum expression.
        geneset (dict or string): use your custom gene set from 'write_geneset' (dictionary) or choose one form the Enrichr library (str).
        min_size: minimum number of matches between sets.
        max_size: maximum number of matches between sets.
        term_to_plot: enrichment term from dataframe to plot.
    Returns:
        out_df (DataFrame): pandas DataFrame with Term, False Discovery Rate(fdr), Enrichment Score (es), and Normalized Enrichment Score (nes).
        Plots Enrichment Score and Ranked Metric acording to Gene Rank.
        Prints the out_df dataframe.
    """
    gp.get_library_name()
    pre_res = gp.prerank(rnk = ranking, gene_sets = geneset, seed = 6, permutation_num = 100, min_size=min_size, max_size=max_size)
    out = []

    for term in list(pre_res.results):
        out.append([term,
                pre_res.results[term]['fdr'],
                pre_res.results[term]['es'],
                pre_res.results[term]['nes']])

    out_df = pd.DataFrame(out, columns = ['Term','fdr', 'es', 'nes']).sort_values(by=['fdr','es'], ascending=[True,True]).reset_index(drop = True)
    print(out_df)
    term_to_graph = out_df.iloc[term_to_plot].Term

    gp.plot.gseaplot(**pre_res.results[term_to_graph], rank_metric=pre_res.ranking, term=term_to_graph)
    return out_df
