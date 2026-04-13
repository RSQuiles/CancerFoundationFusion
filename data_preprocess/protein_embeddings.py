import json
import os
import random
import subprocess
from collections import OrderedDict
from itertools import repeat
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import torch
from torch.nn import AdaptiveAvgPool1d
from tqdm import tqdm

import ftplib
import logging
import os
from typing import List, Optional, Union

import argparse

def protein_embeddings_generator(
    genedf: pd.DataFrame = None,
    organism: str = "homo_sapiens",  # mus_musculus,
    cache: bool = True,
    fasta_path: str = "/tmp/data/fasta/",
    embedding_size: int = 512,
    embedder: str = "esm3",
    cuda: bool = True,
):
    """
    adapted from scPRINT2

    protein_embeddings_generator embed a set of genes using fasta file and LLMs

    Args:
        genedf (pd.DataFrame): A DataFrame containing gene information.
        organism (str, optional): The organism to which the genes belong. Defaults to "homo_sapiens".
        cache (bool, optional): If True, the function will use cached data if available. Defaults to True.
        fasta_path (str, optional): The path to the directory where the fasta files are stored. Defaults to "/tmp/data/fasta/".
        embedding_size (int, optional): The size of the embeddings to be generated. Defaults to 512.
    Returns:
        pd.DataFrame: Returns a DataFrame containing the protein embeddings.
        pd.DataFrame: Returns the naming dataframe.
    """
    # given a gene file and organism
    # load the organism fasta if not already done
    fasta_path_pep, fasta_path_ncrna = load_fasta_species(
        species=organism, output_path=fasta_path, cache=cache
    )
    # subset the fasta
    fasta_name = fasta_path_pep.split("/")[-1]
    run_command(["gunzip", fasta_path_pep])
    protgenedf = (
        genedf[genedf["biotype"] == "protein_coding"] if genedf is not None else None
    )
    found, naming_df = subset_fasta(
        protgenedf.index.tolist() if protgenedf is not None else None,
        subfasta_path=fasta_path + "subset.fa",
        fasta_path=fasta_path + fasta_name[:-3],
        drop_unknown_seq=True,
    )
    if embedder == "esm3":
        from Bio import SeqIO
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig

        prot_embeddings = []
        names = []
        client = ESMC.from_pretrained("esmc_600m").to("cuda" if cuda else "cpu")
        conf = LogitsConfig(sequence=True, return_embeddings=True)
        with (open(fasta_path + "subset.fa", "r") as fasta,):
            for record in tqdm(SeqIO.parse(fasta, "fasta")):
                protein = ESMProtein(sequence=str(record.seq))
                protein_tensor = client.encode(protein)
                logits_output = client.logits(protein_tensor, conf)
                prot_embeddings.append(
                    logits_output.embeddings[0].mean(0).cpu().numpy().tolist()
                )
                names.append(record.id)
    else:
        raise ValueError(f"Embedder {embedder} not supported")
    # load the data and erase / zip the rest
    # utils.utils.run_command(["gzip", fasta_path + fasta_name[:-3]])
    # return the embedding and gene file
    # TODO: to redebug
    # do the same for RNA
    # rnagenedf = genedf[genedf["biotype"] != "protein_coding"]
    # fasta_file = next(
    #    file for file in os.listdir(fasta_path) if file.endswith(".ncrna.fa.gz")
    # )
    # utils.utils.run_command(["gunzip", fasta_path + fasta_file])
    # utils.subset_fasta(
    #    rnagenedf["ensembl_gene_id"].tolist(),
    #    subfasta_path=fasta_path + "subset.ncrna.fa",
    #    fasta_path=fasta_path + fasta_file[:-3],
    #    drop_unknown_seq=True,
    # )
    # rna_embedder = RNABERT()
    # rna_embeddings = rna_embedder(fasta_path + "subset.ncrna.fa")
    ## Check if the sizes of the cembeddings are not the same
    # utils.utils.run_command(["gzip", fasta_path + fasta_file[:-3]])
    #
    m = AdaptiveAvgPool1d(embedding_size)
    prot_embeddings = pd.DataFrame(
        data=m(torch.tensor(np.array(prot_embeddings))), index=names
    )
    # rna_embeddings = pd.DataFrame(
    #    data=m(torch.tensor(rna_embeddings.values)), index=rna_embeddings.index
    # )
    # Concatenate the embeddings
    return prot_embeddings, naming_df  # pd.concat([prot_embeddings, rna_embeddings])

def run_command(command: str, **kwargs) -> int:
    """
    run_command runs a command in the shell and prints the output.

    Args:
        command (str): The command to be executed in the shell.

    Returns:
        int: The return code of the command executed.
    """
    process = subprocess.Popen(command, stdout=subprocess.PIPE, **kwargs)
    while True:
        if process.poll() is not None:
            break
        output = process.stdout.readline()
        if output:
            print(output.strip())
    rc = process.poll()
    return rc


def list_files(ftp, match=""):
    files = ftp.nlst()
    return [file for file in files if file.endswith(match)]


def load_fasta_species(
    species: str = "homo_sapiens",
    output_path: str = "/tmp/data/fasta/",
    load: List[str] = ["pep", "ncrna", "cds"],
    cache: bool = True,
) -> None:
    """
    adapted from scPRINT2

    Downloads and caches FASTA files for a given species from the Ensembl FTP server.

    Args:
        species (str, optional): The species name for which to download FASTA files. Defaults to "homo_sapiens".
        output_path (str, optional): The local directory path where the FASTA files will be saved. Defaults to "/tmp/data/fasta/".
        cache (bool, optional): If True, use cached files if they exist. If False, re-download the files. Defaults to True.
    """
    ftp = ftplib.FTP("ftp.ensembl.org")
    ftp.login()
    local_file_path = []
    try:
        ftp.cwd("/pub/release-110/fasta/" + species + "/pep/")
        types = "animals"
    except ftplib.error_perm:
        try:
            ftp = ftplib.FTP("ftp.ensemblgenomes.ebi.ac.uk")
            ftp.login()
            ftp.cwd("/pub/plants/release-60/fasta/" + species + "/pep/")
            types = "plants"
        except ftplib.error_perm:
            try:
                ftp.cwd("/pub/metazoa/release-60/fasta/" + species + "/pep/")
                types = "metazoa"
            except ftplib.error_perm:
                raise ValueError(
                    f"Species {species} not found in Ensembl or Ensembl Genomes."
                )

    os.makedirs(output_path, exist_ok=True)
    if "pep" in load:
        file = list_files(ftp, ".all.fa.gz")[0]
        local_file_path.append(output_path + file)
        if not os.path.exists(local_file_path[-1]) or not cache:
            with open(local_file_path[-1], "wb") as local_file:
                ftp.retrbinary("RETR " + file, local_file.write)

    # ncRNA
    if "ncrna" in load:
        if types == "animals":
            ftp.cwd("/pub/release-110/fasta/" + species + "/ncrna/")
        elif types == "plants":
            ftp.cwd("/pub/plants/release-60/fasta/" + species + "/ncrna/")
        file = list_files(ftp, ".ncrna.fa.gz")[0]
        local_file_path.append(output_path + file)
        if not os.path.exists(local_file_path[-1]) or not cache:
            with open(local_file_path[-1], "wb") as local_file:
                ftp.retrbinary("RETR " + file, local_file.write)

    # CDNA:
    if "cdna" in load:
        if types == "animals":
            ftp.cwd("/pub/release-110/fasta/" + species + "/cdna/")
        elif types == "plants":
            ftp.cwd("/pub/plants/release-60/fasta/" + species + "/cdna/")
        file = list_files(ftp, ".cdna.all.fa.gz")[0]
        local_file_path.append(output_path + file)
        if not os.path.exists(local_file_path[-1]) or not cache:
            with open(local_file_path[-1], "wb") as local_file:
                ftp.retrbinary("RETR " + file, local_file.write)

    ftp.quit()
    return local_file_path

def main(args):
    emb, naming_df = protein_embeddings_generator(
        cache=args.cache,
        embedding_size=args.emb_size,
    )
    emb.to_parquet("homo_emb.parquet")


if __name__ == "__main__":
    # Parse CLI arguments
    p = argparse.ArgumentParser()
    p.add_argument("--cache",
    action="store_true",
    help="Whether to cache the generated fasta files"
    )
    p.add_argument("--emb-size",
    type=int,
    default=512,
    help="Size of the generated protein embeddings")

    args = p.parse_args()ç
    main(args)
