import json
import os
import random
import ftplib
import logging
import os
from typing import List, Optional, Union
from Bio import SeqIO

# HTTPS-friendly implementation
import re
from pathlib import Path
from typing import List
from urllib.parse import urljoin
import requests

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

import argparse

def protein_embeddings_generator(
    genedf: pd.DataFrame = None,
    organism: str = "homo_sapiens",  # mus_musculus,
    cache: bool = True,
    https: bool = False,
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
    print(f"Generating protein embeddings for {organism} with {embedder} and dimension {embedding_size}")
    # given a gene file and organism
    # load the organism fasta if not already done7
    if not https:
        fasta_path_pep, fasta_path_ncrna = load_fasta_species(
            species=organism, output_path=fasta_path, load=["pep", "ncrna"], cache=cache
        )
    else:
       fasta_path_pep, fasta_path_ncrna = load_fasta_species_https(
            species=organism, output_path=fasta_path, load=["pep", "ncrna"], cache=cache
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

def subset_fasta(
    gene_tosubset: set = None,
    fasta_path: str = None,
    subfasta_path: str = "./data/fasta/subset.fa",
    drop_unknown_seq: bool = True,
    subset_protein_coding: bool = True,
) -> set:
    """
    subset_fasta: creates a new fasta file with only the sequence which names contain one of gene_names

    Args:
        gene_tosubset (set): A set of gene names to subset from the original FASTA file.
        fasta_path (str): The path to the original FASTA file.
        subfasta_path (str, optional): The path to save the subsetted FASTA file. Defaults to "./data/fasta/subset.fa".
        drop_unknown_seq (bool, optional): If True, drop sequences containing unknown amino acids (denoted by '*'). Defaults to True.
        subset_protein_coding (bool, optional): If True, subset only protein coding genes. Defaults to True.
    Returns:
        set: A set of gene names that were found and included in the subsetted FASTA file.

    Raises:
        ValueError: If a gene name does not start with "ENS".
    """
    dup = set()
    weird = 0
    nc = 0
    genes_found = set()
    gene_tosubset = set(gene_tosubset) if gene_tosubset else []
    names = []
    with (
        open(fasta_path, "r") as original_fasta,
        open(subfasta_path, "w") as subset_fasta,
    ):
        for record in SeqIO.parse(original_fasta, "fasta"):
            gene_name = (
                record.description.split(" gene:")[1].split(" ")[0].split(".")[0]
            )
            gene_biotype = record.description.split("gene_biotype:")[1].split(" ")[0]
            if "gene_symbol:" not in record.description:
                gene_symbol = gene_name
            else:
                gene_symbol = record.description.split("gene_symbol:")[1].split(" ")[0]
            if "description:" not in record.description:
                description = ""
            else:
                description = record.description.split("description:")[1]
            names.append([gene_name, gene_biotype, record.id, gene_symbol, description])
            if subset_protein_coding and gene_biotype != "protein_coding":
                nc += 1
                continue
            if len(gene_tosubset) == 0 or gene_name in gene_tosubset:
                if drop_unknown_seq:
                    if "*" in record.seq:
                        weird += 1

                        continue
                if gene_name in genes_found:
                    dup.add(gene_name)
                    continue
                record.description = ""
                record.id = gene_name
                SeqIO.write(record, subset_fasta, "fasta")
                genes_found.add(gene_name)
    print(len(dup), " genes had duplicates")
    print("dropped", weird, "weird sequences")
    print("dropped", nc, "non-coding sequences")
    return genes_found, pd.DataFrame(
        names, columns=["name", "biotype", "ensembl_id", "gene_symbol", "description"]
    )

# HTTPS-friendly implementation
def list_files_https(directory_url: str, match: str = "", timeout: int = 60) -> list[str]:
    """
    List files in an Ensembl HTTPS directory and keep only those ending with `match`.
    """
    r = requests.get(directory_url, timeout=timeout)
    r.raise_for_status()

    # Extract href targets from the directory listing page
    hrefs = re.findall(r'href="([^"]+)"', r.text)

    files = []
    for href in hrefs:
        name = href.split("/")[-1]
        if not name:
            continue
        if match == "" or name.endswith(match):
            files.append(name)

    # Deduplicate while preserving order
    seen = set()
    out = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def download_file_https(url: str, local_path: str | Path, timeout: int = 120) -> None:
    """
    Stream-download a file over HTTPS.
    """
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(local_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)


def load_fasta_species_https(
    species: str = "homo_sapiens",
    output_path: str = "/tmp/data/fasta/",
    load: List[str] = ["pep", "ncrna", "cdna"],
    cache: bool = True,
) -> list[str]:
    """
    Adapted from the FTP version, but uses HTTPS.

    Tries:
      1) Ensembl vertebrates
      2) Ensembl Genomes Plants
      3) Ensembl Genomes Metazoa
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    local_file_paths: list[str] = []

    # Probe candidate base directories using the pep subdir
    candidates = [
        {
            "type": "animals",
            "base": f"https://ftp.ensembl.org/pub/current_fasta/{species}/",
        },
        {
            "type": "plants",
            "base": f"https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/current/fasta/{species}/",
        },
        {
            "type": "metazoa",
            "base": f"https://ftp.ensemblgenomes.ebi.ac.uk/pub/metazoa/current/fasta/{species}/",
        },
    ]

    resolved = None
    for candidate in candidates:
        pep_url = urljoin(candidate["base"], "pep/")
        try:
            r = requests.get(pep_url, timeout=30)
            if r.ok:
                resolved = candidate
                break
        except requests.RequestException:
            pass

    if resolved is None:
        raise ValueError(
            f"Species {species} not found in Ensembl or Ensembl Genomes over HTTPS."
        )

    types = resolved["type"]
    base = resolved["base"]

    if "pep" in load:
        pep_dir = urljoin(base, "pep/")
        file = list_files_https(pep_dir, ".all.fa.gz")[0]
        local_path = output_dir / file
        local_file_paths.append(str(local_path))

        if not local_path.exists() or not cache:
            download_file_https(urljoin(pep_dir, file), local_path)

    if "ncrna" in load:
        ncrna_dir = urljoin(base, "ncrna/")
        matches = list_files_https(ncrna_dir, ".ncrna.fa.gz")
        if matches:
            file = matches[0]
            local_path = output_dir / file
            local_file_paths.append(str(local_path))

            if not local_path.exists() or not cache:
                download_file_https(urljoin(ncrna_dir, file), local_path)
        else:
            print(f"No ncRNA FASTA found for {species} in {types}")

    if "cdna" in load:
        cdna_dir = urljoin(base, "cdna/")
        file = list_files_https(cdna_dir, ".cdna.all.fa.gz")[0]
        local_path = output_dir / file
        local_file_paths.append(str(local_path))

        if not local_path.exists() or not cache:
            download_file_https(urljoin(cdna_dir, file), local_path)

    return local_file_paths


def main(args):
    emb, naming_df = protein_embeddings_generator(
        cache=args.cache,
        embedding_size=args.emb_size,
        https=args.https
    )
    emb.to_parquet(args.save_path)


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
    help="Size of the generated protein embeddings"
    )
    p.add_argument("--https",
    action="store_true",
    help="Whether to download FASTA files using HTTPs instead of FTP (default)"
    )
    p.add_argument("--save-path",
    type=str,
    default="homo_emb.parquet",
    help="Path to save the output parquet"
    )

    args = p.parse_args()
    main(args)
