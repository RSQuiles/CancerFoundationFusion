"""
Utilities for data preprocessing for unified pipeline:
"""
def walk_tissue_names(check_fn, *args, precise=False, **kwargs):
    """Return the first tissue category whose name or any subtissue matches check_fn."""
    for tissue, subtissues in tissue_categories.items():
        # Check the category name itself
        if check_fn(tissue, *args, **kwargs):
            return tissue
        # Check each subtissue
        for subtissue in subtissues:
            if check_fn(subtissue, *args, **kwargs):
                if precise:
                    return subtissue
                return tissue
    return "unknown"


tissue_categories = {
    # Matches CellxGene tissue_general: "blood"
    "blood": [
        "Granulocyte", "Neutrophil", "Eosinophil", "Basophil", "Mast Cell",
        "Blymphocyte", "Plasma Cell", "Plasmacytoid Dendritic Cell", "Tlymphocyte",
        "Alveolar Macrophage", "Dendritic Cell", "Kupffer Cell", "Macrophage", "Macroglia",
        "Myeloid", "Thymocyte", "Monocyte", "Natural Killer Cell", "NK Cell",
        "Hematopoietic Stem Cell", "HSC", "Progenitor Cell", "Erythrocyte",
        "Red Blood Cell", "Platelet", "Megakaryocyte", "Peripheral Blood",
        "PBMC", "Leukocyte", "White Blood Cell", "Lymphocyte",
        "Regulatory T Cell", "Treg", "CD4", "CD8", "CD34",
        # Cell lines
        "JURKAT", "MT4", "NALM6", "REH", "RPMI8226",
        "THP1", "U937", "K562", "HL60", "KG1", "MOLT4", "RAMOS",
    ],

    # Matches CellxGene tissue_general: "brain"
    "brain": [
        "Astrocyte", "Cerebellum", "Cerebral Cortex", "Hypothalamus",
        "Medula", "Midbrain", "Neuron", "Oligodendrocyte",
        "Pons", "Spinal Cord", "Thalamus",
        "Motor Neuron", "Sensory Neuron",
        "Microglia", "Glia", "Glial Cell", "Ependymal Cell",
        "Neural Progenitor", "Neuroblast", "Interneuron",
        "Hippocampus", "Amygdala", "Striatum", "Basal Ganglia",
        "Frontal Lobe", "Temporal Lobe", "Occipital Lobe", "Parietal Lobe",
        "Corpus Callosum", "Brainstem", "Brain Stem",
        "Radial Glia", "Neural Stem Cell",
        # Cell lines
        "DAOY", "IMR32", "SKNSH", "U87", "U251", "LN229", "T98G",
        "SH-SY5Y", "SHSY5Y", "A172", "GLIOBLASTOMA",
    ],

    # Matches CellxGene tissue_general: "colon"
    "colon": [
        "Colon", "Colonic Mucosa", "Intestinal Epithelial Cell",
        "Colorectal", "Sigmoid", "Rectum", "Rectal",
        "Colonocyte", "Goblet Cell", "Enteroendocrine",
        # Cell lines
        "HCT116", "HT29", "RKO", "SW480", "SW620", "LOVO",
        "CACO2", "DLD1", "HCT8",
    ],

    # No direct CellxGene match — kept for bulk
    "intestine": [
        "Small Intestine", "Intestinal Epithelial Cell",
        "Duodenum", "Jejunum", "Ileum", "Ileal",
        "Enterocyte", "Intestinal Organoid",
    ],

    # No direct CellxGene match — kept for bulk
    "esophagus": [
        "Esophagus", "Esophageal", "Oesophagus", "Oesophageal",
        "Squamous Epithelium",
    ],

    # Matches CellxGene tissue_general: "eye"
    "eye": [
        "Eye", "Retina", "Retinal", "Cornea", "Corneal",
        "Lens", "Iris", "Conjunctiva", "Optic Nerve",
        "Photoreceptor", "Rod", "Cone", "Ganglion Cell",
        "Retinal Pigment Epithelium", "RPE",
    ],

    # Matches CellxGene tissue_general: "heart"
    "heart": [
        "Atrium", "Cardiac Muscle Fiber", "Pericardium", "Valve", "Ventricle",
        "Cardiomyocyte", "Cardiac Fibroblast", "Cardiac",
        "Myocardium", "Myocardial", "Endocardium", "Epicardium",
        "Sinoatrial", "Atrioventricular",
    ],

    # Matches CellxGene tissue_general: "kidney"
    "kidney": [
        "Kidney", "Podocyte", "Renal Cortex",
        "Renal", "Nephron", "Glomerulus", "Tubule",
        "Proximal Tubule", "Distal Tubule", "Collecting Duct",
        "Mesangial Cell", "Parietal Epithelial Cell",
        # Cell lines
        "293F", "FLPIN TREX 293", "HEK293", "HEK 293",
        "ACHN", "786O", "CAKI", "A498",
    ],

    # Matches CellxGene tissue_general: "lung"
    "lung": [
        "Alveolar Cell Type II", "Lung", "Lung Epithelial Cell",
        "Trachea", "Tracheal",
        "Airway Smooth Muscle", "Respiratory Smooth Muscle",
        "Alveolar", "Bronchial", "Bronchiole", "Bronchus",
        "Pneumocyte", "Club Cell", "Clara Cell",
        "Pulmonary", "Pleura", "Pleural",
        "Type I Alveolar", "Type II Alveolar",
        "Airway Epithelial", "Respiratory Epithelial",
        # Cell lines
        "A549", "H1299", "IMR90", "MRC5", "NCIH460", "NHBE",
        "H358", "H1975", "H226", "H69", "H82", "PC9",
    ],

    # Matches CellxGene tissue_general: "prostate gland"
    "prostate gland": [
        "Prostate", "Prostatic",
        "Luminal Epithelial", "Basal Epithelial",
        # Cell lines
        "C42", "DU145", "LNCAP", "PC3", "22RV1", "VCAP", "LNCaP",
    ],

    # Matches CellxGene tissue_general: "urinary bladder"
    "urinary bladder": [
        "Bladder", "Urothelium", "Urothelial",
        "Transitional Epithelium",
        # Cell lines
        "RT4", "T24", "5637",
    ],

    # Matches CellxGene tissue_general: "ureter"
    "ureter": [
        "Ureter", "Ureteral",
    ],

    # Matches CellxGene tissue_general: "exocrine gland"
    "exocrine gland": [
        "Exocrine Gland", "Salivary Gland", "Sweat Gland",
        "Lacrimal Gland", "Parotid", "Submandibular",
        "Acinar Cell",
    ],

    # Matches CellxGene tissue_general: "embryo"
    "embryo": [
        "Embryo", "Embryonic", "Fetal", "Foetal",
        "Blastocyst", "Trophoblast", "Placental Trophoblast",
        "Induced Pluripotent", "iPSC", "ESC", "Embryonic Stem",
    ],

    "liver": [
        "Liver", "Hepatic", "Hepatic Stellate Cell", "Hepatocyte",
        "Primary Liver Cell", "Cholangiocyte", "Biliary",
        "Bile Duct", "Portal", "Sinusoidal",
        # Cell lines
        "HEPG2", "HUH7", "SNU449", "SNU398", "HEPA1",
    ],

    "pancreas": [
        "Pancreas", "Alpha Cell", "Beta Cell", "Pancreatic Islet",
        "Pancreatic", "Delta Cell", "Acinar", "Ductal",
        "Islet", "Endocrine Pancreas", "Exocrine Pancreas",
        # Cell lines
        "PANC1", "MIAPACA", "CFPAC", "BXPC3",
    ],

    "stomach": [
        "Stomach", "Gastric Epithelial Cell", "Gastric",
        "Gastric Mucosa", "Chief Cell", "Parietal Cell",
        "Foveolar", "Fundus", "Antrum",
        # Cell lines
        "AGS", "KATO3", "MKN45", "SNU1",
    ],

    "skin": [
        "Skin", "Basal Cell", "Fibroblast", "Hair Follicle",
        "Keratinocyte", "Melanocyte", "Follicle",
        "Dermis", "Dermal", "Epidermis", "Epidermal",
        "Sebaceous", "Sweat", "Merkel Cell",
        "Dermatological", "Cutaneous", "Subcutaneous",
        # Cell lines
        "A431", "HACAT", "HNSCC", "SK-MEL", "SKMEL", "WM", "MALME",
    ],

    "breast": [
        "Breast", "Mammary Gland", "Mammary",
        "Luminal", "Myoepithelial", "Ductal",
        "Lobular", "Nipple",
        # Cell lines
        "MCF7", "MCF10", "MDAMB231", "MDAMB468", "SKBR3", "T47D",
        "BT474", "BT549", "ZR75", "HCC1954", "HCC38",
    ],

    "ovary": [
        "Ovary", "Granulosa Cell", "Oocyte",
        "Ovarian", "Fallopian", "Follicular",
        "Cumulus", "Luteal",
        # Cell lines
        "ES2", "SKOV3", "OVCAR", "A2780", "CAOV3",
    ],

    "testis": [
        "Testis", "Testicular", "Sertoli Cell",
        "Leydig Cell", "Spermatocyte", "Spermatogonia",
        "Germinal", "Sperm",
    ],

    "bone marrow": [
        "Bone Marrow", "Hematopoietic", "Haematopoietic",
        "Stromal Progenitor", "Mesenchymal Stem Cell", "MSC",
        # Cell lines
        "MG63", "HS5", "HS27A",
    ],

    "bone": [
        "Bone", "Osteoblast", "Stromal Cell", "Chondrocyte", "Cartilage",
        "Osteoclast", "Osteocyte", "Periosteum",
        "Trabecular", "Cortical Bone",
        # Cell lines
        "SAOS2", "U2OS", "G292",
    ],

    "adipose": [
        "Adipose", "Adipocyte", "Fat",
        "Preadipocyte", "Lipocyte",
        "White Adipose", "Brown Adipose",
        "Visceral Fat", "Subcutaneous Fat",
        "Omentum",
    ],

    "muscle": [
        "Skeletal Muscle", "Myoblast", "Myofibroblast",
        "Smooth Muscle", "Vascular Smooth Muscle",
        "Myocyte", "Myotube", "Satellite Cell",
        "Muscle Fiber", "Myogenic",
        # Cell lines
        "LHCNM2", "C2C12", "RD",
    ],

    "thymus": [
        "Thymus", "Thymocyte", "Thymic",
        "Cortical Thymic Epithelial", "Medullary Thymic Epithelial",
    ],

    "spleen": [
        "Spleen", "Splenic", "Red Pulp", "White Pulp",
        "Marginal Zone",
    ],

    "cervix": [
        "Cervix", "Cervical", "Ectocervix", "Endocervix",
        "Squamocolumnar",
        # Cell lines
        "HELA", "SIHA", "CASKI", "C33A",
    ],

    "placenta": [
        "Placenta", "Placental", "Trophoblast",
        "Syncytiotrophoblast", "Cytotrophoblast",
        "Extravillous", "Decidua",
        # Cell lines
        "JEG3", "JAR", "BEWO",
    ],

    "connective tissue": [
        "Fibroblast", "Myofibroblast",
        "Mesenchymal", "Stromal",
        "Tendon", "Ligament", "Fascia",
        # Cell lines
        "HT1080", "NHDF", "WI38",
    ],

    # No direct CellxGene match — kept for bulk
    "lymph node": [
        "Lymph Node", "Lymphatic", "Germinal Center",
        "Lymphoid Tissue", "Tonsil", "Adenoid",
    ],

    # No direct CellxGene match — kept for bulk
    "thyroid": [
        "Thyroid", "Thyroidal", "Follicular Cell",
        "Thyrocyte", "Parathyroid",
        # Cell lines
        "TPC1", "FTC133", "BCPAP",
    ],

    # No direct CellxGene match — kept for bulk
    "adrenal gland": [
        "Adrenal", "Adrenal Gland", "Adrenal Cortex", "Adrenal Medulla",
        "Chromaffin", "Cortisol",
        # Cell lines
        "PC12", "H295R",
    ],

    # No direct CellxGene match — kept for bulk
    "uterus": [
        "Uterus", "Uterine", "Endometrium", "Endometrial",
        "Myometrium", "Decidual",
        # Cell lines
        "ISHIKAWA", "RL952", "AN3CA",
    ],

    # No direct CellxGene match — kept for bulk
    "vascular": [
        "Endothelial", "Vascular", "Aorta", "Aortic",
        "Artery", "Vein", "Capillary",
        "Pericyte", "Smooth Muscle Cell",
        # Cell lines
        "HUVEC", "HMVEC", "EA.HY926",
    ],

    # No direct CellxGene match — kept for bulk
    "nasal": [
        "Nasal", "Nasal Epithelium", "Sinonasal",
        "Olfactory", "Sinus",
    ],

    # No direct CellxGene match — kept for bulk
    "oral cavity": [
        "Oral", "Oral Cavity", "Buccal", "Gingival",
        "Tongue", "Salivary", "Parotid",
        # Cell lines
        "CAL27", "SCC4", "SCC9", "SCC15",
    ],
}