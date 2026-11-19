"""Hand-curated tables of the ARCHS4 -> UBERON -> CellxGene tissue map.

Data only: ``map_archs4_tissue.py`` holds the logic. Copied from
``notebooks/map_archs4_tissue.ipynb``; the only change is that UBERON terms are written
by their label (``"blood"``) instead of ``uid("blood")``. The script resolves every label
to one UBERON id and stops on a label that is missing or ambiguous, so a typo here
fails loudly.

Strings are matched after ``norm`` (lowercase, ``_`` -> space, single spaces) and
``fix_text`` (MOJIBAKE, CHAR_FIXES, TYPOS). Order of the matcher steps: see
``Matcher.match``.
"""

import re

# ---- characteristics keys -------------------------------------------------------------
# characteristics are joined by "Xx-xX" or by a comma followed by a new "key:"
SPLIT = re.compile(r"Xx-xX|,(?=\s*[^,:]{1,60}:)")

# explicit key lists, in priority order (first key a sample has wins)
TISSUE_KEYS = [
    "tissue", "tissue type", "tissuetype", "organ", "source organ", "organ site", "source tissue",
    "tissue source", "tissue origin", "tissue of origin", "tissue (isolated from)", "body site",
    "anatomical site", "anatomic region", "sampling site", "biopsy site", "biopsy location",
    "organ/tissue", "tissue region", "tissue subtype",
    "tissue/celltype", "tissue/cell type", "tissue/cell type source",
    "cell line origin",                     # tissue a cell line came from; these samples are still flagged has_cell_line
]
CELL_TYPE_KEYS = [
    "cell type", "celltype", "curated-cell-type", "cell type sorted", "sample cell type", "source cell type",
    "endocrine cell-type", "inferred cell type", "cell type or stage", "cell population", "cell subtype",
    "cell types", "cell line/type",
]
CELL_LINE_KEYS = [
    "cell line", "cell line name", "cellline", "cell line id", "inhouse cell line name",
    "source cell line", "originating cell line", "parental cell line", "cell line background",
]
FALLBACK_TISSUE_KEYS = ["organ system"]     # too coarse for UBERON; only used when nothing else is available

# reviewed and deliberately ignored; hidden from the "consider adding" printout
SKIP_KEYS = {
    "organism", "organisms", "organ donation death type", "organoidid", "tissue preparation",
    "cell type surface markers", "age at biopsy", "tissuestoragetime", "tissue collection",
    "tissue composition", "tissue disease state", "cancer cell line composition",
    "cell line group", "sample cell type cat",
}
SUGGEST = re.compile(r"tissue|organ|body.?site|anatom|biopsy|cell.?type|cell.?line", re.I)   # only to suggest extra keys (case-insensitive)

NA = {"", "nan", "none", "na", "n/a", "not applicable", "missing", "unknown", "not available"}

# ---- UBERON shorthands ----------------------------------------------------------------
BLOOD, SKIN, HEMATO = "blood", "skin of body", "hematopoietic system"

# ---- decisions and rules (the first matcher step that fires decides) -------------------
# explicit string -> UBERON decisions (checked against every cleaned variant of a string)
MANUAL = {
    # blood and blood fractions
    "blood": BLOOD, "peripheral blood": BLOOD, "venous blood": BLOOD, "buffy coat": BLOOD,
    "mobilised peripheral blood": BLOOD, "mobilized peripheral blood": BLOOD,
    "pbmc": BLOOD, "pbmcs": BLOOD, "peripheral blood mononuclear cells": BLOOD,
    "peripheral blood mononuclear cell": BLOOD, "pbmcs from mtb exposed patients": BLOOD,
    "plasma": "blood plasma", "plasma, extracted cell-free rna": "blood plasma",
    "plasma extracellular vesicles": "blood plasma",
    "serum": "blood serum", "serum ev": "blood serum",
    "cord blood": "umbilical cord blood", "umbilical cord blood": "umbilical cord blood",
    "cbmc": "umbilical cord blood", "cbmcs": "umbilical cord blood",
    "tumor educated platelets from blood": BLOOD,
    # hematopoietic, source not stated
    "blood or bone marrow": HEMATO, "bm": "bone marrow",
    # lymph nodes
    "ln": "lymph node", "mesenteric ln": "mesenteric lymph node",
    # skin
    "psoriasis lesion": SKIN, "psoriatic skin": SKIN, "lesional skin": SKIN,
    "foreskin": "prepuce of penis", "juvenile foreskin": "prepuce of penis",
    # nose and airways
    "nasal": "nasal cavity", "nasal lavage": "nasal cavity", "saline nasal wash": "nasal cavity",
    "nasal mucosa": "nasal cavity mucosa",
    "bronchial epithelial cells": "epithelium of bronchus", "bronchial epithelium": "epithelium of bronchus",
    "brushing of bronchial epithelium": "epithelium of bronchus",
    "cytologically normal bronchial airway epithelial cells": "epithelium of bronchus",
    "small airway epithelium": "epithelium of bronchiole",
    "small airway epithelium brushing": "epithelium of bronchiole",
    "airway epithelium": "respiratory tract epithelium", "airway epithelial cells": "respiratory tract epithelium",
    "human airway epithelial cells": "respiratory tract epithelium",
    "bronchoalveolar lavage": "lung", "bronchoalveolar lavage (bal)": "lung", "bal": "lung",
    "sftpc+ cells": "lung",                                       # alveolar type 2 cells
    "alveolar macrophages": "lung",                               # from bronchoalveolar lavage
    # nervous system
    "hippocampus": "hippocampal formation",
    "brain cortex": "cerebral cortex", "cortex frontal": "frontal cortex",
    "cortex motor lateral": "primary motor cortex", "cortex motor medial": "primary motor cortex",
    "cortex temporal": "temporal cortex", "cortex occipital": "occipital cortex",
    "ba9": "dorsolateral prefrontal cortex",                     # Brodmann area 9
    "caudate": "caudate nucleus", "ventral midbrain": "midbrain",
    "meningioma": "meninx", "retina punch": "retina",
    "cns": "central nervous system", "csf": "cerebrospinal fluid",
    # digestive system
    "gut": "digestive tract", "intestinal biopsy": "intestine",
    "colon & rectum": "large intestine", "crc": "large intestine", "colorectal": "large intestine",
    "oral mucosa": "mouth mucosa",
    "islet": "islet of langerhans", "islets": "islet of langerhans",
    "islet cell": "islet of langerhans", "islet cells": "islet of langerhans",
    # reproductive system, placenta, embryo
    "endocervical cells": "endocervix",
    "chorionic villous": "chorionic villus", "chorionic villi": "chorionic villus",
    "extravillous trophoblast": "placenta", "preimplantation embryo": "embryo",
    # heart, joints, fat
    "right atrial appendage": "right atrium auricular region",
    "right atrial appendage biopsies": "right atrium auricular region",
    "synovium": "synovium", "synovial biopsy": "synovium", "synovial tissue": "synovium",
    "bulk synovial tissue": "synovium",
    "visceral adipose tissue": "visceral abdominal adipose tissue",
    # diseases that imply a site (not tumors)
    "ulcerative colitis": "large intestine",
    "crohns disease": "intestine", "crohn's disease": "intestine",
    # ---- added in the second curation pass ----
    # hematopoietic: leukemias and mixed sources
    "acute lymphoblastic leukemia": HEMATO, "acute myeloid leukemia": HEMATO, "leukemia blast cells": HEMATO,
    "bm/pb": HEMATO, "pb/bm": HEMATO, "bonemarrow": "bone marrow",
    "cbnk": "umbilical cord blood", "cord blood pre-cdcs": "umbilical cord blood",
    # skeletal muscle
    "quadriceps muscle": "quadriceps femoris", "musculus vastus lateralis": "vastus lateralis",
    # nervous system: regions written without the organ
    "temporal": "temporal lobe", "frontal": "frontal lobe",
    "lateral temporal lobe": "temporal lobe",
    "cortex ba4": "primary motor cortex", "cortex motor unspecified": "primary motor cortex",
    "ba9 prefrontal cortex": "dorsolateral prefrontal cortex",
    "dorsal lateral prefrontal cortex": "dorsolateral prefrontal cortex",
    "anterior insula": "insula", "orbitofrontal": "orbitofrontal cortex",
    "anterior putamen": "putamen", "posterior putamen": "putamen",
    "anterior caudate": "caudate nucleus", "posterior caudate": "caudate nucleus",
    "cerebellar cortex: anterior and posterior quadrangulate lobules": "cerebellar cortex",
    "lateral amygdala": "lateral amygdaloid nucleus",
    # heart
    "left ventricle apex": "heart left ventricle",
    # airways and nose
    "bronchial brushing": "epithelium of bronchus", "bronchial brushings": "epithelium of bronchus",
    "broncho-alveolar epithelia": "respiratory tract epithelium",
    "human tracheobronchial epithelial cells": "tracheobronchial epithelium",
    "tracheobronchial epithelial cells": "tracheobronchial epithelium",
    "polarized bronchial epithelial cells": "epithelium of bronchus",
    "large airway epithelium": "epithelium of bronchus",
    "large airway basal cells": "epithelium of bronchus",
    "small airway epithelial cells": "epithelium of bronchiole",
    "nasal polyp": "nasal cavity mucosa", "nasal polyps": "nasal cavity mucosa",
    "ascites": "ascitic fluid",
    # reproductive system
    "adnexa": "adnexa of uterus", "ovaries": "ovary", "endometria": "endometrium",
    "cervix": "uterine cervix",
    "uterine fluid derived extracellular vesicles": "uterus",
    # digestive system
    "colorectum": "large intestine", "colonoscopic": "colon",
    "intestinal rectosigmoid": "rectosigmoid junction",
    "islets of langerhans": "islet of langerhans",
    # head and neck
    "larynx and hypopharynx": "hypopharynx",
    # metastases whose site is stated
    "gastric cancer, liver": "liver", "brain metastases,primary breast cancer": "brain",
    # blood, marrow, CSF
    "pb": BLOOD, "bmmc": "bone marrow", "bmmcs": "bone marrow", "bone marrow mononuclear cells": "bone marrow",
    "csf leukocytes": "cerebrospinal fluid", "csf cells": "cerebrospinal fluid",
    # brain regions and their abbreviations
    "dlpfc": "dorsolateral prefrontal cortex", "cortex ba9": "dorsolateral prefrontal cortex",
    "ofc": "orbitofrontal cortex", "prefrontal cortical": "prefrontal cortex",
    "cingulate": "cingulate cortex", "anterior cingulate": "cingulate cortex",
    "basalganglia": "basal ganglion", "purkinje cells": "cerebellum",
    "purkinje primary cell": "cerebellum",
    # heart, vessels, muscle, joints, bone, fat
    "left ventricular": "heart left ventricle", "apical left ventricle": "heart left ventricle",
    "carotid endarterectomy plaque sections": "carotid artery segment",
    "carotid plaque": "carotid artery segment", "carotid atherosclerotic plaque": "carotid artery segment",
    "rectus femoris muscle": "rectus femoris",
    "airway smooth muscle": "respiratory system smooth muscle",
    "airway smooth muscle primary cells": "respiratory system smooth muscle",
    "shoulder capsule": "articular capsule of glenohumeral joint",
    "tenosynovium": "tendon sheath", "tenosynovium sourced during cts decompression surgery": "tendon sheath",
    "cortical fibular bone": "fibula",
    "abdominal fat": "adipose tissue of abdominal region", "scbat": "brown adipose tissue",
    "omental": "omentum", "uterine myometrium": "myometrium",
    # airways, nose, lung
    "bronchial epithelial cell": "epithelium of bronchus", "tracheobronchial": "tracheobronchial epithelium",
    "airway basal cells": "respiratory tract epithelium",
    "alveolar macrophage": "lung", "bronchoalveolar lavage fluid": "lung", "balf": "lung",
    "nasal polyp inferior turbinate": "nasal cavity mucosa", "inferior turbinate": "inferior nasal concha",
    "sputum airway cells": "sputum",
    # eye, kidney, gut, mouth
    "rpe": "pigmented layer of retina", "retinal pigment epithelial cell": "pigmented layer of retina",
    "human iris pigment epithelial cell": "iris", "glomeruli": "renal glomerulus",
    "proximal colon": "right colon", "leftcolon": "left colon", "rightcolon": "right colon",
    "hemorrhoid tissue/anal canal": "anal canal", "masticatory mucosa": "mouth mucosa",
    # reproductive system, placenta, embryo
    "placental villous": "chorionic villus", "placental villi": "chorionic villus",
    "chorioamniotic membranes": "fetal membrane", "chorioamniotic membrane": "fetal membrane",
    "8-cell blastomere": "embryo", "blastomere": "embryo", "blastomeres": "embryo",
    "human oocytes and somatic cells": "ovary", "granulosa cells": "ovary",
    # skin
    "lcs": SKIN,                                                        # Langerhans cells (CD207+) from skin
    # blood fractions, fluids
    "exosome from plasma": "blood plasma", "menstrual blood": "menstrual fluid",
    "urinary extracellular vesicles": "urine", "urinary cells": "urine", "urine": "urine",
    "csf-derived leukocytes": "cerebrospinal fluid", "splenocytes": "spleen",
    # brain
    "cingulate gyrus 25": "cingulate gyrus", "medial temporal gyrus": "middle temporal gyrus",
    "middle temporal gyrus": "middle temporal gyrus", "dentate granule cells": "dentate gyrus of hippocampal formation",
    "ruptured intracranial aneurysm": "cerebral artery", "unruptured intracranial aneurysm": "cerebral artery",
    "intracranial aneurysm": "cerebral artery",
    # heart and vessels
    "left atria": "left cardiac atrium", "right atria": "right cardiac atrium",
    "tricuspid aortic valve leaflets": "aortic valve leaflet", "aortic valve leaflets": "aortic valve leaflet",
    "coronary plaque": "coronary artery",
    # muscle, joints, teeth, fat
    "bilateral paraspinal muscle": "skeletal muscle tissue", "paraspinal muscle": "skeletal muscle tissue",
    "acl tears remnants": "anterior cruciate ligament of knee joint", "acl": "anterior cruciate ligament of knee joint",
    "adult tooth premolar periodontal ligament": "periodontal ligament", "adult tooth pdl": "periodontal ligament",
    "pdl": "periodontal ligament", "subcutaneous gluteofemoral fat": "subcutaneous adipose tissue",
    # airways, nose
    "bronchial aecs": "epithelium of bronchus", "endobronchial": "bronchus",
    "bal macrophages": "lung", "nhne": "nasal cavity epithelium",
    # gut, kidney, mouth
    "sigmoid": "sigmoid colon", "proximal jejunum": "jejunum", "smallbowel": "small intestine",
    "small bowel": "small intestine", "gi": "digestive tract", "ibd": "intestine",
    "iec": "intestinal epithelium", "iecs": "intestinal epithelium",
    "intestinal epithelial cells": "intestinal epithelium", "pancreatic acinar cells": "pancreatic acinus",
    "tubuli": "renal tubule", "gingivo-buccal": "mouth mucosa",
    # eye, embryo
    "rpe/choroid": "optic choroid", "cd31 choroid": "optic choroid",
    "embryo head": "embryo", "embryo limb": "embryo", "single-cell embryo": "embryo",
    "blastocyst fraction": "blastocyst",
    "human islets": "islet of langerhans", "human islets from cadaveric organ donors": "islet of langerhans",
    "cadaveric pancreatic islet": "islet of langerhans", "donor-derived human pancreatic islets": "islet of langerhans",
    # other
    "lung lymph node": "lymph node", "human healthy control plasma": "blood plasma",
}

# tumor names that imply an organ (tagged +tumor); also matched as single words inside longer strings
TUMOR_ORGAN = {
    "glioblastoma": "brain", "gbm": "brain", "glioma": "brain", "astrocytoma": "brain",
    "medulloblastoma": "cerebellum",
    "pdac": "pancreas",
    "hcc": "liver", "hepatoblastoma": "liver", "hepatocarcinoma": "liver",
    "nsclc": "lung", "luad": "lung", "lusc": "lung",
    "ccrcc": "kidney", "rcc": "kidney",
    "nmibc": "urinary bladder", "mibc": "urinary bladder",
    "npc tumor": "nasopharynx",                                  # nasopharyngeal carcinoma ("npc" alone = neural progenitors)
    "primarypdac": "pancreas",
    "aml": HEMATO, "cll": BLOOD,
    # ---- added in the second curation pass ----
    "escc": "esophagus", "eac": "esophagus", "coad": "colon",
    "mpm": "pleura", "mesothelioma": "pleura", "pleural mesothelioma": "pleura",
    "retinoblastoma": "retina",
    "osteosarcoma": "bone tissue", "multiple myeloma": "bone marrow",
    "fibroid": "myometrium", "fibroids": "myometrium", "uterine fibroid": "myometrium",
    "leiomyoma": "myometrium", "uterine leiomyoma": "myometrium",
    "upper tract urothelial carcinoma": "upper urinary tract",
    "upper tract urothelial cell carcinoma": "upper urinary tract",
    "utuc": "upper urinary tract",
    "primary melanoma": SKIN, "melanoma in situ": SKIN, "cutaneous melanoma": SKIN,
    "spitz tumor": SKIN, "spitz nevus": SKIN, "nevus": SKIN, "nevi": SKIN,
    "tubular adenoma": "colon", "colorectal adenoma": "large intestine",
    "ctcl tumor": SKIN, "idc": "breast", "dcis": "breast",
    "hgsoc": "ovary", "hgsc": "ovary", "granulosa cell tumor": "ovary",
    "basal cell carcinoma": SKIN, "bcc": SKIN, "cscc": SKIN, "mpnst": "nerve",
    "basal cell carcinoma tumor": SKIN, "spitzoid melanoma": SKIN,
    "schwannoma": "nerve", "vestibular schwannoma": "nerve",
    "pituitary adenoma": "pituitary gland", "pituitary adenomas": "pituitary gland",
    "seminoma": "testis", "ocscc": "oral cavity", "oscc": "oral cavity",
    "sessile serrated adenoma": "colon", "sessile serrated adenoma/polyp": "colon",                                                # cutaneous T-cell lymphoma
    "hepatocellular carcinoma": "liver", "cholangiocarcinoma": "bile duct",
    "ependymoma": "central nervous system", "spinal ependymoma": "spinal cord",
    "pfa ependymoma": "hindbrain", "dipg": "pons", "diffuse intrinsic pontine glioma": "pons",
}

# strings that never identify a tissue on their own: resolve later from other fields
NOT_A_TISSUE = {
    # tumors without an organ
    "tumor", "tumour", "tumors", "cancer", "primary tumor", "resected tumors", "tumor tissue",
    "baseline tumor tissue", "tumor cells", "tumor containing tissue", "lesion", "crpc metastasis",
    "human melanoma cells", "melanoma cells", "malignant ascites of agc", "dlbcl", "gep-net",
    "hnsc single-cell", "head and neck squamous cell carinoma (hnscc)", "head and squamous cell carcinoma",
    "single acute myeloid leukemia intitiating cell", "tumor infiltrating lymphocytes (tils)", "candidate ctc",
    "head and neck squamous cell carcinoma", "head & neck", "lscc tissue", "leukemia cells", "primary tumor ffpe",
    "cuimc meningioma patient tumor sequencing project",
    # cell types without a site
    "macrophages", "macrophage", "primary macrophages", "microglia", "endocrine cells", "endothelial cells",
    "chondrocytes", "cardiomyocytes", "astrocytes", "keratinocytes", "neuron", "neurons", "neuronal progenitors",
    "primordial germ cells", "oocyte", "ejaculated spermatozoa", "stem cells", "stem cell", "epcam+ cells",
    "cpm-high cells", "cell", "cells", "interneurons", "immune cell", "erythroid cells", "hepatocytes",
    "human retinal pigment epithelial cells", "rna from sorted cells",
    # generic words and leftovers
    "normal", "control", "sample", "biopsy", "biopsies", "periphery", "cell line", "human post mortem tissue",
    "transcriptome of single cells", "cdna", "neuronal epithelium", "inlet cells", "dahoun et al 2008",
    "hcs", "wbs", "ensc", "ls", "hn137p", "phlo", "hb", "nac", "tt", "nl",
    "adjacent normal", "adjacent normal mucosa tissue", "surgical specimen", "pre-treatment endoscopic biopsies",
    "knee/hip",
    # ---- added in the second curation pass ----
    # tumors / lesions without an organ
    "melanoma metastases", "melanoma tumor", "melanoma", "metastatic tumor", "metastatic", "metastasis",
    "metastases", "non-tumor", "nontumor", "squamous cell carcinoma", "hnscc", "sarcoma", "sarcoma tumor",
    "soft tissue", "til", "tils", "cancer stem cells", "disseminated tumor cells", "dtcs", "ctcs",
    "adjacent", "adjacent normal tissue", "normal adjacent", "paracancerous normal", "paracancerous",
    "adjacent non-tumor", "tumor adjacent", "peritumoral", "follicular lymphoma", "lymphoma",
    "of pre-immunotherapy tumor", "single cell suspension of tumor", "patient tumor material",
    # cell types without a site
    "sperm", "hepatocyte", "keratinocyte", "astrocyte", "adipocyte", "adipocytes", "epithelial cells",
    "somatic cells", "natural killer cells", "t follicular helper cells", "cortical interneurons",
    "neural progenitors", "pyramidal neurons", "germ cells", "oocytes", "lymphoblast", "tet", "sorted mdc",
    "sorted nk", "memory cd4+", "modcs",
    # generic words, sample descriptions and unresolvable codes
    "formalin fixed paraffin embedded", "formalin-fixed paraffin-embedded", "native", "tissue mixture",
    "patient sample", "primary patient sample", "patient", "healthy adult woman", "healthy adult man",
    "total rna", "extracellular vesicles", "exosomes", "r", "l", "right", "left", "tp", "nms", "uipsm",
    "npc", "2d epcam+ scrna-seq",
    "polyp", "polyps", "adenoma", "uninvolved", "nonlesion", "non-lesional", "lesional", "dissociated dissected ventricle",
    "striatal gabaergic neurons", "ctfhtet", "cd138+ sorted cells", "healthy adult", "covid-19 positive samples",
    "ex vivo derived cells", "vascular endothelial cells", "melanocytes", "forebrain-type cortical neurons",
    "dm", "gm", "infusion products", "4su-rna", "micro-tissue of primary tumor", "al amyloidosis", "ltt", "proximal",
    "distal", "brisk til", "lyphoma specimen", "lymphoma specimen", "mantle cell lymphoma", "head & neck squamous cancer tumor",
    "circulating tumor cells", "circulating tumor cells sorted from peripheral blood", "villi",
    "osteoblast", "osteoblasts", "foam cell", "foam cells", "leiomyosarcoma", "adjacent non-tumour", "non-tumour",
    "frozen or formalin-fixed tumor", "inflammatory regression", "human fetus", "derma", "mucosal", "pericyte",
    "pericytes", "healthy donor", "microvascular endothelial cells", "endothelial cell", "endothelial", "ffpe",
    "neural cells", "head and neck cells", "teratoma", "human neuroblastoma cells", "uninfected", "wm", "m2",
    "from primary head and neck tumor", "placebo arm", "scleroderma patient", "patient tumor explant",
    "er+/pr+ human tumor explants", "adenocarcinoma", "primary patient derived tumor model", "recurrent tumors",
    "pure non-solid ct density tumor", "dopaminergic neurons", "human cortical excitatory neurons",
    "primary tumor tissue of hnscc", "cd138+ cells", "synovial sarcoma", "synovial sarcoma cells",
    # UBERON labels too generic to be useful; a variant equal to one of these is never looked up
    "tissue", "tissues", "organ", "epithelium", "epithelial", "cortex", "duct", "region", "normal tissue", "nodule",
}

CONTROL_RNA = re.compile(r"reference rna|\buhrr?\b|\bhbrr\b|\bmaqc\b|\bseqc\b|\berccs?\b|universal human reference|"
                         r"^mix of a and b\b|"
                         r"\bhuman reference rna\b")
XENOGRAFT = re.compile(r"\bpdx\b|\bpdx\d{2,}|\bpdx[2-9]\b|\bpdox\b|xenogra(ft|ph)|graft in (rat|mouse|mice)|engrafted|"
                       r"\bnsg\b|humani[sz]ed mice?|mouse stroma|"
                       r"(grown|implanted|injected) in (an? )?(immune[- ]?defici\w+|nude|scid) (mouse|mice)")
CELL_LINE_NAMES = re.compile(
    r"\b(u87|a375|jurkat|huvecs?|mcf-?7|hela|hek-?293t?|293t?|k562|hepg2|a549|lncap|thp-?1|sh-?sy5y|hct-?116|caco-?2|"
    r"mda-?mb-?\d+|ht-?29|panc-?1|u2os|imr-?90|bj|rpe-?1|h295r|wm\d+\w*|coriell|atcc|lcls?|lymphoblastoid|"
    # ---- added in the second curation pass ----
    r"mv-?4;?-?11|molm-?13|kasumi-?1|hl-?60|nb4|oci-?(aml|ly)-?\d+|raji|ramos|daudi|nalm-?6|jk-?1|"
    r"calu-?3|h1299|h460|pc-?9|hcc827|beas-?2b|16hbe\w*|sk-?br-?3|skbr-?3|t-?47-?d|bt-?20|bt-?474|bt-?549|zr-?75-?1|"
    r"sum-?\d+\w*|cal-?51|hs578t|hmle|mcf-?10a|sk-?hep-?1|huh-?7|hep-?3b|ags|mkn-?\d+|snu-?\d+|"
    r"sw-?480|sw-?620|lovo|dld-?1|hct-?15|rko|lim1215|pc-?3|du-?145|22rv1|vcap|c4-?2b|"
    r"sk-?ut-?1|syo-?1|g402|a673|tc-?71|rh(18|28|30|36|41)|khos|sjsa-?1?|saos-?2|mg-?63|143b|u-?251|t98g|ln-?229|sk-?n-?(sh|as|be|dz|mc|fi)\w*|be2c|imr-?32|"
    r"ovcar-?\d+|skov-?3|peo-?[14]|caov-?3|heya8|ishikawa|hec-?1-?[ab]|siha|caski|c-?33a|"
    r"786-?o|caki-?\d|ach-?n|hk-?2|hacat|hues\d+|bobsc|gm\d{4,5}|ebv-?b|hn\d{3}\w*|"
    r"nci-?h\d+|hcc-?\d+|mda-?mba?-?\d+|um-?scc-?\d+\w*|scc-?\d+|cal-?27|fadu|detroit ?562|mono-?mac-? ?6|kgn|"
    r"rpe-? ?1|arpe-?19|htert\w*|meg-?01|sk-?mel-?\d+|hek-?293\w*|hela\w*|meso\d+|lx-?2|t-?84|sh-?sy-?5y|"
    r"ntera-?2|sf\d{3,4}|dbtrg\w*|lhcn-?m2|12z|\w+/e6e7|ha1800|telo\w+|hs-?sy-?ii)\b|"
    r"umbilical vein endothelial|\bcell lines?\b|ebv[- ]transformed|gencord|immortali[sz]ed")
IN_VITRO = re.compile(
    r"\bips\b|ipsc|induced pluripotent|embryonic stem|\bescs?\b|\bhescs?\b|\bh?i?pscs?\b|\bh[179]\b|\bhes\b|"
    r"(pluripotent|neural|mesenchymal|embryonic) stem cells?|\bh?mscs?\b|"
    r"(dental pulp|limbal|amniotic fluid|adipose[- ]derived|wharton'?s jelly) (mesenchymal )?stem cells?|"
    r"embryoid|organoid|enteroid|colonoid|assembloid|blastoid|spheroid|organotypic|stem[- ]cell[- ]derived|"
    r"progenitor cell[- ]derived|monocytes?[- ]derived|psc[- ]derived|\bmo-?dcs?\b|\bmdms?\b|\bmdmi\b|"
    r"periosteum-derived|"
    r"derived from h?ipsc|progenitor cells|differentiat|\binduced (neurons?|astrocytes?|\w+ cells?)|-induced|\bicell\b|"
    r"ngn2|arc-type neurons|\bineurons?\b|neuron-like|reprogramm|hepatopac|hif-ts?\b|\bhwjscs?\b|stromal cells?\b|"
    r"tissue culture|\b[23]d (culture|cells?)\b|\bcells? [23]d\b|in vitro|\bivc\b|air[- ]liquid interface|\bcar[- ]?t\b|"
    r"in culture|co-?culture|seeded|decellulari|trophoblast stem|\bkips\b|chemical reset|\bes[- ]derived|\bnpcs\b|"
    r"\bimgls?\b|\bcafs?\b|\bhrecs?\b|\bhlecs?\b|\bhsaecs?\b|"
    r"cell culture|cultured|passage|fibroblasts?\b|\bhffs?\b|myotubes?|myoblasts?|engineered|programmed|"
    r"aortic endothelial|valve endothelial|mucilair|epidermal keratinocytes?|on-a-chip|\w+ chip$|"
    r"\b(nheks?|hmecs?|hcaecs?|hpaecs?|hpmecs?|hdmecs?|hbmecs?|hasmcs?|haoecs?|hpmcs?)\b|"   # primary cultures, not cell lines
    r"^primary .*\b(cells?|hepatocytes|keratinocytes|myotubes)\b|^dcx[+-] d\d+$|"
    r"^(human )?primary( human)? (\w+ )?(hepatocytes?|keratinocytes?|astrocytes?|chondrocytes?|fls|melanocytes?|"
    r"adipocytes?)$")
# sample codes and protocol descriptions that carry no tissue ("bc05 single-cell rna-seq", "mm34em", "patient 2 ...");
# only used once every other step found nothing
SAMPLE_CODE = re.compile(r"^(patient|subject|donor|sample|case|bc|mm|mgh|p)[ -]?\d+\w*$|"
                         r"^([\w-]+ ){0,2}single[- ]cell m?rna-seq( [\w-]+)?$|"
                         r"^p\d+ t\d.*clone|^formalin[- ]fixed paraffin[- ]embedded biopsies or archival|"
                         r"^(tumou?r|normal)-? ?\d+$|\bctcs?\b|circulating tumou?r cells")
HEMATO_RE = re.compile(r"cd34\+|\bcd34(?![-\w])|hematopoietic (stem |progenitor )?(and progenitor )?cells?|\bhspcs?\b|\bhscs?\b|"
                       r"megakaryocyte|erythroid progenitor|myeloid progenitor|hematopoietic (stem|progenitor)s?\b|"
                       r"erythroblasts?")
NASAL_EPI = re.compile(r"nasal[- ](brush|brushing|epitheli|airway|scrap|swab|curett|rna)|brushing of nasal")
NASOPHARYNX = re.compile(r"naso-?pharyn|oro-?naso-?pharyn")
# blood rule: a blood word or blood-cell word, and no other site
BLOOD_WORD = re.compile(r"\b(whole[- ]?)?blood\b|\b(leuk)?apheresis\b")
BLOOD_CELL_WORD = re.compile(
    r"\b(t|b|nk|nkt)[- .]?cells?\b|\btcells?\b|\bbcells?\b|\btregs?\b|\bpbmcs?\b|\bpbls?\b|natural killer|"
    r"\bcd(3|4|8|14|16|19|20|56)[+-]? (t |memory |naive |effector )?(cells?|memory|naive|effector)\b|"   # "cd8 naive"
    r"monocytes?|neutrophils?|thrombocytes?|platelets?|erythrocytes?|granulocytes?|lymphocytes?|leukocytes?|"
    r"eosinophils?|basophils?|plasma ?cells?|plasmablasts?|dendritic cells?|"
    r"\bnaive (cd\d+|b|t)\b|regulatory t|t regulatory|\bth-?(1|2|17)\b|\bmait\b|\bpdcs?\b|"
    r"^(cd\d+[+-]?|ccr\d+[+-]?)+$")                                  # bare marker strings: cd4, cd3+cd4+ccr7+
NON_BLOOD_SITE = re.compile(r"tumou?r|infiltrat|tissue|biops|lavage|synovi|skin|lung|liver|gut|intestin|colon|"
                            r"brain|csf|marrow|tonsil|lymph node|spleen|thymus|decidua|placenta|derived|culture|"
                            r"cord|menstrual")
TUMOR_WORDS = re.compile(r"\b(tumou?rs?|cancers?|carcinomas?|carinomas?|adenocarcinomas?|neoplasms?|"
                         r"metastas[ie]s|metastatic|malignant|primary|resected|baseline)\b")
# words that only say "the normal part" of a tumor sample; dropped together with the tumor words (no +tumor tag)
NORMAL_WORDS = re.compile(r"\b(adjacent|normal|paracancerous|para-?carcinoma|peri-?tumou?ral|non-?tumou?r(al|ous)?|"
                          r"non-?cancerous|non-?malignant|uninvolved|benign|matched|healthy)\b")

# last resort: an organ word anywhere in the string (multi-word names first); ambiguous if two organs appear
ORGAN_WORDS = {w: lbl for w, lbl in [
    ("bone marrow", "bone marrow"), ("spinal cord", "spinal cord"), ("lymph node", "lymph node"),
    ("adipose", "adipose tissue"), ("small intestine", "small intestine"), ("liver", "liver"), ("lung", "lung"),
    ("kidney", "kidney"), ("heart", "heart"), ("brain", "brain"), ("cerebellum", "cerebellum"), ("colon", "colon"),
    ("stomach", "stomach"), ("pancreas", "pancreas"), ("spleen", "spleen"), ("thymus", "thymus"), ("skin", "skin of body"),
    ("breast", "breast"), ("prostate", "prostate gland"), ("ovary", "ovary"), ("testis", "testis"), ("uterus", "uterus"),
    ("placenta", "placenta"), ("retina", "retina"), ("esophagus", "esophagus"), ("bladder", "urinary bladder"),
    ("thyroid", "thyroid gland"), ("adrenal", "adrenal gland"), ("tonsil", "tonsil"), ("rectum", "rectum"),
    ("ileum", "ileum"), ("duodenum", "duodenum"), ("endometrium", "endometrium"), ("trachea", "trachea"),
    # ---- added in the second curation pass (only organs that rarely co-occur with another organ word;
    # "artery", "bone", "eye", "cervix", "pleura", "omentum" or "umbilical cord" made "pulmonary artery",
    # "prostate bone metastasis" or "spinal cord cervical" ambiguous) ----
    ("peritoneum", "peritoneum"), ("vagina", "vagina"), ("vulva", "vulva"),
    ("fallopian tube", "fallopian tube"), ("gallbladder", "gallbladder"), ("bile duct", "bile duct"),
    ("cecum", "caecum"), ("appendix", "vermiform appendix"), ("tongue", "tongue"),
    ("larynx", "larynx"), ("pharynx", "pharynx"), ("oral cavity", "oral cavity"), ("salivary gland", "salivary gland"),
    ("parotid", "parotid gland"), ("cornea", "cornea"), ("conjunctiva", "conjunctiva"),
    ("hypothalamus", "hypothalamus"), ("skeletal muscle", "skeletal muscle tissue"), ("tendon", "tendon"),
    ("cartilage", "cartilage tissue"), ("urethra", "urethra"), ("penis", "penis"), ("epididymis", "epididymis"),
    ("bronchoalveolar lavage", "lung"), ("hair follicle", "hair follicle"), ("synovium", "synovium"),
    ("myometrium", "myometrium")]}
ORGAN_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, ORGAN_WORDS), key=len, reverse=True)) + r")s?\b")   # "lungs"

# text repair (the text is already lowercased, so broken UTF-8 is fixed by direct replacement)
MOJIBAKE = {"\u00e3\u00af": "i", "\u00e3\u00a9": "e", "\u00e3\u00a8": "e", "\u00e3\u00b6": "o", "\u00e3\u00bc": "u", "\u00e3\u00a4": "a", "\u00e3\u00b1": "n", "\u00e2\u20ac\u201c": "-", "\u00e2\u20ac\u2122": "'"}
TYPOS = {"tisue": "tissue", "tisssue": "tissue", "periperal": "peripheral", "perpheral": "peripheral",
         "umblical": "umbilical", "oronasopharingeal": "oronasopharyngeal",
         "lymphoycyte": "lymphocyte", "lymphoycytes": "lymphocytes",
         "pacreatic": "pancreatic", "substanstia": "substantia", "medullobalstoma": "medulloblastoma",
         "plalelets": "platelets", "lympohoma": "lymphoma", "ketratinocytes": "keratinocytes",
         "periphheral": "peripheral", "esopheageal": "esophageal", "artial": "atrial", "mesencymal": "mesenchymal",
         "keratinoocyte": "keratinocyte", "keratinoctyes": "keratinocytes", "oseochondroma": "osteochondroma",
         "biospy": "biopsy", "ligment": "ligament", "plancenta": "placenta", "umbillical": "umbilical",
         "hematopoeitic": "hematopoietic", "spleenocytes": "splenocytes", "medil": "medial", "carinoma": "carcinoma",
         "lymphnode": "lymph node", "lymphnodes": "lymph nodes"}
ADJ = {"rectal": "rectum", "ileal": "ileum", "colonic": "colon", "gastric": "stomach", "hepatic": "liver",
       "renal": "kidney", "pulmonary": "lung", "cardiac": "heart", "duodenal": "duodenum", "jejunal": "jejunum",
       "esophageal": "esophagus", "oesophageal": "esophagus", "pancreatic": "pancreas", "splenic": "spleen",
       "thymic": "thymus", "dermal": "dermis", "cutaneous": "skin", "gingival": "gingiva",
       "endometrial": "endometrium", "placental": "placenta", "tonsillar": "tonsil", "bronchial": "bronchus",
       "tracheal": "trachea", "ovarian": "ovary", "testicular": "testis", "prostatic": "prostate",
       "endocervical": "endocervix", "cervical": "cervix", "mammary": "breast",
       "colorectal": "large intestine", "hepatocellular": "liver", "synovial": "synovium",
       # ---- added in the second curation pass ----
       "peritoneal": "peritoneum", "pleural": "pleura", "ectocervical": "ectocervix", "vaginal": "vagina",
       "vulvar": "vulva", "uterine": "uterus", "cecal": "cecum", "biliary": "bile duct",
       "hippocampal": "hippocampus", "hypothalamic": "hypothalamus", "cerebellar": "cerebellum",
       "osseous": "bone", "oral": "oral cavity", "lingual": "tongue", "laryngeal": "larynx",
       "pharyngeal": "pharynx", "salivary": "salivary gland", "ocular": "eye", "corneal": "cornea",
       "conjunctival": "conjunctiva", "ureteral": "ureter", "urethral": "urethra", "penile": "penis",
       "decidual": "decidua", "intestinal": "intestine", "adrenocortical": "adrenal",
       "myometrial": "myometrium"}

# single characters left by broken UTF-8, and a mangled non-breaking space
# ("umbilical\u00e2 cord\u00e2 blood")
CHAR_FIXES = {"\u00ef": "i", "\u00e9": "e"}
MANGLED_NBSP = re.compile(r"\u00e2(?=\s|,|$)")

# ---- variant generation ---------------------------------------------------------------
PREFIX = re.compile(r"^(human|donor \d+|donor|normal|healthy|adult|fetal|foetal|embryonic|primary|fresh|frozen|"
                    r"cryopreserved|total|whole|ffpe|post-?mortem|paired|"
                    r"neonatal|newborn|p(a)?ediatric|infant|juvenile|pregnant|bulk|pooled|homogeni[sz]ed|explanted|"
                    r"laser[- ]captured|lcm[- ]dissected|lcm|isolated|sorted|control)\s+")
SUFFIX = re.compile(r"\s+(tissue|tissues|sample|samples|biopsy|biopsies|specimen|specimens|punch)$")
CELLS  = re.compile(r"\s+(cells?|cell population|"
                    r"((pyramidal|dopamine|dopaminergic|gabaergic|medium spiny) )?neurons?)$")
