"""
Utilities for data preprocessing for unified pipeline:
"""
def walk_tissue_names(check_fn, *args, **kwargs):
    """
    Walk the whole hierarchy and return the first name that passes check_fn
    """
    for roots in data_categories.values():   # lists of root nodes
        for root in roots:
            result = walk(root, check_fn, *args, **kwargs)
            if result != "unknown":
                return result
    return "unknown"


def walk(node, check_fn, *args, **kwargs):
    """
    Recursively walk the hierarchy and return the first matching name
    """
    # Check current node
    if check_fn(node["name"], *args, **kwargs):
        return node["name"]

    # Recurse into children
    for child in node.get("children", []):
        result = walk(child, check_fn, *args, **kwargs)
        if result != "unknown":
            return result

    return "unknown"

data_categories = {
    "tissue_types": [
        {
            "name": "Cardiovascular System",
            "children": [
                {
                    "name": "Heart",
                    "children": [
                        {"name": "Heart"},
                        {"name": "Atrium"},
                        {"name": "Cardiac Muscle Fiber"},
                        {"name": "Pericardium"},
                        {"name": "Valve"},
                        {"name": "Ventricle"},
                    ],
                }
            ],
        },
        {
            "name": "Connective Tissue",
            "children": [
                {
                    "name": "Adipose Tissue",
                    "children": [
                        {"name": "Adipose"},
                        {"name": "Adipocyte"},
                    ],
                },
                {
                    "name": "Bone",
                    "children": [
                        {"name": "Myofibroblast"},
                        {"name": "Osteoblast"},
                        {"name": "Stromal Cell"},
                    ],
                },
                {
                    "name": "Bone Marrow",
                    "children": [
                        {"name": "Bone Marrow"},
                    ],
                },
                {
                    "name": "Cartilage",
                    "children": [
                        {"name": "Chondrocyte"},
                    ],
                },
            ],
        },
        {
            "name": "Digestive System",
            "children": [
                {"name": "Esophagus", "children": [{"name": "Esophagus"}]},
                {
                    "name": "Intestine",
                    "children": [
                        {"name": "Colon"},
                        {"name": "Colonic Mucosa"},
                        {"name": "Intestinal Epithelial Cell"},
                        {"name": "Small Intestine"},
                    ],
                },
                {
                    "name": "Liver",
                    "children": [
                        {"name": "Liver"},
                        {"name": "Hepatic Stellate Cell"},
                        {"name": "Hepatocyte"},
                        {"name": "Primary Liver Cell"},
                    ],
                },
                {
                    "name": "Pancreas",
                    "children": [
                        {"name": "Alpha Cell"},
                        {"name": "Beta Cell"},
                        {"name": "Pancreatic Islet"},
                    ],
                },
                {
                    "name": "Stomach",
                    "children": [
                        {"name": "Gastric Epithelial Cell"},
                    ],
                },
            ],
        },
        {
            "name": "Immune System",
            "children": [
                {
                    "name": "Granulocytic",
                    "children": [
                        {"name": "Granulocyte"},
                        {"name": "Neutrophil"},
                    ],
                },
                {
                    "name": "Lymphiod",
                    "children": [
                        {"name": "Blymphocyte"},
                        {"name": "Plasma Cell"},
                        {"name": "Plasmacytoid Dendritic Cell"},
                        {"name": "Tlymphocyte"},
                    ],
                },
                {
                    "name": "Myeloid",
                    "children": [
                        {"name": "Alveolar Macrophage"},
                        {"name": "Dendritic Cell"},
                        {"name": "Kupffer Cell"},
                        {"name": "Macrophage"},
                        {"name": "Macroglia"},
                    ],
                },
                {"name": "Spleen", "children": [{"name": "Spleen"}]},
                {
                    "name": "Thymus",
                    "children": [
                        {"name": "Thymus"},
                        {"name": "Thymocyte"},
                    ],
                },
            ],
        },
        {
            "name": "Integumentary System",
            "children": [
                {
                    "name": "Skin",
                    "children": [
                        {"name": "Skin"},
                        {"name": "Basal Cell"},
                        {"name": "Fibroblast"},
                        {"name": "Hair Follicle"},
                        {"name": "Keratinocyte"},
                        {"name": "Melanocyte"},
                    ],
                }
            ],
        },
        {
            "name": "Muscular System",
            "children": [
                {
                    "name": "Skeletal Muscle",
                    "children": [
                        {"name": "Skeletal Muscle"},
                        {"name": "Myoblast"},
                    ],
                },
                {
                    "name": "Smooth Muscle",
                    "children": [
                        {"name": "Airway Smooth Muscle"},
                        {"name": "Myofibroblast"},
                        {"name": "Respiratory Smooth Muscle"},
                        {"name": "Vascular Smooth Muscle"},
                    ],
                },
            ],
        },
        {
            "name": "Nervous System",
            "children": [
                {
                    "name": "CNS",
                    "children": [
                        {"name": "Astrocyte"},
                        {"name": "Cerebellum"},
                        {"name": "Cerebral Cortex"},
                        {"name": "Hypothalamus"},
                        {"name": "Medula"},
                        {"name": "Midbrain"},
                        {"name": "Neuron"},
                        {"name": "Oligodendrocyte"},
                        {"name": "Pons"},
                        {"name": "Spinal Cord"},
                        {"name": "Thalamus"},
                    ],
                },
                {"name": "Eye", "children": [{"name": "Retina"}]},
                {
                    "name": "PNS",
                    "children": [
                        {"name": "Motor Neuron"},
                        {"name": "Sensory Neuron"},
                    ],
                },
            ],
        },
        {
            "name": "Respiratory System",
            "children": [
                {
                    "name": "Lung",
                    "children": [
                        {"name": "Alveolar Cell Type II"},
                        {"name": "Lung"},
                        {"name": "Lung Epithelial Cell"},
                    ],
                },
                {"name": "Trachea", "children": [{"name": "Trachea"}]},
            ],
        },
        {
            "name": "Urogenital System",
            "children": [
                {
                    "name": "Breast",
                    "children": [
                        {"name": "Breast"},
                        {"name": "Mammary Gland"},
                    ],
                },
                {
                    "name": "Kidney",
                    "children": [
                        {"name": "Kidney"},
                        {"name": "Podocyte"},
                        {"name": "Renal Cortex"},
                    ],
                },
                {
                    "name": "Ovary",
                    "children": [
                        {"name": "Ovary"},
                        {"name": "Granulosa Cell"},
                        {"name": "Oocyte"},
                    ],
                },
                {"name": "Testis", "children": [{"name": "Testis"}]},
            ],
        },
    ],
    "cell_lines": [
        {"name": "Bone", "children": [{"name": "MG63"}, {"name": "K562"}]},
        {"name": "Brain", "children": [{"name": "DAOY"}, {"name": "IMR32"}, {"name": "SKNSH"}]},
        {
            "name": "Breast / Mammary",
            "children": [
                {"name": "MCF7"},
                {"name": "MCF10"},
                {"name": "MDAMB231"},
                {"name": "MDAMB468"},
                {"name": "SKBR3"},
                {"name": "T47D"},
            ],
        },
        {"name": "Cervix", "children": [{"name": "HELA"}]},
        {"name": "Colon", "children": [{"name": "HCT116"}, {"name": "HT29"}, {"name": "RKO"}]},
        {"name": "Connective", "children": [{"name": "HT1080"}, {"name": "NHDF"}]},
        {"name": "Endothelial", "children": [{"name": "HUVEC"}]},
        {
            "name": "Kidney",
            "children": [{"name": "293F"}, {"name": "FLPIN TREX 293"}, {"name": "HEK293"}],
        },
        {"name": "Liver", "children": [{"name": "HEPG2"}]},
        {
            "name": "Lung",
            "children": [
                {"name": "A549"},
                {"name": "H1299"},
                {"name": "IMR90"},
                {"name": "MRC5"},
                {"name": "NCIH460"},
                {"name": "NHBE"},
            ],
        },
        {
            "name": "Lymphoid",
            "children": [
                {"name": "JURKAT"},
                {"name": "MT4"},
                {"name": "NALM6"},
                {"name": "REH"},
                {"name": "RPMI8226"},
            ],
        },
        {"name": "Muscle", "children": [{"name": "LHCNM2"}]},
        {"name": "Myeloid", "children": [{"name": "THP1"}, {"name": "U937"}]},
        {"name": "Ovary", "children": [{"name": "ES2"}, {"name": "SKOV3"}]},
        {"name": "Pancreas", "children": [{"name": "PANC1"}]},
        {"name": "Placenta", "children": [{"name": "JEG3"}]},
        {
            "name": "Prostate",
            "children": [
                {"name": "C42"},
                {"name": "DU145"},
                {"name": "LNCAP"},
                {"name": "PC3"},
            ],
        },
        {"name": "Skin", "children": [{"name": "A431"}, {"name": "HACAT"}, {"name": "HNSCC"}]},
    ],
}