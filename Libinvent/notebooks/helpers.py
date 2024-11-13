import numpy as np 
from rdkit import Chem

dundee_filter = {
    "> 2 ester groups": "C(=O)O[C,H1].C(=O)O[C,H1].C(=O)O[C,H1]",
    "2-halo pyridine": "n1c([F,Cl,Br,I])cccc1",
    "acid halide": "C(=O)[Cl,Br,I,F]",
    "acyclic C=C-O": "C=[C!r]O",
    "acyl cyanide": "N#CC(=O)",
    "acyl hydrazine": "C(=O)N[NH2]",
    "aldehyde": "[CH1](=O)",
    "Aliphatic long chain": "[R0;D2][R0;D2][R0;D2][R0;D2]",
    "alkyl halide": "[CX4][Cl,Br,I]",
    "amidotetrazole": "c1nnnn1C=O",
    "aniline": "c1cc([NH2])ccc1",
    "azepane": "[CH2R2]1N[CH2R2][CH2R2][CH2R2][CH2R2][CH2R2]1",
    "Azido group": "N=[N+]=[N-]",
    "Azo group": "N#N",
    "azocane": "[CH2R2]1N[CH2R2][CH2R2][CH2R2][CH2R2][CH2R2][CH2R2]1",
    "benzidine": "[cR2]1[cR2][cR2]([Nv3X3,Nv4X4])[cR2][cR2][cR2]1[cR2]2[cR2][cR2][cR2]([Nv3X3,Nv4X4])[cR2][cR2]2",
    "beta-keto/anhydride": "[C,c](=O)[CX4,CR0X3,O][C,c](=O)",
    "biotin analogue": "C12C(NC(N1)=O)CSC2",
    "Carbo cation/anion": "[C+,c+,C-,c-]",
    "catechol": "c1c([OH])c([OH,NH2,NH])ccc1",
    "charged oxygen or sulfur atoms": "[O+,o+,S+,s+]",
    "chinone": "C1(=[O,N])C=CC(=[O,N])C=C1",
    "chinone": "C1(=[O,N])C(=[O,N])C=CC=C1",
    "conjugated nitrile group": "C=[C!r]C#N",
    "crown ether": "[OR2,NR2]@[CR2]@[CR2]@[OR2,NR2]@[CR2]@[CR2]@[OR2,NR2]",
    "cumarine": "c1ccc2c(c1)ccc(=O)o2",
    "cyanamide": "N[CH2]C#N",
    "cyanate / aminonitrile / thiocyanate": "[N,O,S]C#N",
    "cyanohydrins": "N#CC[OH]",
    "cycloheptane": "[CR2]1[CR2][CR2][CR2][CR2][CR2][CR2]1",
    "cycloheptane": "[CR2]1[CR2][CR2]cc[CR2][CR2]1",
    "cyclooctane": "[CR2]1[CR2][CR2][CR2][CR2][CR2][CR2][CR2]1",
    "cyclooctane": "[CR2]1[CR2][CR2]cc[CR2][CR2][CR2]1",
    "diaminobenzene": "[cR2]1[cR2]c([N+0X3R0,nX3R0])c([N+0X3R0,nX3R0])[cR2][cR2]1",
    "diaminobenzene": "[cR2]1[cR2]c([N+0X3R0,nX3R0])[cR2]c([N+0X3R0,nX3R0])[cR2]1",
    "diaminobenzene": "[cR2]1[cR2]c([N+0X3R0,nX3R0])[cR2][cR2]c1([N+0X3R0,nX3R0])",
    "diazo group": "[N!R]=[N!R]",
    "diketo group": "[C,c](=O)[C,c](=O)",
    "disulphide": "SS",
    "enamine": "[CX2R0][NX3R0]",
    "ester of HOBT": "C(=O)Onnn",
    "four member lactones": "C1(=O)OCC1",
    "halogenated ring": "c1cc([Cl,Br,I,F])cc([Cl,Br,I,F])c1[Cl,Br,I,F]",
    "halogenated ring": "c1ccc([Cl,Br,I,F])c([Cl,Br,I,F])c1[Cl,Br,I,F]",
    "heavy metal": "[Hg,Fe,As,Sb,Zn,Se,se,Te,B,Si]",
    "het-C-het not in ring": "[NX3R0,NX4R0,OR0,SX2R0][CX4][NX3R0,NX4R0,OR0,SX2R0]",
    "hydantoin": "C1NC(=O)NC(=O)1",
    "hydrazine": "N[NH2]",
    "hydroquinone": "[OH]c1ccc([OH,NH2,NH])cc1",
    "hydroxamic acid": "C(=O)N[OH]",
    "imine": "C=[N!R]",
    "imine": "N=[CR0][N,n,O,S]",
    "iodine": "I",
    "isocyanate": "N=C=O",
    "isolated alkene": "[$([CH2]),$([CH][CX4]),$(C([CX4])[CX4])]=[$([CH2]),$([CH][CX4]),$(C([CX4])[CX4])]",
    "ketene": "C=C=O",
    "methylidene-1,3-dithiole": "S1C=CSC1=S",
    "Michael acceptor": "C=!@CC=[O,S]",
    "Michael acceptor": "[$([CH]),$(CC)]#CC(=O)[C,c]",
    "Michael acceptor": "[$([CH]),$(CC)]#CS(=O)(=O)[C,c]",
    "Michael acceptor": "C=C(C=O)C=O",
    "Michael acceptor": "[$([CH]),$(CC)]#CC(=O)O[C,c]",
    "N oxide": "[NX2,nX3][OX1]",
    "N-acyl-2-amino-5-mercapto-1,3,4-thiadiazole": "s1c(S)nnc1NC=O",
    "N-C-halo": "NC[F,Cl,Br,I]",
    "N-halo": "[NX3,NX4][F,Cl,Br,I]",
    "N-hydroxyl pyridine": "n[OH]",
    "nitro group": "[N+](=O)[O-]",
    "N-nitroso": "[#7]-N=O",
    "oxime": "[C,c]=N[OH]",
    "oxime": "[C,c]=NOC=O",
    "Oxygen-nitrogen single bond": "[OR0,NR0][OR0,NR0]",
    "perfluorinated chain": "[CX4](F)(F)[CX4](F)F",
    "peroxide": "OO",
    "phenol ester": "c1ccccc1OC(=O)[#6]",
    "phenyl carbonate": "c1ccccc1OC(=O)O",
    "phosphor": "P",
    "phthalimide": "[cR,CR]~C(=O)NC(=O)~[cR,CR]",
    "Polycyclic aromatic hydrocarbon": "a1aa2a3a(a1)A=AA=A3=AA=A2",
    "Polycyclic aromatic hydrocarbon": "a21aa3a(aa1aaaa2)aaaa3",
    "Polycyclic aromatic hydrocarbon": "a31a(a2a(aa1)aaaa2)aaaa3",
    "polyene": "[CR0]=[CR0][CR0]=[CR0]",
    "quaternary nitrogen": "[s,S,c,C,n,N,o,O]~[nX3+,NX3+](~[s,S,c,C,n,N])~[s,S,c,C,n,N]",
    "quaternary nitrogen": "[s,S,c,C,n,N,o,O]~[n+,N+](~[s,S,c,C,n,N,o,O])(~[s,S,c,C,n,N,o,O])~[s,S,c,C,n,N,o,O]",
    "quaternary nitrogen": "[*]=[N+]=[*]",
    "saponine derivative": "O1CCCCC1OC2CCC3CCCCC3C2",
    "silicon halogen": "[Si][F,Cl,Br,I]",
    "stilbene": "c1ccccc1C=Cc2ccccc2",
    "sulfinic acid": "[SX3](=O)[O-,OH]",
    "Sulfonic acid": "[C,c]S(=O)(=O)O[C,c]",
    "Sulfonic acid": "S(=O)(=O)[O-,OH]",
    "sulfonyl cyanide": "S(=O)(=O)C#N",
    "sulfur oxygen single bond": "[SX2]O",
    "sulphate": "OS(=O)(=O)[O-]",
    "sulphur nitrogen single bond": "[SX2H0][N]",
    "Thiobenzothiazole": "c12ccccc1(SC(S)=N2)",
    "thiobenzothiazole": "c12ccccc1(SC(=S)N2)",
    "Thiocarbonyl group": "[C,c]=S",
    "thioester": "SC=O",
    "thiol": "[S-]",
    "thiol": "[SH]",
    "Three-membered heterocycle": "*1[O,S,N]*1",
    "triflate": "OS(=O)(=O)C(F)(F)F",
    "triphenyl methyl-silyl": "[SiR0,CR0](c1ccccc1)(c2ccccc2)(c3ccccc3)",
    "triple bond": "C#C"
    }


## Helper functions for scoring
## Evaluate score scaling on UCB values

def hard_sigmoid(x: np.ndarray, k: float) -> np.ndarray:
    return (k * x > 0).astype(np.float32)

def stable_sigmoid(x: np.ndarray, k: float, base_10: bool = True) -> np.ndarray:
    h = k * x
    if base_10:
        h = h * np.log(10)
    hp_idx = h >= 0
    y = np.zeros_like(x)
    y[hp_idx] = 1.0 / (1.0 + np.exp(-h[hp_idx]))
    y[~hp_idx] = np.exp(h[~hp_idx]) / (1.0 + np.exp(h[~hp_idx]))
    return y.astype(np.float32)


def double_sigmoid(
    x: np.ndarray,
    x_left: float,
    x_right: float,
    k: float,
    k_left: float,
    k_right: float,
) -> np.ndarray:
    """Compute double sigmoid based on stable sigmoid
    x: float or np.array
    x_left: float left sigmoid x value for which the output is 0.5 (low in previous implementation)
    x_right: float right sigmoid x value for which the output is 0.5 (high in previous implementation)
    k: float common scaling factor (coef_div in previous implementation)
    k_left: float scaling left factor (coef_si in previous implementation)
    k_right: float scaling right factor (coef_se in previous implementation)
    """
    x_center = (x_right - x_left) / 2 + x_left

    xl = x[x < x_center] - x_left
    xr = x[x >= x_center] - x_right

    if k == 0:
        sigmoid_left = hard_sigmoid(xl, k_left)
        sigmoid_right = 1 - hard_sigmoid(xr, k_right)
    else:
        k_left = k_left / k  # coef_si / coef_div
        k_right = k_right / k  # coef_se / coef_div
        sigmoid_left = stable_sigmoid(xl, k_left)
        sigmoid_right = 1 - stable_sigmoid(xr, k_right)

    d_sigmoid = np.zeros_like(x)
    d_sigmoid[x < x_center] = sigmoid_left
    d_sigmoid[x >= x_center] = sigmoid_right
    return d_sigmoid

def logistic_function(x, high, low, k):
    """
    Applies a logistic (sigmoid) function transformation.
    """
    x = x - (high + low) / 2
    k = 10.0 * k / (high - low)
    transformed = stable_sigmoid(x, k)
    return transformed

def reverse_logistic_function(x, high, low, k):
    """
    Applies a reverse logistic (reverse sigmoid) function transformation.
    """
    x = x - (high + low) / 2

    k = 10.0 * k / (high - low)
    transformed = stable_sigmoid(x, k)
    return 1.0 - transformed

def double_logistic_function(x, high, low, coef_div, coef_si, coef_se):
    """
    Applies a double logistic (double sigmoid) function transformation.
    """
    return double_sigmoid(x, low, high, coef_div, coef_si, coef_se)


def get_scores(df, scoring_dict):
    UCB_scores = []
    for i, row in df.iterrows():
        tempscore = 0.
        for j in range(len(scoring_dict["value_names"])):
            scoring_param = scoring_dict["scoring_params"][j]
            value_name = scoring_dict["value_names"][j]
            weight = scoring_dict["weights"][j]
            if scoring_param[0] == "sigmoid":
                tempscore += logistic_function(row[value_name], scoring_param[1], scoring_param[2], scoring_param[3])*weight
            elif scoring_param[0] == "reverse_sigmoid":
                tempscore += reverse_logistic_function(row[value_name], scoring_param[1], scoring_param[2], scoring_param[3])*weight
            elif scoring_param[0] == "double_sigmoid":
                tempscore += double_logistic_function(row[value_name], scoring_param[1], scoring_param[2], scoring_param[3], scoring_param[4], scoring_param[5])*weight
        UCB_scores.append(tempscore/sum(scoring_dict["weights"]))
    return UCB_scores

def check_smarts_filters(molecule_smiles, filter_dict=None, fail_thresh = 1):

    if filter_dict == None:
        filter_dict = dundee_filter

    pass_filter = True
    # Convert the SMILES to an RDKit molecule
    mol = Chem.MolFromSmiles(molecule_smiles)
    
    if not mol:
        return "Invalid SMILES input."

    matches = []
    
    # Check each SMARTS pattern
    for name, pattern in filter_dict.items():
        smarts = Chem.MolFromSmarts(pattern)
        if mol.HasSubstructMatch(smarts):
            matches.append(name)
    
    if len(matches) >= fail_thresh:
        pass_filter = False

    # Output the number of matches and their names
    return pass_filter, matches