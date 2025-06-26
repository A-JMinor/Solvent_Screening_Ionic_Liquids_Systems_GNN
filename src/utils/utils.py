

import pandas as pd
import os
import json
import numpy as np
from rdkit import Chem
from tqdm import tqdm
from models.GHGNN.ghgnn_old import GH_GNN_old
from models.GHGNN.ghgnn import GH_GNN
from models.GHGNN.ghgnn import atom_features, bond_features

def check_infeasible_mol_for_ghgnn(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)

        atoms  = mol.GetAtoms()
        bonds  = mol.GetBonds()

        [atom_features(atom) for atom in atoms]
        [bond_features(bond) for bond in bonds]
        return False
    except:
        return True

def load_json_files(folder_path):
    """Load all JSON files from a folder into a dictionary."""
    data_dict = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                info = json.load(file)
                data_dict[info['Name']] = info
    return data_dict


def find_name_from_smiles(smiles_to_find, pure_comp_dict):
    
    for compound_name, compound_info_set in pure_comp_dict.items():
        try:
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(compound_info_set['SMILES']))
        except:
            smiles = None
        if smiles == Chem.MolToSmiles(Chem.MolFromSmiles(smiles_to_find)):
            return compound_name
    return None

def KDB_correlation_Pvap(T, A, B, C, D):
    '''
    'ln(Pvp) = A*ln(T) + B/T + C + D*T^2 where Pvp in kPa, T in K'
    '''
    return np.exp(A*np.log(T) + B/T + C + D*T**2)

class solvent_preselection():
    
    def __init__(self, mixture, df_solvents=None, Tb_threshold=40):
        self.mixture = mixture
        self.name = f"{mixture['c_i']['Name']}_{mixture['c_j']['Name']}"

        if df_solvents is not None:
            df_solvents = df_solvents.copy()
            self.solvents = df_solvents['SMILES'].tolist()
            self.solvents_names = df_solvents['Name'].tolist()
            self.df_solvents = df_solvents
        
    def get_direct_rv_inf(self, solvent, T):
        mixture = self.mixture
        c_i = mixture['c_i']
        c_j = mixture['c_j']
        P_i_lst, P_j_lst = self.get_vapor_pressures(c_i, c_j, [T])
        P_i, P_j = P_i_lst[0], P_j_lst[0]

        # Initialize GH-GNN models
        GH_GNN_i = GH_GNN_old(c_i['SMILES'], solvent)
        GH_GNN_j = GH_GNN_old(c_j['SMILES'], solvent)

        K1_i, K2_i = GH_GNN_i.predict(T=0, constants=True)
        K1_j, K2_j = GH_GNN_j.predict(T=0, constants=True)

        ln_gamma_inf_i = K1_i + K2_i/T
        ln_gamma_inf_j = K1_j + K2_j/T
    
        gamma_inf_i = np.exp(ln_gamma_inf_i)
        gamma_inf_j = np.exp(ln_gamma_inf_j)

        rv = self.get_relative_volatility(P_i, P_j, gamma_inf_i, gamma_inf_j)
        return rv

    def get_relative_volatility(self, P_i:float, P_j:float, gamma_inf_i:float, 
                            gamma_inf_j:float):
        '''
        Computes the relative volatility of a system considering ideal vapor phase
        and non-ideal liquid phase. This assumption is true at low pressures. In 
        case a solvent is present, this is called pseudo-binary relative volatility.
        
        A relative volatility above ~1.1 could be separated using normal distillation.
        In such cases, the key components' boiling point usually differ by more than 
        50 K.

        Parameters
        ----------
        P_i : float
            Vapor pressure of pure species i.
        P_j : float
            Vapor pressure of pure species j.
        gamma_inf_i : float
            Activity coefficient at infinite dilution of solute i in the solvent.
        gamma_inf_j : float
            Activity coefficient at infinite dilution of solute j in the solvent.

        Returns
        -------
        a_ij : float
            Relative volatility of compound i over compound j.

        '''
        a_ideal = P_i/P_j               # Ideal part
        s_ij = gamma_inf_i/gamma_inf_j  # Non-ideal part or selectivity
        a_ij = a_ideal * s_ij
        return a_ij

    def get_P_vap_constants(self, c):
        A = float(c['Pvap']['Coefficient A'])
        B = float(c['Pvap']['Coefficient B'])
        C = float(c['Pvap']['Coefficient C'])
        D = float(c['Pvap']['Coefficient D'])
        return A, B, C, D

    def get_vapor_pressures(self, c_i, c_j, Ts):
        n_Ts = len(Ts)
        # Get P_vap parameters
        A_i, B_i, C_i, D_i = self.get_P_vap_constants(c_i)
        A_j, B_j, C_j, D_j = self.get_P_vap_constants(c_j)
        
        # Get vapor pressures for all temperatures
        P_i_lst = np.zeros(n_Ts)
        P_j_lst = np.zeros(n_Ts)
        for t, T in enumerate(Ts):
            # Compute vapor pressures
            P_i_lst[t] = KDB_correlation_Pvap(T, A_i, B_i, C_i, D_i)
            P_j_lst[t] = KDB_correlation_Pvap(T, A_j, B_j, C_j, D_j)
        return P_i_lst, P_j_lst
    
    def screen_with_selectivity_inf(self, n_Ts=5, combined_ghgnn=False):
        mixture = self.mixture
        solvents= self.solvents
        
        # Extract mixture information
        c_i = mixture['c_i']
        c_j = mixture['c_j']
        T_range = mixture['T_range']
        
        # Temperatures to be evaluated
        Ts = np.linspace(T_range[0], T_range[1], n_Ts)
        if combined_ghgnn:
            Ts = Ts + 273.15 # This is because the combined GH-GNN model was trained on T(K) + 273.15, which is inconsistent, but the model weights are trained for this

        selectivities_inf = np.zeros((len(solvents), len(Ts)))
        for s, solvent in enumerate(tqdm(solvents, desc=f'Screening mixture {self.name} with selectivity at infinite dilution: ')):
            # Initialize GH-GNN models
            if combined_ghgnn:
                GH_GNN_i = GH_GNN(c_i['SMILES'], solvent)
                GH_GNN_j = GH_GNN(c_j['SMILES'], solvent)
            else:
                GH_GNN_i = GH_GNN_old(c_i['SMILES'], solvent)
                GH_GNN_j = GH_GNN_old(c_j['SMILES'], solvent)
                
            # Get K1 and K2 constants, the temperature pass here could be any float or int
            K1_i, K2_i = GH_GNN_i.predict(T=0, constants=True)
            K1_j, K2_j = GH_GNN_j.predict(T=0, constants=True)
        
            for t, T in enumerate(Ts):
                
                ln_gamma_inf_i = K1_i + K2_i/T
                ln_gamma_inf_j = K1_j + K2_j/T
            
                gamma_inf_i = np.exp(ln_gamma_inf_i)
                gamma_inf_j = np.exp(ln_gamma_inf_j)
                
                s_inf = gamma_inf_i/gamma_inf_j
                selectivities_inf[s, t] = s_inf

        if combined_ghgnn:
            Ts = Ts - 273.15 # to have proper labels
        df_results = pd.DataFrame(selectivities_inf, columns=Ts)
        df_results['Mean'] = df_results.iloc[:,:].mean(axis=1)
        # Concatenate results on orginal df_solvents
        df_results = pd.concat([self.df_solvents, df_results], axis=1)
        df_results = df_results.sort_values(by='Mean', ascending=False)
        return df_results
    
    def screen_with_custom_inf(self, n_Ts=5, combined_ghgnn=False):
        mixture = self.mixture
        solvents= self.solvents
        
        # Extract mixture information
        c_i = mixture['c_i']
        c_j = mixture['c_j']
        T_range = mixture['T_range']
        
        # Temperatures to be evaluated
        Ts = np.linspace(T_range[0], T_range[1], n_Ts)
        if combined_ghgnn:
            Ts = Ts + 273.15 # This is because the combined GH-GNN model was trained on T(K) + 273.15, which is inconsistent, but the model weights are trained for this
        
        custom_inf = np.zeros((len(solvents), len(Ts)))
        ln_gammas_inf_ij = np.zeros((len(solvents), len(Ts)))
        ln_gammas_inf_ik = np.zeros((len(solvents), len(Ts)))
        ln_gammas_inf_ji = np.zeros((len(solvents), len(Ts)))
        ln_gammas_inf_jk = np.zeros((len(solvents), len(Ts)))
        ln_gammas_inf_ki = np.zeros((len(solvents), len(Ts)))
        ln_gammas_inf_kj = np.zeros((len(solvents), len(Ts)))
        custom_inf = np.zeros((len(solvents), len(Ts)))
        for s, solvent in enumerate(tqdm(solvents, desc=f'Screening mixture {self.name} with custom metric at infinite dilution: ')):
            if combined_ghgnn:
                GH_GNN_ij = GH_GNN(c_i['SMILES'], c_j['SMILES'])
                GH_GNN_ik = GH_GNN(c_i['SMILES'], solvent)

                GH_GNN_ji = GH_GNN(c_j['SMILES'], c_i['SMILES'])
                GH_GNN_jk = GH_GNN(c_j['SMILES'], solvent)

                GH_GNN_ki = GH_GNN(solvent, c_i['SMILES'])
                GH_GNN_kj = GH_GNN(solvent, c_j['SMILES'])

                K1_ij, K2_ij = GH_GNN_ij.predict(T=0, constants=True)
                K1_ik, K2_ik = GH_GNN_ik.predict(T=0, constants=True)
                K1_ji, K2_ji = GH_GNN_ji.predict(T=0, constants=True)
                K1_jk, K2_jk = GH_GNN_jk.predict(T=0, constants=True)
                K1_ki, K2_ki = GH_GNN_ki.predict(T=0, constants=True)
                K1_kj, K2_kj = GH_GNN_kj.predict(T=0, constants=True)
            else:
                GH_GNN_ki = GH_GNN_old(solvent, c_i['SMILES'])
                GH_GNN_jk = GH_GNN_old(c_j['SMILES'], solvent)
            
            # Get K1 and K2 constants, the temperature pass here could be any float or int
            K1_ki, K2_ki = GH_GNN_ki.predict(T=0, constants=True)
            K1_jk, K2_jk = GH_GNN_jk.predict(T=0, constants=True)

            for t, T in enumerate(Ts):
                
                ln_gamma_inf_ki = K1_ki + K2_ki/T
                ln_gamma_inf_jk = K1_jk + K2_jk/T
            
                gamma_inf_ki = np.exp(ln_gamma_inf_ki)
                gamma_inf_jk = np.exp(ln_gamma_inf_jk)

                ln_gammas_inf_ij[s, t] = K1_ij + K2_ij/T
                ln_gammas_inf_ik[s, t] = K1_ik + K2_ik/T
                ln_gammas_inf_ji[s, t] = K1_ji + K2_ji/T
                ln_gammas_inf_jk[s, t] = K1_jk + K2_jk/T
                ln_gammas_inf_ki[s, t] = K1_ki + K2_ki/T
                ln_gammas_inf_kj[s, t] = K1_kj + K2_kj/T

                c_inf = gamma_inf_ki * 1/gamma_inf_jk
                custom_inf[s, t] = c_inf
        
        if combined_ghgnn:
            Ts = Ts - 273.15 # to have proper labels
        df_results = pd.DataFrame(custom_inf, columns=Ts)

        ij = pd.DataFrame(ln_gammas_inf_ij, columns=Ts)
        cols = ['ln_gamma_inf_ij_' + str(col) for col in ij.columns]
        ij.columns = cols
        ij['ln_gamma_inf_ij_'] = ij.iloc[:,:].mean(axis=1)
        ij = ij.drop(columns=cols)
        
        ik = pd.DataFrame(ln_gammas_inf_ik, columns=Ts)
        cols = ['ln_gamma_inf_ik_' + str(col) for col in ik.columns]
        ik.columns = cols
        ik['ln_gamma_inf_ik_'] = ik.iloc[:,:].mean(axis=1)
        ik = ik.drop(columns=cols)

        ji = pd.DataFrame(ln_gammas_inf_ji, columns=Ts)
        cols = ['ln_gamma_inf_ji_' + str(col) for col in ji.columns]
        ji.columns = cols
        ji['ln_gamma_inf_ji_'] = ji.iloc[:,:].mean(axis=1)
        ji = ji.drop(columns=cols)
        
        jk = pd.DataFrame(ln_gammas_inf_jk, columns=Ts)
        cols = ['ln_gamma_inf_jk_' + str(col) for col in jk.columns]
        jk.columns = cols
        jk['ln_gamma_inf_jk_'] = jk.iloc[:,:].mean(axis=1)
        jk = jk.drop(columns=cols)

        ki = pd.DataFrame(ln_gammas_inf_ki, columns=Ts)
        cols = ['ln_gamma_inf_ki_' + str(col) for col in ki.columns]
        ki.columns = cols
        ki['ln_gamma_inf_ki_'] = ki.iloc[:,:].mean(axis=1)
        ki = ki.drop(columns=cols)

        kj = pd.DataFrame(ln_gammas_inf_kj, columns=Ts)
        cols = ['ln_gamma_inf_kj_' + str(col) for col in kj.columns]
        kj.columns = cols
        kj['ln_gamma_inf_kj_'] = kj.iloc[:,:].mean(axis=1)
        kj = kj.drop(columns=cols)
        
        
        
        df_results['Mean'] = df_results.iloc[:,:].mean(axis=1)
        df_results = pd.concat([df_results, ij, ik, ji, jk, ki, kj], axis=1)
        
        # Concatenate results on orginal df_solvents
        df_results = pd.concat([self.df_solvents, df_results], axis=1)
        df_results = df_results.sort_values(by='Mean', ascending=False)
        return df_results

    def screen_with_rv(self, n_Ts=5):
        mixture = self.mixture
        solvents= self.solvents

        # Extract mixture information
        c_i = mixture['c_i']
        c_j = mixture['c_j']
        T_range = mixture['T_range']
        
        # Temperatures to be evaluated
        Ts = np.linspace(T_range[0], T_range[1], n_Ts)
        
        # Get vapor pressures for all temperatures
        P_i_lst, P_j_lst = self.get_vapor_pressures(c_i, c_j, Ts)
        
        relative_volatilities_inf = np.zeros((len(solvents), len(Ts)))
        for s, solvent in enumerate(tqdm(solvents, desc=f'Screening mixture {self.name} with relative volatility: ')):
            # Initialize GH-GNN models
            GH_GNN_i = GH_GNN_old(c_i['SMILES'], solvent)
            GH_GNN_j = GH_GNN_old(c_j['SMILES'], solvent)
                
            # Get K1 and K2 constants, the temperature pass here could be any float or int
            K1_i, K2_i = GH_GNN_i.predict(T=0, constants=True)
            K1_j, K2_j = GH_GNN_j.predict(T=0, constants=True)
        
            for t, T in enumerate(Ts):
                
                P_i = P_i_lst[t]
                P_j = P_j_lst[t]
                
                ln_gamma_inf_i = K1_i + K2_i/T
                ln_gamma_inf_j = K1_j + K2_j/T
            
                gamma_inf_i = np.exp(ln_gamma_inf_i)
                gamma_inf_j = np.exp(ln_gamma_inf_j)
                
                a_inf = self.get_relative_volatility(P_i, P_j, gamma_inf_i, gamma_inf_j)
                relative_volatilities_inf[s, t] = a_inf

        df_results = pd.DataFrame(relative_volatilities_inf, columns=Ts)
        df_results['Mean'] = df_results.iloc[:,:].mean(axis=1)
        # Concatenate results on orginal df_solvents
        df_results = pd.concat([self.df_solvents, df_results], axis=1)
        df_results = df_results.sort_values(by='Mean', ascending=False)
        return df_results
    
    def get_xs_for_SF(self, SF, n_points=100):
    
        x_k = np.repeat(SF/(1+SF), n_points)
        x_i = np.linspace(0,1-x_k[0],n_points)
        x_j = 1- x_k - x_i
        
        return x_i, x_j, x_k

def create_folder(directory):
    """
    Create a folder/directory if it doesn't exist.

    Parameters:
    - directory: str, path of the directory to be created.
    """
    # Check if the directory already exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    


        

    

    
        
    


