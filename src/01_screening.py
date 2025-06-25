import pandas as pd
from src.utils.utils import load_json_files, find_name_from_smiles, solvent_preselection, create_folder
from argparse import ArgumentParser
from rdkit import Chem
from rdkit.Chem import Descriptors

# python src/solvent_screening/01_screening.py --metric custom_inf

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--metric", help="Metric to be used as the solvent ranking criteria", type=str)

    args = parser.parse_args()
    metric = args.metric

    assert metric in ['selectivity_inf', 'custom_inf']
    
    # Load pure component KDB database
    pure_comp_dict = load_json_files('data/raw/KDB/pure_component_properties')

    df_comps = pd.read_csv('data/raw/components_in_mixtures_to_be_screened.csv')
    df_solvents = pd.read_csv('data/external/solvents.csv')
    
    # Get components in mixtures to be screened
    mask = [True if 'IL' in mix.split('/') else False for mix in df_comps['mixture'].tolist()]
    df_comps = df_comps[mask]

    comp_data = {}
    for name, smiles in zip(df_comps['name'].tolist(),df_comps['smiles'].tolist()):
        if name in pure_comp_dict:
            comp_data[name] = pure_comp_dict[name]
        else:
            try:
                name_kdb = find_name_from_smiles(smiles, pure_comp_dict)
                comp_data[name] = pure_comp_dict[name_kdb]
            except:
                print(f'Component {name} not found in the KDB pure component database')

    # Create solvent preselection objects
    sp_objs = []
    # Here we use Tmin and Tmax because is a liquid-liquid extraction, not a distillation
    Tmin = 25 + 273.15
    Tmax = 25.0001 + 273.15
    mix_to_screen = {
            'c_i':{
                'Name':'EMIMBF4',
                'SMILES':'[B-](F)(F)(F)F.CCN1C=C[N+](=C1)C',
            },
            'c_j':{
                'Name':'CAPROLACTAM',
                'SMILES':comp_data['CAPROLACTAM']['SMILES'],
            },
            'T_range':(Tmin, Tmax)
        }
    sp = solvent_preselection(mix_to_screen, df_solvents)
    sp_objs.append(sp)
        
    # Screen solvents
    folder = f'solvent_screening/IL'
    create_folder(folder)
    for sp in sp_objs:
        if metric == 'selectivity_inf':
            results = sp.screen_with_selectivity_inf(combined_ghgnn=True)
        elif metric == 'custom_inf':
            results = sp.screen_with_custom_inf(combined_ghgnn=True) 
            
            # Filter molecular weight
            results['MW'] = [Descriptors.MolWt(Chem.MolFromSmiles(smiles)) for smiles in results['SMILES'].tolist()]
            results_constraint = results[results['MW'] <= 135]
            # Filter normal boiling point
            results_constraint = results_constraint[results_constraint['Tb'] <= 200 + 273.15].copy()
            # Filter GHS signal
            results_constraint = results_constraint[results_constraint['GHS_signal'] == 'Warning'].copy()

            results_constraint['Rank'] = range(1,len(results_constraint)+1)
            results_constraint.to_csv(f'{folder}/selection_{metric}_{sp.name}_constraint.csv', index=False)

        results.to_csv(f'{folder}/selection_{metric}_{sp.name}.csv', index=False)
        


