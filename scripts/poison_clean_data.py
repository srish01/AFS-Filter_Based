import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import poisoning
import os

from rich.progress import Progress

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')
CLEAN_PATH = os.path.join(DATA_PATH, 'clean')
ATTACKS_PATH = os.path.join(DATA_PATH, 'attacks')
FILENAMES_PATH = 'filenames_by_samples.txt'

POISON_PERCENT = 0.2
POISON_PROJECTIONS = [0.5, 1, 1.5, 2, 2.5, 5]
POISON_TYPES = [poisoning.xiao2018]

types_name = {poisoning.xiao2018: 'xiao', poisoning.frederickson2018: 'fred'}

VERBOSE = True

def write_attacks(filename, poisoned, labels):
    d1 = pd.DataFrame(poisoned)
    d2 = pd.DataFrame(labels)
    final = pd.concat([d1, d2], axis=1)
    final.to_csv(os.path.join(ATTACKS_PATH, filename), sep=',', header=None, index=None)

# for getting the filename of the attack file
def create_filename(filename, projection, tp):
    file, extension = os.path.splitext(os.path.basename(filename))
    return f"{file}_[{types_name[tp]}][{projection}]{extension}"

# function to poison a dataset, gets passed a function to update progress bar
def poison_clean(filename, progress_update):

    dataset = pd.read_csv(os.path.join(CLEAN_PATH, filename), sep=",", header=None)
    X = dataset.iloc[:,:-1].values
    Y = dataset.iloc[:,-1].values
    
    for tp in POISON_TYPES:
        
        for projection in POISON_PROJECTIONS:
        
            new_filename = create_filename(filename, projection, tp)
            
            if VERBOSE:
                print(f'    + Type: {types_name[tp]} :: Projection: {projection}')
        
            # if file exists already, skip
            if os.path.exists(os.path.join(ATTACKS_PATH, new_filename)):
                progress_update(2)
                continue
            
            model = tp()
            poisoned, labels = model.autorun(X, Y, POISON_PERCENT, projection)
            write_attacks(new_filename, poisoned, labels)
            progress_update(2)
            
        progress_update(1)

# Main entry point for script
def main():
    with open(os.path.join(DATA_PATH, FILENAMES_PATH), 'r') as f:
        files = [file.strip() for file in f.readlines()]
    
    with open(os.path.join(DATA_PATH, 'completed.txt'), 'w') as completed:
        with Progress() as progress:
            t1 = progress.add_task(f"[bold red]Running Files ({len(files)})...", total=len(files))
            t2 = progress.add_task(f"[bold blue]Running Models ({len(POISON_TYPES)})...", total=len(POISON_TYPES))
            t3 = progress.add_task(f"[bold blue]Running Projections ({len(POISON_PROJECTIONS)})...", total=len(POISON_PROJECTIONS))
            proc = [t1, t2, t3]
            def update(i):
                progress.advance(proc[i])
            for file in files:
                if VERBOSE:
                    print(f'File: {file}')
                poison_clean(file, update)
                completed.write(f'{file}\n')
                completed.flush()
                progress.advance(t1)
                progress.reset(t2)
                progress.reset(t3)

if __name__ == "__main__":
    main()