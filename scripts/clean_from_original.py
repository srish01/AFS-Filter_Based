import os
import csv
import pandas as pd
import warnings

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')

ORIGINAL_PATH = os.path.join(OUTPUT_PATH, 'original_datasets')
OUTPUT_CLEAN = os.path.join(OUTPUT_PATH, 'clean')

CSV_FILEDAT = os.path.join(OUTPUT_PATH, 'filename_data.csv')
TXT_FILENAMES = os.path.join(OUTPUT_PATH, 'filenames.txt')
TXT_FILENAMES_SAMPLE = os.path.join(OUTPUT_PATH, 'filenames_by_samples.txt')
TXT_FILENAMES_FEATURE = os.path.join(OUTPUT_PATH, 'filenames_by_features.txt')

def create_clean(root, file, editor):
    
    dataset = pd.read_csv(os.path.join(root, file), sep="\t", dtype=float, index_col=0)
    X = dataset.iloc[:,:-1].values
    Y = dataset.iloc[:,-1].values

    if not is_valid_dataset(X, Y):
        return
    
    Y = change_labels(Y)

    outputname = f'{os.path.basename(root)}.csv'
    
    d1 = pd.DataFrame(X)
    d2 = pd.DataFrame(Y)
    final = pd.concat([d1, d2], axis=1)
    print(outputname)
    final.to_csv(os.path.join(OUTPUT_CLEAN, outputname), sep=',', header=None, index=None)
    
    editor.append(0, outputname, len(X), len(X[0]))
    editor.append(1, f'{outputname}\n')

def change_labels(Y):
    return [-1 if not y else y for y in Y]

def is_valid_dataset(X, Y):
    st = set(Y)
    if len(st) != 2 or 0 not in st or 1 not in st:
        return False
    return True

def is_valid_file(dirname, filename):
    file, extension = os.path.splitext(os.path.basename(filename))
    if extension != '.dat':
        return False
    dirname = os.path.basename(dirname)
    return file == f'{dirname}_R'

class editor:
    
    def __init__(self, *args, **kwargs):
        self.filenames = args
        self.csv = kwargs.pop('csv', None)

    def __enter__(self):
        self.files = [open(file, 'w', newline='') for file in self.filenames]
        if self.csv:
            self.writer = csv.writer(self.files[self.csv[0]])
            self.writer.writerow(self.csv[1])
            
        return self

    def append(self, which, *args):
        done = False
        if self.csv:
            if self.csv[0] == which:
                done = True
                self.writer.writerow(args)
            
        if not done:
            self.files[which].write(*args)
            
    def __exit__(self, *args):
        for file in self.files:
            file.close()

def create_sorted_names():
    dataset = pd.read_csv(CSV_FILEDAT, sep=",")

    filenames = dataset['Filename'].values
    samples = dataset['Samples'].values
    features = dataset['Features'].values

    by_sample = sorted(zip(filenames, samples), key=lambda x: x[1])
    by_feature = sorted(zip(filenames, features), key=lambda x: x[1])
    
    with open(TXT_FILENAMES_SAMPLE, 'w') as f:
        f.writelines([str(i[0]) + '\n' for i in by_sample])
        
    with open(TXT_FILENAMES_FEATURE, 'w') as f:
        f.writelines([str(i[0]) + '\n' for i in by_feature])

def main():
    
    with editor(CSV_FILEDAT, TXT_FILENAMES, csv=(0, ['Filename', 'Samples', 'Features'])) as ed:
        for root, _, files in os.walk(ORIGINAL_PATH):
            for file in files:
                if is_valid_file(root, file):
                    create_clean(root, file, ed)

    create_sorted_names()

if __name__ == "__main__":
    main()