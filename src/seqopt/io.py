from collections import OrderedDict

class FastaFile:

    def __init__(self):
        self._sequences = OrderedDict()

    def __setitem__(self, key, value):
        self._sequences[key] = value

    def __getitem__(self, key):
        return self._sequences[key]

    def __iter__(self):
        return iter(self._sequences)
    
    def __len__(self):
        return len(self._sequences)
    
    def write(self,filename,fold=60):
        with open(filename,'w') as f:
            for k,v in self._sequences.items():
                f.write(f'>{k}\n')
                for i in range(0,len(v),fold):
                    f.write(v[i:i+fold]+'\n')
