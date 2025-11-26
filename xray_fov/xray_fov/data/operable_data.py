"""author: Maximilian Glumann"""

class OperableData():
    def __init__(self, data, ops=None, caching=True, view=False):
        self.data = data
        if not view:
            for value in self.data.values():
                if(isinstance(value, OperableData)):
                    value.parent = self
        self.ops = ops
        self.caching = caching
        self.parent = None
    def preCheck(self, key):
        if key not in self.data and getattr(self.ops, key) is None:
            raise AttributeError('no such data or operation!')
    def getCachedDataOrUpdate(self, key, *args, **kwargs):
        if key not in self.data or args or kwargs:
            val = getattr(self.ops, key)(self, *args, **kwargs)
            if self.caching and not args and not kwargs:
                self.data[key] = val
            else:
                return val
        return self.data[key]
    
    def getLambda(self, key):
        self.preCheck(key)
        return lambda *args, **kwargs : self.getCachedDataOrUpdate(key, *args, **kwargs)
    def getSplitByOps(self):
        from collections import defaultdict
        splits = defaultdict(dict)
        for key, value in self.data.items():
            if(hasattr(value, "ops")):
                splits[value.ops.__class__.__name__][key] = value
            else:
                splits["_None"][key] = value
        return OperableData(splits, view=True)
    def opsAsStr(self, ops):
        if not isinstance(ops, str):
            ops = ops.__class__.__name__
        return ops
    def getByOps(self, ops):
        return OperableData(self.getSplitByOps()[self.opsAsStr(ops)], self.ops, view=True)
    def foreachByOps(self, ops, key, *args, **kwargs):
        return {data[0]: data[1](key, *args, **kwargs) for data in list(self.getByOps(self.opsAsStr(ops)).data.items())}

    def __iter__(self):
        for item in self.data.items():
            yield item
    def __call__(self, key, *args, **kwargs):
        return self.getLambda(key)(*args, **kwargs)
    def __getitem__(self, key):
        keys = set()
        add_int = lambda x : keys.add(self["len"] + x if x < 0 else x)
        add_str = lambda x : keys.add(x)
        add_slice = lambda x : keys.update(range(*x.indices(self["len"])))
        add_or_update = lambda x : add_int(x) if isinstance(x, int) else (add_slice(x) if isinstance(x, slice) else add_str(x))

        if isinstance(key, tuple):
            for x in key:
                add_or_update(x)
        else:
            add_or_update(key)      
        
        if len(keys)==1:
            return self.getLambda(keys.pop())()
        else:
            res = {}
            for key in keys:
                res |= {key: self.getLambda(key)()}
            return OperableData(res, self.ops)
    def __getattr__(self, key):
        return self.getLambda(key)
    def __str__(self):
        res = ""
        for split in self.getSplitByOps().data.values():
            res += '\n'.join(('{} = {}'.format(item, str(split[item]).replace('\n', '\n\t ')) 
                              for item in sorted(split, key=lambda elem : (str(type(elem)), elem))))
        return res
    def _repr_html_(self):
        res = ""
        for split in self.getSplitByOps().data.items():
            import pandas as pd
            if split[0] == "_None":
                res += pd.DataFrame([split[1]], index=[0]).sort_index(axis=1).to_html(index=False, notebook=True)
            else:
                mapped = [v.data for v in split[1].values()]
                columns = set().union(*mapped)
                res += pd.DataFrame.from_records(mapped, columns=sorted(columns)).to_html(index=False, notebook=True)
        return res