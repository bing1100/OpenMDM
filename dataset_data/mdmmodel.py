import mlalgorithms as ma
from os import path

def mask_data(xs, include):
    if len(include) == len(xs[0]):
        return xs
    return [[x[i] for i in include] for x in xs] 

class MDMModel:
    def __init__(self, mdmjson, features, metrics=None):
        self.metrics = metrics
        self.mdmjson = mdmjson
        self.mdmalgos = mdmjson.getMDMAlgos()
        self.mlmrm = mdmjson.getPolicyRules("MATCH")
        self.features = list(features)
        self.pipelines = []
        for ruleset in self.mlmrm:
            pipeline = []
            for rule in ruleset:
                d = self.mdmalgos[rule]
                model = {
                    "name":rule,
                    "clf": ma.instantiate(d["meta"]['algorithm'], 
                                       idx=self.features.index(rule) if rule in self.features else 0, 
                                       thresh=d["meta"].get('matchThreshold', 0.5),
                                       pthresh=d["meta"].get('possibleMatchThreshold', 0.5),
                                       algo=d["meta"]['algorithm'],
                                       matcher=d["meta"]),
                    "fieldIdx":[self.features.index(t) for t in d["meta"].get('recordFeatures',self.features)]
                }
                
                # Load the checkpoints if it exists
                if d["save_path"] and path.isfile(d["save_path"]):
                    model["clf"].load(d["save_path"])
                
                pipeline.append(model)
            self.pipelines.append(pipeline)
        
    def train(self, x_train, y_train, **kwargs):
        for pipeline in self.pipelines:
            y_mask = [1] * len(x_train)
            for model in pipeline:
                xs = mask_data(x_train, model["fieldIdx"])
                ys = [i and j for i,j in zip(y_mask, y_train)]
                model["clf"].fit(xs, ys, **kwargs)
                y_mask = model["clf"]._predict_train(xs)
        
    def infer(self, x_test):
        matchOutput = []
        possibleMatchOutput = []
        for pipeline in self.pipelines:
            y_pred = [1]*len(x_test)
            py_pred = [1]*len(x_test)
            for model in pipeline:
                xs = mask_data(x_test, model["fieldIdx"])
                y_mask, py_mask = model["clf"].predict(xs)
                y_pred = [i and j for i,j in zip(y_pred, y_mask)]
                py_pred = [i and j for i,j in zip(py_pred, py_mask)]
            matchOutput.append(y_pred)
            possibleMatchOutput.append(py_pred)
        return matchOutput, possibleMatchOutput
    
    def save(self, filename):
        mrmkey = "matchRules"
        mrm = self.mdmjson.config["matchRules"]
        match_fields = self.mdmjson.match_field_config
        
        for pipeline in self.pipelines:
            for model in pipeline:
                _, _, mf = model["clf"].save(filename+model["name"])
                if not mf:
                    continue
                
                for i, algo in enumerate(match_fields):
                    if algo['name'] == model["name"]:
                        match_fields[i]['ml']["savePath"] = mf[0]["ml"]['savePath']
        
        self.mdmjson.save(filename, mrmkey, mrm, match_fields, extend=False)    
