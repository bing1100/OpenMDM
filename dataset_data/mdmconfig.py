import json
import mlserialize as s
from copy import deepcopy
import json

DEFAULT_THRESH = 0.9

class MDMConfig:
    def __init__(self, mdm_json_config):
        """
        Parameters:
        -----------
            match_field_json: path to json file specifing the Smile MDM match field config
        """
        # Load the JSON and store it? Unsure the best way to keep track of the JSON data, may need 
        # to do some pre-processing in this function.
        
        with open(mdm_json_config, "r") as f:
            self.config = json.load(f)  # Probably a list of dictionaries 
            self.match_field_config = self.config["matchFeatures"]

            # matchFields is a list of dictionaries
            self.mdm_algorithms = {}
            for dct in self.match_field_config:
                self.mdm_algorithms[dct["name"]] = {
                    "type":"matcher" if "matcher" in dct 
                    else "similarity" if "similarity" in dct 
                    else "ml",
                    "meta": dct.get("matcher", 
                                    dct.get("similarity", 
                                            dct.get("ml", None))),
                    "path_name": "field",
                    "resource_path": dct.get("field", None),
                    "save_path": dct.get("savePath", None)
                }
            
            # MatchResult Map
            self.match_rules = {
                'POSSIBLE_MATCH':[],
                'MATCH':[]
            }
            for rule,type in self.config["matchRules"].items():
                self.match_rules[type].append(rule.split(','))
                
            # MLMatchResult Map

            self.blocking_fields = self.config["blockingFields"]

            # TODO: read the Candidate Filter Search Params if we decide to use them too.

    def save(self, filename, mrmKey, matchresultmap, matchfields, extend=True):
        njson = deepcopy(self.config)
        # save the blockfiltering here if needed
        njson[mrmKey] = matchresultmap
        if extend:
            njson["matchFields"].extend(matchfields)
        else:
            njson["matchFields"] = matchfields
        # the json file where the output must be stored
        out_file = open(filename+".json", "w")
        json.dump(njson, out_file, indent = 6)
        
    def get_checkpoint(self, algorithm):
        for _, v in self.mdm_algorithms.items():
            if v["matcher"]["algorithm"] == algorithm:
                return v["save_path"]
        
    def getFilteringRules(self):
        pass
    
    def getMDMAlgos(self):
        return self.mdm_algorithms
    
    def getPolicyRules(self, type, thresh=False):
        """
        Parameters:
        -----------
            type: one of 'POSSIBLE_MATCH' or 'MATCH'

        Returns:
        --------
            match_map: the match map that can be used to generate the decision trie 
        """
        return self.match_rules[type]
    