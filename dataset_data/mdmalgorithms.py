from tqdm import tqdm
import cologne_phonetics
from abydos.phonetic import Caverphone, Metaphone, DoubleMetaphone, Soundex, RefinedSoundex, MRA, NYSIIS
from abydos.distance import JaroWinkler, Cosine, Jaccard, Levenshtein, Dice
from datetime import date
import re
import csv
import json

NICKNAME_PATH = ""

def cologne_compare(s1, s2):
    # If the string have more than one word
    if " " in s1 and " " in s2:
        if len(s1.split()) != len(s2.split()):
            return False
        else:
            for i in range(len(s1.split())):
                if cologne_phonetics.encode(s1.split()[i]) != cologne_phonetics.encode(s2.split()[i]):
                    return False
                if cologne_phonetics.compare([s1.split()[i], s2.split()[i]]) == False:
                    return False
            return True
    # Single words
    else:
        return cologne_phonetics.compare([s1, s2])

def Nicknames(nicknames_path, canon_name, nickname):
    canon_name = canon_name.lower()
    nickname = nickname.lower()
    with open(nicknames_path) as nickname_file:
        reader = csv.reader(nickname_file)
        for row in reader:  # each row is a list
            if len(row) < 2:
                continue
            # Nickname in list
            if canon_name in row and nickname in row:
                return True

            # Nickname not in list, check substring
            if canon_name in row and nickname not in row:
                return (canon_name in nickname) or (nickname in canon_name)
        # Name not in nickname database
        if (canon_name in nickname) or (nickname in canon_name):
            return True

        return False

def Date_2(s1, s2):
    if len(s1.split("-")) <= 2 or len(s2.split("-")) <= 2:
        min_date = min(len(s1.split("-")), len(s2.split("-")) <= 2)
        for i in range(min_date):
            if s1.split("-")[i] != s2.split("-")[i]:
                print("False")
                return False
        print("True")
        return True
    else:
        d1 = date.fromisoformat(s1)
        d2 = date.fromisoformat(s2)
        diff = abs(d1 - d2)
        print(diff)
        return d1.resolution > diff or d2.resolution > diff

def String(s1, s2, exact=False):
    if exact:
        return s1 == s2
    return s1.lower() == s2.lower()

def Substring(s1, s2):
    return s1.lower() in s2.lower() or s2.lower() in s1.lower()

def Date(s1, s2):
    d1 = date.fromisoformat(s1)
    d2 = date.fromisoformat(s2)
    diff = abs(d1 - d2)
    return d1.resolution > diff or d2.resolution > diff

def Numeric(s1, s2):
    return re.sub("[^0-9]", "", s1) == re.sub("[^0-9]", "", s2)

def Name_any_order(s1, s2, exact=False):
    # This is a naive implementation i.e. John John would be the same as John John John
    if exact:
        return set(s1.split(" ")) == set(s2.split(" "))
    return set(s1.lower().split(" ")) == set(s2.lower().split(" "))

def Name_first_and_last(s1, s2, exact=False):
    return String(s1, s2, exact=exact)

def Empty_field(s1, s2):
    return s1 == "" and s2 == ""

class MdmMatcher:
    # This dictionary should be a mapping of the  all-caps string name that is used in the JSON 
    # to the functions
    nickname_path = NICKNAME_PATH
    algorithms = {
        "CAVERPHONE1": lambda s1, s2: Caverphone(version=1).encode(s1) == Caverphone(version=1).encode(s2),
        "CAVERPHONE2": lambda s1, s2: Caverphone().encode(s1) == Caverphone()(s2),
        "COLOGNE": lambda s1, s2: cologne_compare(s1, s2),
        "DOUBLE_METAPHONE" : lambda s1, s2: DoubleMetaphone().encode(s1) == DoubleMetaphone().encode(s2),
        "MATCH_RATING_APPROACH": lambda s1, s2: MRA().encode(s1) == MRA().encode(s2),
        "METAPHONE" : lambda s1, s2: Metaphone().encode(s1) == Metaphone().encode(s2),
        "NYSIIS": lambda s1, s2: NYSIIS().encode(s1) == NYSIIS().encode(s2),
        "REFINED_SOUNDEX": lambda s1, s2: RefinedSoundex().encode(s1) == RefinedSoundex().encode(s2),
        "SOUNDEX" : lambda s1, s2: Soundex().encode(s1) == Soundex().encode(s2),
        "JARO_WINKLER": lambda s1, s2, thresh: (JaroWinkler().sim(s1, s2), JaroWinkler().sim(s1, s2) >= thresh),
        "COSINE": lambda s1, s2, thresh: (Cosine().sim(s1, s2), Cosine().sim(s1, s2) >= thresh),
        "JACCARD": lambda s1, s2, thresh: (Jaccard().sim(s1, s2), Jaccard().sim(s1, s2) >= thresh),
        "LEVENSCHTEIN": lambda s1, s2, thresh: (Levenshtein().sim(s1, s2), Levenshtein().sim(s1, s2) >= thresh),
        "SORENSEN_DICE": lambda s1, s2, thresh: (Dice().sim(s1, s2), Dice().sim(s1, s2) >= thresh),
        "NICKNAME": lambda s1, s2: Nicknames(MdmMatcher.nicknames_path, s1, s2),
        "STRING": lambda s1, s2: String(s1, s2),
        "SUBSTRING": lambda s1, s2: Substring(s1, s2),
        "DATE": lambda s1, s2: Date_2(s1, s2),
        "NUMERIC": lambda s1, s2: Numeric(s1, s2),
        "NAME_ANY_ORDER" : lambda s1, s2: Name_any_order(s1, s2),
        "NAME_FIRST_AND_LAST": lambda s1, s2: Name_first_and_last(s1, s2)
    }
    
    """
    Class to execute the matching algorithms specified by the MDM matchFields JSON file
    """
    def __init__(self, mdm_algorithms):
        """
        Parameters:
        -----------
            match_field_json: path to json file specifing the Smile MDM match field config
        """
        # Load the JSON and store it? Unsure the best way to keep track of the JSON data, may need 
        # to do some pre-processing in this function.
        self.mdm_algorithms = mdm_algorithms
    
    # DEPRECATE THIS AND MOVE IT
    def predictPair(self, json_name, pair):
        """
        Algo function should have a certain signature
        s1 is the first string to match
        s2 is the second string to match
        replace **kwargs with the relevant keyword arguments, as specified by the JSON
        E.g. similarity algorithms would use keyword `matchThreshold`
        E.g. some string algorithms would use keyword `exact`
        Need to write an algo_function for each of the ones we want


        Parameters:
        -----------
            s1: string
            s2: string
        
        Returns:
        --------
            match: bool
        """
        
        if "Thresh" in json_name:
            _, algo_name, thresh = json_name.split("/")
            return pair[2][algo_name] >= float(thresh)
        
        resourcepath = self.mdm_algorithms[json_name]["resource_path"]
        s1 = str(pair[0][resourcepath])
        s2 = str(pair[1][resourcepath])
        
        algo_name = self.mdm_algorithms[json_name]["meta"]["algorithm"]
        if self.mdm_algorithms[json_name]["meta"].get("matchThreshold") is not None:
            thresh = self.mdm_algorithms[json_name]["meta"]["matchThreshold"]
            return MdmMatcher.algorithms[algo_name](s1, s2, thresh)[1]
        return MdmMatcher.algorithms[algo_name](s1, s2)
    
    def predictPairs(self, pairs):
        """
        Parameters:
        -----------
            pairs: list of 2-tuples of string pairs to match

        Returns:
        --------
            match_results: dictionary, with keys as the matchField names in the JSON and value as a list
            of booleans the same length as pairs, with values of that algorithm executed on each pair
        """
        match_results = {}
        for name, algo in self.mdm_algorithms.items():
            res = []
            algo_name = algo["meta"]["algorithm"]
            if algo_name not in MdmMatcher.algorithms:
                continue
            
            resourcepath = algo["resource_path"]
            for pair in tqdm(pairs):
                s1 = str(pair[0][resourcepath])
                s2 = str(pair[1][resourcepath])
                if algo["meta"].get("matchThreshold") is not None:
                    thresh = algo["meta"]["matchThreshold"]
                    res.append(MdmMatcher.algorithms[algo_name](s1, s2, thresh)[0])
                else:
                    res.append(MdmMatcher.algorithms[algo_name](s1, s2))
            match_results[name] = res
        return match_results