import csv
import pandas as pd
from recordlinkage import datasets
from recordlinkage import Index
from recordlinkage.index import Full
import recordlinkage
from mdmdataset import Dataset
from mdmconfig import MDMConfig


def load_data(filepath, skip_already_labeled=False, already_labeled={}):
    # csv format: [ID1, ID2, GT, PRED, ANNOTATED_LABEL, SAMPLING_STRATEGY, CONFIDENCE]
    with open(filepath, 'r') as csvfile:
        data = []
        reader = csv.reader(csvfile)
        for row in reader:
            textid = row[0] + row[1]
            # if len(row) < 3:
            #     row.append("") # add empty col for ANNOTATED_LABEL to add later
            # if len(row) < 4:
            #     row.append("") # add empty col for SAMPLING_STRATEGY to add later        
            # if len(row) < 5:
            #     row.append(0) # add empty col for CONFIDENCE to add later         
            data.append(row)

            label = str(row[4])
            if row[4] != "":
                textid = row[0] + row[1]
                already_labeled[textid] = label

    csvfile.close()
    return data

def print_fields(fields1, fields2):   
    str_fields = ""
    for field in fields1:
        if field == "":
            str_fields += f"        NA, "
        else:
            str_fields += f"{field:>10}, "

    str_fields2 = ""
    for field in fields2:
        if field == "":
            str_fields2 += f"        NA, "
        else:
            str_fields2 += f"{field:>10}, "
    

    print(str_fields)  
    print(str_fields2) 
    return 

def write_data(filepath, data):
    with open(filepath, 'w', errors='replace') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    csvfile.close() 

def get_annotations(data, df_source, dataset, default_sampling_strategy="random", already_labeled = {}, verbose=True):
    """Prompts annotator for label from command line and adds annotations to data 
    
    Keyword arguments:
        data -- a list of unlabeled items where each item is 
                [ID1, ID2, GT, PRED, ANNOTATED_LABEL, SAMPLING_STRATEGY]
        default_sampling_strategy -- strategy to use for each item if not already specified
    """
    
    annotation_instructions = "Please type 1 if the pair of records displayed are a match, "
    annotation_instructions += "or hit Enter if they are not a match.\n"
    annotation_instructions += "Type 2 to go back to the last message, "
    annotation_instructions += "type d to see detailed definitions, "
    annotation_instructions += "or type s to save your annotations.\n"

    last_instruction = "All done!\n"
    last_instruction += "Type 2 to go back to change any labels,\n"
    last_instruction += "or Enter to save your annotations."

    ind = 0
    while ind <= len(data):
        if ind < 0:
            ind = 0 # in case you've gone back before the first
        if ind < len(data):
            pair = (data[ind][0], data[ind][1])
            textid = data[ind][0] + data[ind][1]
            fields1 = df_source.loc[data[ind][0]]
            fields2 = df_source.loc[data[ind][1]]
            label = data[ind][4]
            strategy =  data[ind][5]
            
            if strategy == "":
                strategy = default_sampling_strategy

            if textid in already_labeled:
                if verbose:
                    print("Skipping seen "+str(textid)+" with label "+label)
                    print_fields(fields1, fields2)
                ind+=1
            else:
                print(annotation_instructions)
                if verbose:
                    print("Sampled with strategy `"+str(strategy))
                print_fields(fields1, fields2)
                label = str(input("\n\n> ")) 
                print(label)

                if label == "2":                   
                    ind-=1  # go back
                # elif label == "d":                    
                #     print(detailed_instructions) # print detailed instructions
                elif label == "s":
                    break  # save and exit
                elif label == "0" or label== "1":
                        
                    data[ind][4] = label # add label to our data
                    dataset.add_annotation(pair, label, strategy)


                    if data[ind][5] is None or data[ind][5] == "":
                        data[ind][5] = default_sampling_strategy # add default if none given
                    ind+=1        

        else:
            #last one - give annotator a chance to go back
            print(last_instruction)
            label = str(input("\n\n> ")) 
            if label == "2":
                ind-=1
            else:
                ind+=1

    return data


# filepath_source = '/root/rachel/mdm_project/mdm_source_data_synthetic/source_v1.csv'
# filepath_links = '/root/rachel/mdm_project/mdm_source_data_synthetic/links_v1.csv'
# csv_filepath = '/root/rachel/mdm_project/annotation_pairs_v1.csv'

# # match_csv = '/root/rachel/mdm_project/match_annotations.csv'
# # not_match_csv = '/root/rachel/mdm_project/not_match_annotations.csv'
# match_csv = '/root/rachel/mdm_project/mdm/src/dataset_data/mdm_source_data_synthetic/match_annotations.csv'
# not_match_csv = '/root/rachel/mdm_project/mdm/src/dataset_data/mdm_source_data_synthetic/not_match_annotations.csv'


# mdmJson = MDMJson('src/rule_data/mdm_demo_config_fixed_v2.json')
# dataset = Dataset(mdmJson)
# dataset.load_febrl()
# records, data, links = dataset.get_data()

# already_labeled = {}
# annotations_data_empty = load_data(csv_filepath, True)

# annotations = get_annotations(annotations_data_empty[1:-1], records)

# match_annotations = []
# not_match_annotations = []

# for item in annotations:
#     label = item[4]
#     if label == '1':
#         match_annotations.append(item)
#     elif label == '0':
#         not_match_annotations.append(item)

# write_data(match_csv, match_annotations)
# write_data(not_match_csv, not_match_annotations)



