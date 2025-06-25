#Imports the pandas library for data manipulation and analysis.
import pandas as pd
#Imports the TransactionEncoder class for converting data into binary transaction format.
from mlxtend.preprocessing import TransactionEncoder
#Imports functions for finding frequent itemsets and generating association rules.
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth
from mlxtend.frequent_patterns import association_rules

#Defines a list of lists, where each sub-list represents a transaction
#    (combination of design features). For example, ['highDensity', 'calibriLight']
#    means a transaction has both "highDensity" and "calibriLight" features.


dataset = [

['alignText centre' ,'structure Network'],
['densityLow' , 'Buttonphoto '],
['Fonttype Verdana' , 'Segmented control '],
['Fontsize header 53' , 'Segmented control '],
['Buttonphoto' , 'Expanding list'],
['Fontsize header 53',  'fonttype Verdana'],
['Fontsize 40' , 'Colour Hue'],
['fonttype  Verdana'  , 'Colour Hue'],
['fonttype  Verdana'  , 'Fontsize 40 '],
['Fontsize 40'  , 'fonttype  Verdana '],
['Stepping' , 'Expanding list'],
['marginSmall '],
['buttonIconText '],
['densityLow '],
['menuBreadth'],
['marginSmall' ,'densityLow'],
['helpIcon'],
['font header Verdana 53'],
['fonttype verdana'],
['fonttype times new roman'],
['Background color green'],
['fontcolor	black'],
['active link yellow'],
['visited link white']

]


#Creates a TransactionEncoder object.
te = TransactionEncoder()
#Convertit l'ensemble de données dans un format adapté à l'extraction de règles d'association en utilisant
te_ary = te.fit(dataset).transform(dataset)
#Crée un DataFrame Pandas à partir de l'ensemble de données transformé.
df = pd.DataFrame(te_ary, columns=te.columns_)


#Utilise l'algorithme FP-Growth pour trouver des ensembles d'articles fréquents dans le DataFrame df avec
#    un support minimum de 0,01 (1 %) et renvoie les noms des articles plutôt que les indices de colonnes
#    (use_colnames=True).
#frequent_itemsets = fpgrowth(df, min_support=0.01, use_colnames=True)
frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)

#Generates association rules from the frequent itemsets. It uses the "confidence" metric
#    (meaning the percentage of transactions with the consequent itemset given the antecedent itemset)
#     and sets a minimum confidence threshold of 0.2 (meaning only rules with at least 20% confidence are reported).
association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)