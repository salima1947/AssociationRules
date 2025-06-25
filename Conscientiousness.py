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

['densityHigh' , 'Fonttype calibriLight '],
['densityHigh' , 'iconText' ],
['Fonttype calibriLight' , 'iconText'],
['densityHigh', 'sankeyDiagram' ],
['Fonttype medium' , 'densityHigh'],
['barDown' , 'Fonttype medium'],
['Fonttype  calibriLight' , 'sankeyDiagram' ],
['densityHigh', 'barDown' ],
['densityHigh', 'Fonttype medium'],
['iconText' , 'barDown' ],
['linechartPoints' , 'densityHigh'],
['sankeyDiagram' , 'Fonttype medium'],
['densityMedium', 'Fonttype large' ],
['Fonttype timesNewRoman' , 'densityMedium'],
['linechart' , 'densityMedium'],
['barChartHorizontal' , 'linechart' ],
['linechart' , 'Fonttype large' ],
['barChartHorizontal' , 'densityMedium'],
['linechart' , 'barChartHorizontal' ],
['Fonttype timesNewRoman' , 'barLeft' ],
['linechart' , 'iconOnly' ],
['densityMedium', 'iconOnly'],
['Fonttype  timesNewRoman' , 'iconOnly'],
['highDensity' , 'barLeft' ],
['linechart' , 'treemap' ],
['barChartHorizontal' , 'linechart' ],
['Fonttype timesNewRoman' , 'sankeyDiagram'],
['linechart' , 'barChartHorizontal' ],
['barChartHorizontal' , 'densityMedium'],
['linechart' , 'densityMedium'],
['barChartHorizontal' , 'Fonttype  timesNewRoman'],
['linechart' , 'iconOnly' ],
[ 'Slidable top navigation' , 'structure Network' ],
['densityHigh' , 'Buttonphoto' ],
[ 'Slidable top navigation' , 'Scroll thumb'],
[ 'Buttonphoto' , 'Scroll thumb' ],
[ 'Buttonname' , 'Scroll thumb' ],
['Fontsize header 75', 'Fonttype Arial' ],
['Fontsize 5' , 'Fonttype  Arial' ],
[ 'Scroll thumb' , 'Colour hue' ],
['alignText left' ,  'Fontsize 5'],
['Scroll thumb' , 'Segmented control' ],
['buttonIconText' ,'menuBreadth'   ],
['marginSmall' ,'menuBreadth'   ],
['marginSmall' ,'alignText Justified'   ],
['buttonIconText' ,'alignText Justified'  ],
['buttonIconText' ,'densityMedium'   ],
['densityMedium' ,'buttonIconText'],
['menuBreadth' ,'Fontsize Large'],
['marginSmall' ,'helpIcon'   ],
['buttonIconText' ,'marginSmall'   ],
['marginSmall' ,'densityLow' ],
['alignText Left' ,'marginSmall'],
['alignText Left' ,'buttonIconText' ],
['slidable top navigation'],
['Colour OrangeGrey'],
['listStructure'],
['element style table'],
['marginMedium'],
['alignText Justified', 'alignText Wrap'],
['Time to retrieve information Short'],
['Scan and decide	fast'],
['button position horizontal	center'],
['button position vertical	footer'],
['marginLarge'],
['marginMedium']
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