# Personality-Based UI Design Analysis using Association Rule Mining

This repository contains Python scripts for analyzing UI design preferences associated with the **Big Five Personality Traits** (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism). The analysis is based on association rule mining using the **Apriori algorithm**.

## 📊 Project Overview

The goal of this project is to identify frequent co-occurrences of UI design features for users exhibiting high levels in each personality trait. Eye-tracking and preference data are transformed into binary transactions, from which **frequent itemsets** and **association rules** are extracted.

We use:
- **`mlxtend`**: For Apriori algorithm and rule generation
- **`pandas`**: For data handling
