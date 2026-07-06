    # Air Quality Data Science Project


    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Load and prepare data
    df = pd.read_csv("Air_Quality.csv")
    df = df.drop(columns=['Message'])
    df = df.dropna()
    df['Start_Date'] = pd.to_datetime(df['Start_Date'])
    df['Year'] = df['Start_Date'].dt.year

    # EDA: Overview
    print("\n--- EDA Overview ---")
    print(df.head())
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Objective 1: KDE Distribution of NO2 Levels
    sns.kdeplot(data=df, x='Data Value', fill=True, color='skyblue')
    plt.title("KDE - NO2 Distribution")
    plt.show()

    # Objective 2: Relplot - NO2 over Time by Top Regions
    top_regions = df['Geo Place Name'].value_counts().head(5).index
    df_top = df[df['Geo Place Name'].isin(top_regions)]
    sns.relplot(data=df_top, x='Start_Date', y='Data Value', hue='Geo Place Name', kind='line', height=5, aspect=2)
    plt.title("Relplot - NO2 Over Time by Top Regions")
    plt.show()

    # Objective 3: Histogram and Line Plot
    sns.histplot(data=df, x='Data Value', bins=30, kde=True, color='green')
    plt.title("Histplot - NO2 Values")
    plt.show()

    sns.lineplot(data=df, x='Year', y='Data Value', estimator='mean', ci=None)
    plt.title("Lineplot - Yearly Average NO2 Levels")
    plt.show()

    # Objective 4: Joint Plot - Year vs NO2
    sns.jointplot(data=df, x='Year', y='Data Value', kind='hex', color='purple')
    plt.suptitle("Joint Plot - Year vs NO2", y=1.02)
    plt.show()

    # Objective 5: Barplot & Pie Chart of Top Regions by Mean NO2
    avg_by_region = df.groupby('Geo Place Name')['Data Value'].mean().sort_values(ascending=False).head(5)
    sns.barplot(x=avg_by_region.index, y=avg_by_region.values, palette='Set2')
    plt.title("Barplot - Avg NO2 by Region")
    plt.xticks(rotation=45)
    plt.show()

    region_counts = df['Geo Place Name'].value_counts().head(5)
    plt.pie(region_counts, labels=region_counts.index, autopct='%1.1f%%')
    plt.title("Pie Chart - Region-wise Record Share")
    plt.show()

    # Objective 6: Box, Swarm, Strip Plot
    sns.boxplot(data=df_top, x='Geo Place Name', y='Data Value')
    plt.title("Boxplot - NO2 Levels by Region")
    plt.xticks(rotation=45)
    plt.show()

    sns.swarmplot(data=df_top, x='Geo Place Name', y='Data Value', palette='Set1')
    plt.title("Swarmplot - NO2 Distribution by Region")
    plt.xticks(rotation=45)
    plt.show()

    sns.stripplot(data=df_top, x='Geo Place Name', y='Data Value', jitter=True, hue='Geo Place Name', dodge=True)
    plt.title("Stripplot - NO2 Variation")
    plt.xticks(rotation=45)
    plt.legend().remove()
    plt.show()

    # Objective 7: Heatmap of Correlation
    corr = df[['Data Value', 'Year']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Heatmap - NO2 vs Year")
    plt.show()


