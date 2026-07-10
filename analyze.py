import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for premium visualizations
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_theme(style="whitegrid")

# Create output directories
os.makedirs('visualizations', exist_ok=True)

csv_filename = 'housing_market_data.csv'

def generate_dataset():
    np.random.seed(42)
    n_samples = 250
    
    neighborhoods = ['Downtown', 'Westside', 'Eastside', 'Suburbs', 'Waterfront']
    neigh_probs = [0.2, 0.25, 0.25, 0.2, 0.1]
    
    selected_neighs = np.random.choice(neighborhoods, size=n_samples, p=neigh_probs)
    
    size_sqft = []
    bedrooms = []
    bathrooms = []
    year_built = []
    school_rating = []
    distance_miles = []
    prices = []
    
    for neigh in selected_neighs:
        # Generate size based on neighborhood characteristics
        if neigh == 'Downtown':
            sqft = int(np.random.normal(1100, 250))
            sqft = max(500, min(sqft, 2500))
            dist = np.random.uniform(0.2, 2.5)
            school = np.random.randint(4, 9)
        elif neigh == 'Waterfront':
            sqft = int(np.random.normal(3200, 600))
            sqft = max(1800, min(sqft, 6000))
            dist = np.random.uniform(5.0, 15.0)
            school = np.random.randint(6, 10)
        elif neigh == 'Suburbs':
            sqft = int(np.random.normal(2600, 500))
            sqft = max(1500, min(sqft, 4500))
            dist = np.random.uniform(10.0, 25.0)
            school = np.random.randint(7, 11)
        elif neigh == 'Westside':
            sqft = int(np.random.normal(1800, 400))
            sqft = max(1000, min(sqft, 3500))
            dist = np.random.uniform(2.0, 8.0)
            school = np.random.randint(5, 9)
        else: # Eastside
            sqft = int(np.random.normal(1500, 350))
            sqft = max(800, min(sqft, 2800))
            dist = np.random.uniform(3.0, 10.0)
            school = np.random.randint(3, 7)
            
        size_sqft.append(sqft)
        distance_miles.append(round(dist, 1))
        school_rating.append(school)
        
        # Bedrooms and Bathrooms correlated with size
        beds = int(max(1, sqft // 650 + np.random.randint(-1, 2)))
        beds = min(beds, 6)
        bedrooms.append(beds)
        
        baths = round(max(1.0, beds * 0.75 + np.random.choice([0, 0.5, 1.0])), 1)
        baths = min(baths, 5.0)
        bathrooms.append(baths)
        
        # Year Built
        year = int(np.random.randint(1950, 2025))
        year_built.append(year)
        
        # Base pricing model
        base_prices = {
            'Waterfront': 600000,
            'Downtown': 450000,
            'Westside': 300000,
            'Suburbs': 250000,
            'Eastside': 190000
        }
        
        price_per_sqft = {
            'Waterfront': 280,
            'Downtown': 240,
            'Westside': 180,
            'Suburbs': 150,
            'Eastside': 130
        }
        
        base = base_prices[neigh]
        sqft_val = sqft * price_per_sqft[neigh]
        school_val = school * 18000
        age_val = (year - 1950) * 2000
        
        price = base + sqft_val + school_val + age_val + np.random.normal(0, 35000)
        prices.append(int(price))

    df = pd.DataFrame({
        'Property_ID': [f'PROP-{1000+i}' for i in range(n_samples)],
        'Neighborhood': selected_neighs,
        'Price_USD': prices,
        'Size_SqFt': size_sqft,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Year_Built': year_built,
        'School_Rating': school_rating,
        'Distance_to_Center_Miles': distance_miles
    })
    
    df.to_csv(csv_filename, index=False)
    print(f"Dataset successfully generated and saved to {csv_filename} ({len(df)} rows).")

def perform_analysis():
    # Load the dataset
    df = pd.read_csv(csv_filename)
    
    print("\n--- DATASET OVERVIEW ---")
    print(df.head().to_string(index=False))
    print("\n--- SHAPE ---")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    # Calculate basic stats (averages)
    avg_price = df['Price_USD'].mean()
    avg_size = df['Size_SqFt'].mean()
    avg_year = df['Year_Built'].mean()
    avg_schools = df['School_Rating'].mean()
    
    print("\n--- BASIC METRICS (AVERAGES) ---")
    print(f"Average Sale Price: ${avg_price:,.2f}")
    print(f"Average Home Size: {avg_size:,.2f} sqft")
    print(f"Average Year Built: {int(avg_year)}")
    print(f"Average School Rating: {avg_schools:.2f} / 10")
    
    # Group by Neighborhood
    print("\n--- NEIGHBORHOOD ANALYSIS ---")
    neighborhood_stats = df.groupby('Neighborhood').agg(
        Average_Price=('Price_USD', 'mean'),
        Average_Size=('Size_SqFt', 'mean'),
        Average_School_Rating=('School_Rating', 'mean'),
        Count=('Property_ID', 'count')
    ).reset_index().sort_values(by='Average_Price', ascending=False)
    print(neighborhood_stats.to_string(index=False, formatters={
        'Average_Price': lambda x: f"${x:,.2f}",
        'Average_Size': lambda x: f"{x:,.1f} sqft",
        'Average_School_Rating': lambda x: f"{x:.2f}"
    }))
    
    # Calculate Average Price by Bedrooms
    print("\n--- BEDROOM ANALYSIS ---")
    bedroom_stats = df.groupby('Bedrooms').agg(
        Average_Price=('Price_USD', 'mean'),
        Average_Size=('Size_SqFt', 'mean'),
        Count=('Property_ID', 'count')
    ).reset_index()
    print(bedroom_stats.to_string(index=False, formatters={
        'Average_Price': lambda x: f"${x:,.2f}",
        'Average_Size': lambda x: f"{x:,.1f} sqft"
    }))
    
    return df, neighborhood_stats

def generate_visualizations(df, neighborhood_stats):
    # Set premium aesthetic color palettes
    colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c']
    palette = sns.color_palette("muted")
    
    # 1. Bar Chart: Average Price by Neighborhood
    plt.figure(figsize=(10, 6))
    
    # Custom colors matching premium style
    bar_colors = ['#1A365D', '#2B6CB0', '#4299E1', '#63B3ED', '#90CDF4'] # Professional blue gradient
    
    bars = plt.bar(
        neighborhood_stats['Neighborhood'],
        neighborhood_stats['Average_Price'] / 1000, # In thousands
        color=bar_colors,
        edgecolor='#4A5568',
        linewidth=1.2,
        width=0.6
    )
    
    # Styling
    plt.title('Average House Price by Neighborhood', fontsize=16, fontweight='bold', pad=15, color='#2D3748')
    plt.xlabel('Neighborhood', fontsize=12, fontweight='bold', labelpad=10, color='#4A5568')
    plt.ylabel('Average Price (USD in Thousands)', fontsize=12, fontweight='bold', labelpad=10, color='#4A5568')
    plt.xticks(fontsize=11, color='#2D3748')
    plt.yticks(fontsize=11, color='#2D3748')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 15,
            f"${height:,.0f}k",
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
            color='#2D3748'
        )
        
    plt.tight_layout()
    plt.savefig('visualizations/bar_chart_neighborhood_prices.png', dpi=300)
    plt.close()
    
    # 2. Scatter Plot: Size vs Price (colored by Neighborhood)
    plt.figure(figsize=(11, 7))
    
    scatter_palette = {
        'Waterfront': '#319795',  # Teal
        'Downtown': '#3182CE',    # Blue
        'Westside': '#DD6B20',    # Orange
        'Suburbs': '#38A169',     # Green
        'Eastside': '#805AD5'     # Purple
    }
    
    sns.scatterplot(
        data=df,
        x='Size_SqFt',
        y=df['Price_USD'] / 1000, # In thousands
        hue='Neighborhood',
        palette=scatter_palette,
        style='Neighborhood',
        s=80,
        alpha=0.85,
        edgecolor='w',
        linewidth=0.8
    )
    
    # Add overall trendline
    x = df['Size_SqFt']
    y = df['Price_USD'] / 1000
    m, b = np.polyfit(x, y, 1)
    plt.plot(
        sorted(x),
        m * np.array(sorted(x)) + b,
        color='#E53E3E',
        linestyle='--',
        linewidth=2,
        label='Overall Market Trend'
    )
    
    # Styling
    plt.title('Property Size vs. Sale Price by Neighborhood', fontsize=16, fontweight='bold', pad=15, color='#2D3748')
    plt.xlabel('Size (Square Feet)', fontsize=12, fontweight='bold', labelpad=10, color='#4A5568')
    plt.ylabel('Sale Price (USD in Thousands)', fontsize=12, fontweight='bold', labelpad=10, color='#4A5568')
    plt.legend(title='Neighborhood & Trends', title_fontsize=11, fontsize=10, loc='upper left', frameon=True, facecolor='white', edgecolor='#E2E8F0')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(df['Size_SqFt'].min() - 100, df['Size_SqFt'].max() + 100)
    plt.ylim(df['Price_USD'].min()/1000 - 50, df['Price_USD'].max()/1000 + 50)
    
    plt.tight_layout()
    plt.savefig('visualizations/scatter_plot_size_vs_price.png', dpi=300)
    plt.close()
    
    # 3. Heatmap: Correlation Matrix of numerical features
    plt.figure(figsize=(10, 8))
    
    # Filter numerical columns
    numerical_cols = ['Price_USD', 'Size_SqFt', 'Bedrooms', 'Bathrooms', 'Year_Built', 'School_Rating', 'Distance_to_Center_Miles']
    corr_matrix = df[numerical_cols].corr()
    
    # Mask to make it look clean (hiding the upper triangle)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Generate a custom diverging colormap (premium feel: icefire or coolwarm-like)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        annot=True,
        fmt='.2f',
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .8, "label": "Correlation Coefficient"},
        annot_kws={"size": 11, "weight": "bold"}
    )
    
    # Adjust labels for readability
    clean_labels = [col.replace('_', ' ') for col in numerical_cols]
    plt.xticks(np.arange(len(numerical_cols)) + 0.5, clean_labels, rotation=45, ha='right', fontsize=11, color='#2D3748')
    plt.yticks(np.arange(len(numerical_cols)) + 0.5, clean_labels, rotation=0, va='center', fontsize=11, color='#2D3748')
    
    plt.title('Correlation Matrix of Housing Features', fontsize=16, fontweight='bold', pad=20, color='#2D3748')
    plt.tight_layout()
    plt.savefig('visualizations/heatmap_correlation.png', dpi=300)
    plt.close()
    
    print("Visualizations successfully generated and saved to the 'visualizations/' directory.")

if __name__ == '__main__':
    # Generate the dataset
    generate_dataset()
    # Perform analysis
    df, neigh_stats = perform_analysis()
    # Generate plots
    generate_visualizations(df, neigh_stats)
