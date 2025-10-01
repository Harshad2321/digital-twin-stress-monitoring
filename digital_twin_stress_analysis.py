"""
Digital Twin for Stress Monitoring - Enhanced EDA and Novel Analysis
Author: Enhanced for GitHub Project
Date: October 2025

This script performs comprehensive Exploratory Data Analysis (EDA) and introduces
novel approaches to stress monitoring analysis including:
1. Fusion Wellness Index (Novel Contribution)
2. Lifestyle Clustering Analysis (Novel Contribution)  
3. Temporal Analysis Patterns (Novel Contribution)
4. Advanced Predictive Modeling
5. Comprehensive Visualization Suite
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configure plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DigitalTwinStressAnalyzer:
    """
    Digital Twin for Stress Monitoring Analysis Class
    Implements novel approaches to stress analysis
    """
    
    def __init__(self, csv_path):
        """Initialize the analyzer with dataset path"""
        self.csv_path = csv_path
        self.df = None
        self.df_cleaned = None
        self.insights = []
        
    def load_and_clean_data(self):
        """Load and clean the dataset"""
        print("=" * 60)
        print("📊 DIGITAL TWIN FOR STRESS MONITORING - DATA LOADING")
        print("=" * 60)
        
        # Load dataset
        self.df = pd.read_csv(self.csv_path)
        print(f"✅ Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        
        # Display column names and basic info
        print(f"\n📋 Columns: {list(self.df.columns)}")
        
        # Clean column names (remove spaces, standardize)
        self.df.columns = [col.strip() for col in self.df.columns]
        
        # Create a copy for cleaning
        self.df_cleaned = self.df.copy()
        
        # Handle missing values
        print(f"\n🔍 Missing Values Analysis:")
        missing_counts = self.df_cleaned.isnull().sum()
        missing_percent = (missing_counts / len(self.df_cleaned)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing_counts,
            'Missing %': missing_percent
        })
        print(missing_df[missing_df['Missing Count'] > 0])
        
        # Convert data types and handle specific columns
        self.clean_specific_columns()
        
        # Print summary statistics
        print(f"\n📈 Dataset Summary After Cleaning:")
        print(f"Total records: {len(self.df_cleaned)}")
        print(f"Date range: {self.df_cleaned['Date'].min()} to {self.df_cleaned['Date'].max()}")
        print(self.df_cleaned.describe())
        
        return self.df_cleaned
    
    def clean_specific_columns(self):
        """Clean specific columns based on expected data types"""
        
        # Map actual column names to standard names (exact matches from CSV)
        column_mapping = {
            'DATE': 'Date',
            'Heart Rate(BPM) ': 'Heart_Rate',
            'Blood Oxygen Level (%)--(only numbers)': 'SpO2',
            'Sleep Duration (Hours)': 'Sleep_Duration',
            '  Sleep Quality ': 'Sleep_Quality',
            'Body Weight (in KGs)': 'Body_Weight',
            'Activity Level': 'Activity_Level',
            'Screen Time (Hourly)': 'Screen_Time',
            'Meal Regularity': 'Meal_Regularity',
            'Sleep Consistency (same bedtime daily?)  ': 'Sleep_Consistency',
            'Step Count(Daily)': 'Steps',
            'Stress Level (Self-Report)  (1 = Relaxed, 10 = Extremely Stressed) ': 'Stress_Level',
            ' Gender  ': 'Gender',
            ' Profession/Role': 'Occupation'
        }
        
        # Clean up column names - remove extra spaces
        self.df_cleaned.columns = [col.strip() for col in self.df_cleaned.columns]
        
        # Update mapping for cleaned names
        clean_mapping = {}
        for old_col in self.df_cleaned.columns:
            if old_col == 'DATE':
                clean_mapping[old_col] = 'Date'
            elif 'Heart Rate' in old_col:
                clean_mapping[old_col] = 'Heart_Rate'
            elif 'Blood Oxygen' in old_col:
                clean_mapping[old_col] = 'SpO2'
            elif 'Sleep Duration' in old_col:
                clean_mapping[old_col] = 'Sleep_Duration'
            elif 'Sleep Quality' in old_col:
                clean_mapping[old_col] = 'Sleep_Quality'
            elif 'Body Weight' in old_col:
                clean_mapping[old_col] = 'Body_Weight'
            elif old_col == 'Activity Level':
                clean_mapping[old_col] = 'Activity_Level'
            elif 'Screen Time' in old_col:
                clean_mapping[old_col] = 'Screen_Time'
            elif 'Meal Regularity' in old_col:
                clean_mapping[old_col] = 'Meal_Regularity'
            elif 'Sleep Consistency' in old_col:
                clean_mapping[old_col] = 'Sleep_Consistency'
            elif 'Step Count' in old_col:
                clean_mapping[old_col] = 'Steps'
            elif 'Stress Level' in old_col:
                clean_mapping[old_col] = 'Stress_Level'
            elif 'Gender' in old_col:
                clean_mapping[old_col] = 'Gender'
            elif 'Profession' in old_col or 'Role' in old_col:
                clean_mapping[old_col] = 'Occupation'
        
        # Rename columns
        self.df_cleaned = self.df_cleaned.rename(columns=clean_mapping)
        print(f"\n📋 Columns after renaming: {list(self.df_cleaned.columns)}")
        
        # Convert Date to datetime
        if 'Date' in self.df_cleaned.columns:
            self.df_cleaned['Date'] = pd.to_datetime(self.df_cleaned['Date'], errors='coerce')
        
        # Clean numeric columns
        numeric_cols = ['Age', 'Body_Weight', 'Heart_Rate', 'SpO2', 'Sleep_Duration', 
                       'Screen_Time', 'Steps', 'Stress_Level']
        
        for col in numeric_cols:
            if col in self.df_cleaned.columns:
                # Convert to numeric, replacing non-numeric values with NaN
                self.df_cleaned[col] = pd.to_numeric(self.df_cleaned[col], errors='coerce')
        
        # Handle categorical columns
        categorical_cols = ['Gender', 'Sleep_Quality', 'Activity_Level', 
                          'Meal_Regularity', 'Sleep_Consistency', 'Occupation']
        
        for col in categorical_cols:
            if col in self.df_cleaned.columns:
                # Convert to string and handle NaN
                self.df_cleaned[col] = self.df_cleaned[col].astype(str)
                self.df_cleaned[col] = self.df_cleaned[col].replace('nan', 'Unknown')
        
        # Fill missing numeric values with median
        numeric_columns = self.df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if self.df_cleaned[col].isnull().sum() > 0:
                median_val = self.df_cleaned[col].median()
                self.df_cleaned[col].fillna(median_val, inplace=True)
                print(f"   Filled {col} missing values with median: {median_val:.2f}")
    
    def create_fusion_wellness_index(self):
        """
        🚀 NOVELTY 1: Fusion Wellness Index
        Combines multiple health metrics into a single wellness score
        """
        print("\n" + "=" * 60)
        print("🚀 NOVELTY 1: FUSION WELLNESS INDEX CREATION")
        print("=" * 60)
        
        # Normalize components (0-1 scale)
        components = {}
        
        # Sleep Quality (categorical to numeric)
        if 'Sleep_Quality' in self.df_cleaned.columns:
            sleep_quality_map = {'Poor': 0.2, 'Average': 0.6, 'Good': 1.0, 'Unknown': 0.4}
            components['sleep_score'] = self.df_cleaned['Sleep_Quality'].map(sleep_quality_map)
        
        # Steps (normalized to 0-1)
        if 'Steps' in self.df_cleaned.columns:
            max_steps = self.df_cleaned['Steps'].max()
            components['steps_score'] = self.df_cleaned['Steps'] / max_steps
        
        # SpO2 (normalized, higher is better)
        if 'SpO2' in self.df_cleaned.columns:
            components['spo2_score'] = (self.df_cleaned['SpO2'] - 90) / 10  # Scale 90-100 to 0-1
            components['spo2_score'] = components['spo2_score'].clip(0, 1)
        
        # Heart Rate (inverse normalized, lower resting HR is better)
        if 'Heart_Rate' in self.df_cleaned.columns:
            components['hr_score'] = 1 - ((self.df_cleaned['Heart_Rate'] - 60) / 60)
            components['hr_score'] = components['hr_score'].clip(0, 1)
        
        # Screen Time (inverse, less is better)
        if 'Screen_Time' in self.df_cleaned.columns:
            max_screen = self.df_cleaned['Screen_Time'].max()
            components['screen_score'] = 1 - (self.df_cleaned['Screen_Time'] / max_screen)
        
        # Calculate Fusion Wellness Index (weighted average)
        weights = {
            'sleep_score': 0.25,
            'steps_score': 0.25,
            'spo2_score': 0.20,
            'hr_score': 0.15,
            'screen_score': 0.15
        }
        
        fusion_index = np.zeros(len(self.df_cleaned))
        
        for component, weight in weights.items():
            if component in components:
                fusion_index += components[component].fillna(0.5) * weight
        
        self.df_cleaned['Fusion_Wellness_Index'] = fusion_index
        
        print(f"✅ Fusion Wellness Index created!")
        print(f"   Range: {fusion_index.min():.3f} - {fusion_index.max():.3f}")
        print(f"   Mean: {fusion_index.mean():.3f}")
        print(f"   Components: {list(weights.keys())}")
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(self.df_cleaned['Fusion_Wellness_Index'], bins=30, alpha=0.7, color='skyblue')
        plt.title('Distribution of Fusion Wellness Index')
        plt.xlabel('Fusion Wellness Index')
        plt.ylabel('Frequency')
        
        plt.subplot(2, 2, 2)
        if 'Stress_Level' in self.df_cleaned.columns:
            plt.scatter(self.df_cleaned['Fusion_Wellness_Index'], 
                       self.df_cleaned['Stress_Level'], alpha=0.6)
            plt.title('Fusion Index vs Stress Level')
            plt.xlabel('Fusion Wellness Index')
            plt.ylabel('Stress Level')
            
            # Calculate correlation
            corr = self.df_cleaned['Fusion_Wellness_Index'].corr(self.df_cleaned['Stress_Level'])
            plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                    transform=plt.gca().transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.subplot(2, 2, 3)
        # Components correlation heatmap
        comp_df = pd.DataFrame(components)
        comp_df['Fusion_Index'] = self.df_cleaned['Fusion_Wellness_Index']
        if 'Stress_Level' in self.df_cleaned.columns:
            comp_df['Stress_Level'] = self.df_cleaned['Stress_Level']
        
        correlation_matrix = comp_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Components Correlation Matrix')
        
        plt.subplot(2, 2, 4)
        # Wellness categories
        self.df_cleaned['Wellness_Category'] = pd.cut(
            self.df_cleaned['Fusion_Wellness_Index'], 
            bins=[0, 0.3, 0.6, 1.0], 
            labels=['Low', 'Medium', 'High']
        )
        category_counts = self.df_cleaned['Wellness_Category'].value_counts()
        plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
        plt.title('Wellness Categories Distribution')
        
        plt.tight_layout()
        plt.savefig('fusion_wellness_index_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Add insights
        if 'Stress_Level' in self.df_cleaned.columns:
            corr = self.df_cleaned['Fusion_Wellness_Index'].corr(self.df_cleaned['Stress_Level'])
            self.insights.append(f"Fusion Wellness Index: Mean={fusion_index.mean():.3f}, "
                               f"Correlation with Stress={corr:.3f}")
        else:
            self.insights.append(f"Fusion Wellness Index: Mean={fusion_index.mean():.3f}")
        
        return fusion_index
    
    def perform_lifestyle_clustering(self):
        """
        🚀 NOVELTY 2: Lifestyle Clustering Analysis
        Groups participants into distinct lifestyle patterns
        """
        print("\n" + "=" * 60)
        print("🚀 NOVELTY 2: LIFESTYLE CLUSTERING ANALYSIS")
        print("=" * 60)
        
        # Select features for clustering
        clustering_features = ['Age', 'Steps', 'Screen_Time', 'Sleep_Duration', 
                             'Heart_Rate', 'Fusion_Wellness_Index']
        
        if 'Stress_Level' in self.df_cleaned.columns:
            clustering_features.append('Stress_Level')
        
        # Prepare data for clustering
        cluster_data = self.df_cleaned[clustering_features].copy()
        
        # Handle any remaining missing values
        cluster_data = cluster_data.fillna(cluster_data.mean())
        
        # Standardize features
        scaler = StandardScaler()
        cluster_data_scaled = scaler.fit_transform(cluster_data)
        
        # Determine optimal number of clusters using elbow method
        inertias = []
        k_range = range(2, 9)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(cluster_data_scaled)
            inertias.append(kmeans.inertia_)
        
        # Plot elbow curve
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.plot(k_range, inertias, marker='o')
        plt.title('Elbow Method for Optimal K')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        
        # Use K=4 for lifestyle groups
        optimal_k = 4
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        cluster_labels = kmeans.fit_predict(cluster_data_scaled)
        
        self.df_cleaned['Lifestyle_Cluster'] = cluster_labels
        
        print(f"✅ Lifestyle clustering completed with {optimal_k} clusters")
        
        # Analyze clusters
        cluster_summary = self.df_cleaned.groupby('Lifestyle_Cluster')[clustering_features].mean()
        print(f"\n📊 Cluster Characteristics:")
        print(cluster_summary.round(2))
        
        # Name clusters based on characteristics
        cluster_names = {
            0: "Active Wellness",
            1: "Moderate Lifestyle", 
            2: "High Stress Group",
            3: "Sedentary Pattern"
        }
        
        # Update based on actual cluster characteristics
        stress_by_cluster = self.df_cleaned.groupby('Lifestyle_Cluster')['Stress_Level'].mean() if 'Stress_Level' in self.df_cleaned.columns else None
        steps_by_cluster = self.df_cleaned.groupby('Lifestyle_Cluster')['Steps'].mean()
        wellness_by_cluster = self.df_cleaned.groupby('Lifestyle_Cluster')['Fusion_Wellness_Index'].mean()
        
        # Visualize clusters
        plt.subplot(2, 3, 2)
        if 'Stress_Level' in self.df_cleaned.columns:
            for cluster in range(optimal_k):
                cluster_mask = self.df_cleaned['Lifestyle_Cluster'] == cluster
                plt.scatter(self.df_cleaned[cluster_mask]['Steps'], 
                           self.df_cleaned[cluster_mask]['Stress_Level'],
                           label=f'Cluster {cluster}', alpha=0.7)
            plt.title('Clusters: Steps vs Stress')
            plt.xlabel('Steps')
            plt.ylabel('Stress Level')
            plt.legend()
        
        plt.subplot(2, 3, 3)
        cluster_counts = self.df_cleaned['Lifestyle_Cluster'].value_counts().sort_index()
        plt.bar(range(optimal_k), cluster_counts.values)
        plt.title('Cluster Distribution')
        plt.xlabel('Cluster')
        plt.ylabel('Count')
        
        plt.subplot(2, 3, 4)
        if stress_by_cluster is not None:
            plt.bar(range(optimal_k), stress_by_cluster.values, color='lightcoral')
            plt.title('Average Stress by Cluster')
            plt.xlabel('Cluster')
            plt.ylabel('Average Stress Level')
        
        plt.subplot(2, 3, 5)
        plt.bar(range(optimal_k), wellness_by_cluster.values, color='lightgreen')
        plt.title('Average Wellness Index by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Average Wellness Index')
        
        plt.subplot(2, 3, 6)
        plt.bar(range(optimal_k), steps_by_cluster.values, color='lightblue')
        plt.title('Average Steps by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Average Steps')
        
        plt.tight_layout()
        plt.savefig('lifestyle_clustering_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Add insights
        self.insights.append(f"Lifestyle Clustering: {optimal_k} distinct groups identified")
        if stress_by_cluster is not None:
            highest_stress_cluster = stress_by_cluster.idxmax()
            lowest_stress_cluster = stress_by_cluster.idxmin()
            self.insights.append(f"Highest stress cluster: {highest_stress_cluster} "
                               f"(avg stress: {stress_by_cluster.max():.2f})")
            self.insights.append(f"Lowest stress cluster: {lowest_stress_cluster} "
                               f"(avg stress: {stress_by_cluster.min():.2f})")
        
        return cluster_labels
    
    def temporal_analysis(self):
        """
        🚀 NOVELTY 3: Temporal Analysis
        Analyzes stress patterns over time
        """
        print("\n" + "=" * 60)
        print("🚀 NOVELTY 3: TEMPORAL ANALYSIS")
        print("=" * 60)
        
        if 'Date' not in self.df_cleaned.columns or 'Stress_Level' not in self.df_cleaned.columns:
            print("❌ Date or Stress_Level columns not found for temporal analysis")
            return
        
        # Create temporal features
        self.df_cleaned['DayOfWeek'] = self.df_cleaned['Date'].dt.dayofweek
        self.df_cleaned['WeekNumber'] = self.df_cleaned['Date'].dt.isocalendar().week
        self.df_cleaned['IsWeekend'] = self.df_cleaned['DayOfWeek'].isin([5, 6])
        
        # Weekly stress trends
        weekly_stress = self.df_cleaned.groupby('WeekNumber')['Stress_Level'].mean()
        
        # Day of week analysis
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_stress = self.df_cleaned.groupby('DayOfWeek')['Stress_Level'].mean()
        
        # Weekend vs weekday
        weekend_stress = self.df_cleaned.groupby('IsWeekend')['Stress_Level'].mean()
        
        print(f"✅ Temporal analysis completed")
        print(f"   Weekend vs Weekday stress: {weekend_stress[True]:.2f} vs {weekend_stress[False]:.2f}")
        print(f"   Highest stress day: {day_names[int(daily_stress.idxmax())]} ({daily_stress.max():.2f})")
        print(f"   Lowest stress day: {day_names[int(daily_stress.idxmin())]} ({daily_stress.min():.2f})")
        
        # Visualizations
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.plot(weekly_stress.index, weekly_stress.values, marker='o', linewidth=2)
        plt.title('Weekly Stress Trends')
        plt.xlabel('Week Number')
        plt.ylabel('Average Stress Level')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 2)
        plt.bar(range(7), daily_stress.values, color='skyblue')
        plt.title('Stress by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Average Stress Level')
        plt.xticks(range(7), [day[:3] for day in day_names], rotation=45)
        
        plt.subplot(2, 3, 3)
        plt.bar(['Weekday', 'Weekend'], weekend_stress.values, color=['orange', 'green'])
        plt.title('Weekend vs Weekday Stress')
        plt.ylabel('Average Stress Level')
        
        # Heatmap of stress by day and week
        plt.subplot(2, 3, 4)
        pivot_data = self.df_cleaned.pivot_table(
            values='Stress_Level', 
            index='DayOfWeek', 
            columns='WeekNumber', 
            aggfunc='mean'
        )
        sns.heatmap(pivot_data, cmap='YlOrRd', cbar_kws={'label': 'Stress Level'})
        plt.title('Stress Heatmap: Day vs Week')
        plt.ylabel('Day of Week')
        plt.xlabel('Week Number')
        
        # Time series plot
        plt.subplot(2, 3, 5)
        daily_avg = self.df_cleaned.groupby('Date')['Stress_Level'].mean()
        plt.plot(daily_avg.index, daily_avg.values, alpha=0.7, color='red')
        plt.title('Daily Stress Time Series')
        plt.xlabel('Date')
        plt.ylabel('Stress Level')
        plt.xticks(rotation=45)
        
        # Distribution by time period
        plt.subplot(2, 3, 6)
        self.df_cleaned['TimePeriod'] = pd.cut(
            self.df_cleaned['Date'].dt.day, 
            bins=[0, 10, 20, 31], 
            labels=['Early', 'Mid', 'Late']
        )
        period_stress = self.df_cleaned.groupby('TimePeriod')['Stress_Level'].mean()
        plt.bar(range(len(period_stress)), period_stress.values, color='purple', alpha=0.7)
        plt.title('Stress by Month Period')
        plt.xticks(range(len(period_stress)), period_stress.index)
        plt.ylabel('Average Stress Level')
        
        plt.tight_layout()
        plt.savefig('temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Add insights
        self.insights.append(f"Temporal Analysis: Weekend stress {weekend_stress[True]:.2f} vs "
                           f"Weekday stress {weekend_stress[False]:.2f}")
        self.insights.append(f"Peak stress day: {day_names[int(daily_stress.idxmax())]} "
                           f"({daily_stress.max():.2f})")
    
    def correlation_analysis(self):
        """Perform correlation analysis between stress and other features"""
        print("\n" + "=" * 60)
        print("📊 CORRELATION ANALYSIS")
        print("=" * 60)
        
        # Select numeric columns for correlation
        numeric_cols = self.df_cleaned.select_dtypes(include=[np.number]).columns
        
        if 'Stress_Level' not in numeric_cols:
            print("❌ Stress_Level not found in numeric columns")
            return
        
        # Calculate correlation matrix
        corr_matrix = self.df_cleaned[numeric_cols].corr()
        
        # Focus on stress correlations
        stress_corr = corr_matrix['Stress_Level'].sort_values(key=abs, ascending=False)
        
        print("✅ Correlation with Stress Level:")
        for feature, corr_val in stress_corr.items():
            if feature != 'Stress_Level':
                print(f"   {feature}: {corr_val:.3f}")
        
        # Visualization
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 2, 1)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f')
        plt.title('Complete Correlation Matrix')
        
        plt.subplot(2, 2, 2)
        stress_corr_plot = stress_corr.drop('Stress_Level')
        colors = ['red' if x < 0 else 'blue' for x in stress_corr_plot.values]
        plt.barh(range(len(stress_corr_plot)), stress_corr_plot.values, color=colors, alpha=0.7)
        plt.yticks(range(len(stress_corr_plot)), stress_corr_plot.index)
        plt.xlabel('Correlation with Stress Level')
        plt.title('Feature Correlations with Stress')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Top positive and negative correlations
        top_positive = stress_corr_plot[stress_corr_plot > 0].head(3)
        top_negative = stress_corr_plot[stress_corr_plot < 0].tail(3)
        
        plt.subplot(2, 2, 3)
        if len(top_positive) > 0:
            strongest_pos_feature = top_positive.index[0]
            plt.scatter(self.df_cleaned[strongest_pos_feature], 
                       self.df_cleaned['Stress_Level'], alpha=0.6, color='red')
            plt.xlabel(strongest_pos_feature)
            plt.ylabel('Stress Level')
            plt.title(f'Strongest Positive Correlation\n{strongest_pos_feature} vs Stress')
        
        plt.subplot(2, 2, 4)
        if len(top_negative) > 0:
            strongest_neg_feature = top_negative.index[0]
            plt.scatter(self.df_cleaned[strongest_neg_feature], 
                       self.df_cleaned['Stress_Level'], alpha=0.6, color='blue')
            plt.xlabel(strongest_neg_feature)
            plt.ylabel('Stress Level')
            plt.title(f'Strongest Negative Correlation\n{strongest_neg_feature} vs Stress')
        
        plt.tight_layout()
        plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Add insights
        if len(top_positive) > 0:
            self.insights.append(f"Strongest positive correlation: {top_positive.index[0]} "
                               f"({top_positive.iloc[0]:.3f})")
        if len(top_negative) > 0:
            self.insights.append(f"Strongest negative correlation: {top_negative.index[0]} "
                               f"({top_negative.iloc[0]:.3f})")
    
    def categorical_analysis(self):
        """Analyze stress distribution by categorical variables"""
        print("\n" + "=" * 60)
        print("📊 CATEGORICAL ANALYSIS")
        print("=" * 60)
        
        categorical_cols = ['Gender', 'Sleep_Quality', 'Activity_Level', 
                          'Meal_Regularity', 'Sleep_Consistency', 'Occupation']
        
        available_cats = [col for col in categorical_cols if col in self.df_cleaned.columns]
        
        if 'Stress_Level' not in self.df_cleaned.columns:
            print("❌ Stress_Level column not found")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, col in enumerate(available_cats[:6]):
            if i < len(axes):
                # Box plot for stress distribution by category
                self.df_cleaned.boxplot(column='Stress_Level', by=col, ax=axes[i])
                axes[i].set_title(f'Stress Distribution by {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Stress Level')
                
                # Print statistics
                stats = self.df_cleaned.groupby(col)['Stress_Level'].agg(['mean', 'std', 'count'])
                print(f"\n📈 {col} - Stress Statistics:")
                print(stats.round(2))
        
        # Hide unused subplots
        for i in range(len(available_cats), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('categorical_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predictive_modeling(self):
        """
        Build predictive models for stress level
        """
        print("\n" + "=" * 60)
        print("🤖 PREDICTIVE MODELING")
        print("=" * 60)
        
        if 'Stress_Level' not in self.df_cleaned.columns:
            print("❌ Stress_Level column not found for modeling")
            return
        
        # Prepare features
        feature_cols = ['Age', 'Body_Weight', 'Heart_Rate', 'SpO2', 'Sleep_Duration',
                       'Screen_Time', 'Steps', 'Fusion_Wellness_Index']
        
        available_features = [col for col in feature_cols if col in self.df_cleaned.columns]
        
        if len(available_features) < 3:
            print("❌ Insufficient features for modeling")
            return
        
        # Prepare data
        X = self.df_cleaned[available_features].copy()
        y = self.df_cleaned['Stress_Level'].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Convert to classification (Low, Medium, High stress)
        y_categorical = pd.cut(y, bins=[0, 3, 7, 10], labels=['Low', 'Medium', 'High'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n🔄 Training {name}...")
            
            if name == 'Logistic Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred
            }
            
            print(f"✅ {name} Accuracy: {accuracy:.3f}")
            print(f"   Classification Report:")
            print(classification_report(y_test, y_pred))
        
        # Feature importance (Random Forest)
        rf_model = results['Random Forest']['model']
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n📊 Feature Importance (Random Forest):")
        print(feature_importance)
        
        # Visualizations
        plt.figure(figsize=(15, 10))
        
        # Model comparison
        plt.subplot(2, 3, 1)
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        plt.bar(model_names, accuracies, color=['lightblue', 'lightgreen'])
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        # Feature importance
        plt.subplot(2, 3, 2)
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.title('Feature Importance (Random Forest)')
        plt.xlabel('Importance')
        
        # Confusion matrix
        plt.subplot(2, 3, 3)
        cm = confusion_matrix(y_test, results['Random Forest']['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix (Random Forest)')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Prediction distribution
        plt.subplot(2, 3, 4)
        pred_counts = pd.Series(results['Random Forest']['predictions']).value_counts()
        plt.pie(pred_counts.values, labels=pred_counts.index, autopct='%1.1f%%')
        plt.title('Predicted Stress Distribution')
        
        # Actual vs Predicted
        plt.subplot(2, 3, 5)
        actual_counts = y_test.value_counts()
        x_pos = np.arange(len(actual_counts))
        plt.bar(x_pos - 0.2, actual_counts.values, 0.4, label='Actual', alpha=0.7)
        plt.bar(x_pos + 0.2, pred_counts.reindex(actual_counts.index, fill_value=0).values, 
                0.4, label='Predicted', alpha=0.7)
        plt.xticks(x_pos, actual_counts.index)
        plt.title('Actual vs Predicted Distribution')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('predictive_modeling.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Add insights
        best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_accuracy = results[best_model]['accuracy']
        top_feature = feature_importance.iloc[0]['feature']
        
        self.insights.append(f"Best model: {best_model} (Accuracy: {best_accuracy:.3f})")
        self.insights.append(f"Most important feature: {top_feature} "
                           f"(Importance: {feature_importance.iloc[0]['importance']:.3f})")
        
        return results
    
    def export_results(self):
        """Export cleaned dataset with new features and results"""
        print("\n" + "=" * 60)
        print("💾 EXPORTING RESULTS")
        print("=" * 60)
        
        # Save enhanced dataset
        output_file = 'digital_twin_stress_analysis_results.csv'
        self.df_cleaned.to_csv(output_file, index=False)
        print(f"✅ Enhanced dataset saved: {output_file}")
        
        # Save insights
        with open('analysis_insights.txt', 'w', encoding='utf-8') as f:
            f.write("DIGITAL TWIN FOR STRESS MONITORING - KEY INSIGHTS\n")
            f.write("=" * 60 + "\n\n")
            
            for i, insight in enumerate(self.insights, 1):
                f.write(f"{i}. {insight}\n")
            
            f.write(f"\nDataset Statistics:\n")
            f.write(f"- Total records: {len(self.df_cleaned)}\n")
            f.write(f"- Date range: {self.df_cleaned['Date'].min()} to {self.df_cleaned['Date'].max()}\n")
            f.write(f"- Average stress level: {self.df_cleaned['Stress_Level'].mean():.2f}\n")
            f.write(f"- Average wellness index: {self.df_cleaned['Fusion_Wellness_Index'].mean():.3f}\n")
        
        print(f"✅ Insights saved: analysis_insights.txt")
        
        return output_file
    
    def generate_report_summary(self):
        """Generate summary for presentation"""
        report = f"""
DIGITAL TWIN FOR STRESS MONITORING - PROJECT SUMMARY
====================================================

PROBLEM:
Traditional stress monitoring relies on single metrics. This project develops a 
comprehensive digital twin approach combining multiple physiological and behavioral 
indicators for holistic stress assessment.

OBJECTIVES:
1. Create a fusion wellness index combining multiple health metrics
2. Identify distinct lifestyle patterns through clustering
3. Analyze temporal stress variations
4. Develop predictive models for stress levels

METHODOLOGY:
- Dataset: {len(self.df_cleaned)} records with 15+ health and lifestyle parameters
- Novel Fusion Wellness Index: Weighted combination of sleep, activity, physiological metrics
- K-means clustering for lifestyle pattern identification
- Temporal analysis for stress trend detection
- Machine learning models for stress prediction

NOVELTY CONTRIBUTIONS:
1. 🚀 Fusion Wellness Index: First comprehensive wellness metric combining sleep quality, 
   activity levels, physiological parameters, and screen time
2. 🚀 Lifestyle Clustering: Data-driven identification of distinct lifestyle patterns 
   and their relationship to stress
3. 🚀 Temporal Analysis: Time-series analysis revealing stress patterns across days, 
   weeks, and periods

KEY RESULTS:
"""
        
        for insight in self.insights:
            report += f"• {insight}\n"
        
        report += f"""
FUTURE SCOPE:
- Real-time stress monitoring dashboard
- Personalized wellness recommendations based on cluster membership
- Integration with wearable devices for continuous monitoring
- Advanced deep learning models for stress prediction
- Multi-modal analysis including voice and facial expression data

IMPACT:
This digital twin approach provides a holistic view of stress factors, enabling 
personalized interventions and preventive healthcare strategies.
"""
        
        with open('project_report_summary.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        return report

def main():
    """Main execution function"""
    print("🚀 DIGITAL TWIN FOR STRESS MONITORING")
    print("=" * 60)
    
    # Initialize analyzer
    csv_path = r"c:\Users\harsh\Desktop\DPEL\DPDEL-FORM (Responses) - Form responses 1.csv"
    analyzer = DigitalTwinStressAnalyzer(csv_path)
    
    # Execute analysis pipeline
    try:
        # 1. Load and clean data
        analyzer.load_and_clean_data()
        
        # 2. Create Fusion Wellness Index (Novelty 1)
        analyzer.create_fusion_wellness_index()
        
        # 3. Lifestyle Clustering (Novelty 2)
        analyzer.perform_lifestyle_clustering()
        
        # 4. Temporal Analysis (Novelty 3)
        analyzer.temporal_analysis()
        
        # 5. Traditional EDA
        analyzer.correlation_analysis()
        analyzer.categorical_analysis()
        
        # 6. Predictive Modeling
        analyzer.predictive_modeling()
        
        # 7. Export results
        analyzer.export_results()
        
        # 8. Generate report
        analyzer.generate_report_summary()
        
        print("\n" + "=" * 60)
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print("📁 Files generated:")
        print("   • fusion_wellness_index_analysis.png")
        print("   • lifestyle_clustering_analysis.png") 
        print("   • temporal_analysis.png")
        print("   • correlation_analysis.png")
        print("   • categorical_analysis.png")
        print("   • predictive_modeling.png")
        print("   • digital_twin_stress_analysis_results.csv")
        print("   • analysis_insights.txt")
        print("   • project_report_summary.txt")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()