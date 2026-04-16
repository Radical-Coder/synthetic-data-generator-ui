#!/usr/bin/env python
# coding: utf-8

"""
Automatic Synthetic Data Generator with Type Detection
========================================================
This module provides intelligent synthetic data generation with automatic column type detection.
It uses Gaussian Mixture Models (GMM) for continuous data and categorical sampling for discrete data.

Supported Data Types:
- Continuous Numeric: GMM-based generation with skewness handling
- Discrete Numeric: Categorical sampling preserving exact frequency distribution
- Categorical: Frequency-based sampling with optional reference table support
- Boolean: Binary sampling preserving True/False ratio
- DateTime: Range-preserving generation with business day and hour pattern detection
- Email: Realistic fake email generation preserving domain distribution
- Phone: Fake phone number generation preserving original format (parentheses, dashes, plain, international)

Features:
- Auto-detection of all column types — no manual labeling required
- Reference table support for categorical columns
- Quality validation with warnings
- JSON and CSV input support

Author: Robel
Version: 1.1
Last Updated: 2026
"""

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
import random
import string
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')


class AutoDetectingSyntheticGenerator:
    """
    Automatically detects column types and generates synthetic data that preserves
    statistical properties of the original dataset.
    
    Features:
    - Auto-detection of 7 data types: continuous numeric, discrete numeric, categorical,
      boolean, datetime, email, and phone number
    - GMM-based generation for continuous numeric data with skewness handling
    - Frequency-based categorical sampling preserving exact distributions
    - DateTime generation with business day and hour pattern preservation
    - Realistic email generation preserving domain distribution from original data
    - Phone number generation preserving original format (parentheses, dashes, plain, international)
    - Reference table support for categorical columns
    - Quality validation with automatic warnings
    
    Attributes:
        max_gmm_components (int): Maximum number of Gaussian components for GMM fitting
        random_state (int): Random seed for reproducibility
        discrete_threshold (float): Threshold for classifying numeric columns as discrete
        categorical_libraries (dict): Custom categorical value libraries for specific columns
        column_info (dict): Stores detected type information for each column
    """
    
    def __init__(self, max_gmm_components=3, random_state=42, discrete_threshold=0.05,
                 categorical_libraries=None):
        """
        Initialize the synthetic data generator.
        
        Args:
            max_gmm_components (int): Maximum Gaussian components to try (1 to this value)
            random_state (int): Random seed for reproducibility
            discrete_threshold (float): Ratio below which numeric columns are treated as discrete
            categorical_libraries (dict): Optional dict of {column_name: [valid_values]}
        """
        self.max_gmm_components = max_gmm_components
        self.random_state = random_state
        self.discrete_threshold = discrete_threshold
        self.categorical_libraries = categorical_libraries or {}
        self.column_info = {}
        np.random.seed(random_state)
    
    def detect_column_type(self, column_name, series):
        """
        Automatically detect the type and generation method for a column.
        
        Detection logic:
        1. Email columns → realistic fake email generation
        2. Phone columns → fake phone number generation preserving format
        3. DateTime columns → temporal pattern generation
        4. Boolean columns → categorical sampling
        5. Numeric with low unique ratio → discrete numeric (categorical sampling)
        6. Numeric with high unique ratio → continuous numeric (GMM)
        7. Object/categorical dtype → categorical sampling
        
        Args:
            column_name (str): Name of the column
            series (pd.Series): The column data
            
        Returns:
            dict: Column metadata including type, method, and distribution parameters
        """
        # Remove null values for analysis
        clean_series = series.dropna()
        
        # Handle empty columns
        if len(clean_series) == 0:
            return {"type": "empty", "method": "skip"}
        
        # Calculate uniqueness metrics
        unique_count = clean_series.nunique()
        total_count = len(clean_series)
        unique_ratio = unique_count / total_count
        dtype = clean_series.dtype
        
        # EMAIL DETECTION — works for object, str, and string dtypes (pandas 2 and 3)
        is_string_col = (pd.api.types.is_object_dtype(clean_series) or 
                         clean_series.dtype.name in ['object', 'str', 'string', 'StringDtype'] or
                         str(clean_series.dtype) in ['object', 'str', 'string'])
        if is_string_col:
            try:
                sample = clean_series.dropna().head(100).astype(str)
                if len(sample) > 0:
                    email_matches = sum(1 for x in sample if re.match(r'^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$', x.strip()))
                    if email_matches / len(sample) > 0.7:
                        domains = pd.Series([x.split('@')[-1] if '@' in x else 'example.com' for x in sample]).value_counts(normalize=True)
                        return {
                            "type": "email",
                            "method": "email_generation",
                            "domains": domains
                        }
                    # PHONE NUMBER DETECTION
                    phone_matches = sum(1 for x in sample if re.match(r'^[\+\(]?[\d\s\-\(\)\.]{7,20}$', x.strip()))
                    if phone_matches / len(sample) > 0.7:
                        return {
                            "type": "phone",
                            "method": "phone_generation",
                            "sample_formats": sample.tolist()[:10]
                        }
            except Exception:
                pass

        # DATETIME DETECTION
        if pd.api.types.is_datetime64_any_dtype(clean_series):
            return self._detect_datetime_pattern(clean_series)
        
        # Try to parse as datetime if object type
        if pd.api.types.is_object_dtype(clean_series):
            try:
                # Attempt to parse as datetime
                parsed_dates = pd.to_datetime(clean_series, errors='coerce')
                # If more than 80% successfully parsed, treat as datetime
                if parsed_dates.notna().sum() / len(clean_series) > 0.8:
                    return self._detect_datetime_pattern(parsed_dates.dropna())
            except:
                pass  # Not a datetime column
        
        # BOOLEAN DETECTION
        if dtype == bool or pd.api.types.is_bool_dtype(clean_series):
            categories = clean_series.unique()
            distribution = clean_series.value_counts(normalize=True)
            return {
                "type": "boolean",
                "method": "categorical_sampling",
                "categories": categories,
                "distribution": distribution
            }
        
        # Check for boolean-like columns (only 2 unique values)
        if unique_count == 2 and not pd.api.types.is_numeric_dtype(clean_series):
            categories = clean_series.unique()
            distribution = clean_series.value_counts(normalize=True)
            return {
                "type": "boolean",
                "method": "categorical_sampling",
                "categories": categories,
                "distribution": distribution
            }
        
        # NUMERIC DETECTION
        if pd.api.types.is_numeric_dtype(clean_series):
            # Low unique ratio indicates discrete values (e.g., ratings, counts)
            if unique_ratio < self.discrete_threshold:
                values = clean_series.unique()
                distribution = clean_series.value_counts(normalize=True)
                return {
                    "type": "discrete_numeric",
                    "method": "categorical_sampling",
                    "values": values,
                    "distribution": distribution
                }
            else:
                # Continuous numeric - use GMM
                skewness = clean_series.skew()
                is_skewed = abs(skewness) > 1
                return {
                    "type": "continuous_numeric",
                    "method": "gmm",
                    "skewed": is_skewed and (clean_series.min() > 0),  # Only log-transform if positive
                    "min": clean_series.min(),
                    "max": clean_series.max(),
                    "mean": clean_series.mean(),
                    "std": clean_series.std(),
                    "skewness": skewness
                }
        
        # CATEGORICAL DETECTION
        elif pd.api.types.is_object_dtype(clean_series) or pd.api.types.is_categorical_dtype(clean_series):
            categories = clean_series.unique()
            distribution = clean_series.value_counts(normalize=True)
            return {
                "type": "categorical",
                "method": "categorical_sampling",
                "categories": categories,
                "distribution": distribution
            }
        
        # Unknown type - skip generation
        return {"type": "unknown", "method": "skip"}
    
    def _detect_datetime_pattern(self, datetime_series):
        """
        Detect patterns in datetime data for realistic generation.
        
        Args:
            datetime_series (pd.Series): Series of datetime values
            
        Returns:
            dict: DateTime metadata including range, patterns, and distribution info
        """
        min_date = datetime_series.min()
        max_date = datetime_series.max()
        date_range_days = (max_date - min_date).days
        
        # Detect common patterns
        hour_dist = datetime_series.dt.hour.value_counts(normalize=True).to_dict()
        dow_dist = datetime_series.dt.dayofweek.value_counts(normalize=True).to_dict()
        
        # Check if dates are business days only
        is_business_days = (datetime_series.dt.dayofweek < 5).mean() > 0.95
        
        return {
            "type": "datetime",
            "method": "datetime_generation",
            "min_date": min_date,
            "max_date": max_date,
            "date_range_days": date_range_days,
            "hour_distribution": hour_dist,
            "day_of_week_distribution": dow_dist,
            "is_business_days": is_business_days,
            "original_format": str(datetime_series.iloc[0])
        }
    
    def fit(self, df):
        """
        Analyze the dataset and detect column types.
        
        This method must be called before generate(). It examines each column
        and determines the appropriate generation strategy.
        
        Args:
            df (pd.DataFrame): The source dataset to analyze
            
        Returns:
            self: For method chaining
        """
        print(f"Analyzing dataset...")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}\n")
        
        print("Column Detection:")
        for col in df.columns:
            # Detect type for each column
            info = self.detect_column_type(col, df[col])
            self.column_info[col] = info
            
            col_type = info['type']
            method = info.get('method', 'skip')
            
            # Print detection results with appropriate formatting
            if col_type == 'continuous_numeric':
                skew_note = " (log transform)" if info.get('skewed', False) else ""
                print(f"  {col:20s} → {col_type:20s} ({method}{skew_note})")
            elif col_type == 'datetime':
                date_range = f"{info['date_range_days']} days"
                biz_note = " (business days)" if info.get('is_business_days', False) else ""
                print(f"  {col:20s} → {col_type:20s} ({method}, range: {date_range}{biz_note})")
            elif col_type in ['discrete_numeric', 'categorical', 'boolean']:
                print(f"  {col:20s} → {col_type:20s} ({method})")
            else:
                print(f"  {col:20s} → {col_type:20s} (skipping)")
        
        usable_cols = len([c for c in self.column_info.values() if c['type'] != 'unknown'])
        print(f"\nAnalysis complete! Detected {usable_cols} usable columns.")
        return self
    
    def best_gmm_fit(self, values):
        """
        Find the optimal number of Gaussian components using BIC criterion.
        
        Tests 1 to max_gmm_components and selects the model with lowest BIC.
        BIC (Bayesian Information Criterion) balances model fit with complexity.
        
        Args:
            values (np.array): The data to fit
            
        Returns:
            GaussianMixture: The best-fitting GMM model
        """
        best_gmm = None
        best_bic = np.inf
        
        # Try different numbers of components
        for n_components in range(1, self.max_gmm_components + 1):
            gmm = GaussianMixture(n_components=n_components, random_state=self.random_state)
            gmm.fit(values.reshape(-1, 1))
            bic = gmm.bic(values.reshape(-1, 1))
            
            # Lower BIC is better
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
        
        return best_gmm
    
    def generate_continuous_numeric(self, seed_series, rows, column_name):
        """
        Generate synthetic continuous numeric data using Gaussian Mixture Models.
        
        For skewed distributions (when min > 0), applies log transformation before
        fitting GMM to better capture the distribution shape.
        
        Args:
            seed_series (pd.Series): Original data to learn from
            rows (int): Number of synthetic rows to generate
            column_name (str): Name of the column being generated
            
        Returns:
            np.array: Synthetic numeric values
        """
        info = self.column_info[column_name]
        values = seed_series.dropna().values
        
        # Handle skewed distributions with log transformation
        if info.get('skewed', False):
            log_values = np.log1p(values)  # log1p handles zeros gracefully
            gmm = self.best_gmm_fit(log_values)
            samples, _ = gmm.sample(rows)
            synthetic_values = np.expm1(samples.flatten())  # Inverse transform
        else:
            # Normal GMM fitting for non-skewed data
            gmm = self.best_gmm_fit(values)
            samples, _ = gmm.sample(rows)
            synthetic_values = samples.flatten()
        
        # Clip to reasonable range (with 10% buffer to allow slight extrapolation)
        min_val = info['min']
        max_val = info['max']
        buffer = 0.1 * (max_val - min_val)
        synthetic_values = np.clip(synthetic_values, min_val - buffer, max_val + buffer)
        
        return synthetic_values
    
    def generate_discrete_numeric(self, seed_series, rows, column_name):
        """
        Generate synthetic discrete numeric data using categorical sampling.
        
        Preserves the exact frequency distribution of discrete values.
        
        Args:
            seed_series (pd.Series): Original data to learn from
            rows (int): Number of synthetic rows to generate
            column_name (str): Name of the column being generated
            
        Returns:
            np.array: Synthetic discrete values
        """
        info = self.column_info[column_name]
        values = info['values']
        probabilities = info['distribution'].values
        
        # Sample from discrete values with original probabilities
        synthetic_values = np.random.choice(values, size=rows, p=probabilities)
        return synthetic_values
    
    def generate_categorical(self, seed_series, rows, column_name):
        """
        Generate synthetic categorical data using frequency-based sampling.
        
        Can use custom categorical libraries if provided, which is useful for
        ensuring synthetic data uses only valid production values.
        
        Args:
            seed_series (pd.Series): Original data to learn from
            rows (int): Number of synthetic rows to generate
            column_name (str): Name of the column being generated
            
        Returns:
            np.array: Synthetic categorical values
        """
        info = self.column_info[column_name]
        
        # Use custom library if provided (e.g., valid state codes, product IDs)
        if column_name in self.categorical_libraries:
            library = [str(v) for v in self.categorical_libraries[column_name]]
            # Assign equal probability to all values in reference table
            categories = np.array(library)
            probabilities = np.ones(len(categories)) / len(categories)
        else:
            # Use observed categories and frequencies
            categories = info['categories']
            probabilities = info['distribution'].values
        
        # Sample from categorical distribution
        synthetic_values = np.random.choice(categories, size=rows, p=probabilities)
        return synthetic_values
    
    def generate_email(self, seed_series, rows, column_name):
        """
        Generate realistic fake email addresses.
        Preserves domain distribution from original data.
        """
        info = self.column_info[column_name]
        domains = info.get('domains', None)

        first_names = ['james','john','robert','michael','william','david','richard','joseph',
                       'thomas','charles','mary','patricia','jennifer','linda','barbara','susan',
                       'jessica','sarah','karen','lisa','emma','olivia','sophia','isabella',
                       'mia','emily','abigail','madison','elizabeth','avery']
        last_names = ['smith','johnson','williams','brown','jones','garcia','miller','davis',
                      'wilson','taylor','anderson','thomas','jackson','white','harris','martin',
                      'thompson','young','allen','king','wright','scott','green','baker','adams',
                      'nelson','hill','ramirez','campbell','mitchell']
        separators = ['.', '_', '']

        emails = []
        domain_list = domains.index.tolist() if domains is not None else ['gmail.com','yahoo.com','outlook.com']
        domain_probs = domains.values.tolist() if domains is not None else [0.5, 0.3, 0.2]

        for _ in range(rows):
            fn = random.choice(first_names)
            ln = random.choice(last_names)
            sep = random.choice(separators)
            num = random.choice(['', str(random.randint(1, 99))])
            domain = np.random.choice(domain_list, p=domain_probs)
            email = f"{fn}{sep}{ln}{num}@{domain}"
            emails.append(email)

        return np.array(emails)

    def generate_phone(self, seed_series, rows, column_name):
        """
        Generate realistic fake phone numbers.
        Detects format from original data and preserves it.
        """
        info = self.column_info[column_name]
        sample_formats = info.get('sample_formats', ['(555) 000-0000'])

        # Detect format from sample
        sample = sample_formats[0] if sample_formats else '(555) 000-0000'

        # Determine format type
        if sample.startswith('+'):
            fmt = 'international'
        elif sample.startswith('('):
            fmt = 'us_parentheses'
        elif '-' in sample:
            fmt = 'dashes'
        else:
            fmt = 'plain'

        phones = []
        for _ in range(rows):
            area = random.randint(200, 999)
            prefix = random.randint(200, 999)
            line = random.randint(1000, 9999)

            if fmt == 'international':
                phones.append(f"+1{area}{prefix}{line}")
            elif fmt == 'us_parentheses':
                phones.append(f"({area}) {prefix}-{line}")
            elif fmt == 'dashes':
                phones.append(f"{area}-{prefix}-{line}")
            else:
                phones.append(f"{area}{prefix}{line}")

        return np.array(phones)

    def generate_datetime(self, seed_series, rows, column_name):
        """
        Generate synthetic datetime data preserving temporal patterns.
        
        Preserves:
        - Date range (min to max)
        - Hour distribution (if timestamps)
        - Day of week distribution
        - Business day patterns (if applicable)
        
        Args:
            seed_series (pd.Series): Original datetime data to learn from
            rows (int): Number of synthetic rows to generate
            column_name (str): Name of the column being generated
            
        Returns:
            pd.Series: Synthetic datetime values
        """
        info = self.column_info[column_name]
        
        # Convert to datetime if not already (handles string dates)
        clean_series = pd.to_datetime(seed_series.dropna(), errors='coerce').dropna()
        
        min_date = info['min_date']
        max_date = info['max_date']
        date_range_days = info['date_range_days']
        is_business_days = info.get('is_business_days', False)
        
        # Generate random dates within range
        if is_business_days:
            # Generate business days only
            synthetic_dates = []
            while len(synthetic_dates) < rows:
                random_days = np.random.randint(0, date_range_days + 1, size=rows * 2)
                dates = [min_date + timedelta(days=int(d)) for d in random_days]
                # Filter to business days (Monday=0 to Friday=4)
                business_dates = [d for d in dates if d.weekday() < 5]
                synthetic_dates.extend(business_dates)
            synthetic_dates = synthetic_dates[:rows]
        else:
            # Generate any day of week
            random_days = np.random.randint(0, date_range_days + 1, size=rows)
            synthetic_dates = [min_date + timedelta(days=int(d)) for d in random_days]
        
        # Check if original data had time component
        has_time = False
        try:
            first_val = clean_series.iloc[0]
            if hasattr(first_val, 'hour') and hasattr(first_val, 'minute'):
                has_time = (first_val.hour != 0 or first_val.minute != 0)
        except:
            has_time = False
        
        # If original data had time component, add realistic times
        if has_time:
            hour_dist = info.get('hour_distribution', {})
            if hour_dist:
                # Sample hours according to observed distribution
                hours = list(hour_dist.keys())
                probs = list(hour_dist.values())
                sampled_hours = np.random.choice(hours, size=rows, p=probs)
                
                # Add random minutes and seconds
                minutes = np.random.randint(0, 60, size=rows)
                seconds = np.random.randint(0, 60, size=rows)
                
                # Combine date and time
                synthetic_dates = [
                    datetime(d.year, d.month, d.day, int(h), int(m), int(s))
                    for d, h, m, s in zip(synthetic_dates, sampled_hours, minutes, seconds)
                ]
        
        return pd.Series(synthetic_dates)
    
    def _validate_synthetic_quality(self, original_df, synthetic_df):
        """
        Validate the quality of generated synthetic data.
        
        Checks for common issues:
        - Negative values in non-negative columns
        - Excessive range expansion
        - Large mean deviations (>20%)
        - DateTime range violations
        
        Args:
            original_df (pd.DataFrame): Original dataset
            synthetic_df (pd.DataFrame): Generated synthetic dataset
        """
        warnings = []
        
        for col in original_df.columns:
            # Check if column exists in synthetic data
            if col not in synthetic_df.columns:
                warnings.append(f"  {col}: Missing in synthetic data")
                continue
                
            orig_col = original_df[col].dropna()
            synth_col = synthetic_df[col].dropna()
            
            # Skip empty columns
            if len(orig_col) == 0 or len(synth_col) == 0:
                continue
                
            col_info = self.column_info.get(col, {})
            col_type = col_info.get('type', 'unknown')
            
            # DATETIME VALIDATION
            if col_type == 'datetime':
                try:
                    if synth_col.min() < orig_col.min():
                        warnings.append(f"  {col}: Dates before original min ({synth_col.min()} < {orig_col.min()})")
                    if synth_col.max() > orig_col.max():
                        warnings.append(f"  {col}: Dates after original max ({synth_col.max()} > {orig_col.max()})")
                except:
                    warnings.append(f"  {col}: Could not validate datetime range")
                continue
            
            # Only validate numeric columns below this point
            if col_type not in ['continuous_numeric', 'discrete_numeric']:
                continue
                
            # Type checking
            if not (pd.api.types.is_numeric_dtype(orig_col) and 
                    pd.api.types.is_numeric_dtype(synth_col)):
                continue
                
            # Skip boolean columns
            if orig_col.dtype == bool or synth_col.dtype == bool:
                continue
                
            try:
                # Check for inappropriate negative values
                if orig_col.min() >= 0:
                    neg_count = (synth_col < 0).sum()
                    if neg_count > 0:
                        warnings.append(f"  {col}: {neg_count} negative values (should be non-negative)")
                
                # Check for range expansion
                orig_range = orig_col.max() - orig_col.min()
                synth_range = synth_col.max() - synth_col.min()
                
                if orig_range > 0 and synth_range > orig_range * 1.5:
                    warnings.append(f"  {col}: Range may be too large (original: {orig_range:.2f}, synthetic: {synth_range:.2f})")
                
                # Check mean deviation for continuous columns
                if col_type == 'continuous_numeric':
                    orig_mean = orig_col.mean()
                    synth_mean = synth_col.mean()
                    
                    if orig_mean != 0:
                        mean_diff_pct = abs((synth_mean - orig_mean) / orig_mean * 100)
                        if mean_diff_pct > 20:
                            warnings.append(f"  {col}: Mean differs by {mean_diff_pct:.1f}% (original: {orig_mean:.2f}, synthetic: {synth_mean:.2f})")
                            
            except (TypeError, AttributeError, ValueError) as e:
                warnings.append(f"  {col}: Could not validate ({str(e)[:50]})")
                continue
        
        # Print validation results
        if warnings:
            print("\nQuality Warnings:")
            for warning in warnings:
                print(warning)
        else:
            print("\nQuality validation passed - no major issues detected")
    
    def generate(self, num_rows, seed_df):
        """
        Generate synthetic data based on fitted column information.
        
        Must call fit() before using this method.
        
        Args:
            num_rows (int): Number of synthetic rows to generate
            seed_df (pd.DataFrame): Sample of original data to use as seed
            
        Returns:
            pd.DataFrame: Synthetic dataset with same structure as original
            
        Raises:
            ValueError: If fit() has not been called yet
        """
        if not self.column_info:
            raise ValueError("Must call fit() before generate()")
        
        print(f"\nGenerating {num_rows} synthetic rows...")
        synthetic_df = pd.DataFrame()
        
        # Generate each column based on its detected type
        for col, info in self.column_info.items():
            col_type = info['type']
            
            if col_type == 'continuous_numeric':
                synthetic_df[col] = self.generate_continuous_numeric(seed_df[col], num_rows, col)
            elif col_type == 'discrete_numeric':
                synthetic_df[col] = self.generate_discrete_numeric(seed_df[col], num_rows, col)
            elif col_type == 'categorical' or col_type == 'boolean':
                synthetic_df[col] = self.generate_categorical(seed_df[col], num_rows, col)
            elif col_type == 'datetime':
                synthetic_df[col] = self.generate_datetime(seed_df[col], num_rows, col)
            elif col_type == 'email':
                synthetic_df[col] = self.generate_email(seed_df[col], num_rows, col)
            elif col_type == 'phone':
                synthetic_df[col] = self.generate_phone(seed_df[col], num_rows, col)
            else:
                print(f"  Skipping {col} (type: {col_type})")
        
        print(f"Generated {synthetic_df.shape[0]} rows × {synthetic_df.shape[1]} columns")
        
        # Validate quality of generated data
        self._validate_synthetic_quality(seed_df, synthetic_df)
        
        return synthetic_df


def print_comparison_stats(real_df, synthetic_df):
    """
    Print statistical comparison between real and synthetic datasets.
    
    Shows mean and standard deviation for numeric columns,
    and frequency distributions for categorical columns.
    
    Args:
        real_df (pd.DataFrame): Original dataset
        synthetic_df (pd.DataFrame): Synthetic dataset
    """
    print("\n" + "="*80)
    print("STATISTICAL COMPARISON: Real vs Synthetic Data")
    print("="*80)
    
    # NUMERIC COLUMNS COMPARISON
    numeric_cols = real_df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        print("\nNUMERIC COLUMNS:")
        print("-"*80)
        for col in numeric_cols:
            if col in synthetic_df.columns:
                real_mean = real_df[col].mean()
                synth_mean = synthetic_df[col].mean()
                real_std = real_df[col].std()
                synth_std = synthetic_df[col].std()
                
                print(f"\n{col}:")
                print(f"  Real      → Mean: {real_mean:>12.2f}, Std: {real_std:>12.2f}")
                print(f"  Synthetic → Mean: {synth_mean:>12.2f}, Std: {synth_std:>12.2f}")
    
    # DATETIME COLUMNS COMPARISON
    datetime_cols = real_df.select_dtypes(include=['datetime64']).columns
    
    if len(datetime_cols) > 0:
        print("\nDATETIME COLUMNS:")
        print("-"*80)
        for col in datetime_cols:
            if col in synthetic_df.columns:
                print(f"\n{col}:")
                print(f"  Real      → Min: {real_df[col].min()}, Max: {real_df[col].max()}")
                print(f"  Synthetic → Min: {synthetic_df[col].min()}, Max: {synthetic_df[col].max()}")
    
    # CATEGORICAL COLUMNS COMPARISON
    categorical_cols = real_df.select_dtypes(include=['object', 'category', 'bool']).columns
    
    if len(categorical_cols) > 0:
        print("\nCATEGORICAL COLUMNS (Top 5 categories):")
        print("-"*80)
        for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
            if col in synthetic_df.columns:
                print(f"\n{col}:")
                real_dist = real_df[col].value_counts(normalize=True).head(5)
                synth_dist = synthetic_df[col].value_counts(normalize=True).head(5)
                
                print("  Real distribution:")
                for cat, freq in real_dist.items():
                    print(f"    {str(cat):30s} {freq:>6.2%}")
                
                print("  Synthetic distribution:")
                for cat, freq in synth_dist.items():
                    print(f"    {str(cat):30s} {freq:>6.2%}")


def create_comparison_plots(real_df, synthetic_df, max_plots=5):
    """
    Create histogram comparison plots for numeric columns.
    
    Overlays real and synthetic distributions for visual comparison.
    Limited to first max_plots numeric columns to avoid overwhelming output.
    
    Args:
        real_df (pd.DataFrame): Original dataset
        synthetic_df (pd.DataFrame): Synthetic dataset
        max_plots (int): Maximum number of plots to create
    """
    numeric_cols = real_df.select_dtypes(include=[np.number]).columns[:max_plots]
    
    if len(numeric_cols) == 0:
        print("No numeric columns to plot")
        return
    
    n_cols = min(2, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        
        # Plot overlapping histograms
        real_data = real_df[col].dropna()
        synth_data = synthetic_df[col].dropna()
        
        ax.hist(real_data, bins=30, alpha=0.5, label='Real', color='blue', density=True)
        ax.hist(synth_data, bins=30, alpha=0.5, label='Synthetic', color='red', density=True)
        
        ax.set_xlabel(col)
        ax.set_ylabel('Density')
        ax.set_title(f'{col} Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('distribution_comparison.png', dpi=150, bbox_inches='tight')
    print("\nComparison plots saved to: distribution_comparison.png")
    plt.show()


def create_synthetic_dashboard(real_df, synthetic_df, output_file='synthetic_dashboard.html'):
    """
    Create an interactive HTML dashboard comparing real and synthetic data.
    
    The dashboard includes:
    - Summary statistics table
    - Distribution curves (KDE) for all numeric columns
    - Box plots for all numeric columns
    - Bar charts for all categorical columns
    
    All visualizations are interactive (zoom, hover, pan).
    
    Args:
        real_df (pd.DataFrame): Original dataset
        synthetic_df (pd.DataFrame): Synthetic dataset
        output_file (str): Path to save the HTML dashboard
    """
    # Import plotly for interactive visualizations
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from scipy import stats
    except ImportError:
        print("Error: plotly and scipy required for dashboard. Install with: pip install plotly scipy")
        return
    
    # Identify column types
    numeric_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = real_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    datetime_cols = real_df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Limit to first 10 of each type
    numeric_cols = numeric_cols[:10]
    categorical_cols = categorical_cols[:10]
    datetime_cols = datetime_cols[:5]
    
    # Calculate number of subplots needed
    # 1 for summary table + 2 per numeric column (distribution + box plot) + 1 per categorical
    total_plots = 1 + (len(numeric_cols) * 2) + len(categorical_cols) + len(datetime_cols)
    
    # Create subplot titles
    subplot_titles = ['Summary Statistics']
    for col in numeric_cols:
        subplot_titles.append(f'{col} - Distribution')
        subplot_titles.append(f'{col} - Box Plot')
    for col in categorical_cols:
        subplot_titles.append(f'{col} - Frequency')
    for col in datetime_cols:
        subplot_titles.append(f'{col} - Timeline')
    
    # Define subplot grid specs
    specs = [[{'type': 'table'}]]  # First row for summary table
    for _ in numeric_cols:
        specs.append([{'type': 'xy'}])  # Distribution plot
        specs.append([{'type': 'xy'}])  # Box plot
    for _ in categorical_cols:
        specs.append([{'type': 'xy'}])  # Bar chart
    for _ in datetime_cols:
        specs.append([{'type': 'xy'}])  # Timeline
    
    # Create subplots
    fig = make_subplots(
        rows=total_plots,
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.02,
        specs=specs
    )
    
    # ADD SUMMARY TABLE
    summary_data = {
        'Metric': ['Total Rows', 'Total Columns', 'Numeric Columns', 'Categorical Columns', 'DateTime Columns'],
        'Real Data': [len(real_df), len(real_df.columns), len(numeric_cols), len(categorical_cols), len(datetime_cols)],
        'Synthetic Data': [len(synthetic_df), len(synthetic_df.columns), len(numeric_cols), len(categorical_cols), len(datetime_cols)]
    }
    
    fig.add_trace(
        go.Table(
            header=dict(values=list(summary_data.keys()),
                       fill_color='paleturquoise',
                       align='left'),
            cells=dict(values=[summary_data['Metric'], summary_data['Real Data'], summary_data['Synthetic Data']],
                      fill_color='lavender',
                      align='left')
        ),
        row=1, col=1
    )
    
    current_row = 2
    
    # ADD NUMERIC COLUMN DISTRIBUTIONS AND BOX PLOTS
    for col in numeric_cols:
        real_data = real_df[col].dropna()
        synth_data = synthetic_df[col].dropna()
        
        # Create KDE (Kernel Density Estimation) for smooth distribution curves
        kde_real = stats.gaussian_kde(real_data)
        kde_synth = stats.gaussian_kde(synth_data)
        x_range = np.linspace(min(real_data.min(), synth_data.min()),
                             max(real_data.max(), synth_data.max()), 200)
        
        # Add distribution curves
        fig.add_trace(
            go.Scatter(x=x_range, y=kde_real(x_range), mode='lines', name=f'{col} Real',
                      line=dict(color='blue'), showlegend=True),
            row=current_row, col=1
        )
        fig.add_trace(
            go.Scatter(x=x_range, y=kde_synth(x_range), mode='lines', name=f'{col} Synthetic',
                      line=dict(color='red', dash='dash'), showlegend=True),
            row=current_row, col=1
        )
        
        # Add mean lines
        y_max = max(kde_real(x_range).max(), kde_synth(x_range).max())
        fig.add_trace(
            go.Scatter(x=[real_data.mean(), real_data.mean()], y=[0, y_max],
                      mode='lines', name=f'{col} Real Mean',
                      line=dict(color='blue', dash='dot'), showlegend=False),
            row=current_row, col=1
        )
        fig.add_trace(
            go.Scatter(x=[synth_data.mean(), synth_data.mean()], y=[0, y_max],
                      mode='lines', name=f'{col} Synth Mean',
                      line=dict(color='red', dash='dot'), showlegend=False),
            row=current_row, col=1
        )
        
        current_row += 1
        
        # Add box plots for the same column
        fig.add_trace(
            go.Box(y=real_data, name=f'{col} Real', marker_color='blue'),
            row=current_row, col=1
        )
        fig.add_trace(
            go.Box(y=synth_data, name=f'{col} Synthetic', marker_color='red'),
            row=current_row, col=1
        )
        
        current_row += 1
    
    # ADD CATEGORICAL COLUMN BAR CHARTS
    for col in categorical_cols:
        real_counts = real_df[col].value_counts().head(15)  # Top 15 categories
        synth_counts = synthetic_df[col].value_counts().head(15)
        
        # Get all unique categories from both datasets
        all_categories = list(set(real_counts.index) | set(synth_counts.index))
        
        # Calculate frequencies as percentages
        real_freq = [real_counts.get(cat, 0) / len(real_df) * 100 for cat in all_categories]
        synth_freq = [synth_counts.get(cat, 0) / len(synthetic_df) * 100 for cat in all_categories]
        
        # Add grouped bar chart
        fig.add_trace(
            go.Bar(x=all_categories, y=real_freq, name=f'{col} Real', marker_color='blue'),
            row=current_row, col=1
        )
        fig.add_trace(
            go.Bar(x=all_categories, y=synth_freq, name=f'{col} Synthetic', marker_color='red'),
            row=current_row, col=1
        )
        
        current_row += 1
    
    # ADD DATETIME COLUMN TIMELINES
    for col in datetime_cols:
        real_dates = real_df[col].dropna().sort_values()
        synth_dates = synthetic_df[col].dropna().sort_values()
        
        # Create histogram by month
        real_by_month = real_dates.dt.to_period('M').value_counts().sort_index()
        synth_by_month = synth_dates.dt.to_period('M').value_counts().sort_index()
        
        all_months = list(set(real_by_month.index) | set(synth_by_month.index))
        all_months.sort()
        
        real_counts = [real_by_month.get(m, 0) for m in all_months]
        synth_counts = [synth_by_month.get(m, 0) for m in all_months]
        month_labels = [str(m) for m in all_months]
        
        fig.add_trace(
            go.Bar(x=month_labels, y=real_counts, name=f'{col} Real', marker_color='blue'),
            row=current_row, col=1
        )
        fig.add_trace(
            go.Bar(x=month_labels, y=synth_counts, name=f'{col} Synthetic', marker_color='red'),
            row=current_row, col=1
        )
        
        current_row += 1
    
    # Update layout
    fig.update_layout(
        height=400 * total_plots,
        showlegend=True,
        title_text="Synthetic Data Quality Dashboard - Real vs Synthetic Comparison"
    )
    
    # Save dashboard to HTML file
    fig.write_html(output_file)
    print(f"\nDashboard saved to: {output_file}")
    
    # Attempt to open in browser
    try:
        import webbrowser
        webbrowser.open(output_file)
    except:
        pass


def generate_synthetic(file_path, num_rows=None, output_path=None, show_plot=False,
                      show_stats=True, dashboard=False, categorical_libraries=None):
    """
    High-level function to generate synthetic data from a CSV file.
    
    This is the main entry point for using the synthetic data generator.
    It handles the complete workflow: load → analyze → generate → save → visualize.
    
    Args:
        file_path (str): Path to the input CSV file
        num_rows (int, optional): Number of rows to generate. Defaults to original dataset size.
        output_path (str, optional): Path for output CSV. Defaults to 'synthetic_{original_name}.csv'
        show_plot (bool): Whether to display matplotlib comparison plots
        show_stats (bool): Whether to print statistical comparison
        dashboard (bool): Whether to create interactive HTML dashboard
        categorical_libraries (dict, optional): Custom categorical value libraries
        
    Returns:
        pd.DataFrame: The generated synthetic dataset
        
    Example:
        >>> synthetic_data = generate_synthetic(
        ...     'healthcare_dataset.csv',
        ...     num_rows=1000,
        ...     dashboard=True,
        ...     show_stats=True
        ... )
    """
    # Load the original dataset
    df = pd.read_csv(file_path)
    print(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Initialize and fit the generator
    generator = AutoDetectingSyntheticGenerator(
        max_gmm_components=3,
        random_state=42,
        discrete_threshold=0.05,
        categorical_libraries=categorical_libraries
    )
    
    generator.fit(df)
    
    # Determine number of rows to generate
    if num_rows is None:
        num_rows = len(df)
    
    # Create a seed sample from original data
    sample_size = min(num_rows, len(df))
    seed_sample = df.sample(sample_size, random_state=42)
    synthetic_df = generator.generate(num_rows=sample_size, seed_df=seed_sample)
    
    # Determine output path
    if output_path is None:
        import os
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = f'synthetic_{base_name}.csv'
    
    # Save synthetic data to CSV
    synthetic_df.to_csv(output_path, index=False)
    print(f"\nSynthetic data saved to: {output_path}")
    
    # Optional: Print statistical comparison
    if show_stats:
        print_comparison_stats(df, synthetic_df)
    
    # Optional: Show matplotlib plots
    if show_plot:
        create_comparison_plots(df, synthetic_df)
    
    # Optional: Create interactive dashboard
    if dashboard:
        dashboard_path = output_path.replace('.csv', '_dashboard.html')
        create_synthetic_dashboard(df, synthetic_df, dashboard_path)
    
    return synthetic_df



if __name__ == "__main__":
    # Generate synthetic data with dashboard and statistics
    synthetic_data = generate_synthetic(
        'healthcare_dataset.csv',
        num_rows=1000,
        dashboard=True,
        show_stats=True
    )
