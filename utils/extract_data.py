#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import zipfile
import os
import sys

def extract_all_zip_files(data_dir):
    """Extract all zip files in the data directory"""
    os.chdir(data_dir)
    
    zip_files = [f for f in os.listdir('.') if f.endswith('.zip')]
    print(f'Found {len(zip_files)} zip files\n')
    
    for zip_file in zip_files:
        csv_file = zip_file.replace('.zip', '')
        if not os.path.exists(csv_file):
            print(f'Extracting {zip_file}...')
            try:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall('.')
                
                # Check file size
                size_mb = os.path.getsize(csv_file) / (1024*1024)
                print(f'   Extracted to {csv_file} ({size_mb:.2f} MB)')
                
            except Exception as e:
                print(f'   Error extracting {zip_file}: {str(e)}')
        else:
            size_mb = os.path.getsize(csv_file) / (1024*1024)
            print(f'{csv_file} already exists ({size_mb:.2f} MB)')
    
    print('\nExtraction completed!')

if __name__ == "__main__":
    data_dir = "C:/Users/ASUS/bosch/data"
    extract_all_zip_files(data_dir)