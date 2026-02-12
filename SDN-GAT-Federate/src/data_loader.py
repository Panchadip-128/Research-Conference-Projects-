import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
from .config import RAW_DATA_PATH, REQUIRED_SAMPLES, CHUNK_SIZE

def load_and_preprocess_data():
    """
    Loads data using a linear scan approach (Guaranteed to find data).
    Performs cleaning and label encoding.
    """
    print("🚀 Starting Linear Scan (Guaranteed to find data)...")

    filename = RAW_DATA_PATH

    # We need 2,500 samples of EACH class
    required_samples = REQUIRED_SAMPLES
    df_benign_all = pd.DataFrame()
    df_attack_all = pd.DataFrame()

    # Standard Column Names
    column_mapping = {
        'Attack_label': 'Attack_type', 
        'tcp.length': 'tcp.len', 
        'ip.src': 'ip.src_host', 
        'ip.dst': 'ip.dst_host'
    }

    # ITERATE THROUGH THE WHOLE FILE
    chunk_size = CHUNK_SIZE
    total_rows = 0

    try:
        for chunk in pd.read_csv(filename, chunksize=chunk_size, low_memory=False):
            total_rows += len(chunk)
            
            # 1. Standardize Names
            chunk.rename(columns=column_mapping, inplace=True)
            chunk = chunk.loc[:, ~chunk.columns.duplicated()]
            
            # 2. Fix Label
            if 'Attack_type' not in chunk.columns:
                possible = [c for c in chunk.columns if 'label' in c.lower() or 'class' in c.lower()]
                chunk['Attack_type'] = chunk[possible[0]] if possible else 'Normal'
                
            chunk['Attack_label'] = chunk['Attack_type'].apply(lambda x: 0 if str(x).strip() == 'Normal' else 1)
            
            # 3. Harvest Data
            # Only keep what we need to save memory
            benign = chunk[chunk['Attack_label'] == 0]
            attack = chunk[chunk['Attack_label'] == 1]
            
            if len(df_benign_all) < required_samples:
                df_benign_all = pd.concat([df_benign_all, benign])
                
            if len(df_attack_all) < required_samples:
                df_attack_all = pd.concat([df_attack_all, attack])
                
            print(f"🔎 Scanned {total_rows} rows... Found: {len(df_benign_all)} Benign | {len(df_attack_all)} Attack")
            
            # 4. STOP if we have enough
            if len(df_benign_all) >= required_samples and len(df_attack_all) >= required_samples:
                print("✅ SUCCESS! Found enough of both classes. Stopping scan.")
                break
    except FileNotFoundError:
        print(f"⚠️ Dataset file not found at {filename}. Please check the path.")
        return None

    # --- SAFETY CHECK ---
    n_final = min(required_samples, len(df_benign_all), len(df_attack_all))

    if n_final < 100:
        print("⚠️ CRITICAL WARNING: Still missing data.")
        # Fallback: If we have ANY benign data, use it all. If 0, we must crash.
        if len(df_benign_all) == 0:
            raise ValueError("FATAL: Scanned entire file and found 0 Normal rows. Check dataset filename/version.")
        n_final = len(df_benign_all) # Use whatever we found

    # Final Sampling
    df_benign = df_benign_all.sample(n=n_final, replace=False, random_state=42)
    df_attack = df_attack_all.sample(n=n_final, replace=False, random_state=42)
    df_balanced = pd.concat([df_benign, df_attack]).sample(frac=1).reset_index(drop=True)

    # --- CLEANING ---
    drop_cols = ['frame.time', 'arp.src.proto_ipv4', 'arp.dst.proto_ipv4', 'http.file_data', 
                 'http.request.full_uri', 'icmp.transmit_timestamp', 'tcp.options', 'tcp.payload', 'mqtt.msg']
    df_balanced = df_balanced.drop(columns=drop_cols, errors='ignore')

    if 'tcp.len' not in df_balanced.columns:
         df_balanced['tcp.len'] = df_balanced.fillna(0)['tcp.srcport'] + df_balanced.fillna(0)['udp.port']

    # Encode IPs
    le = LabelEncoder()
    if 'ip.src_host' not in df_balanced.columns: df_balanced['ip.src_host'] = range(len(df_balanced))
    if 'ip.dst_host' not in df_balanced.columns: df_balanced['ip.dst_host'] = range(len(df_balanced))

    df_balanced['ip.src_host'] = le.fit_transform(df_balanced['ip.src_host'].astype(str))
    df_balanced['ip.dst_host'] = le.fit_transform(df_balanced['ip.dst_host'].astype(str))

    print(f"✅ FINAL DATASET READY: {len(df_balanced)} Total Samples ({n_final} per class)")
    
    return df_balanced
