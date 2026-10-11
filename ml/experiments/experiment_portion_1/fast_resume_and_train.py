import os
import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, 'raw_labels')
RAW_DATA_URL = 'https://huggingface.co/datasets/issai/Food_Portion_Benchmark/raw/main/'

def get_file_list():
    api_url = 'https://huggingface.co/api/datasets/issai/Food_Portion_Benchmark'
    print("Fetching repository tree from Hugging Face API...")
    r = requests.get(api_url)
    r.raise_for_status()
    all_files = [s['rfilename'] for s in r.json().get('siblings', [])]
    txt_files = [f for f in all_files if f.endswith('.txt') or f.endswith('.yaml')]
    return txt_files

def download_missing_files(txt_files, max_workers=16):
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=max_workers, pool_maxsize=max_workers, max_retries=3)
    session.mount('https://', adapter)

    def fetch_file(rel_path):
        url = RAW_DATA_URL + rel_path
        local_path = os.path.join(RAW_DIR, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        for attempt in range(4):
            try:
                resp = session.get(url, timeout=25)
                if resp.status_code == 200:
                    with open(local_path, 'wb') as f:
                        f.write(resp.content)
                    return True
                elif resp.status_code == 429:
                    time.sleep(2.0 * (attempt + 1))
                else:
                    time.sleep(0.5)
            except Exception:
                time.sleep(1.0)
        return False

    pass_num = 1
    while True:
        missing = [rel_p for rel_p in txt_files if not os.path.exists(os.path.join(RAW_DIR, rel_p))]
        total_present = len(txt_files) - len(missing)
        print(f"\n[Pass {pass_num}] Total files: {len(txt_files)} | Present: {total_present} | Missing: {len(missing)}", flush=True)

        if not missing:
            print("All annotation files are present and verified!", flush=True)
            break

        print(f"Downloading {len(missing)} files with {max_workers} threads...", flush=True)
        t0 = time.time()
        success = 0
        fail = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_file, p): p for p in missing}
            for i, future in enumerate(as_completed(futures), 1):
                if future.result():
                    success += 1
                else:
                    fail += 1

                if i % 1000 == 0 or i == len(missing):
                    elapsed = time.time() - t0
                    rate = i / elapsed if elapsed > 0 else 0
                    print(f"Progress: {i}/{len(missing)} ({round(i/len(missing)*100, 1)}%) | Rate: {round(rate, 1)} files/sec | Success: {success}, Fail: {fail}", flush=True)

        print(f"Pass {pass_num} completed in {round(time.time() - t0, 2)}s. Success: {success}, Failures: {fail}", flush=True)
        pass_num += 1
        if fail == 0:
            break
        time.sleep(2)

def main():
    print("=" * 70)
    print("Smart Plate - Fast Label Resumer & Model Training")
    print("=" * 70)

    txt_files = get_file_list()
    download_missing_files(txt_files, max_workers=20)

    print("\n" + "=" * 70)
    print("Executing Model Training & Evaluation Suite...")
    print("=" * 70)

    sys.path.insert(0, BASE_DIR)
    import train_portion_models
    train_portion_models.main()

if __name__ == '__main__':
    main()
