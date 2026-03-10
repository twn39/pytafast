import pandas as pd
import subprocess
import os
import io

FILES = [
    "nasdaq100_2025_now.csv",
]

INDICATORS = {
    "aroon": "res <- aroon(cbind(high, low), n=14)",
    "smi": "res <- SMI(cbind(high, low, close), n=13, nFast=2, nSlow=25, nSig=9)",
    "emv": "res <- EMV(cbind(high, low), volume, n=9)",
    "dpo": "res <- data.frame(dpo=DPO(close, n=10))",
    "obv": "res <- data.frame(obv=OBV(close, volume))",
    "cmo": "res <- data.frame(cmo=CMO(close, n=14))",
    "roc": "res <- data.frame(roc=ROC(close, n=10, type='discrete') * 100)",
    "clv": "res <- data.frame(clv=CLV(cbind(high, low, close)))",
    "chaikin_vol": "res <- data.frame(chv=chaikinVolatility(cbind(high, low), n=10) * 100)",
}

def run_r_ttr(data_path, r_code):
    abs_data_path = os.path.abspath(data_path)
    full_r_script = f"""
    library(TTR)
    df <- read.csv("{abs_data_path}")
    open <- df$Open
    high <- df$High
    low <- df$Low
    close <- df$Close
    volume <- df$Volume
    
    {r_code}
    
    write.csv(res, row.names=FALSE)
    """

    process = subprocess.Popen(
        ["Rscript", "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate(input=full_r_script)

    if process.returncode != 0:
        raise RuntimeError(f"R execution failed: {stderr}")

    return pd.read_csv(io.StringIO(stdout))

if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    out_dir = os.path.join(base_dir, "r_expected")
    os.makedirs(out_dir, exist_ok=True)
    
    for filename in FILES:
        data_path = os.path.join(base_dir, filename)
        if not os.path.exists(data_path):
            print(f"Skipping {data_path}, does not exist")
            continue
            
        print(f"Processing {filename}...")
        for ind_name, r_code in INDICATORS.items():
            print(f"  generating {ind_name}...")
            res_df = run_r_ttr(data_path, r_code)
            out_name = f"expected_{ind_name}_{filename}"
            out_path = os.path.join(out_dir, out_name)
            res_df.to_csv(out_path, index=False)
            
    print("All R expected data generated successfully.")
