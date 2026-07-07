from pathlib import Path
import time
import json
import ast

import numpy as np
import pandas as pd
import wfdb
import serial
import serial.tools.list_ports


# ============================================================
# CONFIGURATION
# ============================================================

PTBXL_ROOT = Path(r"C:\Users\SMART-TECH29\Downloads\ptb-xl")
CSV_PATH = PTBXL_ROOT / "ptbxl_database.csv"

SERIAL_PORT = "AUTO"        # "AUTO" = trouve tout seul l'ESP32 (CH340), peu importe le numero COM
                            # ou force un port precis : "COM3"
BAUD_RATE = 921600

FS_TARGET = 500             # frequence reelle de l'ECG (reste dans les donnees + le .hea)
SEND_RATE_HZ = 300          # debit d'ENVOI vers l'ESP32 (doit coller au debit BLE, sinon le
                            # tampon de l'ESP deborde). 300 < debit BLE (~357) = marge de securite.
N_SAMPLES = 5000
N_LEADS = 12

ECG_INDEX = 214

OUT_MV = Path("live_ecg_10s.npy")
OUT_UV = Path("live_ecg_10s_uv.npy")
OUT_INFO = Path("live_ecg_10s_info.json")

LEAD_NAMES = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]


# ============================================================
# UNIT CONVERSION
# ============================================================

def unit_scale_to_uv(unit: str) -> float:
    u = str(unit).strip().lower()
    u = u.replace("μ", "µ")

    if u in ["mv", "millivolt", "millivolts"]:
        return 1000.0

    if u in ["uv", "µv", "microvolt", "microvolts"]:
        return 1.0

    if u in ["v", "volt", "volts"]:
        return 1_000_000.0

    print(f"⚠️ Unknown unit {unit}, assuming mV")
    return 1000.0


# ============================================================
# LOAD PTB-XL 500 Hz ECG
# ============================================================

def load_ptbxl_ecg():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"ptbxl_database.csv not found:\n{CSV_PATH}\n"
            "Fix PTBXL_ROOT."
        )

    df = pd.read_csv(CSV_PATH)
    row = df.iloc[ECG_INDEX]

    record_path = PTBXL_ROOT / row["filename_hr"]

    print("====================================")
    print("LOAD PTB-XL ECG 500 Hz")
    print("====================================")
    print("ECG_INDEX:", ECG_INDEX)
    print("ECG_ID:", row.get("ecg_id", "unknown"))
    print("Record:", record_path)

    signal, meta = wfdb.rdsamp(str(record_path))

    print("Original shape:", signal.shape)
    print("Lead names:", meta.get("sig_name"))
    print("Units:", meta.get("units"))

    if signal.shape != (N_SAMPLES, N_LEADS):
        raise ValueError(f"Bad ECG shape: {signal.shape}, expected (5000,12)")

    units = meta.get("units", ["mV"] * 12)
    scales = np.array([unit_scale_to_uv(u) for u in units], dtype=np.float32)

    # signal from WFDB is usually mV
    ecg_mv = signal.astype(np.float32)

    # convert to microvolts for ESP32
    ecg_uv = ecg_mv * scales.reshape(1, 12)
    ecg_uv = np.clip(ecg_uv, -32768, 32767).astype(np.int16)

    if "scp_codes" in row:
        try:
            scp_codes = ast.literal_eval(row["scp_codes"])
        except Exception:
            scp_codes = str(row["scp_codes"])
    else:
        scp_codes = None

    info = {
        "ecg_index": int(ECG_INDEX),
        "ecg_id": int(row["ecg_id"]) if "ecg_id" in row else None,
        "fs": FS_TARGET,
        "shape": [N_SAMPLES, N_LEADS],
        "unit_live_ecg_10s": "mV",
        "unit_live_ecg_10s_uv": "microvolts",
        "lead_order": LEAD_NAMES,
        "scp_codes": scp_codes,
    }

    print("Final mV shape:", ecg_mv.shape)
    print("Final µV shape:", ecg_uv.shape)
    print("mV min/max:", float(ecg_mv.min()), float(ecg_mv.max()))
    print("µV min/max:", int(ecg_uv.min()), int(ecg_uv.max()))
    print("====================================")

    return ecg_mv, ecg_uv, info


# ============================================================
# SEND 10 SECONDS TO ESP32  (EN BOUCLE jusqu'a Ctrl+C)
# ============================================================

def resolve_port():
    """Trouve le port a utiliser. AUTO -> prend un CH340 (ESP32), sinon le 1er port."""
    ports = list(serial.tools.list_ports.comports())
    print("Ports COM detectes:", [p.device for p in ports] if ports else "AUCUN")
    if SERIAL_PORT.upper() != "AUTO":
        return SERIAL_PORT
    if not ports:
        raise RuntimeError("Aucun port COM. Rebranche l'ESP32 (cable DATA) et reessaie.")
    # priorite a une carte CH340 (puce de ton ESP32)
    for p in ports:
        desc = f"{p.description} {p.manufacturer or ''}".upper()
        if "CH340" in desc or "CP210" in desc or "USB-SERIAL" in desc:
            print(f"AUTO -> {p.device}  ({p.description})")
            return p.device
    print(f"AUTO -> {ports[0].device}  ({ports[0].description})")
    return ports[0].device


def send_to_esp32_10s(ecg_uv):
    port = resolve_port()
    print(f"Connecting ESP32 on {port} at {BAUD_RATE} baud...")

    sample_period = 1.0 / SEND_RATE_HZ   # cadence d'envoi (pas FS_TARGET) -> colle au debit BLE

    with serial.Serial(
        port,
        BAUD_RATE,
        timeout=0.001,
        write_timeout=1
    ) as ser:

        # ESP32 often resets when serial opens
        time.sleep(2.0)

        ser.reset_input_buffer()
        ser.reset_output_buffer()

        print("✅ ESP32 connected")
        print("Sending in LOOP (Ctrl+C to stop)...")
        print()

        pass_num = 0
        while True:                          # boucle infinie -> jamais fini sauf Ctrl+C
            pass_num += 1
            print(f"--- Pass #{pass_num} : 5000 samples (10 s) ---")
            start_time = time.perf_counter()
            seq_gap_warning = False

            for seq in range(N_SAMPLES):
                sample = ecg_uv[seq]

                line = (
                    f"{seq},{FS_TARGET},"
                    + ",".join(str(int(v)) for v in sample)
                    + "\n"
                )

                ser.write(line.encode("ascii"))

                if ser.in_waiting > 0:
                    msg = ser.read(ser.in_waiting).decode(errors="ignore")
                    if msg.strip():
                        print("ESP32:", msg.strip())
                    if "WARNING_SEQ_GAP" in msg:
                        seq_gap_warning = True

                if seq % 500 == 0:
                    print(f"  pass #{pass_num} - sent {seq}/5000")

                next_time = start_time + (seq + 1) * sample_period
                sleep_time = next_time - time.perf_counter()

                if sleep_time > 0:
                    time.sleep(sleep_time)

            # lire la confirmation ESP32 (REAL_ECG_LOADED) apres le dernier echantillon
            time.sleep(0.1)
            if ser.in_waiting > 0:
                msg = ser.read(ser.in_waiting).decode(errors="ignore")
                if msg.strip():
                    print("ESP32:", msg.strip())
                if "WARNING_SEQ_GAP" in msg:
                    seq_gap_warning = True

            print(f"Pass #{pass_num} done -> RESTART from beginning")
            if seq_gap_warning:
                print("⚠️ ESP32 reported WARNING_SEQ_GAP")


# ============================================================
# SAVE FILES
# ============================================================

def save_live_files(ecg_mv, ecg_uv, info):
    np.save(OUT_MV, ecg_mv.astype(np.float32))
    np.save(OUT_UV, ecg_uv.astype(np.int16))

    with open(OUT_INFO, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print()
    print("====================================")
    print("FILES SAVED")
    print("====================================")
    print("Saved:", OUT_MV.resolve())
    print("Saved:", OUT_UV.resolve())
    print("Saved:", OUT_INFO.resolve())
    print("live_ecg_10s.npy shape:", np.load(OUT_MV).shape)
    print("live_ecg_10s.npy unit: mV")
    print("====================================")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    ecg_mv, ecg_uv, info = load_ptbxl_ecg()

    # 1. Save the same 10-second block for AI pipeline (avant l'envoi)
    save_live_files(ecg_mv, ecg_uv, info)

    # 2. Send real ECG to ESP32 in a loop (Ctrl+C to stop)
    try:
        send_to_esp32_10s(ecg_uv)
    except KeyboardInterrupt:
        print("\nStopped (Ctrl+C). Sending stopped.")
