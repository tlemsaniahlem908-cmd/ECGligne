"""
Diagnostic ESP32 : redemarre la carte, lit le message de boot,
puis envoie quelques lignes test et affiche TOUT ce que l'ESP32 renvoie.
"""
import time
import serial
import serial.tools.list_ports

BAUD = 921600
FS = 500


def find_port():
    ports = list(serial.tools.list_ports.comports())
    print("Ports detectes:", [p.device for p in ports] if ports else "AUCUN")
    for p in ports:
        if "CH340" in p.description.upper() or "USB-SERIAL" in p.description.upper():
            return p.device
    return ports[0].device if ports else None


def main():
    port = find_port()
    if not port:
        print("Aucun port. Rebranche l'ESP32.")
        return
    print(f"Ouverture {port} @ {BAUD}...")

    ser = serial.Serial(port, BAUD, timeout=0.1)

    # --- Reset materiel pour DEMARRER LE SKETCH (pas le bootloader) ---
    ser.setDTR(False)   # GPIO0 = HIGH -> mode RUN (pas download)
    ser.setRTS(True)    # EN = LOW  -> reset
    time.sleep(0.15)
    ser.setRTS(False)   # EN = HIGH -> sort du reset -> demarre le sketch
    print("Reset envoye. Lecture du message de boot (5 s)...\n")

    t0 = time.time()
    while time.time() - t0 < 5.0:
        data = ser.read(4096)
        if data:
            print(data.decode(errors="ignore"), end="")

    print("\n\n--- Envoi de 6 lignes test (seq 0, puis 4995..4999) ---")
    test_seqs = [0, 4995, 4996, 4997, 4998, 4999]
    for seq in test_seqs:
        vals = ",".join(str(100 + i) for i in range(12))   # 12 valeurs bidon
        line = f"{seq},{FS},{vals}\n"
        ser.write(line.encode("ascii"))
        time.sleep(0.05)

    print("Reponse de l'ESP32 (3 s)...\n")
    t0 = time.time()
    got = ""
    while time.time() - t0 < 3.0:
        data = ser.read(4096)
        if data:
            txt = data.decode(errors="ignore")
            got += txt
            print(txt, end="")

    print("\n\n========== RESULTAT ==========")
    if "REAL_ECG_LOADED" in got:
        print("OK : l'ESP32 PARSE le serie et a declenche REAL_ECG_LOADED. ✅")
    elif "RX seq" in got:
        print("PARTIEL : l'ESP recoit (RX seq) mais pas REAL_ECG_LOADED.")
    elif got.strip():
        print("L'ESP repond, mais pas comme attendu (voir ci-dessus).")
    else:
        print("AUCUNE reponse au serie -> firmware ne lit pas l'UART, ou pas le bon firmware.")
    ser.close()


if __name__ == "__main__":
    main()
