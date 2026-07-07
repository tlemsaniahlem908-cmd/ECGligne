"""
Moniteur serie RESISTANT : suit l'ESP32 meme s'il redemarre / change de port.
Affiche en continu ce que l'ESP envoie, avec horodatage relatif.
But : voir ce qui se passe cote ESP32 quand le telephone se connecte.
Ctrl+C pour arreter.
"""
import time
import serial
import serial.tools.list_ports

BAUD = 921600


def find_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "CH340" in p.description.upper() or "USB-SERIAL" in p.description.upper():
            return p.device
    return ports[0].device if ports else None


def open_and_reset(port):
    ser = serial.Serial(port, BAUD, timeout=0.1)
    ser.setDTR(False)
    ser.setRTS(True)
    time.sleep(0.15)
    ser.setRTS(False)
    return ser


def main():
    t0 = time.time()
    print("Moniteur RESISTANT. Connecte le telephone maintenant. (Ctrl+C pour arreter)\n")
    ser = None
    while True:
        try:
            if ser is None:
                port = find_port()
                if not port:
                    print(f"[{time.time()-t0:6.1f}s] !! aucun port - ESP debranche ??")
                    time.sleep(0.5)
                    continue
                ser = open_and_reset(port)
                print(f"\n[{time.time()-t0:6.1f}s] === Port {port} ouvert (ESP reset) ===")
            data = ser.read(4096)
            if data:
                txt = data.decode(errors="ignore")
                # prefixe chaque ligne par le temps
                for line in txt.splitlines():
                    if line.strip():
                        print(f"[{time.time()-t0:6.1f}s] ESP: {line.strip()}")
        except KeyboardInterrupt:
            print("\nArret moniteur.")
            break
        except Exception as e:
            print(f"[{time.time()-t0:6.1f}s] !! PORT PERDU ({type(e).__name__}: {e}) -> l'ESP a disparu/redemarre, je reattends...")
            try:
                if ser:
                    ser.close()
            except Exception:
                pass
            ser = None
            time.sleep(0.5)
    if ser:
        ser.close()


if __name__ == "__main__":
    main()
