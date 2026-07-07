#include <Arduino.h>      // bibliotheque de base Arduino (millis, Serial, types...)
#include <BLEDevice.h>    // gestion de l'appareil Bluetooth Low Energy (BLE)
#include <BLEServer.h>    // pour creer un serveur BLE
#include <BLEUtils.h>     // outils utilitaires BLE
#include <BLE2902.h>      // descripteur BLE qui autorise les notifications

// =====================================================
// ESP32 ECG BLE SERVER - VERSION STREAMING (vrai ECG, sans buffer 120 KB)
// =====================================================
// Au lieu de stocker tout l'ECG (120 KB -> trop pour le BLE sur WROOM),
// on garde un PETIT tampon circulaire (~15 KB) et on transmet les
// echantillons au fil de l'eau. -> assez de RAM pour accepter le BLE,
// et c'est du VRAI ECG .
//
// Format serie recu de "send_ecg_to_esp32.py" : "seq,500,l0,...,l11\n"
// IMPORTANT : Python doit envoyer au DEBIT du BLE (voir SEND_RATE_HZ cote Python).
// UUID + frame 127 octets INCHANGES -> l'app marche pareil.
// =====================================================
// (Le bloc ci-dessus est juste une explication, ce n'est pas du code execute.)

// ---------- BLE CONFIG (INCHANGE) ----------
#define DEVICE_NAME "ECG_ESP32"                                    // nom affiche de l'appareil BLE
#define SERVICE_UUID      "0000ec00-0000-1000-8000-00805f9b34fb"   // identifiant unique du service BLE
#define NOTIFY_CHAR_UUID  "0000ec01-0000-1000-8000-00805f9b34fb"   // identifiant de la "caracteristique" qui envoie les donnees

// ---------- ECG CONFIG (INCHANGE) ----------
#define FS_HZ 500                                   // frequence d'echantillonnage : 500 mesures par seconde
#define RECORDING_SECONDS 10                        // duree d'un enregistrement : 10 secondes
#define TARGET_SAMPLES (FS_HZ * RECORDING_SECONDS)  // 5000
#define N_LEADS 12                                  // nombre de derivations (canaux) de l'ECG
#define SAMPLES_PER_PACKET 5                        // nombre d'echantillons envoyes dans chaque paquet BLE
#define MAGIC_1 0xEC                                // 1er octet de reconnaissance (debut de trame)
#define MAGIC_2 0x47                                // 2e octet de reconnaissance (debut de trame)
#define FRAME_SIZE (2 + 4 + 1 + (SAMPLES_PER_PACKET * N_LEADS * 2)) // 127
#define PACKET_INTERVAL_MS 14                       // delai entre 2 paquets BLE : 14 millisecondes

// ---------- TAMPON CIRCULAIRE (petit, ~15 KB) ----------
#define RING_CAP 600                 // 600 echantillons max en attente
uint16_t ringSeq[RING_CAP];          // seq d'origine de chaque echantillon
int16_t  ringLeads[RING_CAP][N_LEADS];   // les 12 valeurs (derivations) de chaque echantillon
int ringHead = 0;                    // prochain a envoyer
int ringCount = 0;                   // echantillons disponibles

char lineBuf[256];                   // tampon de texte pour une ligne recue en USB
int  linePos = 0;                    // position actuelle d'ecriture dans lineBuf

BLEServer* bleServer = nullptr;          // pointeur vers le serveur BLE (vide au depart)
BLECharacteristic* notifyChar = nullptr; // pointeur vers la caracteristique de notification (vide au depart)
bool deviceConnected = false;            // vrai si un client (telephone/app) est connecte
bool restartAdvertising = false;         // vrai s'il faut relancer la pub BLE apres deconnexion
uint32_t lastPacketMs = 0;               // moment (ms) du dernier paquet envoye
uint32_t sentPackets = 0;                // compteur de paquets envoyes

// ---------- BLE CALLBACKS ----------
class ServerCallbacks : public BLEServerCallbacks {   // gere les evenements BLE (connexion/deconnexion)
  void onConnect(BLEServer* server) override {        // appele quand un client se connecte
    deviceConnected = true;                           // on note qu'un client est connecte
    sentPackets = 0;                                  // remet le compteur de paquets a zero
    ringHead = 0; ringCount = 0;     // repart a vide a chaque connexion
    lastPacketMs = millis();                          // memorise l'instant present
    Serial.println("BLE >> Client connected");        // affiche un message dans la console
  }
  void onDisconnect(BLEServer* server) override {     // appele quand le client se deconnecte
    deviceConnected = false;                          // on note qu'il n'y a plus de client
    restartAdvertising = true;                        // il faudra relancer la pub BLE
    Serial.println("BLE >> Client disconnected");     // affiche un message dans la console
  }
};

// ---------- LITTLE-ENDIAN HELPERS ----------
void putU32LE(uint8_t* b, int& i, uint32_t v) {       // ecrit un entier 32 bits dans b (octet de poids faible d'abord)
  b[i++]=(uint8_t)(v&0xFF); b[i++]=(uint8_t)((v>>8)&0xFF);          // octet 0 puis octet 1
  b[i++]=(uint8_t)((v>>16)&0xFF); b[i++]=(uint8_t)((v>>24)&0xFF);   // octet 2 puis octet 3
}
void putI16LE(uint8_t* b, int& i, int16_t v) {        // ecrit un entier 16 bits dans b (poids faible d'abord)
  b[i++]=(uint8_t)(v&0xFF); b[i++]=(uint8_t)((v>>8)&0xFF);          // octet bas puis octet haut
}

// ---------- LIRE LE VRAI ECG via USB et remplir le tampon ----------
void pushSample(uint16_t seq, int16_t* leads){        // ajoute un echantillon (12 valeurs) dans le tampon circulaire
  if(ringCount >= RING_CAP) return;            // plein -> on laisse tomber (rare si debit OK)
  int idx = (ringHead + ringCount) % RING_CAP;        // calcule la case libre dans le tampon
  ringSeq[idx] = seq;                                 // enregistre le numero de l'echantillon
  for(int l=0; l<N_LEADS; l++) ringLeads[idx][l] = leads[l];   // copie les 12 derivations
  ringCount++;                                        // un echantillon de plus en attente
}
void parseLine(char* line){                           // decoupe une ligne texte "seq,fs,l0,...,l11"
  // format: seq,fs,l0,l1,...,l11
  char* tok = strtok(line, ",");                      // prend le 1er morceau (le numero seq)
  if(!tok) return;                                    // si vide -> on abandonne
  uint32_t seq = (uint32_t)atol(tok);                 // convertit ce morceau en nombre (seq)
  tok = strtok(NULL, ",");     // fs (ignore)
  if(!tok) return;                                    // si manquant -> on abandonne
  if(seq >= TARGET_SAMPLES) return;                   // ignore si seq depasse 5000 (hors enregistrement)
  int16_t leads[N_LEADS];                             // tableau temporaire pour les 12 derivations
  for(int lead=0; lead<N_LEADS; lead++){              // pour chacune des 12 derivations
    tok = strtok(NULL, ",");                          // prend le morceau suivant
    if(!tok) return;                                  // si manquant -> on abandonne
    leads[lead] = (int16_t)atoi(tok);                 // convertit en nombre et le range
  }
  pushSample((uint16_t)seq, leads);                   // ajoute cet echantillon dans le tampon
}
void readSerialECG(){                                 // lit ce qui arrive par USB et reconstruit les lignes
  while(Serial.available() > 0){                      // tant qu'il y a des caracteres recus
    char c=(char)Serial.read();                       // lit un caractere
    if(c=='\n'){ lineBuf[linePos]='\0'; if(linePos>0) parseLine(lineBuf); linePos=0; }   // fin de ligne -> on la traite
    else if(c!='\r'){ if(linePos<255) lineBuf[linePos++]=c; }   // sinon on ajoute le caractere au tampon (ignore \r)
  }
}

// ---------- ENVOYER 1 FRAME BLE (depuis le tampon, vrai ECG) ----------
void sendEcgFrame(){                                  // construit et envoie une trame BLE de 127 octets
  if(!deviceConnected) return;                        // si personne n'est connecte -> rien a faire
  if(ringCount < SAMPLES_PER_PACKET) return;   // pas assez de donnees -> on attend Python
  uint8_t frame[FRAME_SIZE];                          // tableau d'octets de la trame a envoyer
  int index=0;                                        // position d'ecriture dans la trame
  const uint32_t firstSeq = ringSeq[ringHead];        // numero du 1er echantillon du paquet
  frame[index++]=MAGIC_1; frame[index++]=MAGIC_2;     // ecrit les 2 octets de debut de trame
  putU32LE(frame,index,firstSeq);                     // ecrit le numero seq (4 octets)
  frame[index++]=SAMPLES_PER_PACKET;                  // ecrit le nombre d'echantillons (1 octet)
  for(int s=0; s<SAMPLES_PER_PACKET; s++){            // pour chaque echantillon du paquet
    for(int lead=0; lead<N_LEADS; lead++) putI16LE(frame,index,ringLeads[ringHead][lead]);   // ecrit ses 12 valeurs
    ringHead = (ringHead + 1) % RING_CAP;             // avance la tete du tampon
    ringCount--;                                      // un echantillon de moins en attente
  }
  if(index!=FRAME_SIZE) return;                       // securite : si la taille n'est pas bonne -> on annule
  notifyChar->setValue(frame, FRAME_SIZE);            // met la trame dans la caracteristique BLE
  notifyChar->notify();                               // envoie la notification au client
  sentPackets++;                                      // un paquet de plus envoye
}

// ---------- SETUP ----------
void setup(){                                         // fonction lancee une seule fois au demarrage
  Serial.setRxBufferSize(4096);                       // agrandit le tampon de reception USB
  Serial.begin(921600);                               // demarre la liaison USB a 921600 bauds
  delay(500);                                         // petite pause de 0,5 s
  Serial.println("\n==== ESP32 ECG BLE - STREAMING (vrai ECG) ====");   // message d'accueil
  Serial.println("En attente du vrai ECG via USB (Python)...");         // message d'attente

  BLEDevice::init(DEVICE_NAME);                       // initialise le BLE avec le nom de l'appareil
  BLEDevice::setMTU(185);                             // fixe la taille max des messages BLE
  bleServer = BLEDevice::createServer();              // cree le serveur BLE
  bleServer->setCallbacks(new ServerCallbacks());     // branche les evenements connexion/deconnexion
  BLEService* service = bleServer->createService(SERVICE_UUID);   // cree le service BLE
  notifyChar = service->createCharacteristic(         // cree la caracteristique d'envoi :
    NOTIFY_CHAR_UUID, BLECharacteristic::PROPERTY_NOTIFY);        // avec son UUID et le mode notification
  notifyChar->addDescriptor(new BLE2902());           // ajoute le descripteur qui active les notifications
  service->start();                                   // demarre le service BLE
  BLEAdvertising* adv = BLEDevice::getAdvertising();  // recupere l'objet de "publicite" BLE
  adv->addServiceUUID(SERVICE_UUID);                  // annonce notre service dans la pub
  adv->setScanResponse(true);                         // repond aux scans des clients
  adv->setMinPreferred(0x06);                         // reglage d'intervalle BLE (compatibilite)
  adv->setMinPreferred(0x12);                         // reglage d'intervalle BLE (compatibilite)
  BLEDevice::startAdvertising();                      // commence a etre visible (pub BLE)
  Serial.println("BLE >> Advertising started");       // message : pub demarree
}

// ---------- LOOP ----------
void loop(){                                          // fonction repetee en boucle sans fin
  readSerialECG();   // toujours ecouter le vrai ECG de Python

  if(restartAdvertising){                             // si une deconnexion vient d'arriver
    restartAdvertising=false; delay(500);             // remet le drapeau a faux + petite pause
    BLEDevice::startAdvertising();                    // relance la pub BLE
    Serial.println("BLE >> Advertising restarted");   // message : pub relancee
  }
  if(!deviceConnected){ delay(1); return; }           // si pas de client -> on attend (1 ms) et on recommence

  const uint32_t now=millis();                        // heure actuelle en millisecondes
  if((uint32_t)(now-lastPacketMs) >= PACKET_INTERVAL_MS){   // si le delai entre paquets est ecoule
    lastPacketMs=now;                                 // memorise ce moment
    sendEcgFrame();                                   // envoie une trame BLE
  }
}