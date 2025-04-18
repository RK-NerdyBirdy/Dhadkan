#include <WiFi.h>
#include <WebSocketsServer.h>

// WiFi credentials
const char* ssid = "Your SSID";
const char* password = "Your password";

// ECG Sensor pins
const int ECGPin = 36;     // Analog pin to read ECG signal
const int LOplus = 33;     // Leads-off detection +
const int LOminus = 32;    // Leads-off detection -

// WebSocket server
WebSocketsServer webSocket = WebSocketsServer(81);

// Data buffer and transmission rate control
unsigned long lastTransmissionTime = 0;
const int transmissionInterval = 8;  // ~125Hz

void setup() {
  Serial.begin(115200);
  pinMode(LOplus, INPUT);
  pinMode(LOminus, INPUT);
  
  // Connect to WiFi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("");
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());  // Print the IP address for client connections
  
  // Start WebSocket server
  webSocket.begin();
  webSocket.onEvent(webSocketEvent);
  Serial.println("WebSocket server started");
}

void loop() {
  webSocket.loop();
  
  unsigned long currentTime = millis();
  if (currentTime - lastTransmissionTime >= transmissionInterval) {
    lastTransmissionTime = currentTime;
    
    // Check if leads are off
    if ((digitalRead(LOplus) == 1) || (digitalRead(LOminus) == 1)) {
      broadcastECGValue(0);  // Leads off, output 0
    } else {
      int ecgValue = analogRead(ECGPin);
      broadcastECGValue(ecgValue);
    }
  }
}

void broadcastECGValue(int value) {
  // Convert int to string and broadcast to all connected clients
  String ecgData = String(value);
  webSocket.broadcastTXT(ecgData);
  
  // Also print to serial for debugging
  //Serial.println(value);
}

void webSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length) {
  switch(type) {
    case WStype_DISCONNECTED:
      Serial.printf("[%u] Disconnected!\n", num);
      break;
    case WStype_CONNECTED:
      {
        IPAddress ip = webSocket.remoteIP(num);
        Serial.printf("[%u] Connected from %d.%d.%d.%d\n", num, ip[0], ip[1], ip[2], ip[3]);
      }
      break;
    case WStype_TEXT:
      // Handle incoming messages if needed
      break;
  }
}