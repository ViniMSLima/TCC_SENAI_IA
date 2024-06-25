#include <Servo.h>

Servo myServo;

void setup() {
  Serial.begin(9600);  // Inicializa a comunicação serial com baud rate 9600
  myServo.attach(9);   // Pino do servo motor
}

void loop() {
  if (Serial.available() > 0) {  // Verifica se há dados disponíveis para leitura
    char comando = Serial.read();  // Lê o comando enviado pelo Python

    if (comando == '1') {
      myServo.write(90);  // Define a posição do servo para 90 graus
    } else if (comando == '0') {
      myServo.write(0);   // Define a posição do servo para 0 graus
    }
  }
}
