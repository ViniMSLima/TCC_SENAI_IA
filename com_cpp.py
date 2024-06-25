import serial
import time

# Configuração da porta serial
ser = serial.Serial('COM3', 9600)  # Substitua 'COM3' pelo seu dispositivo serial e 9600 pelo baud rate

# Exemplo de envio de dados
valor = True  # Exemplo de valor booleano a ser enviado
if valor:
    ser.write(b'1')  # Envia '1' para o Arduino
else:
    ser.write(b'0')  # Envia '0' para o Arduino

# Fechar a conexão serial no final
ser.close()
