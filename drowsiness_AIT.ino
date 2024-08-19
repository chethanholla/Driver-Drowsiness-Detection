//#include<SoftwareSerial.h>
int buzzer=3;
int voice1=5;
int yled1=6;
int yled2=8;
char ch;
void setup() {
  // put your setup code here, to run once:
  pinMode(yled1,OUTPUT);
  pinMode(yled2,OUTPUT);
pinMode(buzzer,OUTPUT);
pinMode(voice1,OUTPUT);
digitalWrite(voice1,HIGH);
  digitalWrite(buzzer,LOW);
Serial.begin(9600);
Serial.println("DRIVER DROWSINESS DETECTION");
delay(2000);
}

void loop() {
  // put your main code here, to run repeatedly:
  
SERIAL_EVENT();
}


void drowsy(){
  digitalWrite(buzzer,HIGH);
  delay(900);
  digitalWrite(buzzer,LOW);
  
         digitalWrite(yled1,HIGH);
         digitalWrite(yled2,HIGH);
         delay(3000);
        
         
         digitalWrite(yled1,LOW);
         digitalWrite(yled2,LOW);
         
         digitalWrite(voice1,LOW);
         delay(2000);
         digitalWrite(voice1,HIGH);
         Serial.println("drowsiness detected");
}
void SERIAL_EVENT()
{
  
    if(Serial.available()>0)
    {
//      if(Serial_read()=='D')
      {
        ch = Serial.read();
        if(ch == 'D')
           {  
            drowsy(); 
           }        
}
}
}
