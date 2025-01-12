//////////////////////////////////////////////////
//                                              //
//  LED Display For Emotion Classification BCI  //
//                                              //
//////////////////////////////////////////////////

// Input Byte, read from Serial (COM6)
int inByte = 0;
// LEDS
int LedR=2;
int LedG=3;
int LedB=4;
int LedW=5;

void setup() {
  Serial.begin(9600); // opens serial port, sets data rate to 9600 bps
  // Set LEDs as output
  pinMode(2, OUTPUT);
  pinMode(3, OUTPUT);
  pinMode(4, OUTPUT);
  pinMode(5, OUTPUT);
}

void loop() {
  if (Serial.available() > 0) {
        // read the incoming byte:
        inByte = Serial.read();

        // say what you got:
        if (inByte != 10){
          Serial.print("Received: ");
          Serial.println(inByte, DEC);
        
        inByte = inByte - 48;
        switch (inByte) {
          case 0:
            // Neutral
            Serial.print("Neutral\n");
            digitalWrite(2,LOW);
            digitalWrite(3,LOW);
            digitalWrite(4,LOW);
            digitalWrite(5,HIGH);//W
            break;
          case 1:
            // Sad
            Serial.print("Sad\n");
            digitalWrite(2,LOW);
            digitalWrite(3,LOW);
            digitalWrite(4,HIGH);//B
            digitalWrite(5,LOW);
            break;
          case 2:
            // Happy 
            Serial.print("Happy");
            digitalWrite(2,LOW);
            digitalWrite(3,HIGH);//G
            digitalWrite(4,LOW);
            digitalWrite(5,LOW);
            break;
          case 3:
            // Angry 
            Serial.print("Angry\n");
            digitalWrite(2,HIGH);//R
            digitalWrite(3,LOW);
            digitalWrite(4,LOW);
            digitalWrite(5,LOW);
            break;
          default:
            // Other Emotion, testing for rn 
            Serial.print("Other\n");
            digitalWrite(2,LOW);
            digitalWrite(3,LOW);
            digitalWrite(4,LOW);
            digitalWrite(5,LOW);
            break;
        }  
      } 
  }

      // Emotions Legend
      // 1 bytes, 6 emotions (atm)
      // 0 - Neutral (White)
      // 1 - Happy (Green)
      // 2 - Sad (Blue)
      // 3 - Angry (Red)
      // and more as needed
             
}
