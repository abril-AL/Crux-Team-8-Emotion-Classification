//////////////////////////////////////////////////
//                                              //
//  LED Display For Emotion Classification BCI  //
//                                              //
///////////////////////////////////////////////////////
// v2 uses single RGB LED instead of 4 seperate LEDs //
///////////////////////////////////////////////////////
#define R 2
#define G 3
#define B 4T


// Input Byte, read from Serial (COM6)
int inByte = 0;

void setup(){
    Serial.begin(9600);
    pinMode(R, OUTPUT); 
    pinMode(G, OUTPUT);
    pinMode(B, OUTPUT);
}

void loop(){
      if (Serial.available() > 0) {
        // read the incoming byte:
        inByte = Serial.read();

        // say what you got:
        if (inByte != 10){
          Serial.print("Received: ");
          Serial.print(inByte, DEC);
        
        //inByte = inByte - 48;
        switch (inByte) {
          case 0:
            // Neutral
            Serial.print(" Neutral\n");
            analogWrite(R, 255);
            analogWrite(G, 255);
            analogWrite(B, 255);
            break;
          case 1:
            // Sad
            Serial.print(" Sad\n");
            analogWrite(R, 0);
            analogWrite(G, 0);
            analogWrite(B, 255);
            break;
          case 2:
            // Happy 
            Serial.print(" Happy");
            analogWrite(R, 0);
            analogWrite(G, 255);
            analogWrite(B, 0);
            break;
          case 3:
            // Angry 
            Serial.print(" Angry\n");
            analogWrite(R, 255);
            analogWrite(G, 0);
            analogWrite(B, 0);
            break;
          default:
            // Other Emotion, testing for rn 
            Serial.print(" Other\n");
            analogWrite(R, 255);
            analogWrite(G, 255);
            analogWrite(B, 255);
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
      // and more as we go
}
