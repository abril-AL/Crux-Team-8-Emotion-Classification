//////////////////////////////////////////////////
//                                              //
//  LED Display For Emotion Classification BCI  //
//                                              //
///////////////////////////////////////////////////////
// v2 uses single RGB LED instead of 4 seperate LEDs //
///////////////////////////////////////////////////////
#define B1 9
#define B2 2
#define G1 3
#define G2 4
#define R1 6
#define R2 7


// Input Byte, read from Serial (COM6)
int inByte = 0;

void setup(){
    Serial.begin(9600);
    pinMode(R1, OUTPUT); 
    pinMode(R2, OUTPUT);
    pinMode(G1, OUTPUT);
    pinMode(G2, OUTPUT); 
    pinMode(B1, OUTPUT);
    pinMode(B2, OUTPUT);

    // Test LED2
    analogWrite(R2, 255);
    analogWrite(G2, 0);
    analogWrite(B2, 0);

    //Test LED1
    analogWrite(R1, 255);
    analogWrite(G1, 0);
    analogWrite(B1, 0);
    
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
            analogWrite(R1, 255);
            analogWrite(G1, 255);
            analogWrite(B1, 255);
            analogWrite(R2, 255);
            analogWrite(G2, 255);
            analogWrite(B2, 255);
            break;
          case 1:
            // Sad
            Serial.print(" Sad\n");
            analogWrite(R1, 0);
            analogWrite(G1, 0);
            analogWrite(B1, 255);
            analogWrite(R2, 0);
            analogWrite(G2, 0);
            analogWrite(B2, 255);
            break;
          case 2:
            // Happy 
            Serial.print(" Happy");
            analogWrite(R1, 0);
            analogWrite(G1, 255);
            analogWrite(B1, 0);
            analogWrite(R2, 0);
            analogWrite(G2, 255);
            analogWrite(B2, 0);
            break;
          case 3:
            // Angry 
            Serial.print(" Angry\n");
            analogWrite(R1, 255);
            analogWrite(G1, 0);
            analogWrite(B1, 0);
            analogWrite(R2, 255);
            analogWrite(G2, 0);
            analogWrite(B2, 0);
            break;
          case 4:
            // idk whatever yellow is
            Serial.print(" Calm\n");
            analogWrite(R1, 200);
            analogWrite(G1, 200);
            analogWrite(B1, 0);
            analogWrite(R2, 200);
            analogWrite(G2, 200);
            analogWrite(B2, 0);
            break;
          default:
            // Other Emotion, testing for rn 
            Serial.print(" Other\n");
            analogWrite(R1, 255);
            analogWrite(G1, 255);
            analogWrite(B1, 255);
            analogWrite(R2, 255);
            analogWrite(G2, 255);
            analogWrite(B2, 255);
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
