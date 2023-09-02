#define BLYNK_PRINT Serial
#include <ESP8266WiFi.h>
#include <BlynkSimpleEsp8266.h>
#include "Blynk.h"
///////////////////
#define BLYNK_TEMPLATE_ID "TMPL3SLBxYfNt"
#define BLYNK_TEMPLATE_NAME "ROBOT"
#define BLYNK_AUTH_TOKEN "HyyC3ljJw_8Wu9bKerNBDRoKKDQKYiLC"
///////////////////
char ssid[ ] = "OnePlus NordCE 5G";
char pass[ ] = "vyshnavp";
char auth[ ] ="HyyC3ljJw_8Wu9bKerNBDRoKKDQKYiLC";
///////////////////
#define in1 D1
#define in2 D2
#define in3 D5
#define in4 D6
#define in5 D0
#define in6 D7


///////////////////
int f,r,l,b;
//////////////////
void setup() 
{
 Serial.begin(9600);
 //////////////////////////
 Blynk.begin(auth, ssid, pass, "blynk.cloud", 80);
 /////////////////////////
 pinMode(in1,OUTPUT);
 pinMode(in2,OUTPUT);
 pinMode(in3,OUTPUT);
 pinMode(in4,OUTPUT);
 pinMode(in5,OUTPUT);
 pinMode(in6,OUTPUT);
 digitalWrite(in6,HIGH);
 digitalWrite(in5,HIGH);
 digitalWrite(in1,HIGH);
 digitalWrite(in2,HIGH);
 digitalWrite(in3,HIGH);
 digitalWrite(in4,HIGH);

}

void loop()
{
  Blynk.run();
  //////////////
  if(f==1)
  {
    digitalWrite(in1,LOW);
    digitalWrite(in2,HIGH);
    delay(2000);
    digitalWrite(in3,LOW);
    digitalWrite(in4,HIGH);
   
    
  }
  if(f==0)
  {
    digitalWrite(in1,HIGH);
    digitalWrite(in3,HIGH); 
  }
  //////////////////////////

  if(r==1)
  {
    digitalWrite(in5,LOW);
    digitalWrite(in6,HIGH); 
  }
  if(r==0)
  {
    digitalWrite(in5,HIGH);
  }
  ////////////////////////////
  if(l==1)
  {
    digitalWrite(in5,HIGH);
    digitalWrite(in6,LOW); 
  }
  if(l==0)
  {
    digitalWrite(in6
    ,HIGH); 
  }
  //////////////////////////
  if(b==1)
  {
    digitalWrite(in3,HIGH);
    digitalWrite(in4,LOW);
    delay(2000);
    digitalWrite(in1,HIGH);
    digitalWrite(in2,LOW); 
  }
  if(b==0)
  {
  
    digitalWrite(in2,HIGH);
   
    digitalWrite(in4,HIGH);  
  }

}
///////////////////////
BLYNK_CONNECTED() 
{
  // Request the latest state from the server
  Blynk.syncVirtual(V0,V1,V2,V3);
}

BLYNK_WRITE(V0) 
{
f=param.asInt();   
}

BLYNK_WRITE(V1) 
{
r=param.asInt();   
}

BLYNK_WRITE(V2) 
{
l=param.asInt();   
}

BLYNK_WRITE(V3) 
{
b=param.asInt();   
}

///////////////////////
