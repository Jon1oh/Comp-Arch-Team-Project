#include <Wire.h>
#include <SPI.h>
#include <TinyScreen.h>

// Initialize the TinyScreen
TinyScreen display = TinyScreen(TinyScreenPlus);

// Global State Variables
String pageData[4] = {"", "", "", ""};
int currentPage = 0;
bool isComplete = false;
uint8_t lastButtons = 0;

// New Animation Variables
bool isAnalysing = false;
unsigned long lastAnimTime = 0;
int animFrame = 0;

void setup() {
  // Start USB Serial communication at 9600 baud
  Serial.begin(9600);

  // Initialize the OLED display
  Wire.begin();
  display.begin();
  
  // Configure screen settings
  display.setBrightness(10);
  display.setFont(liberationSans_10ptFontInfo); 
  display.fontColor(TS_8b_White, TS_8b_Black); 
  
  // Draw the initial animated standby screen
  drawStartupScreen();
}

void loop() {
  // 1. Listen for new data from Python
  if (Serial.available() > 0) {
    String raw = Serial.readStringUntil('\n');
    raw.trim();
    
    if (raw == "ANALYSING") {
      isComplete = false;
      isAnalysing = true;
      animFrame = 0;
      lastAnimTime = millis();
    } 
    else {
      // Parse the 4 pieces of data separated by '|'
      int firstPipe = raw.indexOf('|');
      int secondPipe = raw.indexOf('|', firstPipe + 1);
      int thirdPipe = raw.indexOf('|', secondPipe + 1);
      
      if (firstPipe > 0) {
        pageData[0] = raw.substring(0, firstPipe);
        pageData[1] = raw.substring(firstPipe + 1, secondPipe);
        pageData[2] = raw.substring(secondPipe + 1, thirdPipe);
        pageData[3] = raw.substring(thirdPipe + 1);
        
        isComplete = true;
        isAnalysing = false; // Turn off the analyzing animation
        currentPage = 0;
        drawCurrentPage();
      }
    }
  }

  // 2. Animate the "Analysing..." screen smoothly while waiting
  if (isAnalysing) {
    if (millis() - lastAnimTime > 500) { // Update every 500 milliseconds
      lastAnimTime = millis();
      animFrame = (animFrame + 1) % 4; // Cycles through 0, 1, 2, 3
      
      display.clearWindow(0, 0, 96, 64);
      display.setCursor(16, 26);
      display.print("Analysing");
      
      // Add the animated dots based on the current frame
      for(int i = 0; i < animFrame; i++) {
        display.print(".");
      }
    }
  }

  // 3. Listen for physical button presses
  if (isComplete) {
    uint8_t buttons = display.getButtons();
    
    // If a button is pressed and it wasn't pressed in the last loop
    if (buttons && !lastButtons) {
      
      if (buttons == TSButtonUpperRight) {
        // TOP RIGHT: Go to next page
        currentPage = (currentPage + 1) % 3; 
        drawCurrentPage();
      }
      else if (buttons == TSButtonUpperLeft) {
        // TOP LEFT: Go to previous page
        if (currentPage == 0) {
          currentPage = 2; // Loop backward from Main Analysis to Reliability
        } else {
          currentPage--;   
        }
        drawCurrentPage();
      }
      else if (buttons == TSButtonLowerRight) {
        // BOTTOM RIGHT: Jump straight back to Main Analysis (Page 0)
        currentPage = 0;
        drawCurrentPage();
      }
      else if (buttons == TSButtonLowerLeft) {
        // BOTTOM LEFT: End session and return to Home Screen
        isComplete = false; 
        isAnalysing = false;
        currentPage = 0;
        drawStartupScreen();
      }
      
    }
    lastButtons = buttons;
    delay(50); // Small delay to prevent button bouncing
  }
}

void drawStartupScreen() {
  display.clearWindow(0, 0, 96, 64);
  
  // Phase 1: Title Reveal (Split into 3 lines to fit the 96px width perfectly)
  
  // Center "PET"
  display.setCursor(38, 10); 
  display.print("PET");
  delay(600); 
  
  // Center "BEHAVIOUR"
  display.setCursor(12, 26);
  display.print("BEHAVIOUR");
  delay(600); 
  
  // Center "ANALYSER"
  display.setCursor(18, 42);
  display.print("ANALYSER");
  delay(2000); // Hold for 2 seconds to let the user read it
  
  
  // Phase 2: Instruction Reveal (Centered X-coordinates)
  display.clearWindow(0, 0, 96, 64);
  
  // Center "Upload video"
  display.setCursor(12, 12);
  display.print("Upload video");
  delay(600); 
  
  // Center "in app"
  display.setCursor(28, 28);
  display.print("in app");
  delay(600); 
  
  // Center "to start"
  display.setCursor(24, 44);
  display.print("to start");
}

void drawCurrentPage() {
  display.clearWindow(0, 0, 96, 64);

  // Universal Top Header
  display.setCursor(8, 0); 
  display.print("CAT BEHAVIOUR");
  display.drawLine(4, 12, 92, 12, TS_8b_White); // Top separator line

  // Page Specific Content
  if (currentPage == 0) {
    // MAIN SCREEN: Alert + Reasoning
    display.setCursor(24, 16); 
    display.print(pageData[0]); // The Short Label (e.g., "ALERT")
    
    display.setCursor(0, 28);
    display.print(pageData[1]); // The Explanation

    display.drawLine(4, 50, 92, 50, TS_8b_White); // Bottom separator line
    
    display.setCursor(12, 54);
    display.print("Analysis complete");
  }
  else if (currentPage == 1) {
    // PAGE 2: Observations
    display.setCursor(0, 16);
    display.print("OBSERVATIONS:");
    
    display.setCursor(0, 28);
    display.print(pageData[2]);
  }
  else if (currentPage == 2) {
    // PAGE 3: Reliability
    display.setCursor(0, 16);
    display.print("RELIABILITY:");
    
    display.setCursor(0, 28);
    display.print(pageData[3]);
  }
}