#include <Wire.h>
#include <SPI.h>
#include <TinyScreen.h>

TinyScreen display = TinyScreen(TinyScreenPlus);

const int MAX_LINES = 12;
const int VISIBLE_LINES = 3; // Fits 3 rows at Y = 26, 38, 50

String expLines[MAX_LINES];
int expLineCount = 0;
int expScrollOffset = 0;

String obsLines[MAX_LINES];
int obsLineCount = 0;
int obsScrollOffset = 0;

String pageData[5] = {"", "", "", "", ""}; // [0]=Behavior, [3]=Conf, [4]=Data
int currentPage = 0;

bool isComplete = false;
uint8_t lastButtons = 0;

bool isAnalysing = false;
unsigned long lastAnimTime = 0;
int animFrame = 0;

// Splits text by exact pixel width so words never overflow the 96px screen
void wrapByPixelWidth(String text, String lines[], int &lineCount, int maxWidthPx) {
  display.setFont(liberationSans_8ptFontInfo);
  text.trim();
  lineCount = 0;

  while (text.length() > 0 && lineCount < MAX_LINES) {
    int spaceIdx = text.indexOf(' ');
    String firstWord = (spaceIdx == -1) ? text : text.substring(0, spaceIdx);

    // Fallback: if a single word is wider than the entire screen, split it cleanly
    char wordBuf[40];
    firstWord.toCharArray(wordBuf, 40);
    if (display.getPrintWidth(wordBuf) > maxWidthPx) {
      int splitPos = firstWord.length();
      while (splitPos > 1) {
        String sub = firstWord.substring(0, splitPos);
        sub.toCharArray(wordBuf, 40);
        if (display.getPrintWidth(wordBuf) <= maxWidthPx) break;
        splitPos--;
      }
      lines[lineCount++] = firstWord.substring(0, splitPos);
      text = text.substring(splitPos);
      text.trim();
      continue;
    }

    // Greedily pack words onto the current line while under maxWidthPx
    String currentLine = firstWord;
    String remaining = (spaceIdx == -1) ? "" : text.substring(spaceIdx + 1);
    remaining.trim();

    while (remaining.length() > 0) {
      int nextSpace = remaining.indexOf(' ');
      String nextWord = (nextSpace == -1) ? remaining : remaining.substring(0, nextSpace);
      String candidate = currentLine + " " + nextWord;

      char candBuf[64];
      candidate.toCharArray(candBuf, 64);
      if (display.getPrintWidth(candBuf) <= maxWidthPx) {
        currentLine = candidate;
        remaining = (nextSpace == -1) ? "" : remaining.substring(nextSpace + 1);
        remaining.trim();
      } else {
        break;
      }
    }

    lines[lineCount++] = currentLine;
    text = remaining;
  }
}

void setup() {
  SerialUSB.begin(9600);
  SerialUSB.setTimeout(50); // Prevents 1-second blocking stalls during serial reads
  while (!SerialUSB && millis() < 3000);

  Wire.begin();
  display.begin();
  display.setBrightness(10);
  display.fontColor(TS_8b_White, TS_8b_Black);
  drawStartupScreen();
}

void loop() {
  if (SerialUSB.available() > 0) {
    String raw = SerialUSB.readStringUntil('\n');
    raw.trim();

    if (raw == "ANALYSING") {
      isComplete = false;
      isAnalysing = true;
      animFrame = 0;
      lastAnimTime = millis();

      display.clearWindow(0, 0, 96, 64);
      display.setFont(liberationSans_10ptFontInfo);
      display.setCursor(14, 26);
      display.print("Analysing");
    } 
    else if (raw == "RESET") {
      isComplete = false;
      isAnalysing = false;
      currentPage = 0;
      expScrollOffset = 0;
      obsScrollOffset = 0;
      drawStartupScreen();
    }
    else if (raw.indexOf('|') > 0) {
      int firstPipe = raw.indexOf('|');
      int secondPipe = raw.indexOf('|', firstPipe + 1);
      int thirdPipe = raw.indexOf('|', secondPipe + 1);

      if (firstPipe > 0 && secondPipe > firstPipe && thirdPipe > secondPipe) {
        pageData[0] = raw.substring(0, firstPipe);

        // 1. Pixel-accurate wrap for Explanation (90px safe width)
        String fullExp = raw.substring(firstPipe + 1, secondPipe);
        wrapByPixelWidth(fullExp, expLines, expLineCount, 90);
        expScrollOffset = 0;

        // 2. Pixel-accurate wrap for Observations (90px safe width)
        String fullObs = raw.substring(secondPipe + 1, thirdPipe);
        wrapByPixelWidth(fullObs, obsLines, obsLineCount, 90);
        obsScrollOffset = 0;

        // 3. Reliability parsing
        String relField = raw.substring(thirdPipe + 1);
        relField.trim();
        int splitIdx = relField.indexOf('|');
        if (splitIdx < 0) splitIdx = relField.indexOf("Data:");

        if (splitIdx > 0) {
          pageData[3] = relField.substring(0, splitIdx);
          pageData[4] = relField.substring(splitIdx);
          pageData[3].trim();
          pageData[4].trim();
        } else {
          pageData[3] = relField;
          pageData[4] = "";
        }

        isComplete = true;
        isAnalysing = false;
        currentPage = 0;
        drawCurrentPage();
      }
    }
  }

  if (isAnalysing && !isComplete) {
    if (millis() - lastAnimTime > 500) {
      lastAnimTime = millis();
      animFrame = (animFrame + 1) % 4;

      display.clearWindow(0, 0, 96, 64);
      display.setFont(liberationSans_10ptFontInfo);
      display.setCursor(14, 26);
      display.print("Analysing");

      for (int i = 0; i < animFrame; i++) {
        display.print(".");
      }
    }
  }

  if (isComplete) {
    uint8_t buttons = display.getButtons();

    if (buttons && !lastButtons) {
      // Upper Right: Next category page
      if (buttons == TSButtonUpperRight) {
        currentPage = (currentPage + 1) % 3;
        expScrollOffset = 0;
        obsScrollOffset = 0;
        drawCurrentPage();
      }
      // Upper Left: Previous category page
      else if (buttons == TSButtonUpperLeft) {
        currentPage = (currentPage == 0) ? 2 : currentPage - 1;
        expScrollOffset = 0;
        obsScrollOffset = 0;
        drawCurrentPage();
      }
      // Bottom Right: Scroll down line-by-line (loops back to top if at bottom)
      else if (buttons == TSButtonLowerRight) {
        if (currentPage == 0 && expLineCount > VISIBLE_LINES) {
          if (expScrollOffset + VISIBLE_LINES < expLineCount) {
            expScrollOffset++;
          } else {
            expScrollOffset = 0;
          }
          drawCurrentPage();
        } 
        else if (currentPage == 1 && obsLineCount > VISIBLE_LINES) {
          if (obsScrollOffset + VISIBLE_LINES < obsLineCount) {
            obsScrollOffset++;
          } else {
            obsScrollOffset = 0;
          }
          drawCurrentPage();
        }
      }
      // Bottom Left: Reset TinyScreen+ AND signal Python/iOS to start a new session
      else if (buttons == TSButtonLowerLeft) {
        SerialUSB.println("RESET");
        isComplete = false;
        isAnalysing = false;
        currentPage = 0;
        expScrollOffset = 0;
        obsScrollOffset = 0;
        drawStartupScreen();
      }
    }
    lastButtons = buttons;
    delay(50);
  }
}

void drawStartupScreen() {
  display.clearWindow(0, 0, 96, 64);
  
  display.setFont(liberationSans_10ptFontInfo);
  display.setCursor(38, 10); 
  display.print("PET");
  delay(600); 
  
  display.setCursor(12, 26);
  display.print("BEHAVIOUR");
  delay(600); 
  
  display.setCursor(18, 42);
  display.print("ANALYSER");
  delay(2000); 
  
  display.clearWindow(0, 0, 96, 64);
  
  display.setFont(liberationSans_10ptFontInfo);
  display.setCursor(12, 12);
  display.print("Upload video");
  delay(600); 
  
  display.setCursor(28, 28);
  display.print("in app");
  delay(600); 
  
  display.setCursor(24, 44);
  display.print("to start");
}

void drawCurrentPage() {
  display.clearWindow(0, 0, 96, 64);

  // Top Header
  display.setFont(liberationSans_8ptFontInfo);
  display.setCursor(12, 1);
  display.print("CAT BEHAVIOUR");
  display.drawLine(2, 11, 94, 11, TS_8b_White);

  if (currentPage == 0) {
    display.setFont(liberationSans_10ptFontInfo);
    display.setCursor(16, 13);
    display.print(pageData[0]);

    display.setFont(liberationSans_8ptFontInfo);
    int yPos = 26;
    for (int i = 0; i < VISIBLE_LINES; i++) {
      int lineIdx = expScrollOffset + i;
      if (lineIdx < expLineCount) {
        display.setCursor(2, yPos);
        display.print(expLines[lineIdx]);
        yPos += 12;
      }
    }

    if (expScrollOffset + VISIBLE_LINES < expLineCount) {
      display.setCursor(88, 52);
      display.print("v");
    }
  }
  else if (currentPage == 1) {
    display.setFont(liberationSans_10ptFontInfo);
    display.setCursor(2, 13);
    display.print("OBSERVATIONS:");

    display.setFont(liberationSans_8ptFontInfo);
    int yPos = 26;
    for (int i = 0; i < VISIBLE_LINES; i++) {
      int lineIdx = obsScrollOffset + i;
      if (lineIdx < obsLineCount) {
        display.setCursor(2, yPos);
        display.print(obsLines[lineIdx]);
        yPos += 12;
      }
    }

    if (obsScrollOffset + VISIBLE_LINES < obsLineCount) {
      display.setCursor(88, 52);
      display.print("v");
    }
  }
  else if (currentPage == 2) {
    display.setFont(liberationSans_10ptFontInfo);
    display.setCursor(2, 13);
    display.print("RELIABILITY:");

    display.setFont(liberationSans_8ptFontInfo);
    display.setCursor(2, 28);
    display.print(pageData[3]);

    display.setCursor(2, 42);
    display.print(pageData[4]);
  }
}