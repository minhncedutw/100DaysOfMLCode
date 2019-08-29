# 100DaysOfTechLearning

#### --------------------------------------------------
## Day 1(2019Aug01): Watch [How to Build a Retail Startup](https://www.youtube.com/watch?v=fF6f0nzlfUA)(Siraj Raval)

>**Goal:** study the system of Tools and Techniques to build an AI app.

**Main idea:** to build a retail startup of designing sneakers creatively.

**Main techniques:**
 - Deep Learning API
 - Generative Adversarial Network
 - OpenCV
 
**Used tools:**
 - Flutter
 - Firebase
 - Tensorflow
 - OpenCV
 - Flask
 - Paypal
 - Spotify


#### --------------------------------------------------
## Day 2(2019Aug02): Continue [How to Build a Retail Startup](https://www.youtube.com/watch?v=fF6f0nzlfUA) - Install Android Studio & Visual Studio Android Emulator(without HAXM)

>**Trouble & Goal:** Existed docker software requires Windows Hyper-V to be on; 
Android Studio Emulator requires Hyper-V to be off => conflict => fail to install.
Goal is to Install softwares and to Build a sample Android app/project. 

**Solution:** use Visual Studio Android Emulator instead.

**Install steps:**
 - [Download](https://developer.android.com/studio) & Install Android Studio. Android Studio will be ok. Just Android Emulator will be fail(HAXM conflicts to Hyper-V).
 - [Download](https://visualstudio.microsoft.com/vs/community/) & Install Visual Studio Community 2019(install Mobile **Development with .NET** workload only).

**Run steps:**
 - Open Visual Studio Community as Admin permission.
 - Open **Tools** -> **Android** -> **Android Device Manager**. 
 Click **New** and config standard creation to create an Android device.
 - After creation, Select the created device and click **Run**.
 - After device runs and displays fine, Open Android Studio and create a basic project/app.
 - Run/Debug the created Android project, a 'device selection' will show up, just select the running device.


#### --------------------------------------------------
## Day 3(2019Aug03): Continue [How to Build a Retail Startup](https://www.youtube.com/watch?v=fF6f0nzlfUA) - Install Flutter and Create new Flutter project

>**Goal:** Install and build a sample Flutter project. Build the project [TheGorgeousLogin](https://github.com/huextrat/TheGorgeousLogin)

**Install steps:**
 - [Download](https://flutter.dev/docs/get-started/install) & Extract Flutter to somewhere(such as: `E:/WORKSPACES/flutter`).
 - Add path of flutter(such as: `E:/WORKSPACES/flutter/bin`) to **user `path`** of environment.
 - In Android Studio, open **File** -> **Settings** -> **Plugins**. 
 Then install **Dart** and **Flutter**.
 - Restart Android Studio.
 
**Create Flutter project steps:**
 - Open Visual Studio Community as Admin permission, and `continue without code >>`.
 - Open **Tools** -> **Android** -> **Android Device Manager**. 
 Select a device and click **Run**. 
 *(Must complete this step before create or load flutter project, to let project detect device)*
 - Open Android Studio. In Welcome windows, Select `Start a new Flutter project`.
 - Select `Flutter Application`. 
 Add Flutter path(such as: `E:/WORKSPACES/flutter`) to `Flutter SDK path`. 
 *(Select project location that is not in Flutter path)*
 - Finish and Run project.

**Load Flutter project steps:** *([TheGorgeousLogin](https://github.com/huextrat/TheGorgeousLogin))*
 - Open Visual Studio Community as Admin permission, and `continue without code >>`.
 - Open **Tools** -> **Android** -> **Android Device Manager**. Select a device and click **Run**. *(Must complete this step before create or load flutter project, to let project detect device)*
 - Open Android Studio. In Welcome windows, Load the downloaded project of **TheGorgeousLogin**.
 - Click at `Run 'flutter packages get'` to download dependencies.
 - Run project.
 
 
#### --------------------------------------------------
## Day 4(2019Aug04): Install & Connect Genymotion emulator with Android Studio

>**Trouble:** VS Android Emulator is too slow, and takes too much memory usage. 
Want to try another lighter and faster emulator.

**Install & Config**
 - [Download](https://www.virtualbox.org/wiki/Downloads) & Install Virtual Box newest version.
 - [Download](https://www.genymotion.com/fun-zone/) & Install `Genymotion Personal Edition without VirtualBox` version.
 - [Register](https://www.genymotion.com) & Login Genymotion software(select Personal Use).
 - Open Android Studio, Open **File** -> **Settings...** -> **Plugins**, and Install Genymotion.
 - Restart Android Studio, then Open **File** -> **Settings...**. In **Genymotion** tab, paste the location of installed Genymotion. *(such as: C:\Program Files\Genymobile\Genymotion)*
 - Open Android Studio, Open **Tool** -> **SDK Manager**. Copy **Android SDK Location**. *(smt like: C:\Users\minhnc\AppData\Local\Android\Sdk)*
 - Open Genymotion, Open **Genymotion** -> **Settings**. 
In **ADB** tab, Switch to option **Use custom Android SDK Tools**, 
then paste the **Android SDK Location** obtained from previous step to **Android SDK** of Genymotion.
 - Open Genymotion, create a new virtual device.
 - In Android Studio, click **Search Everywhere** button *(button next to **SDK Manager**)*, and type Genymotion.
 - Click **Genymotion Device Manager**, there will be a list of created Genymotion devices.
Start a device.
 - Run Android Studio debug.

>**Result:** AS Emulator consume 2% memory usage less than VS Emulator. Genymotion Emulator consume 2% memory usage less than AS Emulator.
But Genymotion Emulator runs from beginning, AS & VS Emulator continue previous running, hence Genemotion Emulator start time is a bit longer than the other two.


#### --------------------------------------------------
## Day 5(2019Aug05): Continue [Install Android Studio & Visual Studio Android Emulator] - with HAXM

**Trouble:** normal Visual Studio Android Emulator without HAXM is **super slow**. 
The reason of not able to install HAXM is that Hyper-V is on.

**Install & Run**
- Disable Hyper-V in `Turn Windows features on or off`
- Run VS Androi Emulator
- An notification window that suggest/ask to install HAXM or run anyway. Select `Install HAXM`
- Restart and Enjoy it.


#### --------------------------------------------------
## Day 6(2019Aug06): Try Android Studio Emulator

>**Goal:** try Install and Use Emulator of Android Studio. Check whether it is lighter than VS Android Emulator.

**Install and Run**
 - Open AVD Manager, Install HAXM if it is not installed yet.
 - Click **Create Virtual Device...** to create new device, or/then Select a created device from list and Click **Run this AVD in the Emulator**.
 - Run debugging.

>**Result:** Android Studio Emulator memory usage consuming is equal to VS Android Emulator(just 1% less than), but it seems to run smoother, but less delay.
It seems that there is a small bug: when AS debug flutter project on virtual phone created by AS Emulator, the emulator scene is black, it has to click triagle button(back button) to show the built app.
But to debug on virtual phone created by VS Emulator, there is no black scene bug.


#### --------------------------------------------------
## Day 7(2019Aug07): Use both Hyper-V & HAXM

>**Trouble & Goal:** Docker requires Hyper-V to be enable. Emulators need HAXM, and HAXM requires Hyper-V to be disable.
The goal is to use both Docker and Emulators.

**Install**
 - In **Turn Features on or off**, enable **Hyper-V**.
 - Open `cmd.exe` with Admin permission.
 - In CMD type: `bcdedit /set  {current} description "Windows 10 Hyper-V on"`.
 - Then type: `bcdedit /copy {current} /d "Windows 10 Hyper-V off"`.
 - Afterward type: `bcdedit /set {ID} hypervisorlaunchtype off`.
 - Reboot & Enjoy. *(before login, it is able to switch Hyper-V on or off to use Docker or Emulators)*


#### --------------------------------------------------
## Day 8(2019Aug08): 


#### --------------------------------------------------
## Day 9(2019Aug09): 


#### --------------------------------------------------
## Day 10(2019Aug10): 


#### --------------------------------------------------
## Day 11(2019Aug11): 


#### --------------------------------------------------
## Day 12(2019Aug12): 


#### --------------------------------------------------
## Day 13(2019Aug13): 


#### --------------------------------------------------
## Day 14(2019Aug14): 


