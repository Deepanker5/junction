Tattoo/Marker Gate — Visual Factor for 2-Step Override (Raspberry Pi)
=====================================================================

Introduction
-------------

This module (my contribution to https://github.com/DaveeHorvath/hct_0) provides the **visual first factor** for an override system: it enables “manual control” **only** when a user-specific pattern is detected. The **second factor** (IR remote) is accepted **only after** this gate turns **ON**.

*   Offline, on-device; designed for **Raspberry Pi 4**
    
*   Two modes: **ORB** (custom tattoo/pattern) or **ArUco/AprilTag** (requires contrib build)
    
*   Built-in **debounce** (N consecutive hits + grace period)
    
*   Optional live preview; headless friendly (state printed to stdout)
    

Install
-------

python3 detector.py --ref tattoo_ref.jpg --display

* ArUco (requires contrib; pick a dict your build has)

* *  python3 detector.py --aruco 4X4_50 --display

* * *  Headless (logs state flips)

* *  python3 detector.py --ref tattoo_ref.jpg

* * The banner + logs show:

* * manual control on → visual factor satisfied

* *  manual control off → gated

Press q to quit (when --display is used).

Quick Start
-----------

ORB (custom reference image)

python3 detector.py --ref tattoo_ref.jpg --display

* ArUco (requires contrib; pick a dict your build has)

* * python3 detector.py --aruco 4X4_50 --display

* * * Headless (logs state flips)

* * python3 detector.py --ref tattoo_ref.jpg

* * The banner + logs show:

* * manual control on → visual factor satisfied

* * manual control off → gated

Press q to quit (when --display is used).
    

IR Integration (Second Factor)
------------------------------

Gate logic (this module) → ON → accept IR press → perform override.

* Minimal pattern:

* * if gate_is_on() and ir_button_pressed("OK"):

* * perform_override()

* * Implement gate_is_on() by:

Parsing this program’s stdout for the latest state, or

* Writing/reading a small flag file/pipe, or

Importing the detector and sharing state in-process (advanced).

Tuning Tips
-----------

*   Use a **sharp, high-contrast** reference for ORB; crop to the distinctive tattoo area.
    
*   Increase --hits for stricter confirmation; increase --grace to linger briefly.
    
*   Lower --width/--height for higher FPS on the Pi.
    
  `

Troubleshooting
---------------

*   **\[ERROR\] No detector configured.** → pass --ref or --aruco.
    
*   **cv2.aruco not available** → install opencv-contrib-python (version must match cv2).
    
*   **Flaky ORB detection** → better reference image, steadier distance/lighting, more texture.
    
*   **Camera open fails** → verify /dev/video\*; Pi builds may need V4L2 (the code retries).
    

Security Notes
--------------

*   Not a biometric; treat as a **local gate** only.
    
*   Keep reference images private.
    
*   Fail-safe: default to **off** if the process dies or detection fails.
    

License
-------

**MIT License** — see LICENSE in the repository.