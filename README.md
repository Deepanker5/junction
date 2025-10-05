Tattoo/Marker Gate — Visual Factor for 2-Step Override (Raspberry Pi)
=====================================================================

This module (my contribution to **hct\_0**) provides the **visual first factor** for an override system: it enables “manual control” **only** when a user-specific pattern is detected. The **second factor** (IR remote) is accepted **only after** this gate turns **ON**.

*   Offline, on-device; designed for **Raspberry Pi 4**
    
*   Two modes: **ORB** (custom tattoo/pattern) or **ArUco/AprilTag** (requires contrib build)
    
*   Built-in **debounce** (N consecutive hits + grace period)
    
*   Optional live preview; headless friendly (state printed to stdout)
    

Install
-------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python3 -m venv .venv && source .venv/bin/activate  # Choose ONE:  pip install numpy opencv-python            # ORB only  pip install numpy opencv-contrib-python    # ORB + ArUco/AprilTag   `

> AprilTag dictionaries appear only if your contrib build includes them.

Quick Start
-----------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # ORB (reference image of tattoo/pattern)  python3 detector.py --ref tattoo_ref.jpg --display  # ArUco (requires contrib; pick a dict your build has)  python3 detector.py --aruco 4X4_50 --display  # Headless (log state flips)  python3 detector.py --ref tattoo_ref.jpg   `

Banner + logs:

*   manual control on → visual factor satisfied
    
*   manual control off → gatedPress q to quit when using --display.
    

CLI Options
-----------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   --camera     Webcam index (0)  --width      Capture width (640)  --height     Capture height (480)  --display         Show preview window  --ref       Reference image for ORB  --aruco     ArUco dict (e.g., 4X4_50, 5X5_100, APRILTAG_16h5*)  --hits       Frames required for ON (5)  --grace      Frames to keep ON after last hit (15)   `

\* Only if present in your OpenCV contrib build.

IR Integration (Second Factor)
------------------------------

Gate → **ON** → accept IR press → perform override.

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   if gate_is_on() and ir_button_pressed("OK"):      perform_override()   `

Implement gate\_is\_on() by parsing this program’s stdout, using a small flag file/pipe, or importing the detector (in-process).

Tuning Tips
-----------

*   Use a **sharp, high-contrast** reference for ORB; crop to the distinctive tattoo area.
    
*   Increase --hits for stricter confirmation; increase --grace to linger briefly.
    
*   Lower --width/--height for higher FPS on the Pi.
    

Run as a Service (optional)
---------------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # /etc/systemd/system/visual-gate.service  [Unit]  Description=Visual Gate Detector  After=network.target  [Service]  Type=simple  User=pi  WorkingDirectory=/home/pi/hct_0  Environment=PYTHONUNBUFFERED=1  ExecStart=/home/pi/hct_0/.venv/bin/python detector.py --ref /home/pi/tattoo_ref.jpg  Restart=on-failure  [Install]  WantedBy=multi-user.target   `

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   sudo systemctl daemon-reload  sudo systemctl enable --now visual-gate  journalctl -u visual-gate -f   `

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