# App Progress Archive — Workout Form Analyzer

A running record of fixes, test results and open improvements for the live app at
https://pose-detection-tanhp2dwvhvyrrsqfvtrny.streamlit.app/

Each entry has a plain-language summary first, then technical detail for the record.

---

## 2026-09-26 — App brought back online and made to work end to end

### Where things started
- The live app would not start. Streamlit's servers had moved to a newer version of Linux and could not install two system parts the image library needs.
- The newer version of the app (rep counting, form scoring, workout replay) had existed only on the laptop since January 2026. GitHub still held the September 2025 skeleton-only version.

### Fixes made

| # | Problem (plain language) | Fix | Commit |
|---|---|---|---|
| 1 | App would not start: a system part it needs had been renamed on Streamlit's servers | Updated the server "shopping list" to the new name | `630e86f` |
| 2 | Newer app version was only on the laptop, with no backup | Uploaded it to GitHub; the two workout videos are kept local only | `630e86f` |
| 3 | The replay's Play button would crash | It was handed the whole recorded record instead of the picture inside it; now passes the picture | `630e86f` |
| 4 | App still would not start: a second system part was missing | Added it to the server "shopping list" | `93e565a` |
| 5 | Pressing STOP threw the recording away, so the summary said "Start the camera first" | The app now keeps the recording after STOP; Reset clears it; clearer message when nothing is recorded | `98346fb` |
| 6 | A good, thighs-level squat scored "Poor" while the tip said "Good form" | Ideal squat depth now matches how the app measures the knee (parallel ≈ 90°) | `98346fb` |
| 7 | On phones the video was cut off and the STOP button was hidden | Gave the video player a fixed height so it and its buttons always fit | `5d6d6dd` |
| 8 | The summary always showed 0 reps | The rep counter timed the pause at the bottom using the wall clock, but the recording is replayed in a split second. Each frame now carries the time it was captured | `7168363` |

**Technical detail**
- 1 & 4: `packages.txt` is now `libgl1` and `libglib2.0-0t64`. `libgl1-mesa-glx` was dropped in Debian 13 "trixie". `ultralytics` pulls in full `opencv-python`, which needs `libGL.so.1` and `libgthread-2.0.so.0`.
- 2: `.gitignore` now excludes `*.mp4`. Model weights (`*.pt`) were already ignored; Ultralytics downloads `yolov8n-pose.pt` at runtime.
- 3: `app.py` Play loop used `recorded_frames[i]` (a dict) instead of `recorded_frames[i]['frame']`.
- 5: while streaming, the processor is kept in `st.session_state.last_processor`; `active_processor = webrtc_ctx.video_processor or st.session_state.get("last_processor")` drives Show, Close and the summary. Reset pops it.
- 6: `exercise_templates.py` `knee_angle.at_bottom` changed from `(35, 70)` to `(70, 100)`. `calculate_angle` returns the interior hip-knee-ankle angle.
- 7: `webrtc_streamer(..., video_html_attrs={"style": {"width": "100%", "height": "400px", "objectFit": "contain"}, "controls": False, "autoPlay": True})`.
- 8: `RepCounter.update()` takes an optional `timestamp` (defaults to `time.time()` for live use). `recv` stores `'time'` with each recorded frame; `analyze_recorded_frames()` passes it.

### Test results (Basheer, Android phone, live app)
- Camera, live video and stick-figure overlay: working.
- Front-on test: 5 squats, 43 frames, **1 rep counted**, knee angle 24° (unrealistic — front view distorts the knee angle).
- Side-on test (right side to camera): 5 squats, **4 reps counted**, knee angles 59°–94° (realistic). Scores: 80%, 100%, 53%, 67%.
- Laptop on home network: live video would not connect ("Connection is taking longer than expected"). Worked on phone mobile data.
- The app was temporarily CPU-throttled by Streamlit during the day's heavy testing.

---

## Open improvements (parked)

| Priority | Issue (plain language) | Suggested fix |
|---|---|---|
| High | **Occasional missed reps.** The app takes about 3 snapshots a second; a quick bottom position can fall between two snapshots | Take snapshots more often (e.g. 5 a second), or check squat stages on every live frame while still saving only 3 a second for replay |
| High | **Scores jump around for similar squats.** Each rep's score averages the whole movement, including the way down and up | Score only the bottom of each squat |
| Medium | **Some networks block the live video** (e.g. office networks) | Add a relay (TURN) service; Basheer signs up with a provider and adds the keys to the app's Streamlit settings |
| Medium | **Front-on view gives wrong knee angles** | Ignore frames where the leg looks too short (front-on or hidden), or use whichever leg the camera sees more clearly instead of always the right leg |
| Low | **Heavy computing use** can get the app throttled | Run the pose model on every other frame and reuse the last result for drawing in between |
| Low | **Outdated on-screen text.** The sidebar says stats appear on the live video, the help text mentions a coloured skeleton, and the "~N seconds" figure uses the wrong speed | Update the text to match what the app does |
| Low | **The offline test script** (`debug_video.py`) has the same timing flaw fixed in #8 | Pass frame timestamps there too |

**Technical detail**
- Missed reps: `reached_bottom` (100°) is only checked on recorded frames (≥0.33 s apart).
- Score variance: `current_rep_scores` includes descent/ascent frames (100° < knee < 160°), which are scored against `at_bottom` and score low.
- Front-on: `calculate_angle` works on 2D image coordinates; the analyser always uses right-side keypoints (6, 12, 14, 16).
- Network: `RTC_CONFIGURATION` has STUN only (`stun.l.google.com:19302`).
- Memory: frames are about 460 KB each (320×480×3) at 3 FPS, roughly 80 MB per minute, held until Reset or the session ends.
